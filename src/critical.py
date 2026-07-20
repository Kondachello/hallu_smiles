"""Claim-level evidence checks used by the ``support-critical`` detector.

This module intentionally does not alter the historical relation verifier.
It adds a separate, versioned closed-world protocol in which a clear factual
claim that has no direct textual support is ``unsupported`` rather than the
legacy verifier's catch-all ``unknown``.  Every LLM-backed component is cached
independently so a completed critical run can be replayed fully offline.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from .cache import CacheOnlyMissError, config_value, llm_runtime_fingerprint
from .config import resolve_api_key
from .dspy_adapter import (
    StructuredOutputParseError,
    StructuredOutputSchemaError,
    is_retryable_llm_exception,
    json_schema_response_format,
    strict_json_loads,
    structured_output_settings,
    validate_json_document,
)
from .matching import Embedder, normalize
from .verifier import EvidenceSpan, _sentences


CRITICAL_VERDICTS = frozenset({"entailed", "unknown", "unsupported", "contradicted"})
CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 1},
                },
                "required": ["text", "start", "end"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}
VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": sorted(CRITICAL_VERDICTS)},
    },
    "required": ["verdict"],
    "additionalProperties": False,
}

_WORD_RE = re.compile(r"[\w]+", re.UNICODE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "did",
    "do", "does", "for", "from", "has", "have", "had", "in", "is", "it", "of",
    "on", "or", "the", "to", "was", "were", "with",
}


class CriticalProtocolError(RuntimeError):
    """Raised when a critical component cannot produce its strict artifact."""


@dataclass(frozen=True)
class AtomicClaim:
    text: str
    start: int
    end: int
    sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "start": self.start, "end": self.end, "sources": list(self.sources)}


@dataclass(frozen=True)
class CriticalVerdict:
    verdict: str
    evidence: tuple[EvidenceSpan, ...]
    cache_hit: bool = False

    def __post_init__(self) -> None:
        if self.verdict not in CRITICAL_VERDICTS:
            raise ValueError(f"unsupported critical verdict {self.verdict!r}")


def claim_risk(verdict: str | None, unknown_risk: float) -> float:
    """Map a final claim verdict to the critical detector's fixed risk scale."""
    if verdict == "entailed":
        return 0.0
    if verdict in {"unsupported", "contradicted"}:
        return 1.0
    if verdict == "unknown":
        return float(unknown_risk)
    # Ungrounded graph edges are strict candidates without a textual verdict.
    return 1.0


def merge_claims(*collections: Iterable[AtomicClaim]) -> list[AtomicClaim]:
    """Deduplicate exact response spans while retaining candidate provenance."""
    merged: dict[tuple[int, int, str], AtomicClaim] = {}
    for collection in collections:
        for claim in collection:
            key = (int(claim.start), int(claim.end), normalize(claim.text))
            if not key[2]:
                continue
            previous = merged.get(key)
            if previous is None:
                merged[key] = claim
                continue
            merged[key] = replace(
                previous,
                sources=tuple(sorted(set(previous.sources) | set(claim.sources))),
            )
    return sorted(merged.values(), key=lambda c: (c.start, c.end, normalize(c.text)))


def _validate_claims(payload: Any, response: str, source: str) -> list[AtomicClaim]:
    try:
        validate_json_document(payload, CLAIM_SCHEMA)
    except StructuredOutputSchemaError:
        raise
    claims: list[AtomicClaim] = []
    for raw in payload["claims"]:
        start, end, text = int(raw["start"]), int(raw["end"]), str(raw["text"])
        if not (0 <= start < end <= len(response)):
            raise StructuredOutputParseError(f"{source} claim offsets are outside the response")
        if response[start:end] != text:
            raise StructuredOutputParseError(
                f"{source} claim text does not exactly equal response[start:end]"
            )
        claims.append(AtomicClaim(text=text, start=start, end=end, sources=(source,)))
    return merge_claims(claims)


def _response_field(value: Any, field: str) -> Any:
    return value.get(field) if isinstance(value, dict) else getattr(value, field, None)


class _CachedComponent:
    """Common cache, transport, and retry contract for critical components."""

    component: str = "critical_component"
    protocol: str = "support-critical-v1"

    def __init__(self, cfg, section_name: str, usage=None, *, cache_only: bool = False):
        critical_cfg = getattr(cfg, "support_critical", None)
        if critical_cfg is None:
            raise ValueError("support_critical config is required")
        section = getattr(critical_cfg, section_name, None)
        if section is None:
            raise ValueError(f"support_critical.{section_name} config is required")
        self.cfg = cfg
        self.section = section
        self.model = cfg.llm.model
        self.api_base = getattr(cfg.llm, "api_base", None)
        self.structured_output = structured_output_settings(cfg.llm)
        self.temperature = float(cfg.llm.temperature)
        self.max_tokens = int(config_value(section, "max_tokens", 512))
        self.max_retries = int(cfg.llm.max_retries)
        self.backoff_base = float(cfg.llm.retry_backoff_base_s)
        self.backoff_max = float(getattr(cfg.llm, "retry_backoff_max_s", 60))
        self.request_timeout_s = float(getattr(cfg.llm, "request_timeout_s", 90))
        self.prompt_version = str(config_value(section, "prompt_version", "v1"))
        self.cache_dir = Path(str(config_value(section, "cache_dir")))
        self.cache_only = bool(cache_only)
        self.usage = usage
        if self.max_tokens <= 0 or self.max_retries <= 0 or self.request_timeout_s <= 0:
            raise ValueError(f"invalid {self.component} runtime limits")
        if self.backoff_max < self.backoff_base:
            raise ValueError("llm.retry_backoff_max_s must be at least retry_backoff_base_s")
        if not self.cache_only:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, payload: dict[str, Any]) -> str:
        envelope = {
            "protocol": self.protocol,
            "component": self.component,
            "prompt_version": self.prompt_version,
            "max_tokens": self.max_tokens,
            "llm": llm_runtime_fingerprint(self.cfg),
            "api_base": self.api_base,
            "payload": payload,
        }
        return hashlib.sha256(
            json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _load(self, key: str) -> dict[str, Any] | None:
        try:
            payload = json.loads((self.cache_dir / f"{key}.json").read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            return None

    def _save(self, key: str, payload: dict[str, Any]) -> None:
        dest = self.cache_dir / f"{key}.json"
        tmp = dest.with_name(f"{key}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, dest)

    def _record(self, key: str, elapsed: float, cached: bool) -> None:
        if self.usage is not None:
            self.usage.record_call(self.component, key, elapsed, cached=cached)

    def _call_json(self, messages: list[dict[str, str]], schema: dict[str, Any], name: str) -> dict[str, Any]:
        try:
            from litellm import completion  # type: ignore
        except Exception as exc:  # pragma: no cover - live-only dependency
            raise CriticalProtocolError("litellm is required for critical verification") from exc
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.request_timeout_s,
            "num_retries": 0,
        }
        api_key = resolve_api_key(self.cfg)
        if api_key:
            kwargs["api_key"] = api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.structured_output.transport == "response_format":
            kwargs["response_format"] = json_schema_response_format(schema, name=name)
            if self.structured_output.request_backend is not None:
                kwargs["extra_body"] = {
                    "guided_decoding_backend": self.structured_output.request_backend
                }
        elif self.structured_output.transport == "guided_json":
            kwargs["extra_body"] = {"guided_json": schema}
        response = completion(**kwargs)
        choices = _response_field(response, "choices")
        if not isinstance(choices, (list, tuple)) or len(choices) != 1:
            raise StructuredOutputParseError(f"{self.component} completion must contain one choice")
        choice = choices[0]
        if _response_field(choice, "finish_reason") != "stop":
            raise StructuredOutputParseError(f"{self.component} completion did not finish cleanly")
        content = _response_field(_response_field(choice, "message"), "content")
        if not isinstance(content, str):
            raise StructuredOutputParseError(f"{self.component} response has no text content")
        return strict_json_loads(content.strip(), label=f"{self.component} response")

    def _retry_json(self, messages: list[dict[str, str]], schema: dict[str, Any], name: str) -> dict[str, Any]:
        result: dict[str, Any] | None = None
        for attempt in Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=self.backoff_base, max=self.backoff_max),
            retry=retry_if_exception(is_retryable_llm_exception),
            reraise=True,
        ):
            with attempt:
                result = self._call_json(messages, schema, name)
        assert result is not None
        return result


class AtomicClaimExtractor(_CachedComponent):
    component = "critical_claim_extractor"
    protocol = "support-critical-claims-v1"

    def __init__(self, cfg, usage=None, *, cache_only: bool = False):
        super().__init__(cfg, "claim_extractor", usage, cache_only=cache_only)

    def extract(self, response: str) -> list[AtomicClaim]:
        if not response.strip():
            return []
        key = self._cache_key({"response": response})
        cached = self._load(key)
        if cached is not None:
            claims = _validate_claims(cached, response, "atomic")
            self._record(key, 0.0, cached=True)
            return claims
        if self.cache_only:
            raise CacheOnlyMissError(self.component, key, self.cache_dir / f"{key}.json")
        messages = [
            {
                "role": "system",
                "content": (
                    "Extract every minimal independently checkable factual claim in the answer. "
                    "Keep quantities, dates, negation, comparison, condition, modality, and scope in the claim. "
                    "Do not include greetings, opinions, or purely stylistic text. Each text value must be an exact "
                    "substring of the answer and start/end are zero-based Python string offsets, end exclusive."
                ),
            },
            {"role": "user", "content": f"Answer:\n{response}"},
        ]
        start = time.perf_counter()
        try:
            payload = self._retry_json(messages, CLAIM_SCHEMA, "atomic_claims")
            claims = _validate_claims(payload, response, "atomic")
        except Exception as exc:  # noqa: BLE001
            raise CriticalProtocolError("atomic claim extraction failed") from exc
        self._save(key, {"claims": [{"text": c.text, "start": c.start, "end": c.end} for c in claims]})
        self._record(key, time.perf_counter() - start, cached=False)
        return claims


class FullContextReviewer(_CachedComponent):
    component = "critical_coverage_reviewer"
    protocol = "support-critical-coverage-v1"

    def __init__(self, cfg, usage=None, *, cache_only: bool = False):
        super().__init__(cfg, "coverage_reviewer", usage, cache_only=cache_only)

    def review(
        self, response: str, context: str, query: str | None, known_claims: Sequence[AtomicClaim]
    ) -> list[AtomicClaim]:
        if not response.strip():
            return []
        key = self._cache_key({
            "response": response,
            "context": context,
            "query": query or "",
            "known_claims": [c.to_dict() for c in known_claims],
        })
        cached = self._load(key)
        if cached is not None:
            claims = _validate_claims(cached, response, "global_review")
            self._record(key, 0.0, cached=True)
            return claims
        if self.cache_only:
            raise CacheOnlyMissError(self.component, key, self.cache_dir / f"{key}.json")
        known = "\n".join(f"- {claim.text}" for claim in known_claims) or "(none)"
        messages = [
            {
                "role": "system",
                "content": (
                    "Read the entire source and answer under a closed-world rule. Find factual answer fragments that "
                    "may be unsupported or contradicted by the source, especially a single subtle extra fact. "
                    "Return only exact answer substrings with Python offsets. Do not return claims that are directly "
                    "supported. This is a candidate-generation review, not a final verdict."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuery:\n{query or ''}\n\nAnswer:\n{response}\n\n"
                f"Already extracted claims:\n{known}",
            },
        ]
        start = time.perf_counter()
        try:
            payload = self._retry_json(messages, CLAIM_SCHEMA, "coverage_candidates")
            claims = _validate_claims(payload, response, "global_review")
        except Exception as exc:  # noqa: BLE001
            raise CriticalProtocolError("full-context coverage review failed") from exc
        self._save(key, {"claims": [{"text": c.text, "start": c.start, "end": c.end} for c in claims]})
        self._record(key, time.perf_counter() - start, cached=False)
        return claims


def select_claim_evidence(
    context: str,
    query: str | None,
    claim: str,
    *,
    max_sentences: int,
    stopwords: Iterable[str] = (),
    embedder: Embedder | None = None,
) -> list[EvidenceSpan]:
    """Deterministically rank lexical and local-S-BERT sentence evidence."""
    blocked = set(stopwords) | _STOPWORDS
    claim_norm = normalize(claim)
    tokens = [t for t in _WORD_RE.findall(claim_norm) if t not in blocked]
    candidates: list[EvidenceSpan] = []
    for source, text in (("context", context or ""), ("query", query or "")):
        for index, start, end, sentence in _sentences(text):
            sentence_norm = normalize(sentence)
            token_hits = sum(
                int(re.search(r"(?<!\w)" + re.escape(token) + r"(?!\w)", sentence_norm) is not None)
                for token in tokens
            )
            phrase = int(bool(claim_norm and claim_norm in sentence_norm))
            candidates.append(EvidenceSpan(source, index, start, end, sentence, 4 * phrase + token_hits))
    if not candidates:
        return []
    semantic = np.zeros(len(candidates), dtype=float)
    if embedder is not None:
        try:
            vectors = embedder.encode([claim] + [span.text for span in candidates])
            if len(vectors) == len(candidates) + 1:
                semantic = vectors[1:] @ vectors[0]
        except Exception:
            # Retrieval can remain lexical if an optional test embedder cannot encode a value.
            semantic = np.zeros(len(candidates), dtype=float)
    ranked = list(enumerate(candidates))
    ranked.sort(
        key=lambda item: (
            -item[1].rank,
            -float(semantic[item[0]]),
            0 if item[1].source == "context" else 1,
            item[1].start,
            item[1].index,
        )
    )
    return [span for _, span in ranked[: max(0, int(max_sentences))]]


class CriticalClaimVerifier(_CachedComponent):
    component = "critical_claim_verifier"
    protocol = "support-critical-verdict-v1"

    def __init__(self, cfg, usage=None, *, cache_only: bool = False, embedder: Embedder | None = None):
        super().__init__(cfg, "claim_verifier", usage, cache_only=cache_only)
        self.max_sentences = int(config_value(self.section, "max_evidence_sentences", 8))
        self.stopwords = set(getattr(cfg.matching, "stopwords", []) or [])
        self.embedder = embedder

    def verify_claim(self, claim: str, context: str, query: str | None) -> CriticalVerdict:
        evidence = select_claim_evidence(
            context, query, claim, max_sentences=self.max_sentences,
            stopwords=self.stopwords, embedder=self.embedder,
        )
        key = self._cache_key({
            "claim": claim,
            "evidence": [span.to_dict() for span in evidence],
            "max_evidence_sentences": self.max_sentences,
            "embedding_model": config_value(self.cfg.matching, "embedding_model"),
            "embedding_model_revision": config_value(self.cfg.matching, "embedding_model_revision"),
        })
        cached = self._load(key)
        if cached is not None and cached.get("verdict") in CRITICAL_VERDICTS:
            self._record(key, 0.0, cached=True)
            return CriticalVerdict(str(cached["verdict"]), tuple(evidence), cache_hit=True)
        if self.cache_only:
            raise CacheOnlyMissError(self.component, key, self.cache_dir / f"{key}.json")
        evidence_text = "\n".join(f"[{span.source}:{span.index}] {span.text}" for span in evidence)
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify the claim using only the supplied evidence. Entailed requires direct support for every "
                    "material detail, including quantity, date, negation, condition, modality, comparison, and scope. "
                    "Contradicted requires direct incompatible evidence. Unsupported means this is a clear factual claim "
                    "but the evidence does not directly establish it; do not use world knowledge or plausible inference. "
                    "Unknown is reserved for a non-factual or genuinely ambiguous fragment, not merely missing evidence."
                ),
            },
            {"role": "user", "content": f"Claim:\n{claim}\n\nEvidence:\n{evidence_text or '(no evidence retrieved)'}"},
        ]
        start = time.perf_counter()
        try:
            payload = self._retry_json(messages, VERDICT_SCHEMA, "critical_claim_verdict")
            validate_json_document(payload, VERDICT_SCHEMA)
            verdict = str(payload["verdict"])
        except Exception as exc:  # noqa: BLE001
            raise CriticalProtocolError(f"critical claim verification failed for {claim!r}") from exc
        self._save(key, {"verdict": verdict})
        self._record(key, time.perf_counter() - start, cached=False)
        return CriticalVerdict(verdict, tuple(evidence), cache_hit=False)

    # Interface parity with RelationVerifier for graph-edge scoring.
    def verify(
        self,
        canonical_triple: tuple[str, str, str],
        context: str,
        query: str | None,
        *,
        matching_params: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> CriticalVerdict:
        claim = f"{canonical_triple[0]} {canonical_triple[1]} {canonical_triple[2]}."
        return self.verify_claim(claim, context, query)


class CriticalClaimPipeline:
    """Build, review, deduplicate, and verify response-level atomic claims."""

    def __init__(self, cfg, usage=None, *, cache_only: bool = False, embedder: Embedder | None = None):
        self.extractor = AtomicClaimExtractor(cfg, usage, cache_only=cache_only)
        self.reviewer = FullContextReviewer(cfg, usage, cache_only=cache_only)
        self.verifier = CriticalClaimVerifier(cfg, usage, cache_only=cache_only, embedder=embedder)

    def assess(self, response: str, context: str, query: str | None) -> list[dict[str, Any]]:
        atomic = self.extractor.extract(response)
        reviewed = self.reviewer.review(response, context, query, atomic)
        claims = merge_claims(atomic, reviewed)
        audits: list[dict[str, Any]] = []
        for claim in claims:
            decision = self.verifier.verify_claim(claim.text, context, query)
            audits.append({
                "claim": claim.to_dict(),
                "evidence": [span.to_dict() for span in decision.evidence],
                "verdict": decision.verdict,
                "verifier_cache_hit": decision.cache_hit,
            })
        return audits


class FakeCriticalClaimPipeline:
    """Deterministic offline fixture for ``support-critical`` tests."""

    def __init__(self, claims: Sequence[AtomicClaim] | None = None, verdicts: dict[str, str] | None = None):
        self.claims = list(claims or [])
        self.verdicts = {normalize(k): v for k, v in (verdicts or {}).items()}

    def assess(self, response: str, context: str, query: str | None) -> list[dict[str, Any]]:  # noqa: ARG002
        return [
            {
                "claim": claim.to_dict(),
                "evidence": [],
                "verdict": self.verdicts.get(normalize(claim.text), "unknown"),
                "verifier_cache_hit": False,
            }
            for claim in self.claims
        ]
