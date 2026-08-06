"""Likelihood-weighted semantic entropy for black-box RAGTruth evaluation.

This implements the SemanticEntropy baseline in TOHA for a gateway-hosted
generator.  For every original RAGTruth prompt, it samples ``n`` answers,
obtains each answer's selected-token log probabilities, groups samples by
mutual NLI entailment, and computes entropy over the resulting semantic-class
probability mass.  The algorithm matches TOHA's ``get_semantic_ids``,
``logsumexp_by_id(..., agg='sum_normalized')``, and
``predictive_entropy_rao`` functions; only the transport differs.

Raw prompts and completions are never written to progress, usage, reports, or
archives.  They are retained only in content-addressed external-disk caches so
that a cache-only replay can reproduce the score without a network request.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import platform
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

import numpy as np
from tenacity import Retrying, retry_if_exception

from .cache import CacheOnlyMissError, config_value, llm_runtime_fingerprint
from .retry import (
    RequestPacer,
    RetryHeartbeat,
    StopAfterAttemptsExceptRateLimit,
    WaitRetryAfterOrExponentialJitter,
    is_rate_limit_error,
    transient_http_status,
)


SEMANTIC_ENTROPY_PROTOCOL = "toha-semantic-entropy-v1-gemini-selected-logprobs"
NLI_PROTOCOL = "toha-mutual-entailment-greedy-v1"
CANDIDATE_AGREEMENT_PROTOCOL = "ragtruth-candidate-agreement-v1"


class SemanticEntropyError(RuntimeError):
    """A non-retryable semantic-entropy protocol error."""


class SemanticEntropyTruncatedError(SemanticEntropyError):
    """A sampled answer reached its documented output cap."""


class CandidateAgreementEmptyCandidateError(SemanticEntropyError):
    """The labelled historical response is empty and cannot be compared."""


class GatewayRequestError(RuntimeError):
    """Redacted transport failure carrying retry-safe HTTP metadata."""

    def __init__(self, status_code: int | None, message: str, headers: Any = None):
        self.status_code = status_code
        self.headers = headers or {}
        super().__init__(message)


@dataclass(frozen=True)
class CompletionSample:
    """One sampled completion and its TOHA-compatible mean token log-likelihood."""

    text: str
    mean_logprob: float
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class SemanticEntropyScore:
    entropy: float
    semantic_ids: tuple[int, ...]
    class_logprobs: tuple[float, ...]
    n_samples: int
    n_classes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_entropy": self.entropy,
            "semantic_ids": list(self.semantic_ids),
            "class_logprobs": list(self.class_logprobs),
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
        }


@dataclass(frozen=True)
class CandidateAgreementScore:
    """Likelihood-weighted semantic support for one historical response.

    ``agreement_mass`` is the normalized likelihood mass of sampled answers
    mutually equivalent to the candidate.  ``disagreement`` is its
    response-conditioned hallucination-risk score, ``1 - agreement_mass``.
    No candidate or sampled text belongs in the serialized representation.
    """

    agreement_mass: float
    disagreement: float
    n_samples: int
    matched_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "agreement_mass": self.agreement_mass,
            "candidate_disagreement": self.disagreement,
            "n_samples": self.n_samples,
            "matched_samples": self.matched_samples,
        }


class EntailmentBackend(Protocol):
    def classify_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[int]:
        """Return MNLI class IDs (0 contradiction, 1 neutral, 2 entailment)."""


class DebertaEntailment:
    """Batched local equivalent of TOHA's ``EntailmentDeberta`` helper."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        model_path: str | None = None,
        device: str = "auto",
        batch_size: int = 2,
        max_length: int = 512,
    ):
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("NLI batch_size and max_length must be positive")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:  # pragma: no cover - live dependency path
            raise SemanticEntropyError(
                "Semantic entropy requires torch and transformers; install the semantic runtime dependencies"
            ) from exc
        source = str(model_path or model_name)
        kwargs: dict[str, Any] = {}
        if revision:
            kwargs["revision"] = revision
        self.torch = torch
        self.device = self._resolve_device(torch, device)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        # Newer Transformers releases detect a legacy SentencePiece regex
        # pattern in this DeBERTa snapshot. Opt into their corrective path so
        # the local NLI classifier does not silently tokenize with it.
        self.tokenizer = AutoTokenizer.from_pretrained(source, fix_mistral_regex=True, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(source, **kwargs)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(torch: Any, requested: str) -> str:
        value = str(requested).strip().lower()
        if value == "auto":
            if bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
                return "mps"
            if bool(torch.cuda.is_available()):
                return "cuda"
            return "cpu"
        if value == "mps" and not bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise SemanticEntropyError("semantic_entropy.nli_device=mps but MPS is unavailable")
        if value == "cuda" and not bool(torch.cuda.is_available()):
            raise SemanticEntropyError("semantic_entropy.nli_device=cuda but CUDA is unavailable")
        if value not in {"cpu", "mps", "cuda"}:
            raise ValueError("semantic_entropy.nli_device must be auto, cpu, mps, or cuda")
        return value

    def classify_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[int]:
        labels: list[int] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            left, right = zip(*batch)
            encoded = self.tokenizer(
                list(left), list(right), padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits
            labels.extend(int(value) for value in logits.argmax(dim=1).detach().cpu().tolist())
        return labels


class SemanticUsageLogger:
    """Redacted append-only request accounting independent of LiteLLM hooks."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.calls = 0
        self.requests = 0
        self.cache_hits = 0
        self.retries = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.retry_reasons: dict[str, int] = {}
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, payload: dict[str, Any]) -> None:
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def record_call(
        self, *, cache_key: str, seconds: float, cached: bool,
        prompt_tokens: int = 0, completion_tokens: int = 0,
    ) -> None:
        with self._lock:
            self.requests += 1
            self.cache_hits += int(cached)
            self.calls += 0 if cached else 1
            self.prompt_tokens += int(prompt_tokens)
            self.completion_tokens += int(completion_tokens)
            self._write({
                "kind": "semantic_entropy_generation", "cache_key": cache_key,
                "cached": bool(cached), "seconds": round(float(seconds), 4),
                "cum_calls": self.calls, "cum_requests": self.requests,
                "cum_cache_hits": self.cache_hits, "cum_retries": self.retries,
                "cum_prompt_tokens": self.prompt_tokens,
                "cum_completion_tokens": self.completion_tokens,
            })

    def record_retry(self, component: str, exc: BaseException) -> None:
        status = transient_http_status(exc)
        reason = f"http_{status}" if status is not None else type(exc).__name__
        with self._lock:
            self.retries += 1
            self.retry_reasons[reason] = self.retry_reasons.get(reason, 0) + 1
            self._write({
                "kind": component, "event": "retry", "retry_reason": reason,
                "cum_calls": self.calls, "cum_requests": self.requests,
                "cum_cache_hits": self.cache_hits, "cum_retries": self.retries,
                "cum_prompt_tokens": self.prompt_tokens,
                "cum_completion_tokens": self.completion_tokens,
            })

    def summary(self) -> dict[str, Any]:
        return {
            "api_calls": self.calls,
            "requests_total": self.requests,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(self.cache_hits / self.requests, 6) if self.requests else 0.0,
            "retries": self.retries,
            "retry_reasons": dict(sorted(self.retry_reasons.items())),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class GatewaySampler:
    """Small, dependency-light client for the authenticated OpenAI gateway."""

    def __init__(self, cfg: Any, entropy_cfg: Any):
        llm = cfg.llm
        api_base = str(config_value(llm, "api_base", "")).rstrip("/")
        if not api_base:
            raise SemanticEntropyError("llm.api_base is required for semantic entropy")
        key_name = str(config_value(llm, "api_key_env", "HALLU_GATEWAY_API_KEY"))
        self.api_key = os.environ.get(key_name, "").strip()
        if not self.api_key:
            raise SemanticEntropyError(f"{key_name} is not set")
        self.url = api_base + "/chat/completions"
        self.model = str(llm.model)
        self.temperature = float(config_value(entropy_cfg, "temperature", 1.0))
        self.max_tokens = int(config_value(entropy_cfg, "max_tokens", 512))
        self.timeout_s = float(config_value(entropy_cfg, "request_timeout_s", 120))
        if not 0.0 <= self.temperature <= 2.0 or self.max_tokens <= 0 or self.timeout_s <= 0:
            raise ValueError("invalid semantic entropy generation settings")

    def sample(self, prompt: str) -> CompletionSample:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "logprobs": True,
            "top_logprobs": 0,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=body, method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            # Deliberately do not include the server body: it can contain a
            # provider diagnostic and must not leak through a redacted monitor.
            raise GatewayRequestError(int(exc.code), "gateway rejected semantic entropy request", exc.headers) from exc
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            raise GatewayRequestError(None, "gateway request failed") from exc
        try:
            payload = json.loads(raw)
            choices = payload["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("choice count")
            choice = choices[0]
            if choice.get("finish_reason") == "length":
                raise SemanticEntropyTruncatedError(
                    f"semantic entropy sample reached max_tokens={self.max_tokens}"
                )
            if choice.get("finish_reason") != "stop":
                raise SemanticEntropyError("semantic entropy sample did not finish cleanly")
            text = choice["message"]["content"]
            entries = choice["logprobs"]["content"]
            if not isinstance(text, str) or not text or not isinstance(entries, list) or not entries:
                raise ValueError("missing completion or logprobs")
            logprobs = [float(entry["logprob"]) for entry in entries]
            if not all(math.isfinite(value) for value in logprobs):
                raise ValueError("non-finite logprob")
            usage = payload.get("usage") or {}
            return CompletionSample(
                text=text,
                # This reproduces TOHA's causal-LM ``loss`` reduction: mean
                # token log-likelihood, rather than length-biased total mass.
                mean_logprob=float(sum(logprobs) / len(logprobs)),
                prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                completion_tokens=int(usage.get("completion_tokens", len(logprobs)) or len(logprobs)),
            )
        except SemanticEntropyError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SemanticEntropyError("gateway returned malformed semantic entropy response") from exc


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def mutual_nli_equivalent(
    forward_label: int, backward_label: int, *, strict_entailment: bool
) -> bool:
    """Apply the TOHA mutual, non-contradictory NLI equivalence rule.

    The default rule intentionally permits entailment/neutral combinations
    while rejecting any contradiction and a pair that is neutral in both
    directions.  Candidate agreement must use this exact rule rather than a
    stronger post-hoc notion of textual equality.
    """

    if forward_label not in {0, 1, 2} or backward_label not in {0, 1, 2}:
        raise SemanticEntropyError("NLI classifier returned invalid MNLI labels")
    if strict_entailment:
        return forward_label == 2 and backward_label == 2
    return (
        forward_label != 0
        and backward_label != 0
        and (forward_label, backward_label) != (1, 1)
    )


class SemanticEntropyDetector:
    """Cache-backed TOHA semantic entropy scoring for a set of original prompts."""

    def __init__(
        self,
        cfg: Any,
        *,
        usage: SemanticUsageLogger | None = None,
        cache_only: bool = False,
        entailment_backend: EntailmentBackend | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.cfg = cfg
        self.settings = getattr(cfg, "semantic_entropy", None)
        if self.settings is None:
            raise ValueError("semantic_entropy config is required")
        self.protocol = str(config_value(self.settings, "protocol", SEMANTIC_ENTROPY_PROTOCOL))
        if self.protocol != SEMANTIC_ENTROPY_PROTOCOL:
            raise ValueError(f"unsupported semantic entropy protocol: {self.protocol}")
        self.cache_root = Path(str(config_value(self.settings, "cache_dir")))
        self.cache_read_roots = [Path(str(item)) for item in (config_value(self.settings, "cache_read_dirs", []) or [])]
        self.n_samples = int(config_value(self.settings, "n_samples", 15))
        if self.n_samples < 2:
            raise ValueError("semantic_entropy.n_samples must be at least 2")
        self.strict_entailment = bool(config_value(self.settings, "strict_entailment", False))
        self.cache_only = bool(cache_only)
        self.usage = usage or SemanticUsageLogger(None)
        self.progress_callback = progress_callback
        self.entailment_backend = entailment_backend
        self._loaded_nli: EntailmentBackend | None = entailment_backend
        self.sampler = None if self.cache_only else GatewaySampler(cfg, self.settings)
        self.pacer = RequestPacer(float(config_value(cfg.llm, "request_min_interval_s", 0)))
        self.max_retries = int(config_value(cfg.llm, "max_retries", 0))
        self.retry_base = float(config_value(cfg.llm, "retry_backoff_base_s", 2))
        self.retry_max = float(config_value(cfg.llm, "retry_backoff_max_s", 60))
        self.rate_limit_cooldown = float(config_value(cfg.llm, "rate_limit_cooldown_max_s", 900))
        self.rate_limit_deadline = float(config_value(cfg.llm, "rate_limit_retry_deadline_s", 1800))
        self.retry_deadline = float(config_value(cfg.llm, "retry_deadline_s", 1800))
        if not self.cache_only:
            self.cache_root.mkdir(parents=True, exist_ok=True)

    def _emit(self, **payload: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(dict(payload))

    def _nli_identity(self) -> dict[str, Any]:
        return {
            "protocol": NLI_PROTOCOL,
            "model": config_value(self.settings, "nli_model"),
            "revision": config_value(self.settings, "nli_revision"),
            "model_path": config_value(self.settings, "nli_model_path"),
            "max_length": int(config_value(self.settings, "nli_max_length", 512)),
            "strict_entailment": self.strict_entailment,
        }

    def _cache_key(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        llm_identity: dict[str, Any] | None = None,
        api_base: str | None = None,
    ) -> str:
        return _canonical_hash({
            "protocol": self.protocol,
            "kind": kind,
            "llm": llm_runtime_fingerprint(self.cfg) if llm_identity is None else llm_identity,
            "api_base": config_value(self.cfg.llm, "api_base") if api_base is None else api_base,
            "payload": payload,
        })

    def _load_from_roots(
        self, kind: str, key: str, roots: Iterable[Path]
    ) -> dict[str, Any] | None:
        for root in roots:
            path = root / kind / f"{key}.json"
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
                payload = envelope.get("payload")
                if (
                    isinstance(envelope, dict)
                    and envelope.get("protocol") == self.protocol
                    and envelope.get("kind") == kind
                    and envelope.get("cache_key") == key
                    and isinstance(payload, dict)
                    and envelope.get("payload_sha256") == _canonical_hash(payload)
                ):
                    return payload
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return None

    def _load(self, kind: str, key: str) -> dict[str, Any] | None:
        return self._load_from_roots(
            kind, key, [self.cache_root, *self.cache_read_roots]
        )

    def _save(self, kind: str, key: str, payload: dict[str, Any]) -> None:
        _atomic_json(self.cache_root / kind / f"{key}.json", {
            "protocol": self.protocol,
            "kind": kind,
            "cache_key": key,
            "payload": payload,
            "payload_sha256": _canonical_hash(payload),
        })

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        return isinstance(exc, GatewayRequestError) and (
            exc.status_code is None or exc.status_code == 429 or 500 <= exc.status_code <= 599
        )

    def _live_sample(self, prompt: str) -> CompletionSample:
        assert self.sampler is not None
        result: CompletionSample | None = None
        for attempt in Retrying(
            stop=StopAfterAttemptsExceptRateLimit(
                None if self.max_retries == 0 else self.max_retries,
                rate_limit_retry_deadline_seconds=self.rate_limit_deadline,
                retry_deadline_seconds=self.retry_deadline,
            ),
            wait=WaitRetryAfterOrExponentialJitter(
                self.retry_base, self.retry_max,
                rate_limit_cooldown_max_seconds=self.rate_limit_cooldown,
                rate_limit_retry_deadline_seconds=self.rate_limit_deadline,
                retry_deadline_seconds=self.retry_deadline,
            ),
            retry=retry_if_exception(self._is_retryable),
            before_sleep=RetryHeartbeat("semantic_entropy_generation", self.usage, self.progress_callback),
            reraise=True,
        ):
            with attempt:
                self.pacer.wait_for_turn()
                result = self.sampler.sample(prompt)
        assert result is not None
        return result

    def _sample(self, prompt: str, index: int) -> CompletionSample:
        generation = {
            "prompt": prompt,
            "sample_index": int(index),
            "n_samples": self.n_samples,
            "temperature": float(config_value(self.settings, "temperature", 1.0)),
            "max_tokens": int(config_value(self.settings, "max_tokens", 512)),
            "prompt_version": str(config_value(self.settings, "prompt_version", "ragtruth-original-prompt-v1")),
            "likelihood_normalization": str(config_value(self.settings, "likelihood_normalization")),
        }
        key = self._cache_key("samples", generation)
        cached = self._load("samples", key)
        cache_key_used = key
        reused_from_max_tokens: int | None = None
        if cached is None:
            # The gateway persists an entry only after a ``finish_reason=stop``
            # response.  Such a completed sequence has the same probability
            # under a larger output ceiling; only a sequence that hit the old
            # ceiling would differ, and it is deliberately never cached.
            current_cap = int(generation["max_tokens"])
            legacy_contracts = config_value(
                self.settings, "legacy_sample_cache_contracts", []
            ) or []
            for previous_cap in config_value(self.settings, "max_tokens_read_through", []) or []:
                if not isinstance(previous_cap, int) or isinstance(previous_cap, bool):
                    continue
                if previous_cap <= 0 or previous_cap >= current_cap:
                    continue
                previous_generation = dict(generation, max_tokens=int(previous_cap))
                previous_key = self._cache_key("samples", previous_generation)
                previous_cached = self._load("samples", previous_key)
                if previous_cached is not None:
                    cached = previous_cached
                    cache_key_used = previous_key
                    reused_from_max_tokens = int(previous_cap)
                if cached is None:
                    for contract in legacy_contracts:
                        if not isinstance(contract, dict):
                            continue
                        caps = contract.get("max_tokens", [])
                        if not isinstance(caps, list) or int(previous_cap) not in caps:
                            continue
                        identity = contract.get("llm_identity")
                        api_base = contract.get("api_base")
                        cache_dir = contract.get("cache_dir")
                        if (
                            not isinstance(identity, dict)
                            or not isinstance(api_base, str)
                            or not api_base
                            or not isinstance(cache_dir, str)
                            or not cache_dir
                        ):
                            continue
                        previous_key = self._cache_key(
                            "samples",
                            previous_generation,
                            llm_identity=identity,
                            api_base=api_base,
                        )
                        previous_cached = self._load_from_roots(
                            "samples", previous_key, [Path(cache_dir)]
                        )
                        if previous_cached is not None:
                            cached = previous_cached
                            cache_key_used = previous_key
                            reused_from_max_tokens = int(previous_cap)
                            break
                if cached is not None:
                    break
        if cached is not None:
            try:
                sample = CompletionSample(
                    text=str(cached["text"]), mean_logprob=float(cached["mean_logprob"]),
                    prompt_tokens=int(cached.get("prompt_tokens", 0)),
                    completion_tokens=int(cached.get("completion_tokens", 0)),
                )
                if sample.text and math.isfinite(sample.mean_logprob):
                    if reused_from_max_tokens is not None and not self.cache_only:
                        self._save("samples", key, {
                            "text": sample.text,
                            "mean_logprob": sample.mean_logprob,
                            "prompt_tokens": sample.prompt_tokens,
                            "completion_tokens": sample.completion_tokens,
                            "reused_completed_sample_from_max_tokens": reused_from_max_tokens,
                            "reused_completed_sample_from_cache_key": cache_key_used,
                        })
                    self.usage.record_call(cache_key=cache_key_used, seconds=0.0, cached=True)
                    return sample
            except (KeyError, TypeError, ValueError):
                pass
        if self.cache_only:
            raise CacheOnlyMissError("semantic_entropy_samples", key, self.cache_root / "samples" / f"{key}.json")
        started = time.monotonic()
        sample = self._live_sample(prompt)
        self._save("samples", key, {
            "text": sample.text,
            "mean_logprob": sample.mean_logprob,
            "prompt_tokens": sample.prompt_tokens,
            "completion_tokens": sample.completion_tokens,
        })
        self.usage.record_call(
            cache_key=key, seconds=time.monotonic() - started, cached=False,
            prompt_tokens=sample.prompt_tokens, completion_tokens=sample.completion_tokens,
        )
        self._emit(event="sample_completed", sample_index=index)
        return sample

    def _nli(self) -> EntailmentBackend:
        if self._loaded_nli is None:
            self._loaded_nli = DebertaEntailment(
                str(config_value(self.settings, "nli_model")),
                revision=config_value(self.settings, "nli_revision"),
                model_path=config_value(self.settings, "nli_model_path"),
                device=str(config_value(self.settings, "nli_device", "auto")),
                batch_size=int(config_value(self.settings, "nli_batch_size", 2)),
                max_length=int(config_value(self.settings, "nli_max_length", 512)),
            )
        return self._loaded_nli

    def _semantic_ids(self, samples: Sequence[CompletionSample]) -> tuple[int, ...]:
        cluster_input = {
            "nli": self._nli_identity(),
            "samples": [{"text": sample.text, "mean_logprob": sample.mean_logprob} for sample in samples],
        }
        key = self._cache_key("clusters", cluster_input)
        cached = self._load("clusters", key)
        if cached is None:
            # Cluster records do not depend on the generation ceiling.  Their
            # payload key still binds the NLI identity and the sampled texts,
            # while this explicit legacy lookup preserves the LLM identity
            # present in historical cache keys.
            for contract in config_value(
                self.settings, "legacy_sample_cache_contracts", []
            ) or []:
                if not isinstance(contract, dict):
                    continue
                identity = contract.get("llm_identity")
                api_base = contract.get("api_base")
                cache_dir = contract.get("cache_dir")
                if (
                    not isinstance(identity, dict)
                    or not isinstance(api_base, str)
                    or not api_base
                    or not isinstance(cache_dir, str)
                    or not cache_dir
                ):
                    continue
                legacy_key = self._cache_key(
                    "clusters", cluster_input, llm_identity=identity, api_base=api_base
                )
                cached = self._load_from_roots("clusters", legacy_key, [Path(cache_dir)])
                if cached is not None:
                    key = legacy_key
                    break
        if cached is not None:
            raw = cached.get("semantic_ids")
            if isinstance(raw, list) and len(raw) == len(samples) and all(isinstance(value, int) and value >= 0 for value in raw):
                return tuple(raw)
        if self.cache_only:
            raise CacheOnlyMissError("semantic_entropy_clusters", key, self.cache_root / "clusters" / f"{key}.json")
        texts = [sample.text for sample in samples]
        ids = [-1] * len(texts)
        next_id = 0
        for index, source in enumerate(texts):
            if ids[index] != -1:
                continue
            ids[index] = next_id
            candidates = [other for other in range(index + 1, len(texts)) if ids[other] == -1]
            pairs = [(source, texts[other]) for other in candidates] + [(texts[other], source) for other in candidates]
            labels = self._nli().classify_pairs(pairs) if pairs else []
            if len(labels) != len(pairs) or any(value not in {0, 1, 2} for value in labels):
                raise SemanticEntropyError("NLI classifier returned invalid MNLI labels")
            forward, backward = labels[:len(candidates)], labels[len(candidates):]
            for other, first, second in zip(candidates, forward, backward):
                if mutual_nli_equivalent(
                    first, second, strict_entailment=self.strict_entailment
                ):
                    ids[other] = next_id
            next_id += 1
        if -1 in ids:
            raise SemanticEntropyError("semantic clustering left an unassigned sample")
        self._save("clusters", key, {"semantic_ids": ids})
        self._emit(event="semantic_clusters_completed", n_classes=next_id)
        return tuple(ids)

    @staticmethod
    def _entropy(semantic_ids: Sequence[int], log_likelihoods: Sequence[float]) -> tuple[float, tuple[float, ...]]:
        if not semantic_ids or len(semantic_ids) != len(log_likelihoods):
            raise ValueError("semantic ids and likelihoods must be non-empty and aligned")
        values = np.asarray(log_likelihoods, dtype=float)
        if not np.all(np.isfinite(values)):
            raise SemanticEntropyError("non-finite semantic entropy log-likelihood")
        maximum = float(values.max())
        log_normalizer = maximum + math.log(float(np.exp(values - maximum).sum()))
        class_logprobs: list[float] = []
        for class_id in range(max(semantic_ids) + 1):
            selected = values[np.asarray(semantic_ids) == class_id] - log_normalizer
            if not len(selected):
                raise SemanticEntropyError("semantic class IDs must be contiguous")
            peak = float(selected.max())
            class_logprobs.append(peak + math.log(float(np.exp(selected - peak).sum())))
        probabilities = np.exp(np.asarray(class_logprobs, dtype=float))
        entropy = -float(np.sum(probabilities * np.asarray(class_logprobs, dtype=float)))
        return entropy, tuple(class_logprobs)

    def score_prompt(self, prompt: str) -> SemanticEntropyScore:
        if not isinstance(prompt, str) or not prompt.strip():
            raise SemanticEntropyError("RAGTruth source prompt is required for semantic entropy")
        samples = self.samples_for_prompt(prompt)
        semantic_ids = self._semantic_ids(samples)
        entropy, class_logprobs = self._entropy(semantic_ids, [sample.mean_logprob for sample in samples])
        return SemanticEntropyScore(
            entropy=entropy, semantic_ids=semantic_ids, class_logprobs=class_logprobs,
            n_samples=len(samples), n_classes=len(set(semantic_ids)),
        )

    def samples_for_prompt(self, prompt: str) -> tuple[CompletionSample, ...]:
        """Load or generate the fixed sample set without semantic clustering.

        This is deliberately a narrow public seam for response-conditioned
        baselines.  It preserves the existing sample key and cache contract:
        calling it does not create a semantic-cluster cache entry and it does
        not modify the entropy formula.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise SemanticEntropyError("RAGTruth source prompt is required for semantic entropy")
        return tuple(self._sample(prompt, index) for index in range(self.n_samples))


class CandidateAgreementDetector:
    """Response-conditioned agreement baseline over SemanticEntropy samples.

    The generator sample cache remains owned by :class:`SemanticEntropyDetector`.
    This class adds an independent, content-addressed comparison cache whose
    entries contain only MNLI verdicts and aggregates.  In particular, raw
    historical candidates and raw generated completions are never emitted to
    checkpoints, public outputs, or archives.
    """

    def __init__(
        self,
        cfg: Any,
        *,
        sample_detector: SemanticEntropyDetector | None = None,
        cache_only: bool = False,
        entailment_backend: EntailmentBackend | None = None,
    ):
        self.cfg = cfg
        self.settings = getattr(cfg, "candidate_agreement", None)
        if self.settings is None:
            raise ValueError("candidate_agreement config is required")
        self.protocol = str(config_value(
            self.settings, "protocol", CANDIDATE_AGREEMENT_PROTOCOL
        ))
        if self.protocol != CANDIDATE_AGREEMENT_PROTOCOL:
            raise ValueError(f"unsupported candidate-agreement protocol: {self.protocol}")
        self.cache_root = Path(str(config_value(self.settings, "cache_dir")))
        self.cache_read_roots = [
            Path(str(item))
            for item in (config_value(self.settings, "cache_read_dirs", []) or [])
        ]
        self.cache_only = bool(cache_only)
        self.sample_detector = sample_detector or SemanticEntropyDetector(
            cfg, cache_only=self.cache_only, entailment_backend=entailment_backend
        )
        # A response-conditioned evaluation may deliberately read generator
        # samples cache-only while filling its separate *local NLI* comparison
        # cache.  The inverse is unsafe: a candidate cache-only replay must
        # never be backed by a sample detector that could call Gemini.
        if self.cache_only and not self.sample_detector.cache_only:
            raise ValueError("candidate cache-only replay requires a sample cache-only detector")
        self.entailment_backend = entailment_backend
        self._loaded_nli: EntailmentBackend | None = entailment_backend
        self.nli_pair_evaluations = 0
        self.cache_hits = 0
        self.cache_misses = 0
        if not self.cache_only:
            self.cache_root.mkdir(parents=True, exist_ok=True)

    def _nli(self) -> EntailmentBackend:
        if self._loaded_nli is None:
            # Reuse the exact model identity and lazy-loading behavior used by
            # the semantic-entropy implementation.
            self._loaded_nli = self.sample_detector._nli()  # noqa: SLF001
        return self._loaded_nli

    def _comparison_key(
        self, candidate: str, samples: Sequence[CompletionSample]
    ) -> tuple[str, list[str], np.ndarray]:
        if not samples:
            raise SemanticEntropyError("candidate agreement requires at least one sample")
        likelihoods = np.asarray([sample.mean_logprob for sample in samples], dtype=float)
        if not np.all(np.isfinite(likelihoods)):
            raise SemanticEntropyError("non-finite candidate-agreement log-likelihood")
        maximum = float(likelihoods.max())
        weights = np.exp(likelihoods - maximum)
        weights /= float(weights.sum())
        sample_hashes = [
            _canonical_hash({
                "text": sample.text,
                "mean_logprob": float(sample.mean_logprob),
                "completion_tokens": int(sample.completion_tokens),
            })
            for sample in samples
        ]
        key = _canonical_hash({
            "protocol": self.protocol,
            "candidate_answer_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
            "sample_generation_protocol": self.sample_detector.protocol,
            "sample_runtime": llm_runtime_fingerprint(self.cfg),
            "sample_identities": [
                {
                    "sample_sha256": sample_hash,
                    "mean_logprob": float(sample.mean_logprob),
                    "normalized_weight": float(weight),
                }
                for sample_hash, sample, weight in zip(sample_hashes, samples, weights, strict=True)
            ],
            "nli": self.sample_detector._nli_identity(),  # noqa: SLF001
            "equivalence": "bidirectional_noncontradictory_nli",
            "strict_entailment": self.sample_detector.strict_entailment,
        })
        return key, sample_hashes, weights

    def _load(self, key: str) -> dict[str, Any] | None:
        for root in [self.cache_root, *self.cache_read_roots]:
            path = root / "comparisons" / f"{key}.json"
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
                payload = envelope.get("payload")
                if (
                    isinstance(envelope, dict)
                    and envelope.get("protocol") == self.protocol
                    and envelope.get("kind") == "comparisons"
                    and envelope.get("cache_key") == key
                    and isinstance(payload, dict)
                    and envelope.get("payload_sha256") == _canonical_hash(payload)
                ):
                    return payload
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return None

    def _save(self, key: str, payload: dict[str, Any]) -> None:
        _atomic_json(self.cache_root / "comparisons" / f"{key}.json", {
            "protocol": self.protocol,
            "kind": "comparisons",
            "cache_key": key,
            "payload": payload,
            "payload_sha256": _canonical_hash(payload),
        })

    @staticmethod
    def _score_from_payload(payload: dict[str, Any], n_samples: int) -> CandidateAgreementScore | None:
        try:
            agreement_mass = float(payload["agreement_mass"])
            disagreement = float(payload["candidate_disagreement"])
            matched_samples = int(payload["matched_samples"])
            forward = payload["forward_labels"]
            backward = payload["backward_labels"]
            if (
                not math.isfinite(agreement_mass)
                or not math.isfinite(disagreement)
                or not 0.0 <= agreement_mass <= 1.0
                or not 0.0 <= disagreement <= 1.0
                or not math.isclose(disagreement, 1.0 - agreement_mass, abs_tol=1e-12)
                or not 0 <= matched_samples <= n_samples
                or not isinstance(forward, list)
                or not isinstance(backward, list)
                or len(forward) != n_samples
                or len(backward) != n_samples
                or any(value not in {0, 1, 2} for value in forward + backward)
            ):
                return None
            return CandidateAgreementScore(
                agreement_mass=agreement_mass,
                disagreement=disagreement,
                n_samples=n_samples,
                matched_samples=matched_samples,
            )
        except (KeyError, TypeError, ValueError):
            return None

    def score_candidate(self, prompt: str, candidate: str) -> CandidateAgreementScore:
        """Score a labelled candidate against cached or newly sampled answers.

        Scores never observe RAGTruth labels.  A cache-only invocation may
        read samples and a comparison record, but any missing comparison is a
        hard miss before the local NLI model is loaded.
        """

        if not isinstance(candidate, str) or not candidate.strip():
            raise CandidateAgreementEmptyCandidateError(
                "historical candidate answer is empty for candidate agreement"
            )
        samples = self.sample_detector.samples_for_prompt(prompt)
        key, sample_hashes, weights = self._comparison_key(candidate, samples)
        cached = self._load(key)
        if cached is not None:
            score = self._score_from_payload(cached, len(samples))
            if score is not None:
                self.cache_hits += 1
                return score
        self.cache_misses += 1
        if self.cache_only:
            raise CacheOnlyMissError(
                "candidate_agreement_comparisons", key,
                self.cache_root / "comparisons" / f"{key}.json",
            )
        pairs = [(candidate, sample.text) for sample in samples] + [
            (sample.text, candidate) for sample in samples
        ]
        labels = self._nli().classify_pairs(pairs)
        self.nli_pair_evaluations += len(pairs)
        if len(labels) != len(pairs):
            raise SemanticEntropyError("NLI classifier returned the wrong number of candidate labels")
        forward = [int(value) for value in labels[:len(samples)]]
        backward = [int(value) for value in labels[len(samples):]]
        matches = [
            mutual_nli_equivalent(
                first, second, strict_entailment=self.sample_detector.strict_entailment
            )
            for first, second in zip(forward, backward, strict=True)
        ]
        agreement_mass = float(sum(
            float(weight) for weight, matched in zip(weights, matches, strict=True) if matched
        ))
        # Keep rounding out of the scientific calculation.  Clamp only the
        # unavoidable floating-point endpoints.
        agreement_mass = min(1.0, max(0.0, agreement_mass))
        payload = {
            "candidate_answer_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
            "sample_sha256": sample_hashes,
            "forward_labels": forward,
            "backward_labels": backward,
            "agreement_mass": agreement_mass,
            "candidate_disagreement": 1.0 - agreement_mass,
            "matched_samples": int(sum(matches)),
            "n_samples": len(samples),
        }
        self._save(key, payload)
        return CandidateAgreementScore(
            agreement_mass=agreement_mass,
            disagreement=1.0 - agreement_mass,
            n_samples=len(samples),
            matched_samples=int(sum(matches)),
        )


def semantic_runtime_metadata(cfg: Any) -> dict[str, Any]:
    """Secret-free identity recorded beside a run and in its score cache."""
    section = cfg.semantic_entropy
    return {
        "semantic_entropy_protocol": SEMANTIC_ENTROPY_PROTOCOL,
        "llm": llm_runtime_fingerprint(cfg),
        "n_samples": int(config_value(section, "n_samples", 15)),
        "temperature": float(config_value(section, "temperature", 1.0)),
        "max_tokens": int(config_value(section, "max_tokens", 512)),
        "max_tokens_read_through": list(config_value(section, "max_tokens_read_through", []) or []),
        "legacy_sample_cache_contracts": [
            {
                "cache_dir": str(contract.get("cache_dir", "")),
                "max_tokens": list(contract.get("max_tokens", [])),
                "contract_sha256": _canonical_hash(contract),
            }
            for contract in config_value(section, "legacy_sample_cache_contracts", []) or []
            if isinstance(contract, dict)
        ],
        "likelihood_normalization": config_value(section, "likelihood_normalization"),
        "nli": {
            "protocol": NLI_PROTOCOL,
            "model": config_value(section, "nli_model"),
            "revision": config_value(section, "nli_revision"),
            "model_path": config_value(section, "nli_model_path"),
            "device": config_value(section, "nli_device"),
            "batch_size": int(config_value(section, "nli_batch_size", 2)),
            "max_length": int(config_value(section, "nli_max_length", 512)),
        },
        "platform": {"python": platform.python_version(), "machine": platform.machine()},
    }
