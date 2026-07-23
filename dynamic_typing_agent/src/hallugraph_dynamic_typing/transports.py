"""Injected structured-model and NLI adapters; fake mode is deterministic and offline."""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator, ValidationError
from langchain_core.messages import BaseMessage

from .errors import ModelProtocolError, TransportError
from .models import EvidenceLevel, NliResult, stable_id


_VERTEX_UNSUPPORTED_SCHEMA_KEYS = frozenset({"$schema", "pattern", "minLength", "maxLength", "uniqueItems"})
_MARKDOWN_JSON_FENCE = re.compile(r"^```(?:json)?\s*\n(?P<body>[\s\S]*?)\n```\s*$", re.IGNORECASE)


def vertex_compatible_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Project a validation schema to Vertex's documented response-schema subset.

    The original schema remains the local acceptance contract. Vertex supports a smaller
    response-json-schema vocabulary, so unsupported string and array constraints are
    omitted only on the wire; ``const`` becomes the supported one-item string enum.
    """

    def visit(value: Any) -> Any:
        if isinstance(value, Mapping):
            converted: dict[str, Any] = {}
            for key, child in value.items():
                if key in _VERTEX_UNSUPPORTED_SCHEMA_KEYS:
                    continue
                if key == "const":
                    if isinstance(child, (str, int, float)) and not isinstance(child, bool):
                        converted["enum"] = [child]
                    continue
                converted[str(key)] = visit(child)
            return converted
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    result = visit(schema)
    if not isinstance(result, dict):  # defensive: the caller promises a mapping
        raise ValueError("response schema must be an object")
    return result


def parse_json_completion(content: Any, output_schema: Mapping[str, Any], operation: str) -> Mapping[str, Any]:
    """Parse a JSON-only answer, tolerating an otherwise-valid Markdown fence."""
    text = str(content).strip()
    fenced = _MARKDOWN_JSON_FENCE.match(text)
    if fenced:
        text = fenced.group("body").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelProtocolError(f"{operation}: completion is not valid JSON ({exc.msg})") from exc
    try:
        Draft202012Validator(output_schema).validate(parsed)
    except ValidationError as exc:
        location = "/".join(str(part) for part in exc.absolute_path) or "<root>"
        raise ModelProtocolError(
            f"{operation}: JSON does not satisfy the required schema at {location}: {exc.message}"
        ) from exc
    except Exception as exc:
        raise ModelProtocolError(f"{operation}: JSON does not satisfy the required schema") from exc
    return parsed


class FakeStructuredModel:
    """Offline operation recorder with optional schema-valid scripted results."""

    def __init__(self, responses: Mapping[str, Mapping[str, Any]] | None = None):
        self.responses = dict(responses or {})
        self.calls: list[dict[str, Any]] = []

    def invoke(
        self,
        *,
        operation: str,
        messages: tuple[BaseMessage, ...],
        output_schema: Mapping[str, Any],
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        self.calls.append(
            {
                "operation": operation,
                "idempotency_key": idempotency_key,
                "message_count": len(messages),
            }
        )
        response = self.responses.get(operation)
        if response is None:
            return {}
        Draft202012Validator(output_schema).validate(response)
        return response


@dataclass
class LiteLLMStructuredModel:
    """Optional production adapter with bounded retries for transient live failures.

    LiteLLM retries are disabled deliberately: this adapter applies one observable,
    idempotency-key-preserving retry policy instead.  A 429 (rate limit) and temporary
    transport/server failures are retried; malformed requests and authentication failures
    are returned immediately with a redacted diagnostic.
    """

    model: str
    api_base: str | None
    api_key: str | None
    timeout_seconds: float
    max_attempts: int = 5
    temperature: float = 0.0
    retry_backoff_base_seconds: float = 2.0
    retry_backoff_max_seconds: float = 60.0
    retry_jitter_seconds: float = 1.0
    structured_schema_profile: str = "native"
    completion_callable: Callable[..., Any] | None = field(default=None, repr=False)
    sleep_callable: Callable[[float], None] = field(default=time.sleep, repr=False)
    random_callable: Callable[[], float] = field(default=random.random, repr=False)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.retry_backoff_base_seconds < 0 or self.retry_backoff_max_seconds < 0:
            raise ValueError("retry backoff values must be non-negative")
        if self.retry_backoff_max_seconds < self.retry_backoff_base_seconds:
            raise ValueError("retry_backoff_max_seconds must be at least retry_backoff_base_seconds")
        if self.retry_jitter_seconds < 0:
            raise ValueError("retry_jitter_seconds must be non-negative")
        if self.structured_schema_profile not in {"native", "vertex", "prompt"}:
            raise ValueError("structured_schema_profile must be 'native', 'vertex' or 'prompt'")

    def _completion(self) -> Callable[..., Any]:
        if self.completion_callable is not None:
            return self.completion_callable
        try:
            from litellm import completion  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional extra
            raise TransportError("live backend requires the litellm optional dependency") from exc
        return completion

    def _status_code(self, exc: Exception) -> int | None:
        for candidate in (exc, getattr(exc, "response", None)):
            value = getattr(candidate, "status_code", None)
            if isinstance(value, int):
                return value
        match = re.search(r"(?:status(?:[_ ]code)?|http)\D{0,12}(\d{3})", str(exc), flags=re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _is_transient(self, exc: Exception) -> bool:
        status = self._status_code(exc)
        if status == 429 or status in {408, 409, 425} or status is not None and 500 <= status <= 599:
            return True
        text = str(exc).casefold()
        return isinstance(exc, (TimeoutError, ConnectionError)) or any(
            marker in text
            for marker in ("timeout", "timed out", "connection reset", "connection refused", "rate limit", "too many requests")
        )

    def _redacted_detail(self, exc: Exception) -> str:
        detail = " ".join(str(exc).split())[:600] or "no diagnostic text supplied"
        for secret in (self.api_key,):
            if secret:
                detail = detail.replace(secret, "***")
        detail = re.sub(r"(?i)(authorization|api[_-]?key|token)(\s*[=:]\s*)([^\s,;]+)", r"\1\2***", detail)
        status = self._status_code(exc)
        suffix = f" (status={status})" if status is not None else ""
        return f"{type(exc).__name__}{suffix}: {detail}"

    def _delay_seconds(self, retry_number: int) -> float:
        # retry_number starts at one after the first failed call.
        delay = min(self.retry_backoff_max_seconds, self.retry_backoff_base_seconds * (2 ** (retry_number - 1)))
        return delay + self.retry_jitter_seconds * self.random_callable()

    def invoke(
        self,
        *,
        operation: str,
        messages: tuple[BaseMessage, ...],
        output_schema: Mapping[str, Any],
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        roles = {"human": "user", "ai": "assistant"}
        payload = [{"role": roles.get(message.type, message.type), "content": str(message.content)} for message in messages]
        if self.structured_schema_profile == "prompt":
            schema_instruction = (
                "\n\nThe required JSON Schema follows. Return only one JSON object that validates against it; "
                "do not use Markdown fences or add prose.\n"
                + json.dumps(dict(output_schema), ensure_ascii=False, separators=(",", ":"))
            )
            if payload and payload[0]["role"] == "system":
                payload[0]["content"] += schema_instruction
            else:
                payload.insert(0, {"role": "system", "content": schema_instruction})
        wire_schema = vertex_compatible_schema(output_schema) if self.structured_schema_profile == "vertex" else dict(output_schema)
        completion = self._completion()
        response: Any = None
        for attempt in range(1, self.max_attempts + 1):
            try:  # pragma: no cover - live-only
                request: dict[str, Any] = {
                    "model": self.model,
                    "messages": payload,
                    "api_base": self.api_base,
                    "api_key": self.api_key,
                    "timeout": self.timeout_seconds,
                    "temperature": self.temperature,
                    "num_retries": 0,
                    "metadata": {"idempotency_key": idempotency_key},
                }
                if self.structured_schema_profile != "prompt":
                    request["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": operation, "strict": True, "schema": wire_schema},
                    }
                response = completion(
                    **request,
                )
            except Exception as exc:
                if attempt >= self.max_attempts or not self._is_transient(exc):
                    raise TransportError(
                        f"{operation}: live completion failed after {attempt}/{self.max_attempts} attempt(s): "
                        f"{self._redacted_detail(exc)}"
                    ) from exc
                self.sleep_callable(self._delay_seconds(attempt))
                continue
            content: Any = None
            try:
                choices = getattr(response, "choices", None) or response.get("choices")
                if not isinstance(choices, list) or len(choices) != 1:
                    raise ModelProtocolError(f"{operation}: expected one completion choice")
                choice = choices[0]
                finish_reason = getattr(choice, "finish_reason", None) or choice.get("finish_reason")
                if finish_reason != "stop":
                    raise ModelProtocolError(f"{operation}: completion did not finish cleanly")
                message = getattr(choice, "message", None) or choice.get("message")
                content = getattr(message, "content", None) if message is not None else None
                if content is None and isinstance(message, dict):
                    content = message.get("content")
                return parse_json_completion(content, output_schema, operation)
            except ModelProtocolError as exc:
                if attempt >= self.max_attempts:
                    raise
                # Prompt-mode schemas are validated locally. Preserve the invalid
                # model answer as conversation context and require one complete
                # replacement object; this is a protocol retry, not a silent
                # normalisation of the evidence-bearing output.
                payload.extend(
                    [
                        {"role": "assistant", "content": str(content or "")},
                        {
                            "role": "user",
                            "content": (
                                "The previous answer violated the required JSON contract "
                                f"({exc}). Return a complete corrected JSON object only. "
                                "Do not explain the correction and do not omit required fields."
                            ),
                        },
                    ]
                )
                self.sleep_callable(self._delay_seconds(attempt))
        raise RuntimeError("completion retry loop ended unexpectedly")


class DeterministicNli:
    """Conservative offline NLI: explicit string support only; otherwise neutral."""

    def verify(
        self,
        *,
        hypothesis_kind: str,
        premise: str,
        hypothesis: str,
        evidence_span_ids: tuple[str, ...],
        idempotency_key: str,
    ) -> NliResult:
        premise_normalized = " ".join(premise.casefold().split())
        hypothesis_normalized = " ".join(hypothesis.casefold().split())
        if hypothesis_normalized and hypothesis_normalized in premise_normalized:
            verdict = "entailed"
            rationale = "The exact normalized hypothesis occurs in the supplied premise."
            level = EvidenceLevel.SOURCE_ENTAILED
        else:
            verdict = "neutral"
            rationale = "The supplied premise does not directly establish or negate the hypothesis."
            level = EvidenceLevel.UNKNOWN
        return NliResult(
            request_id=stable_id("nli", {"key": idempotency_key, "kind": hypothesis_kind}),
            verdict=verdict,
            evidence_span_ids=evidence_span_ids,
            rationale=rationale,
            evidence_level=level,
        )


@dataclass
class HhemNli:
    """Lazy, local-only HHEM adapter with conservative three-way thresholds.

    HHEM returns a consistency score, not native three-way NLI labels.  Scores in the
    middle interval therefore stay ``neutral``; only the two explicitly configured
    tails become entailed or contradicted.  The snapshot is never downloaded at runtime.
    """

    model_path: str | Path
    revision: str
    entailment_threshold: float = 0.80
    contradiction_threshold: float = 0.20
    batch_size: int = 8
    _model: Any = field(default=None, repr=False)
    _loader: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.contradiction_threshold < self.entailment_threshold <= 1.0:
            raise ValueError("HHEM thresholds must satisfy 0 <= contradiction < entailment <= 1")
        if not self.revision.strip():
            raise ValueError("HHEM revision must be an exact pinned commit SHA")

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        path = Path(self.model_path)
        if not (path / "config.json").is_file():
            raise TransportError(f"HHEM local snapshot is unavailable: {path}")
        # ``trust_remote_code`` caches the checked-in local HHEM Python modules. Keep
        # that cache beside the ignored snapshot rather than in a user-profile cache.
        os.environ.setdefault("HF_MODULES_CACHE", str(path.parent / ".hf_modules_cache"))
        try:
            from transformers import AutoModelForSequenceClassification  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise TransportError("HHEM requires the optional transformers and torch dependencies") from exc
        self._model = self._loader() if self._loader is not None else AutoModelForSequenceClassification.from_pretrained(
            str(path),
            trust_remote_code=True,
            revision=self.revision,
            local_files_only=True,
        )
        return self._model

    def verify(
        self,
        *,
        hypothesis_kind: str,
        premise: str,
        hypothesis: str,
        evidence_span_ids: tuple[str, ...],
        idempotency_key: str,
    ) -> NliResult:
        model = self._ensure_model()
        try:
            raw_scores = model.predict([(premise, hypothesis)])
            if hasattr(raw_scores, "tolist"):
                raw_scores = raw_scores.tolist()
            score = float(raw_scores[0])
        except Exception as exc:
            raise TransportError("HHEM prediction failed") from exc
        if not 0.0 <= score <= 1.0:
            raise ModelProtocolError("HHEM consistency score must be within [0, 1]")
        if score >= self.entailment_threshold:
            verdict, level = "entailed", EvidenceLevel.SOURCE_ENTAILED
        elif score <= self.contradiction_threshold:
            verdict, level = "contradicted", EvidenceLevel.UNKNOWN
        else:
            verdict, level = "neutral", EvidenceLevel.UNKNOWN
        return NliResult(
            request_id=stable_id("hhem-nli", {"key": idempotency_key, "kind": hypothesis_kind}),
            verdict=verdict,
            evidence_span_ids=evidence_span_ids,
            rationale=(
                f"HHEM consistency score={score:.4f}; thresholds: "
                f"contradicted<={self.contradiction_threshold:.2f}, "
                f"entailed>={self.entailment_threshold:.2f}."
            ),
            evidence_level=level,
        )
