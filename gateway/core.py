"""Dependency-light request, manifest, and response helpers for the gateway.

Keeping this module independent of FastAPI and the Google SDK makes the wire
contract fully unit-testable without credentials or a network connection.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping


GATEWAY_PROTOCOL = "hallu-vertex-openai-gateway-v1"
API_PATH = "/v1"


class GatewayError(ValueError):
    """A client-safe gateway error carrying an HTTP status."""

    def __init__(self, status_code: int, message: str, error_type: str = "invalid_request_error"):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type

    def envelope(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.status_code,
            }
        }


def _required_env(name: str, env: Mapping[str, str] | None = None) -> str:
    value = (os.environ if env is None else env).get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be configured")
    return value


def vertex_model_from_logical(logical_model: str) -> str:
    """Translate the only configured LiteLLM model name into Vertex's ID."""
    prefix = "openai/"
    if not logical_model.startswith(prefix) or logical_model == prefix:
        raise RuntimeError(
            "HALLU_LOGICAL_MODEL must be an OpenAI-compatible LiteLLM slug such as "
            "'openai/gemini-2.5-flash'"
        )
    return logical_model[len(prefix):]


@dataclass(frozen=True)
class GatewaySettings:
    api_key: str
    logical_model: str
    vertex_model: str
    project: str
    location: str
    release: str
    cloud_run_revision: str

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "GatewaySettings":
        source = os.environ if env is None else env
        logical_model = _required_env("HALLU_LOGICAL_MODEL", source)
        return cls(
            api_key=_required_env("HALLU_GATEWAY_API_KEY", source),
            logical_model=logical_model,
            vertex_model=vertex_model_from_logical(logical_model),
            project=_required_env("GOOGLE_CLOUD_PROJECT", source),
            location=_required_env("GOOGLE_CLOUD_LOCATION", source),
            release=_required_env("HALLU_GATEWAY_RELEASE", source),
            cloud_run_revision=source.get("K_REVISION", "local-dev").strip() or "local-dev",
        )


def gateway_manifest(settings: GatewaySettings) -> dict[str, str]:
    """Return the immutable identity that DataSphere fingerprints into caches."""
    return {
        "protocol": GATEWAY_PROTOCOL,
        "api_path": API_PATH,
        "logical_model": settings.logical_model,
        "vertex_model": settings.vertex_model,
        "vertex_location": settings.location,
        "gateway_release": settings.release,
        "cloud_run_revision": settings.cloud_run_revision,
    }


def canonical_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(manifest), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def authenticate(authorization: str | None, expected_key: str) -> None:
    """Validate a bearer credential without exposing which part was wrong."""
    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        raise GatewayError(401, "missing or invalid bearer credential", "authentication_error")
    supplied = authorization[len(prefix):]
    if not supplied or not hmac.compare_digest(supplied, expected_key):
        raise GatewayError(401, "missing or invalid bearer credential", "authentication_error")


def _content_as_text(content: Any, *, field: str) -> str:
    if not isinstance(content, str):
        raise GatewayError(400, f"{field} must be a string; multipart content is unsupported")
    if not content:
        raise GatewayError(400, f"{field} must not be empty")
    return content


def _parse_response_format(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or value.get("type") != "json_schema":
        raise GatewayError(400, "response_format.type must be 'json_schema'")
    json_schema = value.get("json_schema")
    if not isinstance(json_schema, Mapping):
        raise GatewayError(400, "response_format.json_schema must be an object")
    schema = json_schema.get("schema")
    if not isinstance(schema, Mapping):
        raise GatewayError(400, "response_format.json_schema.schema must be an object")
    if json_schema.get("strict") not in (None, True):
        raise GatewayError(400, "only strict JSON Schema responses are supported")
    # Preserve the caller's schema unchanged: Vertex's response_json_schema accepts
    # JSON Schema, including the $defs/$ref dynamic contracts emitted by KGGen.
    return dict(schema)


def _request_model_matches(model: str, settings: GatewaySettings) -> bool:
    return model in {settings.logical_model, settings.vertex_model}


def parse_chat_request(payload: Any, settings: GatewaySettings) -> dict[str, Any]:
    """Validate the supported OpenAI Chat Completions subset and normalise it."""
    if not isinstance(payload, Mapping):
        raise GatewayError(400, "request body must be a JSON object")
    if payload.get("stream") not in (None, False):
        raise GatewayError(400, "streaming is unsupported")
    if payload.get("tools") is not None or payload.get("tool_choice") is not None:
        raise GatewayError(400, "tools and tool_choice are unsupported")
    if payload.get("n") not in (None, 1):
        raise GatewayError(400, "only n=1 is supported")
    model = payload.get("model")
    if not isinstance(model, str) or not _request_model_matches(model, settings):
        raise GatewayError(400, "request model does not match the configured gateway model")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise GatewayError(400, "messages must be a non-empty array")

    system_parts: list[str] = []
    contents: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise GatewayError(400, f"messages[{index}] must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise GatewayError(400, f"messages[{index}].role is unsupported")
        text = _content_as_text(message.get("content"), field=f"messages[{index}].content")
        if role == "system":
            system_parts.append(text)
        else:
            # Vertex uses "model" where OpenAI uses "assistant".
            contents.append({"role": "model" if role == "assistant" else "user", "text": text})
    if not contents:
        raise GatewayError(400, "at least one user or assistant message is required")

    temperature = payload.get("temperature", 1.0)
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise GatewayError(400, "temperature must be a number")
    if not 0.0 <= float(temperature) <= 2.0:
        raise GatewayError(400, "temperature must be between 0 and 2")
    max_tokens = payload.get("max_tokens", payload.get("max_completion_tokens"))
    if max_tokens is not None and (
        not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0
    ):
        raise GatewayError(400, "max_tokens must be a positive integer")

    return {
        "contents": contents,
        "system_instruction": "\n\n".join(system_parts) or None,
        "temperature": float(temperature),
        "max_output_tokens": max_tokens,
        "response_json_schema": _parse_response_format(payload.get("response_format")),
    }


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _candidate_text(candidate: Any) -> str:
    content = _field(candidate, "content")
    parts = _field(content, "parts", [])
    texts = [str(_field(part, "text", "")) for part in parts if _field(part, "text", None) is not None]
    text = "".join(texts)
    if not text:
        raise GatewayError(502, "Vertex returned no text content", "api_error")
    return text


def _finish_reason(candidate: Any) -> str:
    raw = str(_field(candidate, "finish_reason", "")).upper()
    # The SDK may expose a protobuf enum (``FinishReason.STOP``), an enum
    # value, or a plain string depending on its release.  Normalise all of
    # those shapes without ever treating an incomplete answer as successful.
    if raw == "STOP" or raw.endswith(".STOP") or raw.endswith("_STOP"):
        return "stop"
    if raw == "LENGTH" or "MAX_TOKENS" in raw or raw.endswith(".LENGTH"):
        return "length"
    return "content_filter"


def openai_response(vertex_response: Any, settings: GatewaySettings) -> dict[str, Any]:
    candidates = _field(vertex_response, "candidates", [])
    if not isinstance(candidates, (list, tuple)) or len(candidates) != 1:
        raise GatewayError(502, "Vertex returned an invalid candidate envelope", "api_error")
    candidate = candidates[0]
    usage = _field(vertex_response, "usage_metadata", {})
    prompt_tokens = int(_field(usage, "prompt_token_count", 0) or 0)
    completion_tokens = int(_field(usage, "candidates_token_count", 0) or 0)
    total_tokens = int(_field(usage, "total_token_count", prompt_tokens + completion_tokens) or 0)
    model_version = str(_field(vertex_response, "model_version", settings.vertex_model) or settings.vertex_model)
    manifest_hash = canonical_manifest_sha256(gateway_manifest(settings))
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": settings.vertex_model,
        "system_fingerprint": f"{manifest_hash[:16]}:{model_version}",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": _candidate_text(candidate)},
                "finish_reason": _finish_reason(candidate),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def translate_vertex_error(exc: BaseException) -> GatewayError:
    """Map SDK/HTTP failures to the retry boundary used by KGExtractor."""
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    try:
        status = int(status)
    except (TypeError, ValueError):
        status = None
    name = type(exc).__name__.lower()
    if status == 429 or "resourceexhausted" in name or "toomanyrequests" in name:
        return GatewayError(429, "Vertex capacity is temporarily exhausted", "rate_limit_error")
    if status in {400, 401, 403, 404}:
        return GatewayError(status, "Vertex rejected the request", "invalid_request_error")
    if status is not None and 500 <= status <= 599:
        return GatewayError(503, "Vertex is temporarily unavailable", "api_error")
    if any(token in name for token in ("timeout", "connection", "serviceunavailable", "internalserver")):
        return GatewayError(503, "Vertex is temporarily unavailable", "api_error")
    return GatewayError(502, "Vertex request failed", "api_error")
