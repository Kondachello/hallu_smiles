#!/usr/bin/env python3
"""Cheap liveness probe for the typing gateway.

Answers one question before a 750-record pass is worth submitting: does the
gateway still accept our credentials and actually return a completion?  The
probe reuses the production typing config, transport and URL normalisation, so
a pass here means the typed metric pass will reach the model the same way.

Exit codes: 0 = gateway live, 1 = probe failed (reason on stdout/stderr).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "dynamic_typing_agent" / "src"))

from hallugraph_dynamic_typing.transports import LiteLLMStructuredModel  # noqa: E402


def _openai_api_base(gateway_url: str) -> str:
    """Mirror of DynamicTypingAgent._openai_api_base.

    Duplicated rather than imported: pulling in the agent module drags langgraph
    into a probe that only needs the transport, and the probe should not fail for
    a dependency the thing it is testing never touches.
    """
    base = gateway_url.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"

_LIVE_CONFIG = _REPO_ROOT / "dynamic_typing_agent" / "config" / "live-gateway-hhem.yaml"
_DEFAULT_MODEL = "openai/gemini-2.5-flash"
_MANIFEST_TIMEOUT_SECONDS = 30

# Deliberately tiny: one entity, one type, so a pass costs a handful of tokens.
_PROBE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["type_label"],
    "properties": {"type_label": {"type": "string"}},
}


def _redact(text: str, secret: str | None) -> str:
    return text.replace(secret, "***") if secret else text


def _emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def _check_manifest(gateway_url: str, api_key: str) -> tuple[bool, str]:
    """Auth/reachability check that costs no model tokens."""
    url = f"{gateway_url.rstrip('/')}/v1/hallu/manifest"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=_MANIFEST_TIMEOUT_SECONDS) as response:
            body = response.read(4096).decode("utf-8", errors="replace")
            elapsed = time.monotonic() - started
            _emit(
                "manifest_ok",
                status=response.status,
                latency_seconds=round(elapsed, 2),
                body_head=body[:400],
            )
            return True, "ok"
    except urllib.error.HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", errors="replace")
        _emit("manifest_http_error", status=exc.code, detail=_redact(detail, api_key)[:400])
        hint = {
            401: "credentials rejected - HALLU_GATEWAY_API_KEY is stale or wrong project",
            403: "credentials rejected - key lacks access to this gateway",
            404: "gateway URL wrong or manifest endpoint moved",
        }.get(exc.code, f"gateway returned HTTP {exc.code}")
        return False, hint
    except Exception as exc:  # noqa: BLE001 - probe reports every failure shape
        _emit("manifest_error", detail=_redact(str(exc), api_key)[:400])
        return False, "gateway unreachable (DNS/TLS/network or URL wrong)"


def _check_completion(model_name: str, api_base: str, api_key: str, model_config: dict[str, Any]) -> tuple[bool, str]:
    """One real completion through the exact transport the typing pass uses."""
    transport = LiteLLMStructuredModel(
        model=model_name,
        api_base=api_base,
        api_key=api_key,
        timeout_seconds=float(model_config.get("timeout_seconds", 90)),
        # A liveness probe must fail fast: the production pass retries 429 forever,
        # which would turn "quota exhausted" into a hang instead of a verdict.
        max_attempts=2,
        temperature=float(model_config.get("temperature", 0.0)),
        retry_backoff_base_seconds=1.0,
        retry_backoff_max_seconds=4.0,
        retry_jitter_seconds=0.5,
        structured_schema_profile=str(model_config.get("structured_schema_profile", "prompt")),
    )
    messages = (
        SystemMessage(content="You assign a short ontology type label to an entity."),
        HumanMessage(content='Entity: "Paris". Return its type label.'),
    )
    started = time.monotonic()
    try:
        result = transport.invoke(
            operation="gateway_smoke_probe",
            messages=messages,
            output_schema=_PROBE_SCHEMA,
            idempotency_key="gateway-smoke-probe",
        )
    except Exception as exc:  # noqa: BLE001 - probe reports every failure shape
        elapsed = time.monotonic() - started
        _emit(
            "completion_failed",
            latency_seconds=round(elapsed, 2),
            error_type=type(exc).__name__,
            detail=_redact(str(exc), api_key)[:600],
        )
        return False, f"model call failed: {type(exc).__name__}"
    elapsed = time.monotonic() - started
    _emit("completion_ok", latency_seconds=round(elapsed, 2), response=dict(result))
    return True, "ok"


def main() -> int:
    gateway_url = os.environ.get("HALLU_GATEWAY_URL", "").strip()
    api_key = os.environ.get("HALLU_GATEWAY_API_KEY", "").strip()
    model_name = os.environ.get("HALLU_TYPING_MODEL", "").strip() or _DEFAULT_MODEL

    missing = [name for name, value in (("HALLU_GATEWAY_URL", gateway_url), ("HALLU_GATEWAY_API_KEY", api_key)) if not value]
    if missing:
        _emit("verdict", ok=False, reason=f"missing environment: {', '.join(missing)}")
        return 1

    model_config = yaml.safe_load(_LIVE_CONFIG.read_text(encoding="utf-8")).get("model", {})
    api_base = _openai_api_base(gateway_url)
    _emit(
        "probe_start",
        gateway_url=gateway_url,
        api_base=api_base,
        model=model_name,
        api_key_fingerprint=f"len={len(api_key)},tail={api_key[-4:]}",
    )

    manifest_ok, manifest_reason = _check_manifest(gateway_url, api_key)
    if not manifest_ok:
        _emit("verdict", ok=False, stage="manifest", reason=manifest_reason)
        return 1

    completion_ok, completion_reason = _check_completion(model_name, api_base, api_key, model_config)
    if not completion_ok:
        _emit("verdict", ok=False, stage="completion", reason=completion_reason)
        return 1

    _emit("verdict", ok=True, reason="gateway reachable and model answering; typed pass is safe to submit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
