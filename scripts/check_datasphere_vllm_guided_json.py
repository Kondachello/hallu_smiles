#!/usr/bin/env python3
"""Verify vLLM's native JSON-schema constrained-decoding contract.

This is deliberately a direct, tiny OpenAI-compatible request.  It proves the
exact ``guided_json`` transport used by the DataSphere DSPy adapter before the
Job reaches any KGGen extraction or clustering call.  A plain chat healthcheck
cannot establish that guarantee.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dspy_adapter import canonicalize_vllm_guided_json_schema


def kggen_fallback_relation_schema() -> dict[str, Any]:
    """Build the exact dynamic fallback schema used by KGGen 0.4.

    KGGen first tries a relation signature, then uses this fallback when a
    typed relation response is malformed.  It contains the DSPy-only metadata
    that previously differentiated a passing toy probe from the failing real
    reference.  Exercise the production canonicaliser itself, not a hand-made
    approximation.
    """
    from dspy.adapters.json_adapter import _get_structured_outputs_response_format
    from kg_gen.steps._2_get_relations import fallback_extraction_sig

    _, signature = fallback_extraction_sig(
        ["Swiss chard", "spinach", "beetroot", "sea beet"],
        is_conversation=False,
    )
    output_model = _get_structured_outputs_response_format(signature)
    schema = canonicalize_vllm_guided_json_schema(output_model.model_json_schema())
    serialized = json.dumps(schema, sort_keys=True)
    if "#/$defs/" in serialized or "__dspy_" in serialized or '"desc"' in serialized:
        raise RuntimeError("KGGen fallback schema was not fully canonicalised for vLLM")
    return schema


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_probe(*, port: int, model_id: str, timeout_s: float) -> dict[str, Any]:
    if port <= 0:
        raise ValueError("port must be positive")
    if timeout_s <= 0:
        raise ValueError("timeout must be positive")
    schema = kggen_fallback_relation_schema()
    payload = {
        "model": model_id,
        "temperature": 0.0,
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": "Return exactly one JSON object with a relations array. The array may be empty.",
            }
        ],
        # vLLM 0.6.3's documented native structured-output parameter.  This
        # avoids its known response_format compatibility regression.
        "guided_json": schema,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local-datasphere-key"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # nosec B310 - fixed localhost URL
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise TypeError("vLLM response content is not text")
    parsed = json.loads(content)
    if not isinstance(parsed, dict) or set(parsed) != {"relations"}:
        raise ValueError("guided_json response did not contain exactly the required relations field")
    if not isinstance(parsed["relations"], list):
        raise ValueError("guided_json response relations is not an array")
    for relation in parsed["relations"]:
        if not isinstance(relation, dict) or set(relation) != {"subject", "predicate", "object"}:
            raise ValueError("guided_json response relation does not have the closed KGGen triple shape")
        if not all(isinstance(relation[field], str) for field in ("subject", "predicate", "object")):
            raise ValueError("guided_json response relation fields are not strings")
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "ready",
        "model_id": model_id,
        "request_parameter": "guided_json",
        "schema_transport": "local_defs_inlined_and_dspy_annotations_removed",
        "schema_sha256": hashlib.sha256(
            json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "relations_count": len(parsed["relations"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report_path = Path(args.report)
    try:
        report = run_probe(port=args.port, model_id=args.model_id, timeout_s=args.timeout)
    except Exception as exc:
        _atomic_json(
            report_path,
            {
                "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "request_parameter": "guided_json",
            },
        )
        raise
    _atomic_json(report_path, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
