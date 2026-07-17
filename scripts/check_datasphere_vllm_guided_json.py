#!/usr/bin/env python3
"""Gate vLLM on the exact nested KGGen relation contract.

The historical filename is retained so existing bounded DataSphere Jobs keep
working.  The transport is intentionally no longer ``guided_json``: this
probe sends native OpenAI ``response_format.type=json_schema`` to the new
runtime before any three-QA extraction is allowed to start.
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

from src.dspy_adapter import (
    STRUCTURED_OUTPUT_PROTOCOL_VERSION,
    XGRAMMAR_STRICT_REQUEST_BACKEND,
    dspy_output_schema,
    json_schema_response_format,
    strict_json_schema_adapter,
    validate_json_document,
)


SWISS_SOURCE_ID = "15138"
SWISS_ENTITIES = [
    "Swiss chard",
    "spinach",
    "beetroot",
    "Beta vulgaris subsp. maritima",
    "sea beet",
    "pizzoccheri",
    "vegetable garden",
    "storage root",
    "leaf stalks",
    "crisper",
    "refrigerator",
    "plastic bags",
]
TWO_FACT_TEXT = (
    "Ada Lovelace wrote notes about Charles Babbage's Analytical Engine. "
    "Marie Curie was born in Warsaw."
)
TWO_FACT_ENTITIES = [
    "Ada Lovelace",
    "Charles Babbage",
    "Analytical Engine",
    "Marie Curie",
    "Warsaw",
]
DEFAULT_MAX_TOKENS = 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class ContractProbeError(RuntimeError):
    """A contract failure carrying the raw request/response evidence."""

    def __init__(self, message: str, evidence: dict[str, Any]):
        super().__init__(message)
        self.evidence = evidence


def _content_diagnostic(content: Any) -> dict[str, Any]:
    """Return bounded evidence for failures whose full artifact is unavailable."""
    if not isinstance(content, str):
        return {"content_type": type(content).__name__}
    longest_whitespace_run = 0
    current_whitespace_run = 0
    for character in content:
        if character.isspace():
            current_whitespace_run += 1
            longest_whitespace_run = max(
                longest_whitespace_run, current_whitespace_run
            )
        else:
            current_whitespace_run = 0
    return {
        "content_chars": len(content),
        "non_whitespace_chars": sum(not character.isspace() for character in content),
        "longest_whitespace_run": longest_whitespace_run,
        "prefix": content[:240],
        "suffix": content[-240:],
    }


def _load_swiss_source(data_dir: str | Path) -> str:
    source_path = Path(data_dir) / "source_info.jsonl"
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("source_id")) != SWISS_SOURCE_ID:
                continue
            passages = (row.get("source_info") or {}).get("passages")
            if not isinstance(passages, str) or not passages.strip():
                raise ValueError(f"source {SWISS_SOURCE_ID} has no passage text")
            return passages.strip()
    raise ValueError(f"source {SWISS_SOURCE_ID} not found in {source_path}")


def _relation_signature(entities: list[str]) -> Any:
    """Build KGGen 0.4's real fallback signature, not a toy schema."""
    from kg_gen.steps._2_get_relations import fallback_extraction_sig

    _, signature = fallback_extraction_sig(entities, is_conversation=False)
    return signature


def kggen_fallback_relation_schema(entities: list[str] | None = None) -> dict[str, Any]:
    """Expose the exact raw DSPy schema for offline contract tests/preflight."""
    return dspy_output_schema(_relation_signature(entities or SWISS_ENTITIES))


def _request_fixture(source_text: str, entities: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    signature = _relation_signature(entities)
    schema = dspy_output_schema(signature)
    messages = strict_json_schema_adapter().format(
        signature=signature,
        demos=[],
        inputs={"source_text": source_text, "entities": entities},
    )
    return messages, schema


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local-datasphere-key"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # nosec B310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM returned HTTP {exc.code}: {body}") from exc


def _semantic_assertion(case: str, relations: list[dict[str, str]]) -> None:
    if case == "two_fact":
        if len(relations) < 2:
            raise ValueError(f"two-fact contract returned only {len(relations)} relation(s)")
        endpoints = [
            (relation["subject"].casefold(), relation["object"].casefold())
            for relation in relations
        ]

        def contains(left: str, right: str) -> bool:
            return any(
                (left in subject and right in obj)
                or (left in obj and right in subject)
                for subject, obj in endpoints
            )

        if not (
            contains("ada lovelace", "analytical engine")
            or contains("ada lovelace", "charles babbage")
        ):
            raise ValueError("two-fact contract omitted the Ada-Lovelace fact")
        if not contains("marie curie", "warsaw"):
            raise ValueError("two-fact contract omitted the Marie-Curie/Warsaw fact")
        return
    if not relations:
        raise ValueError("Swiss-chard contract returned no relations")
    anchors = [
        (relation["subject"].casefold(), relation["object"].casefold())
        for relation in relations
    ]
    if not any(
        ("chard" in subject and ("spinach" in obj or "beet" in obj))
        or ("chard" in obj and ("spinach" in subject or "beet" in subject))
        for subject, obj in anchors
    ):
        raise ValueError("Swiss-chard relations omitted the spinach/beet endpoint fact")


def _run_case(
    *,
    case: str,
    source_text: str,
    entities: list[str],
    url: str,
    model_id: str,
    timeout_s: float,
    repeat: int,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    request_backend: str = XGRAMMAR_STRICT_REQUEST_BACKEND,
) -> dict[str, Any]:
    messages, schema = _request_fixture(source_text, entities)
    schema_sha256 = hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, repeat + 1):
        payload = {
            "model": model_id,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "messages": messages,
            "response_format": json_schema_response_format(
                schema, name=f"kggen_{case}_relations"
            ),
            "guided_decoding_backend": request_backend,
        }
        evidence: dict[str, Any] = {
            "attempt": attempt,
            "request": payload,
            "schema_sha256": schema_sha256,
        }
        attempts.append(evidence)
        try:
            body = _post_json(url, payload, timeout_s)
            evidence["response"] = body
            choices = body.get("choices") if isinstance(body, dict) else None
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError(f"{case} response must contain exactly one choice")
            choice = choices[0]
            finish_reason = choice.get("finish_reason")
            evidence["finish_reason"] = finish_reason
            if finish_reason != "stop":
                raise ValueError(f"{case} completion finish_reason={finish_reason!r}")
            content = (choice.get("message") or {}).get("content")
            if not isinstance(content, str):
                raise TypeError(f"{case} response content is not text")
            parsed = json.loads(content)
            evidence["parsed"] = parsed
            validate_json_document(parsed, schema)
            relations = parsed["relations"]
            _semantic_assertion(case, relations)
            evidence["relations_count"] = len(relations)
        except Exception as exc:
            choices = (
                evidence.get("response", {}).get("choices")
                if isinstance(evidence.get("response"), dict)
                else None
            )
            content = None
            if isinstance(choices, list) and choices:
                content = (choices[0].get("message") or {}).get("content")
            evidence["response_content_diagnostic"] = _content_diagnostic(content)
            evidence["error_type"] = type(exc).__name__
            evidence["error"] = str(exc)
            print(
                "[contract-probe:error] "
                + json.dumps(
                    {
                        "case": case,
                        "attempt": attempt,
                        "finish_reason": evidence.get("finish_reason"),
                        "response_content_diagnostic": evidence[
                            "response_content_diagnostic"
                        ],
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
            raise ContractProbeError(
                f"{case} contract attempt {attempt} failed: {exc}",
                {
                    "case": case,
                    "schema_sha256": schema_sha256,
                    "attempts": attempts,
                },
            ) from exc
    return {
        "case": case,
        "schema_sha256": schema_sha256,
        "attempts": attempts,
    }


def run_probe(
    *,
    port: int,
    model_id: str,
    data_dir: str | Path,
    timeout_s: float,
    repeat: int = 2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    request_backend: str = XGRAMMAR_STRICT_REQUEST_BACKEND,
) -> dict[str, Any]:
    if port <= 0:
        raise ValueError("port must be positive")
    if timeout_s <= 0:
        raise ValueError("timeout must be positive")
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if request_backend != XGRAMMAR_STRICT_REQUEST_BACKEND:
        raise ValueError(
            "request backend must pin XGrammar no-fallback with bounded whitespace"
        )
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    swiss_text = _load_swiss_source(data_dir)
    specifications = [
        ("swiss_chard", swiss_text, SWISS_ENTITIES),
        ("two_fact", TWO_FACT_TEXT, TWO_FACT_ENTITIES),
    ]
    cases: list[dict[str, Any]] = []
    for case, source_text, entities in specifications:
        try:
            cases.append(
                _run_case(
                    case=case,
                    source_text=source_text,
                    entities=entities,
                    url=url,
                    model_id=model_id,
                    timeout_s=timeout_s,
                    repeat=repeat,
                    max_tokens=max_tokens,
                    request_backend=request_backend,
                )
            )
        except ContractProbeError as exc:
            exc.evidence = {
                "completed_cases": cases,
                "failed_case": exc.evidence,
            }
            raise
    return {
        "checked_at_utc": _utc_now(),
        "status": "ready",
        "model_id": model_id,
        "source_id": SWISS_SOURCE_ID,
        "request_parameter": "response_format",
        "structured_output_protocol": STRUCTURED_OUTPUT_PROTOCOL_VERSION,
        "guided_decoding_request_backend": request_backend,
        "xgrammar_any_whitespace": False,
        "max_tokens": max_tokens,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--request-backend",
        choices=(XGRAMMAR_STRICT_REQUEST_BACKEND,),
        default=XGRAMMAR_STRICT_REQUEST_BACKEND,
    )
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report_path = Path(args.report)
    try:
        report = run_probe(
            port=args.port,
            model_id=args.model_id,
            data_dir=args.data_dir,
            timeout_s=args.timeout,
            repeat=args.repeat,
            max_tokens=args.max_tokens,
            request_backend=args.request_backend,
        )
    except Exception as exc:
        evidence = getattr(exc, "evidence", None)
        _atomic_json(
            report_path,
            {
                "checked_at_utc": _utc_now(),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "request_parameter": "response_format",
                "structured_output_protocol": STRUCTURED_OUTPUT_PROTOCOL_VERSION,
                "guided_decoding_request_backend": args.request_backend,
                "xgrammar_any_whitespace": False,
                "evidence": evidence,
            },
        )
        raise
    _atomic_json(report_path, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
