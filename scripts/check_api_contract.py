#!/usr/bin/env python3
"""Run the exact real KGGen Swiss-chard relation contract three times.

This is intentionally not a toy ``{"ok": true}`` JSON request.  It calls
``KGExtractor.relation_contract`` which uses KGGen 0.4's primary typed relation
signature inside the same DSPy JSON-object adapter as normal extraction.  A
bare Relation, repaired wrapper, code fence, extra field or malformed root is
therefore a hard failure before any RAGTruth pilot work begins.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


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


class ContractProbeError(RuntimeError):
    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_swiss_source(data_dir: str | Path) -> str:
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


def _relations_list(relations: Iterable[tuple[str, str, str]]) -> list[list[str]]:
    normalized: list[list[str]] = []
    for relation in relations:
        if not isinstance(relation, (tuple, list)) or len(relation) != 3:
            raise ValueError(f"relation contract returned a non-triple: {relation!r}")
        triple = [str(value) for value in relation]
        if any(not value.strip() for value in triple):
            raise ValueError(f"relation contract returned an empty triple field: {triple!r}")
        normalized.append(triple)
    normalized.sort()
    return normalized


def _require_swiss_anchor(relations: list[list[str]]) -> None:
    if not relations:
        raise ValueError("Swiss-chard contract returned no relations")
    for subject, _predicate, obj in relations:
        left, right = subject.casefold(), obj.casefold()
        if (
            ("chard" in left and ("spinach" in right or "beet" in right))
            or ("chard" in right and ("spinach" in left or "beet" in left))
        ):
            return
    raise ValueError("Swiss-chard relations omitted the spinach/beet endpoint fact")


def run_contract_probe(
    extractor: Any,
    data_dir: str | Path,
    *,
    repeat: int = 3,
) -> dict[str, Any]:
    if repeat != 3:
        raise ValueError("the API gate requires exactly three independent contract attempts")
    source_text = load_swiss_source(data_dir)
    attempts: list[dict[str, Any]] = []
    for number in range(1, repeat + 1):
        started = time.perf_counter()
        try:
            relations = _relations_list(
                extractor.relation_contract(source_text, list(SWISS_ENTITIES))
            )
            _require_swiss_anchor(relations)
        except Exception as exc:
            attempts.append({
                "attempt": number,
                "status": "error",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            raise ContractProbeError(
                f"Swiss relation contract attempt {number} failed",
                {
                    "protocol": "hallu-api-json-object-contract-v1",
                    "checked_at_utc": _utc_now(),
                    "status": "error",
                    "source_id": SWISS_SOURCE_ID,
                    "transport": "json_object",
                    "repair_allowed": False,
                    "passed": sum(attempt["status"] == "ready" for attempt in attempts),
                    "attempts": attempts,
                },
            ) from exc
        attempts.append({
            "attempt": number,
            "status": "ready",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "relations_count": len(relations),
            "relations": relations,
        })
    return {
        "protocol": "hallu-api-json-object-contract-v1",
        "checked_at_utc": _utc_now(),
        "status": "ready",
        "source_id": SWISS_SOURCE_ID,
        "transport": "json_object",
        "repair_allowed": False,
        "passed": len(attempts),
        "attempts": attempts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--cache-dir")
    args = parser.parse_args()

    from src.config import load_config, resolve_api_key
    from src.extract import KGExtractor, UsageLogger

    cfg = load_config(args.config)
    if args.cache_dir:
        cfg._data["cache_dir"] = args.cache_dir  # noqa: SLF001
        cfg.cache_dir = args.cache_dir
    if not resolve_api_key(cfg):
        raise SystemExit(f"required API secret {cfg.llm.api_key_env!r} is absent or empty")
    report_path = Path(args.report)
    usage = UsageLogger(
        report_path.with_name("contract_usage.jsonl"),
        provider_calls_path=report_path.with_name("provider_calls.jsonl"),
    )
    extractor = KGExtractor(cfg, usage=usage)
    try:
        report = run_contract_probe(extractor, args.data_dir)
    except Exception as exc:
        failure = getattr(exc, "report", {
            "protocol": "hallu-api-json-object-contract-v1",
            "checked_at_utc": _utc_now(),
            "status": "error",
            "repair_allowed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        failure["usage"] = usage.summary()
        _atomic_json(report_path, failure)
        raise
    report["usage"] = usage.summary()
    _atomic_json(report_path, report)
    print(json.dumps({"status": "ready", "passed": report["passed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
