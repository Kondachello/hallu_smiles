#!/usr/bin/env python3
"""Exercise the support verifier's strict JSON-schema path against local vLLM."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.verifier import RelationVerifier, VERDICT_SCHEMA, select_evidence


CLAIM = ("Paris", "is capital of", "France")
EVIDENCE = "Paris is the capital of France."


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_probe(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    verifier = RelationVerifier(cfg)
    if verifier.structured_output.transport != "response_format":
        raise RuntimeError(
            "verifier probe requires llm.structured_output_transport=response_format"
        )
    evidence = select_evidence(
        EVIDENCE,
        None,
        CLAIM,
        max_sentences=verifier.max_sentences,
        stopwords=verifier.stopwords,
    )
    if not evidence:
        raise RuntimeError("verifier probe failed to select its deterministic evidence")
    started = time.perf_counter()
    verdict = verifier._call_llm(CLAIM, evidence)
    if verdict != "entailed":
        raise RuntimeError(f"verifier probe expected entailed, received {verdict!r}")
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "ready",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "model": verifier.model,
        "transport": verifier.structured_output.transport,
        "backend": verifier.structured_output.backend,
        "claim": list(CLAIM),
        "evidence": [span.to_dict() for span in evidence],
        "schema": VERDICT_SCHEMA,
        "schema_sha256": hashlib.sha256(
            json.dumps(VERDICT_SCHEMA, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "finish_reason": "stop",
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report_path = Path(args.report)
    try:
        report = run_probe(args.config)
    except Exception as exc:
        _atomic_json(
            report_path,
            {
                "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    _atomic_json(report_path, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
