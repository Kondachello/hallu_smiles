"""Prediction record + JSONL writer (gold is attached later, by the experiment framework)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .types import DetectionInput, DetectionResult


def prediction_record(result: DetectionResult, item: DetectionInput) -> dict:
    return {
        "response_id": result.response_id,
        "source_id": result.source_id,
        "method": result.method,
        "raw_score": result.raw_score,
        "status": result.status,
        "flagged_unit_ids": list(result.flagged_unit_ids),
        "components": dict(result.components),
        "failure": dict(result.failure) if result.failure else None,
        "usage": dict(result.usage),
        "query_present": item.query is not None,  # preserved, never used as evidence
    }


def write_predictions_jsonl(
    path: str | Path, records: Iterable[dict]
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
