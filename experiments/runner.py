"""Leak-safe paired detector orchestration with a compact immutable prediction archive."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifacts import RunArchive, read_jsonl, utc_now
from .contracts import DetectionResult, DetectorProtocol, STATUS_FAILED, assert_no_gold, make_detection_input, result_record


def load_no_gold_instances(path: str | Path) -> list[dict[str, Any]]:
    instances = read_jsonl(path)
    ids: set[str] = set()
    for record in instances:
        assert_no_gold(record)
        response_id = str(record.get("response_id", ""))
        if not response_id or response_id in ids:
            raise ValueError(f"duplicate or empty response_id in no-gold input: {response_id!r}")
        ids.add(response_id)
    return instances


def _failed_result(item, method: str, exc: Exception) -> DetectionResult:
    return DetectionResult(
        item.response_id,
        item.source_id,
        method,
        None,
        {},
        (),
        STATUS_FAILED,
        {"stage": "predict", "error": repr(exc)},
        {},
        {},
    )


def run_paired(
    archive: RunArchive,
    *,
    instances_path: str | Path,
    detectors: Mapping[str, DetectorProtocol],
    resume: bool = False,
) -> dict[str, Any]:
    """Run each supplied adapter on exactly the same immutable no-gold records."""
    if not detectors:
        raise ValueError("at least one detector is required")
    instances = load_no_gold_instances(instances_path)
    existing = archive.read_jsonl("predictions/raw_predictions.jsonl") if resume else []
    completed = {(str(row["method"]), str(row["response_id"])) for row in existing}
    predictions = list(existing)
    stages = archive.read_jsonl("stages/stage_calls.jsonl") if resume else []

    archive.update_status("running_predictions", started_at_utc=utc_now(), gold_access_state="hidden")
    for method_key, detector in detectors.items():
        method = getattr(detector, "method_name", method_key)
        variant = getattr(detector, "variant_name", method_key)
        for record in instances:
            response_id = str(record["response_id"])
            if (method, response_id) in completed:
                continue
            item = make_detection_input(record)
            started = time.perf_counter()
            try:
                result = detector.predict(item)
            except Exception as exc:  # a per-item failure is not a positive prediction
                result = _failed_result(item, method, exc)
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
            row = result_record(result, variant=variant)
            row.update(
                {
                    "run_id": archive.run_id,
                    "method_run_id": f"{archive.run_id}:{method}:{variant}",
                    "prediction_id": f"{archive.run_id}:{method}:{response_id}",
                    "dataset_record_id": record.get("dataset_record_id"),
                    "gold_access_state": "hidden",
                    "completed_at_utc": utc_now(),
                }
            )
            predictions.append(row)
            stages.append(
                {
                    "stage_call_id": f"{archive.run_id}:{method}:{response_id}:predict",
                    "run_id": archive.run_id,
                    "method_run_id": row["method_run_id"],
                    "source_id": result.source_id,
                    "response_id": result.response_id,
                    "stage_name": "detector_predict",
                    "component_name": method,
                    "status": "ok" if result.status != STATUS_FAILED else "model_failed",
                    "wall_time_ms": elapsed_ms,
                    "cached": False,
                    "gold_access_state": "hidden",
                }
            )

    predictions.sort(key=lambda row: (row["method"], row["response_id"]))
    stages.sort(key=lambda row: (row["method_run_id"], row["response_id"]))
    archive.write_jsonl("predictions/raw_predictions.jsonl", predictions)
    archive.write_jsonl("stages/stage_calls.jsonl", stages)
    paired = pair_predictions(predictions)
    archive.write_jsonl("predictions/paired_predictions.jsonl", paired)
    archive.update_status("predictions_complete", finished_predictions_at_utc=utc_now())
    return {"n_instances": len(instances), "n_predictions": len(predictions), "methods": sorted({row["method"] for row in predictions})}


def pair_predictions(predictions: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_response: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in predictions:
        by_response.setdefault(str(row["response_id"]), {})[str(row["method"])] = row
    paired: list[dict[str, Any]] = []
    for response_id, by_method in sorted(by_response.items()):
        hallu = by_method.get("hallugraph")
        graph = by_method.get("grapheval")
        statuses = [row.get("status") for row in (hallu, graph) if row]
        both_ok = len(statuses) == 2 and all(status == "ok" for status in statuses)
        if hallu and graph and both_ok:
            h_score, g_score = hallu.get("raw_score"), graph.get("raw_score")
            disagreement = None if h_score is None or g_score is None else abs(float(h_score) - float(g_score))
        else:
            disagreement = None
        paired.append(
            {
                "response_id": response_id,
                "source_id": (hallu or graph or {}).get("source_id"),
                "hallugraph_prediction_id": hallu.get("prediction_id") if hallu else None,
                "grapheval_prediction_id": graph.get("prediction_id") if graph else None,
                "hallugraph_score": hallu.get("raw_score") if hallu else None,
                "grapheval_score": graph.get("raw_score") if graph else None,
                "both_status_ok": both_ok,
                "absolute_score_difference": disagreement,
                "gold_access_state": "hidden",
            }
        )
    return paired


def seal_run(archive: RunArchive, instances_path: str | Path) -> dict[str, Any]:
    instances = load_no_gold_instances(instances_path)
    return archive.seal_predictions(expected_response_ids=[row["response_id"] for row in instances])
