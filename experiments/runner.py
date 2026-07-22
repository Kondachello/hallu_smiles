"""Leak-safe paired detector orchestration with a compact immutable prediction archive."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

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
    shared_graph_provider: Any | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run each adapter on immutable no-gold records.

    A controlled track supplies ``shared_graph_provider``.  Its response graph is
    materialized before either detector and is then attached to both prediction rows.
    This makes graph identity, rather than matching configuration text, the proof that
    extraction was shared.
    """
    if not detectors:
        raise ValueError("at least one detector is required")
    instances = load_no_gold_instances(instances_path)
    existing = archive.read_jsonl("predictions/raw_predictions.jsonl") if resume else []
    completed = {(str(row["method"]), str(row["response_id"])) for row in existing}
    predictions = list(existing)
    stages = archive.read_jsonl("stages/stage_calls.jsonl") if resume else []

    archive.update_status("running_predictions", started_at_utc=utc_now(), gold_access_state="hidden")
    total_instances = len(instances)
    for index, record in enumerate(instances, start=1):
        item = make_detection_input(record)
        response_id = str(record["response_id"])
        if progress_callback is not None:
            progress_callback({
                "event": "record_started", "index": index, "total": total_instances,
                "response_id": response_id, "source_id": item.source_id,
            })
        pending = [
            (method_key, detector, getattr(detector, "method_name", method_key), getattr(detector, "variant_name", method_key))
            for method_key, detector in detectors.items()
            if (getattr(detector, "method_name", method_key), response_id) not in completed
        ]
        if not pending:
            continue
        shared_failure: Exception | None = None
        shared_ref: dict[str, str] = {}
        shared_cached = False
        if shared_graph_provider is not None:
            started_shared = time.perf_counter()
            try:
                shared = shared_graph_provider.prepare_response(item)
                shared_ref = shared.reference()
                shared_cached = bool(shared.cache_hit)
                shared_status = "ok"
            except Exception as exc:  # common graph failure is never a prediction
                shared_failure = exc
                shared_status = "model_failed"
            stages.append(
                {
                    "stage_call_id": f"{archive.run_id}:shared_response:{response_id}",
                    "run_id": archive.run_id,
                    "method_run_id": f"{archive.run_id}:shared_preprocessing",
                    "source_id": item.source_id,
                    "response_id": item.response_id,
                    "stage_name": "shared_response_extraction",
                    "component_name": "shared_kggen",
                    "status": shared_status,
                    "wall_time_ms": round((time.perf_counter() - started_shared) * 1000.0, 3),
                    "cached": shared_cached,
                    "gold_access_state": "hidden",
                }
            )
            if progress_callback is not None:
                progress_callback({
                    "event": "shared_graph_ready", "index": index, "total": total_instances,
                    "response_id": response_id, "status": shared_status, "cached": shared_cached,
                    "wall_time_ms": stages[-1]["wall_time_ms"],
                })
        for method_key, detector, method, variant in pending:
            started = time.perf_counter()
            if shared_failure is not None:
                result = _failed_result(item, method, RuntimeError(f"shared_response_extraction: {shared_failure!r}"))
            else:
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
            if shared_graph_provider is not None and shared_ref:
                row["artifact_refs"] = {**dict(row.get("artifact_refs", {})), **shared_ref}
                row.update(shared_ref)
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
            if progress_callback is not None:
                progress_callback({
                    "event": "detector_finished", "index": index, "total": total_instances,
                        "response_id": response_id, "method": method, "status": result.status,
                        "score": result.raw_score, "wall_time_ms": elapsed_ms,
                })
        if progress_callback is not None:
            progress_callback({
                "event": "record_finished", "index": index, "total": total_instances,
                "response_id": response_id,
            })

    predictions.sort(key=lambda row: (row["method"], row["response_id"]))
    stages.sort(key=lambda row: (row["method_run_id"], row["response_id"]))
    archive.write_jsonl("predictions/raw_predictions.jsonl", predictions)
    archive.write_jsonl("stages/stage_calls.jsonl", stages)
    if shared_graph_provider is not None:
        archive.write_json("cache/cache_inventory.json", shared_graph_provider.inspection())
        archive.write_jsonl("cache/cache_resolution.jsonl", shared_graph_provider.resolution_records())
        archive.write_jsonl("shared_graphs/graph_index.jsonl", shared_graph_provider.artifact_records())
    paired = pair_predictions(predictions)
    archive.write_jsonl("predictions/paired_predictions.jsonl", paired)
    archive.update_status("predictions_complete", finished_predictions_at_utc=utc_now())
    return {"n_instances": len(instances), "n_predictions": len(predictions), "methods": sorted({row["method"] for row in predictions}), "shared_graphs": len(shared_graph_provider.artifact_records()) if shared_graph_provider is not None else 0}


def pair_predictions(predictions: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_response: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in predictions:
        by_response.setdefault(str(row["response_id"]), {})[str(row["method"])] = row
    paired: list[dict[str, Any]] = []
    for response_id, by_method in sorted(by_response.items()):
        hallu = by_method.get("hallugraph")
        graph = by_method.get("grapheval")
        statuses = [row.get("status") for row in (hallu, graph) if row]
        shared_graphs = {
            str(row.get("shared_graph_sha256"))
            for row in (hallu, graph) if row and row.get("shared_graph_sha256")
        }
        shared_graph_consistent = len(shared_graphs) in {0, 1}
        both_ok = len(statuses) == 2 and all(status == "ok" for status in statuses) and shared_graph_consistent
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
                "shared_response_graph_consistent": shared_graph_consistent,
                "shared_response_graph_sha256": next(iter(shared_graphs)) if len(shared_graphs) == 1 else None,
                "absolute_score_difference": disagreement,
                "gold_access_state": "hidden",
            }
        )
    return paired


def seal_run(archive: RunArchive, instances_path: str | Path) -> dict[str, Any]:
    instances = load_no_gold_instances(instances_path)
    return archive.seal_predictions(expected_response_ids=[row["response_id"] for row in instances])
