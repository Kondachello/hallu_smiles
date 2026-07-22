"""Leak-safe paired detector orchestration with a compact immutable prediction archive."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifacts import RunArchive, read_jsonl, utc_now
from .contracts import DetectionResult, DetectorProtocol, STATUS_FAILED, assert_no_gold, make_detection_input, result_record


def _variant_id(method_key: str, detector: DetectorProtocol) -> str:
    """Return a stable identity for one configuration, not merely a method family."""
    return str(
        getattr(detector, "variant_id", None)
        or getattr(detector, "variant_name", None)
        or method_key
    )


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
) -> dict[str, Any]:
    """Run each adapter on immutable no-gold records.

    A controlled track supplies ``shared_graph_provider``.  It materializes the
    complete ``(context, query, response)`` graph bundle before any detector and
    attaches that bundle identity to every prediction row.  This makes graph identity,
    rather than matching configuration text, the proof that extraction was shared.
    """
    if not detectors:
        raise ValueError("at least one detector is required")
    instances = load_no_gold_instances(instances_path)
    existing = archive.read_jsonl("predictions/raw_predictions.jsonl") if resume else []
    completed = {
        (str(row.get("variant") or row["method"]), str(row["response_id"]))
        for row in existing
    }
    predictions = list(existing)
    stages = archive.read_jsonl("stages/stage_calls.jsonl") if resume else []

    archive.update_status("running_predictions", started_at_utc=utc_now(), gold_access_state="hidden")
    for record in instances:
        item = make_detection_input(record)
        response_id = str(record["response_id"])
        pending = [
            (
                method_key,
                detector,
                str(getattr(detector, "method_name", method_key)),
                _variant_id(method_key, detector),
            )
            for method_key, detector in detectors.items()
            if (_variant_id(method_key, detector), response_id) not in completed
        ]
        if not pending:
            continue
        shared_failure: Exception | None = None
        shared_ref: dict[str, str] = {}
        shared_cached = False
        if shared_graph_provider is not None:
            started_shared = time.perf_counter()
            try:
                shared = shared_graph_provider.prepare_instance(item)
                shared_ref = shared.reference()
                shared_cached = all(
                    artifact.cache_hit
                    for artifact in (shared.context, shared.query, shared.response)
                )
                shared_status = "ok"
            except Exception as exc:  # common graph failure is never a prediction
                shared_failure = exc
                shared_status = "model_failed"
            stages.append(
                {
                    "stage_call_id": f"{archive.run_id}:shared_graph_bundle:{response_id}",
                    "run_id": archive.run_id,
                    "method_run_id": f"{archive.run_id}:shared_preprocessing",
                    "source_id": item.source_id,
                    "response_id": item.response_id,
                    "stage_name": "shared_graph_bundle_materialization",
                    "component_name": "shared_kggen",
                    "status": shared_status,
                    "wall_time_ms": round((time.perf_counter() - started_shared) * 1000.0, 3),
                    "cached": shared_cached,
                    **shared_ref,
                    "gold_access_state": "hidden",
                }
            )
        for method_key, detector, method, variant in pending:
            started = time.perf_counter()
            if shared_failure is not None:
                result = _failed_result(item, method, RuntimeError(f"shared_graph_bundle: {shared_failure!r}"))
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
                    "method_run_id": f"{archive.run_id}:{variant}",
                    "prediction_id": f"{archive.run_id}:{variant}:{response_id}",
                    "method_family": method,
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

    predictions.sort(key=lambda row: (str(row.get("variant") or row["method"]), row["response_id"]))
    stages.sort(key=lambda row: (row["method_run_id"], row["response_id"]))
    archive.write_jsonl("predictions/raw_predictions.jsonl", predictions)
    archive.write_jsonl("stages/stage_calls.jsonl", stages)
    if shared_graph_provider is not None:
        archive.write_json("cache/cache_inventory.json", shared_graph_provider.inspection())
        archive.write_jsonl("cache/cache_resolution.jsonl", shared_graph_provider.resolution_records())
        archive.write_jsonl("shared_graphs/graph_index.jsonl", shared_graph_provider.artifact_records())
        archive.write_jsonl("shared_graphs/bundles.jsonl", shared_graph_provider.bundle_records())
    _write_detector_artifacts(archive, detectors)
    paired = pair_predictions(predictions)
    archive.write_jsonl("predictions/paired_predictions.jsonl", paired)
    archive.update_status("predictions_complete", finished_predictions_at_utc=utc_now())
    return {
        "n_instances": len(instances),
        "n_predictions": len(predictions),
        "methods": sorted({row["method"] for row in predictions}),
        "variants": sorted({str(row.get("variant") or row["method"]) for row in predictions}),
        "shared_graphs": len(shared_graph_provider.artifact_records()) if shared_graph_provider is not None else 0,
        "shared_graph_bundles": len(shared_graph_provider.bundle_records()) if shared_graph_provider is not None else 0,
    }


def _write_detector_artifacts(
    archive: RunArchive, detectors: Mapping[str, DetectorProtocol]
) -> None:
    """Let an adapter publish its own trace without making the runner method-aware."""
    collected: dict[str, list[dict[str, Any]]] = {}
    for detector in detectors.values():
        exporter = getattr(detector, "export_artifact_records", None)
        if not callable(exporter):
            continue
        for relative, rows in exporter().items():
            collected.setdefault(str(relative), []).extend(dict(row) for row in rows)
    for relative, rows in collected.items():
        unique = {json.dumps(row, ensure_ascii=False, sort_keys=True): row for row in rows}
        archive.write_jsonl(relative, [unique[key] for key in sorted(unique)])


def pair_predictions(predictions: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build a comparison matrix for two or more variants on immutable graph bundles."""
    by_response: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in predictions:
        variant = str(row.get("variant") or row["method"])
        by_response.setdefault(str(row["response_id"]), {})[variant] = row
    paired: list[dict[str, Any]] = []
    for response_id, by_variant in sorted(by_response.items()):
        rows = list(by_variant.values())
        bundle_ids = {
            str(row.get("shared_graph_bundle_id"))
            for row in rows
            if row.get("shared_graph_bundle_id")
        }
        response_graphs = {
            str(row.get("shared_graph_sha256"))
            for row in rows
            if row.get("shared_graph_sha256")
        }
        bundle_present_for_all = bool(bundle_ids) and all(
            row.get("shared_graph_bundle_id") for row in rows
        )
        bundle_consistent = bundle_present_for_all and len(bundle_ids) == 1
        response_consistent = len(response_graphs) in {0, 1}
        all_ok = bool(rows) and all(row.get("status") == "ok" for row in rows) and bundle_consistent
        by_family: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            by_family.setdefault(str(row["method"]), []).append(row)
        hallu_rows = by_family.get("hallugraph", [])
        graph_rows = by_family.get("grapheval", [])
        hallu = hallu_rows[0] if len(hallu_rows) == 1 else None
        graph = graph_rows[0] if len(graph_rows) == 1 else None
        both_ok = (
            hallu is not None
            and graph is not None
            and hallu.get("status") == "ok"
            and graph.get("status") == "ok"
            and response_consistent
        )
        disagreement = (
            abs(float(hallu["raw_score"]) - float(graph["raw_score"]))
            if both_ok and hallu.get("raw_score") is not None and graph.get("raw_score") is not None
            else None
        )
        paired.append(
            {
                "comparison_id": f"comparison:{response_id}",
                "response_id": response_id,
                "source_id": rows[0].get("source_id"),
                "variant_prediction_ids": {variant: row.get("prediction_id") for variant, row in sorted(by_variant.items())},
                "variant_statuses": {variant: row.get("status") for variant, row in sorted(by_variant.items())},
                "variant_scores": {variant: row.get("raw_score") for variant, row in sorted(by_variant.items())},
                "all_variants_status_ok": all_ok,
                "all_variants_have_graph_bundle": bundle_present_for_all,
                "all_variants_same_graph_bundle": bundle_consistent,
                "shared_graph_bundle_id": next(iter(bundle_ids)) if len(bundle_ids) == 1 else None,
                "hallugraph_prediction_id": hallu.get("prediction_id") if hallu else None,
                "grapheval_prediction_id": graph.get("prediction_id") if graph else None,
                "hallugraph_score": hallu.get("raw_score") if hallu else None,
                "grapheval_score": graph.get("raw_score") if graph else None,
                "both_status_ok": both_ok,
                "shared_response_graph_consistent": response_consistent,
                "shared_response_graph_sha256": next(iter(response_graphs)) if len(response_graphs) == 1 else None,
                "absolute_score_difference": disagreement,
                "gold_access_state": "hidden",
            }
        )
    return paired


def seal_run(archive: RunArchive, instances_path: str | Path) -> dict[str, Any]:
    instances = load_no_gold_instances(instances_path)
    return archive.seal_predictions(expected_response_ids=[row["response_id"] for row in instances])
