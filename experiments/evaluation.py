"""Post-seal gold join and dependency-light response-level metrics.

This module never invokes a detector.  It operates only on sealed predictions plus the
separate gold file, preserving the prediction/evaluation trust boundary.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifacts import RunArchive, read_jsonl, utc_now


def join_gold(archive: RunArchive, *, response_gold_path: str) -> list[dict[str, Any]]:
    """Attach response-level labels only after prediction sealing."""
    if not (archive.path / "prediction_seal.json").exists():
        raise ValueError("gold join requires prediction_seal.json")
    candidate = Path(response_gold_path)
    gold_rows = read_jsonl(candidate if candidate.is_absolute() else archive.path / candidate)
    gold_by_id = {str(row["response_id"]): row for row in gold_rows}
    predictions = archive.read_jsonl("predictions/raw_predictions.jsonl")
    joined: list[dict[str, Any]] = []
    for prediction in predictions:
        gold = gold_by_id.get(str(prediction["response_id"]))
        if gold is None:
            raise ValueError(f"prediction has no response-level gold: {prediction['response_id']}")
        joined.append(
            {
                **prediction,
                "gold_response_label": int(gold["gold_response_label"]),
                "gold_quality_raw": gold.get("quality_raw"),
                "gold_access_state": "joined_for_evaluation",
            }
        )
    joined.sort(key=lambda row: (str(row.get("variant") or row["method"]), row["response_id"]))
    archive.write_jsonl("evaluation/predictions_with_gold.jsonl", joined)
    archive.update_status("gold_joined", gold_joined_at_utc=utc_now(), gold_access_state="joined_for_evaluation")
    return joined


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    return None if not denominator else float(numerator) / float(denominator)


def _auroc(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    # Mann–Whitney U with average ranks for ties.
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][1] == ranked[index][1]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        for original_index, _ in ranked[index:end]:
            ranks[original_index] = avg_rank
        index = end
    sum_positive_ranks = sum(rank for rank, label in zip(ranks, labels) if label)
    return (sum_positive_ranks - positives * (positives + 1) / 2.0) / (positives * negatives)


def _auprc(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    if not positives:
        return None
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    true_positive = 0
    area = 0.0
    previous_recall = 0.0
    for index, (_, label) in enumerate(ordered, start=1):
        true_positive += int(label)
        recall = true_positive / positives
        precision = true_positive / index
        if label:
            area += (recall - previous_recall) * precision
            previous_recall = recall
    return area


def metrics_at_threshold(rows: Iterable[Mapping[str, Any]], *, threshold: float) -> dict[str, Any]:
    """Compute metrics only on score-bearing, status-ok rows and expose coverage."""
    rows = list(rows)
    usable = [row for row in rows if row.get("status") == "ok" and row.get("raw_score") is not None]
    scores = [float(row["raw_score"]) for row in usable]
    labels = [int(row["gold_response_label"]) for row in usable]
    decisions = [score > threshold for score in scores]
    tp = sum(decision and label == 1 for decision, label in zip(decisions, labels))
    fp = sum(decision and label == 0 for decision, label in zip(decisions, labels))
    tn = sum(not decision and label == 0 for decision, label in zip(decisions, labels))
    fn = sum(not decision and label == 1 for decision, label in zip(decisions, labels))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    balanced = None if recall is None or specificity is None else (recall + specificity) / 2.0
    return {
        "threshold": float(threshold),
        "threshold_comparator": ">",
        "n_total": len(rows),
        "n_scored": len(usable),
        "n_unscorable_or_failed": len(rows) - len(usable),
        "coverage": _safe_div(len(usable), len(rows)),
        "AUROC": _auroc(scores, labels),
        "AUPRC": _auprc(scores, labels),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced,
        "F1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def evaluate_joined_predictions(archive: RunArchive, *, thresholds: Mapping[str, float]) -> list[dict[str, Any]]:
    """Write one metric row per immutable variant from already joined predictions.

    ``thresholds`` is keyed by ``variant``.  The method-family fallback preserves
    compatibility with the existing two-method callers while variants are adopted.
    """
    rows = archive.read_jsonl("evaluation/predictions_with_gold.jsonl")
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row.get("variant") or row["method"])].append(row)
    metrics: list[dict[str, Any]] = []
    for variant, variant_rows in sorted(by_variant.items()):
        method = str(variant_rows[0]["method"])
        threshold = thresholds.get(variant, thresholds.get(method))
        if threshold is None:
            raise ValueError(f"no frozen/explicit threshold supplied for variant={variant!r}")
        metrics.append(
            {
                "metric_id": f"{archive.run_id}:{variant}:overall",
                "run_id": archive.run_id,
                "method": method,
                "variant": variant,
                "split": "unknown_from_prediction_archive",
                "slice_definition": "overall",
                "gold_policy": "primary_all_labels",
                "failure_policy": "complete_case_with_coverage",
                "bootstrap_unit": "source_id",
                "gold_access_state": "joined_for_evaluation",
                **metrics_at_threshold(variant_rows, threshold=float(threshold)),
            }
        )
    archive.write_jsonl("evaluation/metrics.jsonl", metrics)
    return metrics
