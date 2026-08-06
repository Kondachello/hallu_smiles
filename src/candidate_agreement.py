"""Redacted paired evaluation utilities for the candidate-agreement baseline.

The detector itself lives beside SemanticEntropy because it reuses that
sample-cache contract.  This module deliberately knows only response IDs,
labels and scalar scores: it validates the exact graph-scored pairing from R12
and calculates train-only thresholds and paired uncertainty without loading a
prompt, candidate answer, completion, or graph.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .tune import prf_at_threshold, safe_auc, select_f1_threshold


GRAPH_REFERENCE_PROTOCOL = "ragtruth-r12-paired-graph-reference-v1"
GRAPH_METHODS = ("strict", "support", "support_critical")


class GraphReferenceError(ValueError):
    """The historical reference is not a safe paired-comparison contract."""


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class GraphReferenceRow:
    source_id: str
    response_id: str
    split: str
    y: int
    scores: Mapping[str, float]


@dataclass(frozen=True)
class GraphReference:
    manifest_sha256: str
    rows: tuple[GraphReferenceRow, ...]
    frozen_thresholds: Mapping[str, float]
    archive_sha256: str

    @property
    def response_ids(self) -> frozenset[str]:
        return frozenset(row.response_id for row in self.rows)


def _finite_score(value: Any, name: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise GraphReferenceError(f"graph reference has no finite {name} score") from exc
    if not math.isfinite(score):
        raise GraphReferenceError(f"graph reference has no finite {name} score")
    return score


def load_graph_reference(path: str | Path, *, manifest_sha256: str) -> GraphReference:
    """Load a redacted, scalar-only extraction of the verified R12 archive."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise GraphReferenceError("cannot read the redacted R12 graph reference") from exc
    if not isinstance(payload, dict) or payload.get("protocol") != GRAPH_REFERENCE_PROTOCOL:
        raise GraphReferenceError("graph reference protocol mismatch")
    if payload.get("manifest_sha256") != manifest_sha256:
        raise GraphReferenceError("graph reference manifest is not the fixed historical 750-row manifest")
    archive_sha256 = payload.get("archive_sha256")
    if not isinstance(archive_sha256, str) or len(archive_sha256) != 64:
        raise GraphReferenceError("graph reference has no verified archive checksum")
    thresholds = payload.get("frozen_thresholds")
    if not isinstance(thresholds, dict):
        raise GraphReferenceError("graph reference has no frozen train-only thresholds")
    frozen_thresholds = {method: _finite_score(thresholds.get(method), method) for method in GRAPH_METHODS}
    raw_rows = payload.get("records")
    if not isinstance(raw_rows, list):
        raise GraphReferenceError("graph reference has no records")
    rows: list[GraphReferenceRow] = []
    ids: set[str] = set()
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise GraphReferenceError("graph reference record is malformed")
        source_id, response_id, split = raw.get("source_id"), raw.get("response_id"), raw.get("split")
        if not all(isinstance(item, str) and item for item in (source_id, response_id)):
            raise GraphReferenceError("graph reference record lacks source or response ID")
        if split not in {"train", "test"}:
            raise GraphReferenceError("graph reference record has invalid split")
        if raw.get("y") not in {0, 1}:
            raise GraphReferenceError("graph reference record has invalid label")
        if response_id in ids:
            raise GraphReferenceError("graph reference repeats a response ID")
        ids.add(response_id)
        raw_scores = raw.get("scores")
        if not isinstance(raw_scores, dict):
            raise GraphReferenceError("graph reference record has no scalar scores")
        rows.append(GraphReferenceRow(
            source_id=source_id,
            response_id=response_id,
            split=split,
            y=int(raw["y"]),
            scores={method: _finite_score(raw_scores.get(method), method) for method in GRAPH_METHODS},
        ))
    reference = GraphReference(
        manifest_sha256=manifest_sha256,
        rows=tuple(sorted(rows, key=lambda item: (item.split, item.source_id, item.response_id))),
        frozen_thresholds=frozen_thresholds,
        archive_sha256=archive_sha256,
    )
    _validate_r12_pairing(reference)
    return reference


def _validate_r12_pairing(reference: GraphReference) -> None:
    """Freeze the known R12 paired denominator before candidate scoring."""

    train = [row for row in reference.rows if row.split == "train"]
    test = [row for row in reference.rows if row.split == "test"]
    # Source 12448 was the sole training quarantine.  The three empty answer
    # graphs occurred on held-out factual rows, giving the audited 147-row
    # denominator used by every historical headline method.
    if len(train) != 599 or len(test) != 147:
        raise GraphReferenceError("R12 graph reference does not have the required 599-train / 147-test pairing")
    if sum(row.y for row in train) != 300 or sum(row.y for row in test) != 75:
        raise GraphReferenceError("R12 graph reference label balance is not the verified pairing")
    if len(test) - sum(row.y for row in test) != 72:
        raise GraphReferenceError("R12 graph reference must contain 72 factual held-out responses")


def validate_reference_against_candidate_rows(
    reference: GraphReference, candidate_rows: Iterable[Mapping[str, Any]]
) -> None:
    """Ensure pair IDs, splits and labels match before tuning any threshold."""

    by_id = {str(row["response_id"]): row for row in candidate_rows}
    if len(by_id) < len(reference.rows):
        raise GraphReferenceError("candidate rows have incomplete graph-scored coverage")
    for graph_row in reference.rows:
        candidate = by_id.get(graph_row.response_id)
        if candidate is None:
            raise GraphReferenceError("candidate rows omit a graph-scored response")
        if (
            str(candidate.get("source_id")) != graph_row.source_id
            or str(candidate.get("split")) != graph_row.split
            or int(candidate.get("y")) != graph_row.y
        ):
            raise GraphReferenceError("candidate and graph reference IDs, split, or label disagree")


def _bootstrap_metric(
    scores: np.ndarray, labels: np.ndarray, *, metric: str, threshold: float | None,
    n_bootstrap: int, seed: int,
) -> tuple[float, float]:
    if len(scores) < 3 or len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(scores))
    values: list[float] = []
    for _ in range(n_bootstrap):
        sample = rng.choice(indices, size=len(indices), replace=True)
        if len(np.unique(labels[sample])) < 2:
            continue
        value = (
            safe_auc(scores[sample], labels[sample])
            if metric == "roc_auc"
            else prf_at_threshold(scores[sample], labels[sample], float(threshold))[2]
        )
        if math.isfinite(value):
            values.append(float(value))
    if not values:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.percentile(np.asarray(values), [2.5, 97.5]))


def _method_metrics(
    scores: np.ndarray, labels: np.ndarray, *, threshold: float, n_bootstrap: int, seed: int
) -> dict[str, Any]:
    precision, recall, f1 = prf_at_threshold(scores, labels, threshold)
    return {
        "roc_auc": float(safe_auc(scores, labels)),
        "roc_auc_ci95": list(_bootstrap_metric(
            scores, labels, metric="roc_auc", threshold=None,
            n_bootstrap=n_bootstrap, seed=seed,
        )),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f1_ci95": list(_bootstrap_metric(
            scores, labels, metric="f1", threshold=threshold,
            n_bootstrap=n_bootstrap, seed=seed + 1,
        )),
    }


def _paired_difference(
    scores: np.ndarray, support_critical: np.ndarray, labels: np.ndarray,
    *, score_threshold: float, support_critical_threshold: float, metric: str,
    n_bootstrap: int, seed: int,
) -> dict[str, Any]:
    def calculate(left: np.ndarray, right: np.ndarray, y: np.ndarray) -> float:
        if metric == "roc_auc":
            return float(safe_auc(left, y) - safe_auc(right, y))
        return float(
            prf_at_threshold(left, y, score_threshold)[2]
            - prf_at_threshold(right, y, support_critical_threshold)[2]
        )

    observed = calculate(scores, support_critical, labels)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    values: list[float] = []
    for _ in range(n_bootstrap):
        sample = rng.choice(indices, size=len(indices), replace=True)
        if len(np.unique(labels[sample])) < 2:
            continue
        values.append(calculate(scores[sample], support_critical[sample], labels[sample]))
    interval = [float("nan"), float("nan")]
    if values:
        interval = [float(value) for value in np.percentile(np.asarray(values), [2.5, 97.5])]
    return {"estimate": observed, "paired_bootstrap_ci95": interval}


def evaluate_paired_candidate_agreement(
    candidate_rows: Iterable[Mapping[str, Any]], reference: GraphReference, *,
    n_bootstrap: int = 1000, seed: int = 42,
) -> dict[str, Any]:
    """Evaluate four methods on R12's exact 147 held-out response IDs."""

    rows = list(candidate_rows)
    validate_reference_against_candidate_rows(reference, rows)
    candidate_by_id = {str(row["response_id"]): row for row in rows}
    paired_train = [row for row in reference.rows if row.split == "train"]
    paired_test = [row for row in reference.rows if row.split == "test"]
    candidate_train_scores = np.asarray([
        float(candidate_by_id[row.response_id]["candidate_disagreement"])
        for row in paired_train
    ], dtype=float)
    candidate_train_y = np.asarray([row.y for row in paired_train], dtype=int)
    candidate_threshold, candidate_train_f1 = select_f1_threshold(
        candidate_train_scores, candidate_train_y
    )
    labels = np.asarray([row.y for row in paired_test], dtype=int)
    scores: dict[str, np.ndarray] = {
        "candidate_agreement": np.asarray([
            float(candidate_by_id[row.response_id]["candidate_disagreement"])
            for row in paired_test
        ], dtype=float),
        **{
            method: np.asarray([row.scores[method] for row in paired_test], dtype=float)
            for method in GRAPH_METHODS
        },
    }
    thresholds = {
        "candidate_agreement": float(candidate_threshold),
        **{method: float(reference.frozen_thresholds[method]) for method in GRAPH_METHODS},
    }
    methods = {
        method: _method_metrics(
            method_scores, labels, threshold=thresholds[method],
            n_bootstrap=n_bootstrap, seed=seed + position * 23,
        )
        for position, (method, method_scores) in enumerate(scores.items())
    }
    paired = {
        method: {
            metric: _paired_difference(
                method_scores, scores["support_critical"], labels,
                score_threshold=thresholds[method],
                support_critical_threshold=thresholds["support_critical"],
                metric=metric, n_bootstrap=n_bootstrap,
                seed=seed + 101 + method_index * 29 + metric_index,
            )
            for metric_index, metric in enumerate(("roc_auc", "f1"))
        }
        for method_index, (method, method_scores) in enumerate(scores.items())
        if method != "support_critical"
    }
    return {
        "protocol": "ragtruth-candidate-agreement-paired-evaluation-v1",
        "comparison": "common_r12_graph_scored_responses",
        "threshold_selection": {
            "candidate_agreement": {
                "split": "graph_scored_train_only",
                "objective": "max_f1",
                "theta": float(candidate_threshold),
                "train_f1": float(candidate_train_f1),
                "n": len(paired_train),
            },
            "graph_methods": {
                "source": "verified_r12_train_only_freeze",
                "thresholds": {method: thresholds[method] for method in GRAPH_METHODS},
            },
        },
        "heldout_test": {
            "n": len(paired_test),
            "n_hallucinated": int(labels.sum()),
            "n_factual": int(len(labels) - labels.sum()),
            "methods": methods,
        },
        "paired_vs_support_critical": paired,
        "graph_reference_archive_sha256": reference.archive_sha256,
    }
