#!/usr/bin/env python3
"""Score cached entropy samples with candidate agreement on the R12 ID overlap.

The script is deliberately sample-cache-only: Gemini sampling is disabled at
construction time, so a missing sample is a hard failure before any network
request.  It writes only scalar scores, IDs, hashes, and aggregate metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache import CacheOnlyMissError, config_value
from src.candidate_agreement import GRAPH_METHODS, GraphReferenceError, load_graph_reference
from src.config import load_config
from src.data import Instance, load_instances
from src.semantic_entropy import (
    CANDIDATE_AGREEMENT_PROTOCOL,
    CandidateAgreementDetector,
    CandidateAgreementEmptyCandidateError,
    SemanticEntropyDetector,
    SemanticUsageLogger,
)
from src.tune import prf_at_threshold, safe_auc, select_f1_threshold


PROTOCOL = "ragtruth-candidate-agreement-r12-paired-entropy-subset-v1"
R12_MANIFEST_SHA256 = "19cb9472e1662ac029dab7e144e07267c9e43f7ca50556aa92123a5e268e4f86"
EXPECTED_ENTROPY_ROWS = 559
EXPECTED_PAIRED_ROWS = 496
EXPECTED_PAIRED_TRAIN = 405
EXPECTED_PAIRED_TEST = 91


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _entropy_ids(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        response_id = row.get("response_id")
        if row.get("state") != "scored" or not isinstance(response_id, str) or response_id in rows:
            raise ValueError("semantic-entropy source is not an exact scored-ID contract")
        if row.get("split") not in {"train", "test"} or row.get("y") not in {0, 1}:
            raise ValueError("semantic-entropy source has invalid split or label")
        rows[response_id] = {
            "source_id": str(row.get("source_id")), "response_id": response_id,
            "split": str(row["split"]), "y": int(row["y"]),
        }
    if len(rows) != EXPECTED_ENTROPY_ROWS:
        raise ValueError("semantic-entropy source does not contain the completed 559 scored rows")
    return rows


def _checkpoint_path(root: Path, response_id: str) -> Path:
    return root / "score-checkpoints" / f"{hashlib.sha256(response_id.encode()).hexdigest()}.json"


def _valid_checkpoint(path: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("response_id") != row["response_id"] or value.get("source_id") != row["source_id"]:
            return None
        if value.get("state") != "scored":
            return None
        if not math.isfinite(float(value.get("candidate_disagreement"))):
            return None
        return value
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _bootstrap(scores: np.ndarray, labels: np.ndarray, *, metric: str, threshold: float | None, n: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    indices = np.arange(len(labels))
    for _ in range(n):
        sample = rng.choice(indices, size=len(indices), replace=True)
        if len(np.unique(labels[sample])) < 2:
            continue
        value = safe_auc(scores[sample], labels[sample]) if metric == "roc_auc" else prf_at_threshold(scores[sample], labels[sample], float(threshold))[2]
        if math.isfinite(value):
            values.append(float(value))
    return [float(value) for value in np.percentile(np.asarray(values), [2.5, 97.5])] if values else [float("nan"), float("nan")]


def _method_metrics(scores: np.ndarray, labels: np.ndarray, *, threshold: float, n_bootstrap: int, seed: int) -> dict[str, Any]:
    precision, recall, f1 = prf_at_threshold(scores, labels, threshold)
    return {
        "roc_auc": float(safe_auc(scores, labels)),
        "roc_auc_ci95": _bootstrap(scores, labels, metric="roc_auc", threshold=None, n=n_bootstrap, seed=seed),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "f1_ci95": _bootstrap(scores, labels, metric="f1", threshold=threshold, n=n_bootstrap, seed=seed + 1),
    }


def _paired_difference(left: np.ndarray, right: np.ndarray, labels: np.ndarray, *, left_threshold: float, right_threshold: float, metric: str, n_bootstrap: int, seed: int) -> dict[str, Any]:
    def value(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
        if metric == "roc_auc":
            return float(safe_auc(a, y) - safe_auc(b, y))
        return float(prf_at_threshold(a, y, left_threshold)[2] - prf_at_threshold(b, y, right_threshold)[2])
    observed = value(left, right, labels)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    indices = np.arange(len(labels))
    for _ in range(n_bootstrap):
        sample = rng.choice(indices, size=len(indices), replace=True)
        if len(np.unique(labels[sample])) == 2:
            values.append(value(left[sample], right[sample], labels[sample]))
    interval = [float(value) for value in np.percentile(np.asarray(values), [2.5, 97.5])] if values else [float("nan"), float("nan")]
    return {"estimate": float(observed), "paired_bootstrap_ci95": interval}


def _load_subset(args: argparse.Namespace, *, exclude_implicit_true: bool) -> tuple[list[dict[str, Any]], dict[str, Instance], Any]:
    entropy = _entropy_ids(Path(args.semantic_entropy_scores).resolve())
    reference = load_graph_reference(args.graph_reference, manifest_sha256=R12_MANIFEST_SHA256)
    graph_by_id = {row.response_id: row for row in reference.rows}
    response_ids = sorted(set(entropy).intersection(graph_by_id))
    if len(response_ids) != EXPECTED_PAIRED_ROWS:
        raise GraphReferenceError("semantic entropy and R12 do not have the expected 496 paired IDs")
    instances = {item.response_id: item for item in load_instances(args.data_dir, exclude_implicit_true=exclude_implicit_true)}
    subset: list[dict[str, Any]] = []
    for response_id in response_ids:
        entropy_row, graph_row, instance = entropy[response_id], graph_by_id[response_id], instances.get(response_id)
        if instance is None or (entropy_row["source_id"], entropy_row["split"], entropy_row["y"]) != (graph_row.source_id, graph_row.split, graph_row.y):
            raise GraphReferenceError("paired ID provenance disagreement")
        if (instance.source_id, instance.split, instance.y) != (entropy_row["source_id"], entropy_row["split"], entropy_row["y"]):
            raise GraphReferenceError("dataset disagrees with completed entropy row")
        subset.append({**entropy_row, "graph_scores": dict(graph_row.scores)})
    train = [row for row in subset if row["split"] == "train"]
    test = [row for row in subset if row["split"] == "test"]
    if len(train) != EXPECTED_PAIRED_TRAIN or len(test) != EXPECTED_PAIRED_TEST or sum(row["y"] for row in train) != 224 or sum(row["y"] for row in test) != 55:
        raise GraphReferenceError("paired split denominator or label balance is invalid")
    return subset, instances, reference


def _run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    configured_manifest = config_value(getattr(cfg, "vertex_gateway", None), "manifest_sha256")
    if configured_manifest != args.required_gateway_manifest_sha256:
        raise ValueError("runtime gateway manifest is not cache-compatible")
    subset, instances, reference = _load_subset(
        args, exclude_implicit_true=bool(config_value(cfg.data, "exclude_implicit_true", False))
    )
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    usage = SemanticUsageLogger(output / "usage.jsonl")
    # This is the no-Gemini invariant: even a cache miss cannot construct or
    # call a gateway sampler.  Candidate comparison cache misses may run local NLI.
    samples = SemanticEntropyDetector(cfg, usage=usage, cache_only=True)
    if args.preflight:
        sample_count = 0
        try:
            for row in subset:
                sample_count += len(samples.samples_for_prompt(instances[row["response_id"]].prompt))
        except CacheOnlyMissError as exc:
            raise RuntimeError("paired candidate preflight found a missing semantic sample") from exc
        _atomic_json(output / "preflight.json", {
            "protocol": PROTOCOL, "state": "completed_sample_cache_preflight",
            "paired_sources": EXPECTED_PAIRED_ROWS, "sample_cache_hits": int(usage.summary()["cache_hits"]),
            "samples_loaded": sample_count, "gemini_api_calls": 0, "nli_pair_evaluations": 0,
        })
        return 0
    detector = CandidateAgreementDetector(cfg, sample_detector=samples, cache_only=bool(args.replay))
    records: list[dict[str, Any]] = []
    for index, row in enumerate(subset):
        checkpoint = _checkpoint_path(output, row["response_id"])
        existing = _valid_checkpoint(checkpoint, row)
        if existing is not None and not args.replay:
            records.append(existing)
            continue
        instance = instances[row["response_id"]]
        try:
            score = detector.score_candidate(instance.prompt, instance.response)
        except CacheOnlyMissError as exc:
            raise RuntimeError("sample-cache-only candidate run encountered a missing cache entry") from exc
        except CandidateAgreementEmptyCandidateError as exc:
            raise RuntimeError("paired candidate agreement has an unexpected empty historical answer") from exc
        record = {
            "protocol": PROTOCOL, "state": "scored", "source_id": row["source_id"],
            "response_id": row["response_id"], "split": row["split"], "y": row["y"],
            "candidate_agreement_mass": float(score.agreement_mass),
            "candidate_disagreement": float(score.disagreement), "matched_samples": int(score.matched_samples),
            "n_samples": int(score.n_samples),
        }
        if not args.replay:
            _atomic_json(checkpoint, record)
        records.append(record)
        _atomic_json(output / "progress.json", {
            "protocol": PROTOCOL, "event": "source_completed", "sources_completed": index + 1,
            "sources_total": EXPECTED_PAIRED_ROWS, "gemini_api_calls": 0,
            "sample_cache_hits": int(usage.summary()["cache_hits"]),
            "nli_pair_evaluations": int(detector.nli_pair_evaluations),
            "candidate_comparison_cache_hits": int(detector.cache_hits),
        })
    records.sort(key=lambda row: (row["split"], row["source_id"], row["response_id"]))
    target = output / ("candidate_scores.replay.jsonl" if args.replay else "candidate_scores.jsonl")
    _write_jsonl(target, records)
    if args.replay:
        original = output / "candidate_scores.jsonl"
        if not original.exists() or _sha256_file(original) != _sha256_file(target):
            raise RuntimeError("cache-only replay changed candidate scores")
        _atomic_json(output / "replay.json", {
            "protocol": PROTOCOL, "state": "completed_cache_only_replay", "gemini_api_calls": 0,
            "nli_pair_evaluations": 0, "byte_identical_scores": True,
        })
        return 0
    by_id = {row["response_id"]: row for row in records}
    train = [row for row in subset if row["split"] == "train"]
    test = [row for row in subset if row["split"] == "test"]
    methods = ("candidate_agreement", *GRAPH_METHODS)
    train_y, test_y = np.asarray([row["y"] for row in train]), np.asarray([row["y"] for row in test])
    train_scores = {"candidate_agreement": np.asarray([by_id[row["response_id"]]["candidate_disagreement"] for row in train])}
    test_scores = {"candidate_agreement": np.asarray([by_id[row["response_id"]]["candidate_disagreement"] for row in test])}
    for method in GRAPH_METHODS:
        train_scores[method] = np.asarray([row["graph_scores"][method] for row in train])
        test_scores[method] = np.asarray([row["graph_scores"][method] for row in test])
    thresholds, train_f1 = {}, {}
    for method in methods:
        thresholds[method], train_f1[method] = select_f1_threshold(train_scores[method], train_y)
    metrics = {
        "protocol": PROTOCOL,
        "interpretation": "retrospective exact-ID paired comparison; method thresholds selected only on 405 common training rows",
        "provenance": {
            "r12_archive_sha256": reference.archive_sha256,
            "semantic_entropy_scores_sha256": _sha256_file(Path(args.semantic_entropy_scores)),
            "paired_response_id_set_sha256": _canonical_hash(sorted(by_id)),
        },
        "coverage": {"paired": 496, "entropy_without_graph_score": 63, "train": {"n": 405, "positive": 224, "negative": 181}, "test": {"n": 91, "positive": 55, "negative": 36}},
        "threshold_selection": {method: {"split": "paired_train_only", "threshold": float(thresholds[method]), "train_f1": float(train_f1[method])} for method in methods},
        "heldout_test": {method: _method_metrics(test_scores[method], test_y, threshold=thresholds[method], n_bootstrap=args.n_bootstrap, seed=42 + offset * 31) for offset, method in enumerate(methods)},
        "paired_difference_vs_support_critical": {method: {metric: _paired_difference(test_scores[method], test_scores["support_critical"], test_y, left_threshold=thresholds[method], right_threshold=thresholds["support_critical"], metric=metric, n_bootstrap=args.n_bootstrap, seed=314 + offset * 17) for metric in ("roc_auc", "f1")} for offset, method in enumerate(methods) if method != "support_critical"},
        "execution": {"gemini_api_calls": 0, "nli_pair_evaluations": int(detector.nli_pair_evaluations), "candidate_comparison_cache_hits": int(detector.cache_hits)},
    }
    _atomic_json(output / "metrics.json", metrics)
    _atomic_json(output / "run_metadata.json", {"protocol": PROTOCOL, "state": "completed", "sources_completed": 496, "gemini_api_calls": 0, "nli_pair_evaluations": int(detector.nli_pair_evaluations)})
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--semantic-entropy-scores", required=True)
    parser.add_argument("--graph-reference", required=True)
    parser.add_argument("--required-gateway-manifest-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--replay", action="store_true")
    raise SystemExit(_run(parser.parse_args()))
