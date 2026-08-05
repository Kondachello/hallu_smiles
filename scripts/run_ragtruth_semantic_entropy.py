#!/usr/bin/env python3
"""Evaluate likelihood-weighted SemanticEntropy on a fixed RAGTruth QA sample.

The score is a prompt-level, non-graph uncertainty signal: every selected QA
source is sampled independently from the configured Gemini gateway, then the
generated answers are grouped with local mutual NLI entailment.  This runner
never reads a held-out label while computing a score or selecting its decision
threshold.  It writes no prompts or generated text outside the external cache
configured for the detector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import Instance, load_instances
from src.retry import RateLimitRetryDeadlineExceeded, RetryDeadlineExceeded
from src.sampling import load_manifest_instances, qa_sample_quotas, select_qa_sample, write_manifest
from src.semantic_entropy import (
    GatewayRequestError,
    SemanticEntropyDetector,
    SemanticEntropyTruncatedError,
    SemanticUsageLogger,
    semantic_runtime_metadata,
)
from src.tune import prf_at_threshold, safe_auc, select_f1_threshold


PROTOCOL = "ragtruth-qa-semantic-entropy-evaluation-v1"
SELECTION_PROTOCOL = "ragtruth-qa-source-balanced-v1"
QUARANTINED_SOURCE_IDS = ("12448",)
# The checksum of the 750-row source-level manifest used for the verified
# strict/support/support-critical RAGTruth result.  We recreate it from the
# public corpus rather than copying a historical artifact into this branch.
HISTORICAL_750_MANIFEST_SHA256 = "19cb9472e1662ac029dab7e144e07267c9e43f7ca50556aa92123a5e268e4f86"


class WallClockBudgetExceeded(RuntimeError):
    """The caller-set maximum wall clock was reached before another source."""


def _is_retryable_gateway_failure(error: GatewayRequestError) -> bool:
    """Keep persistent failures out of the cache-resume loop.

    ``GatewayRequestError`` deliberately redacts the response body, but its
    status code is safe to use for control flow.  Authentication and malformed
    request responses need operator attention; retry only a missing transport
    status, quota pressure, and server-side failures.
    """

    return error.status_code is None or error.status_code == 429 or 500 <= error.status_code <= 599


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_hash(payload: dict[str, Any]) -> str:
    # Hash exactly the established ``src.sampling.manifest_dict`` payload.  It
    # keeps the 750-row baseline byte-compatible with the historical graph
    # manifest while allowing this runner to add non-scientific provenance
    # fields beside it.
    clean = {
        key: payload[key]
        for key in ("version", "task", "seed", "quotas", "records")
        if key in payload
    }
    return _canonical_hash(clean)


class ProgressReporter:
    """Redacted liveness state suitable for an unattended local monitor."""

    def __init__(self, output_dir: Path, usage: SemanticUsageLogger, total_sources: int):
        self.output_dir = output_dir
        self.usage = usage
        self.total_sources = int(total_sources)
        self.completed_sources = 0
        self.current_samples_completed = 0
        self.phase = "initializing"
        self.path = output_dir / "progress.json"
        self.journal = output_dir / "progress.jsonl"

    def set_source_state(self, *, completed: int, phase: str, samples_completed: int = 0) -> None:
        self.completed_sources = int(completed)
        self.phase = phase
        self.current_samples_completed = int(samples_completed)

    def callback(self, payload: dict[str, Any]) -> None:
        event = payload.get("event")
        if event == "sample_completed":
            self.current_samples_completed = int(payload.get("sample_index", -1)) + 1
            self.emit("generation_progress")
        elif event == "semantic_clusters_completed":
            self.emit("semantic_clustering_completed", n_classes=int(payload.get("n_classes", 0)))
        elif event == "llm_retry_wait":
            safe = {
                key: payload[key]
                for key in ("component", "reason", "attempt", "sleep_seconds", "retry_seconds", "continuous_429_seconds")
                if key in payload
            }
            self.emit("retry_heartbeat", **safe)

    def emit(self, event: str, **extra: Any) -> None:
        usage = self.usage.summary()
        payload: dict[str, Any] = {
            "protocol": "ragtruth-semantic-entropy-progress-v1",
            "at_utc": _utc(),
            "event": event,
            "phase": self.phase,
            "sources_completed": self.completed_sources,
            "sources_total": self.total_sources,
            "current_source_samples_completed": self.current_samples_completed,
            "api_calls": int(usage["api_calls"]),
            "cache_hits": int(usage["cache_hits"]),
            "retries": int(usage["retries"]),
            "prompt_tokens": int(usage["prompt_tokens"]),
            "completion_tokens": int(usage["completion_tokens"]),
        }
        payload.update({key: value for key, value in extra.items() if value is not None})
        _atomic_json(self.path, payload)
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        print("[semantic-entropy-progress] " + json.dumps(payload, sort_keys=True), flush=True)


def _make_or_load_manifest(
    path: Path, instances: Iterable[Instance], *, total_sources: int, seed: int
) -> tuple[list[Instance], str]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("selection_protocol") != SELECTION_PROTOCOL:
            raise ValueError("existing manifest does not use the RAGTruth source-level entropy protocol")
        if tuple(payload.get("analysis_quarantined_source_ids", [])) != QUARANTINED_SOURCE_IDS:
            raise ValueError("existing manifest has a different quarantined-source policy")
        if payload.get("manifest_sha256") != _manifest_hash(payload):
            raise ValueError("existing manifest checksum mismatch")
        selected = load_manifest_instances(path, instances)
        if len(selected) != total_sources:
            raise ValueError("existing manifest has a different source count")
        if total_sources == 750 and str(payload["manifest_sha256"]) != HISTORICAL_750_MANIFEST_SHA256:
            raise ValueError("750-row entropy manifest is not the verified historical RAGTruth manifest")
        return selected, str(payload["manifest_sha256"])

    train_sources, test_sources = qa_sample_quotas(total_sources, "0.2")
    selected = select_qa_sample(
        instances, seed=seed, train_sources=train_sources, test_sources=test_sources
    )
    write_manifest(
        path, selected, seed=seed, train_sources=train_sources, test_sources=test_sources
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update({
        "selection_protocol": SELECTION_PROTOCOL,
        "analysis_quarantined_source_ids": list(QUARANTINED_SOURCE_IDS),
        "source_level": True,
    })
    payload["manifest_sha256"] = _manifest_hash(payload)
    _atomic_json(path, payload)
    if total_sources == 750 and payload["manifest_sha256"] != HISTORICAL_750_MANIFEST_SHA256:
        raise ValueError("deterministic 750-row selection does not match the verified historical manifest")
    return selected, str(payload["manifest_sha256"])


def _score_path(output_dir: Path, instance: Instance) -> Path:
    digest = hashlib.sha256(instance.response_id.encode("utf-8")).hexdigest()
    return output_dir / "score-checkpoints" / f"{digest}.json"


def _load_checkpoint(path: Path, instance: Instance, manifest_sha256: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("protocol") != PROTOCOL
            or payload.get("manifest_sha256") != manifest_sha256
            or payload.get("response_id") != instance.response_id
            or payload.get("source_id") != instance.source_id
            or int(payload.get("y")) != int(instance.y)
        ):
            return None
        state = str(payload.get("state", "scored"))
        if state == "unscorable_output_length":
            return payload
        if state != "scored" or not math.isfinite(float(payload["semantic_entropy"])):
            return None
        return payload
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _checkpoint_payload(instance: Instance, manifest_sha256: str, score: Any, elapsed_s: float) -> dict[str, Any]:
    # IDs, labels and aggregate NLI assignments are reproducibility metadata;
    # neither the source prompt nor a generated completion is serialized here.
    return {
        "protocol": PROTOCOL,
        "manifest_sha256": manifest_sha256,
        "response_id": instance.response_id,
        "source_id": instance.source_id,
        "split": instance.split,
        "y": int(instance.y),
        "gen_model": instance.gen_model,
        "state": "scored",
        "semantic_entropy": float(score.entropy),
        "n_samples": int(score.n_samples),
        "n_semantic_classes": int(score.n_classes),
        "semantic_ids": list(score.semantic_ids),
        "seconds": round(float(elapsed_s), 6),
    }


def _unscorable_output_length_checkpoint(instance: Instance, manifest_sha256: str, elapsed_s: float) -> dict[str, Any]:
    """Record a model-maximum truncation without silently discarding a source."""

    return {
        "protocol": PROTOCOL,
        "manifest_sha256": manifest_sha256,
        "response_id": instance.response_id,
        "source_id": instance.source_id,
        "split": instance.split,
        "y": int(instance.y),
        "gen_model": instance.gen_model,
        "state": "unscorable_output_length",
        "seconds": round(float(elapsed_s), 6),
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            # Wall-clock seconds are checkpoint diagnostics.  Excluding them
            # makes live and cache-only ``scores.jsonl`` byte-comparable while
            # retaining the scientific score and all provenance fields.
            public = {key: value for key, value in row.items() if key != "seconds"}
            handle.write(json.dumps(public, sort_keys=True) + "\n")


def _bootstrap(
    score: np.ndarray, labels: np.ndarray, metric: str, *, threshold: float | None, n: int, seed: int
) -> tuple[float, float]:
    if len(score) < 3 or len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    indices = np.arange(len(score))
    for _ in range(n):
        sample = rng.choice(indices, size=len(indices), replace=True)
        if len(np.unique(labels[sample])) < 2:
            continue
        if metric == "auc":
            value = safe_auc(score[sample], labels[sample])
        elif metric == "f1" and threshold is not None:
            value = prf_at_threshold(score[sample], labels[sample], threshold)[2]
        else:  # pragma: no cover - caller invariant
            raise ValueError("unsupported bootstrap metric")
        if math.isfinite(value):
            values.append(float(value))
    if not values:
        return float("nan"), float("nan")
    lo, hi = np.percentile(np.asarray(values), [2.5, 97.5])
    return float(lo), float(hi)


def _evaluate(rows: list[dict[str, Any]], *, n_bootstrap: int, seed: int) -> dict[str, Any]:
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    if not train or not test:
        raise ValueError("both train and test splits are required")
    train_scores = np.asarray([float(row["semantic_entropy"]) for row in train], dtype=float)
    train_y = np.asarray([int(row["y"]) for row in train], dtype=int)
    test_scores = np.asarray([float(row["semantic_entropy"]) for row in test], dtype=float)
    test_y = np.asarray([int(row["y"]) for row in test], dtype=int)
    theta, train_f1 = select_f1_threshold(train_scores, train_y)
    precision, recall, test_f1 = prf_at_threshold(test_scores, test_y, theta)
    test_auc = safe_auc(test_scores, test_y)
    auc_ci = _bootstrap(test_scores, test_y, "auc", threshold=None, n=n_bootstrap, seed=seed)
    f1_ci = _bootstrap(test_scores, test_y, "f1", threshold=theta, n=n_bootstrap, seed=seed + 1)
    return {
        "threshold_selection": {
            "split": "train_only",
            "objective": "max_f1",
            "theta": float(theta),
            "train_f1": float(train_f1),
        },
        "heldout_test": {
            "n": int(len(test)),
            "n_hallucinated": int(test_y.sum()),
            "n_factual": int(len(test_y) - test_y.sum()),
            "roc_auc": float(test_auc),
            "roc_auc_ci95": list(auc_ci),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(test_f1),
            "f1_ci95": list(f1_ci),
        },
    }


def _write_report(path: Path, metadata: dict[str, Any], metrics: dict[str, Any]) -> None:
    heldout = metrics["heldout_test"]
    threshold = metrics["threshold_selection"]
    text = "\n".join([
        "# RAGTruth QA SemanticEntropy baseline",
        "",
        "This is a non-graph, prompt-level uncertainty baseline. For each selected QA source, "
        "Gemini samples 15 answers at temperature 1.0; local DeBERTa groups mutually entailed "
        "answers, and likelihood-weighted semantic entropy is the hallucination score (higher is riskier).",
        "",
        f"- Manifest SHA-256: `{metadata['manifest_sha256']}`",
        f"- Fixed source-level manifest: {metadata['manifest_rows']} QA rows; after the historical quarantine, "
        f"{metadata['sources_total']} selected sources ({metadata['train_sources']} train / {metadata['test_sources']} held-out test); "
        f"{metadata.get('sources_scored', metadata['sources_total'])} scored and "
        f"{metadata.get('sources_unscorable_output_length', 0)} unscorable at the model output maximum.",
        f"- Quarantined source IDs: {', '.join(metadata['quarantined_source_ids'])}",
        f"- Threshold: θ = {threshold['theta']:.6f}, selected on train only by F1 (train F1 = {threshold['train_f1']:.4f}).",
        "",
        "## Single held-out evaluation",
        "",
        f"- ROC-AUC: {heldout['roc_auc']:.4f} (bootstrap 95% CI {heldout['roc_auc_ci95'][0]:.4f}–{heldout['roc_auc_ci95'][1]:.4f})",
        f"- Precision / recall / F1: {heldout['precision']:.4f} / {heldout['recall']:.4f} / {heldout['f1']:.4f} "
        f"(F1 95% CI {heldout['f1_ci95'][0]:.4f}–{heldout['f1_ci95'][1]:.4f})",
        "",
        "The RAGTruth label belongs to a single original response selected per source; this baseline "
        "therefore measures the generator's uncertainty for the source prompt, not a graph audit of that response. "
        "No test labels are used for score construction or threshold selection.",
        "",
    ])
    path.write_text(text, encoding="utf-8")


def _run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cfg = load_config(args.config)
    instances = load_instances(args.data_dir, exclude_implicit_true=bool(cfg.data.exclude_implicit_true))
    manifest_path = Path(args.manifest).resolve()
    manifest_selected, manifest_sha256 = _make_or_load_manifest(
        manifest_path, instances, total_sources=args.total_sources, seed=args.seed
    )
    selected = [item for item in manifest_selected if item.source_id not in QUARANTINED_SOURCE_IDS]
    if not selected:
        raise ValueError("all selected sources were quarantined")
    train_sources = sum(item.split == "train" for item in selected)
    test_sources = sum(item.split == "test" for item in selected)
    usage = SemanticUsageLogger(output_dir / "usage.jsonl")
    progress = ProgressReporter(output_dir, usage, len(selected))
    detector = SemanticEntropyDetector(
        cfg, usage=usage, cache_only=bool(args.cache_only), progress_callback=progress.callback
    )
    metadata = {
        "protocol": PROTOCOL,
        "state": "running",
        "started_at_utc": _utc(),
        "cache_only": bool(args.cache_only),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "manifest_rows": len(manifest_selected),
        "sources_total": len(selected),
        "train_sources": train_sources,
        "test_sources": test_sources,
        "quarantined_source_ids": list(QUARANTINED_SOURCE_IDS),
        "output_length_failure_policy": (
            "record_unscorable_output_length_at_the_fixed_model_maximum; "
            "retain_in_manifest_and_report_coverage"
        ),
        "data": {
            "source_info_sha256": _sha256_file(Path(args.data_dir) / "source_info.jsonl"),
            "response_sha256": _sha256_file(Path(args.data_dir) / "response.jsonl"),
        },
        "runtime": semantic_runtime_metadata(cfg),
        "wall_clock_budget_seconds": args.wall_clock_budget_s,
    }
    _atomic_json(output_dir / "run_metadata.json", metadata)

    rows: list[dict[str, Any]] = []
    unscorable: list[dict[str, Any]] = []
    try:
        for index, instance in enumerate(selected):
            progress.set_source_state(completed=index, phase="cache_replay" if args.cache_only else "live")
            path = _score_path(output_dir, instance)
            checkpoint = None if args.cache_only or args.recompute_from_cache else _load_checkpoint(path, instance, manifest_sha256)
            if checkpoint is not None:
                if checkpoint.get("state") == "unscorable_output_length":
                    unscorable.append(checkpoint)
                else:
                    rows.append(checkpoint)
                progress.set_source_state(completed=index + 1, phase="checkpoint_reused")
                progress.emit("source_checkpoint_reused")
                continue
            if (
                not args.cache_only
                and args.wall_clock_budget_s is not None
                and time.monotonic() - started >= args.wall_clock_budget_s
            ):
                raise WallClockBudgetExceeded()
            progress.emit("source_started")
            source_started = time.monotonic()
            try:
                score = detector.score_prompt(instance.prompt)
            except SemanticEntropyTruncatedError:
                if args.cache_only:
                    raise
                checkpoint = _unscorable_output_length_checkpoint(
                    instance, manifest_sha256, time.monotonic() - source_started
                )
                _atomic_json(path, checkpoint)
                unscorable.append(checkpoint)
                progress.set_source_state(completed=index + 1, phase="unscorable_output_length")
                progress.emit("source_unscorable", error_type=SemanticEntropyTruncatedError.__name__)
                continue
            row = _checkpoint_payload(instance, manifest_sha256, score, time.monotonic() - source_started)
            if not args.cache_only:
                _atomic_json(path, row)
            rows.append(row)
            progress.set_source_state(completed=index + 1, phase="cache_replay" if args.cache_only else "live")
            progress.emit("source_completed")
    except WallClockBudgetExceeded:
        metadata.update({
            "state": "wall_clock_budget_exhausted",
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "sources_completed": len(rows) + len(unscorable),
            "sources_scored": len(rows),
            "sources_unscorable_output_length": len(unscorable),
            "usage": usage.summary(),
            "finished_at_utc": _utc(),
        })
        _atomic_json(output_dir / "run_metadata.json", metadata)
        progress.emit("wall_clock_budget_exhausted")
        return 76
    except (RateLimitRetryDeadlineExceeded, RetryDeadlineExceeded) as exc:
        metadata.update({
            "state": "retryable_gateway_pause",
            "error_type": type(exc).__name__,
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "sources_completed": len(rows) + len(unscorable),
            "sources_scored": len(rows),
            "sources_unscorable_output_length": len(unscorable),
            "usage": usage.summary(),
            "finished_at_utc": _utc(),
        })
        _atomic_json(output_dir / "run_metadata.json", metadata)
        progress.emit("retryable_gateway_pause", error_type=type(exc).__name__)
        return 75
    except GatewayRequestError as exc:
        if _is_retryable_gateway_failure(exc):
            metadata.update({
                "state": "retryable_gateway_pause",
                "error_type": type(exc).__name__,
                "elapsed_seconds": round(time.monotonic() - started, 4),
                "sources_completed": len(rows) + len(unscorable),
                "sources_scored": len(rows),
                "sources_unscorable_output_length": len(unscorable),
                "usage": usage.summary(),
                "finished_at_utc": _utc(),
            })
            _atomic_json(output_dir / "run_metadata.json", metadata)
            progress.emit("retryable_gateway_pause", error_type=type(exc).__name__)
            return 75
        metadata.update({
            "state": "error",
            "error_type": type(exc).__name__,
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "sources_completed": len(rows) + len(unscorable),
            "sources_scored": len(rows),
            "sources_unscorable_output_length": len(unscorable),
            "usage": usage.summary(),
            "finished_at_utc": _utc(),
        })
        _atomic_json(output_dir / "run_metadata.json", metadata)
        progress.emit("error", error_type=type(exc).__name__)
        raise
    except Exception as exc:  # noqa: BLE001 - terminal state must remain redacted
        metadata.update({
            "state": "error",
            "error_type": type(exc).__name__,
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "sources_completed": len(rows) + len(unscorable),
            "sources_scored": len(rows),
            "sources_unscorable_output_length": len(unscorable),
            "usage": usage.summary(),
            "finished_at_utc": _utc(),
        })
        _atomic_json(output_dir / "run_metadata.json", metadata)
        progress.emit("error", error_type=type(exc).__name__)
        raise

    rows.sort(key=lambda row: (str(row["split"]), str(row["source_id"]), str(row["response_id"])))
    unscorable.sort(key=lambda row: (str(row["split"]), str(row["source_id"]), str(row["response_id"])))
    _write_jsonl(output_dir / "scores.jsonl", rows)
    _write_jsonl(output_dir / "unscorable.jsonl", unscorable)
    if args.cache_only:
        metadata.update({
            "state": "completed_cache_replay",
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "sources_completed": len(rows) + len(unscorable),
            "sources_scored": len(rows),
            "sources_unscorable_output_length": len(unscorable),
            "usage": usage.summary(),
            "finished_at_utc": _utc(),
        })
        _atomic_json(output_dir / "run_metadata.json", metadata)
        progress.emit("cache_replay_completed")
        return 0

    metrics = _evaluate(rows, n_bootstrap=args.n_bootstrap, seed=args.seed)
    metrics.update({
        "protocol": PROTOCOL,
        "score": "likelihood_weighted_semantic_entropy",
        "higher_score_means": "higher_hallucination_risk",
        "manifest_sha256": manifest_sha256,
        "n_scored": len(rows),
        "n_unscorable_output_length": len(unscorable),
        "coverage": {
            "manifest_sources": len(selected),
            "scored_sources": len(rows),
            "unscorable_output_length": len(unscorable),
        },
        "usage": usage.summary(),
    })
    _atomic_json(output_dir / "metrics.json", metrics)
    metadata.update({
        "state": "completed",
        "elapsed_seconds": round(time.monotonic() - started, 4),
        "sources_completed": len(rows) + len(unscorable),
        "sources_scored": len(rows),
        "sources_unscorable_output_length": len(unscorable),
        "usage": usage.summary(),
        "finished_at_utc": _utc(),
    })
    _atomic_json(output_dir / "run_metadata.json", metadata)
    _write_report(output_dir / "report.md", metadata, metrics)
    progress.emit("completed")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--total-sources", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--wall-clock-budget-s", type=float)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--recompute-from-cache", action="store_true")
    args = parser.parse_args()
    if args.n_bootstrap < 1:
        raise SystemExit("n-bootstrap must be positive")
    if args.wall_clock_budget_s is not None and args.wall_clock_budget_s <= 0:
        raise SystemExit("wall-clock-budget-s must be positive")
    try:
        qa_sample_quotas(args.total_sources, "0.2")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.recompute_from_cache and not args.cache_only:
        raise SystemExit("--recompute-from-cache requires --cache-only")
    raise SystemExit(_run(args))


if __name__ == "__main__":
    main()
