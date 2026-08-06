#!/usr/bin/env python3
"""Run the response-conditioned candidate-agreement baseline on RAGTruth QA.

This is intentionally separate from prompt-only semantic entropy.  It reuses
only the verified Gemini sample cache, compares each historical labelled answer
with those samples locally through bidirectional NLI, and serializes scalar
scores plus IDs only.  No prompt, candidate, completion, raw graph, or gateway
credential is written outside the external cache.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache import CacheOnlyMissError, config_value
from src.candidate_agreement import (
    GraphReferenceError,
    evaluate_paired_candidate_agreement,
    load_graph_reference,
    validate_reference_against_candidate_rows,
)
from src.config import load_config
from src.data import Instance, load_instances
from src.retry import RateLimitRetryDeadlineExceeded, RetryDeadlineExceeded
from src.sampling import load_manifest_instances, qa_sample_quotas, select_qa_sample, write_manifest
from src.semantic_entropy import (
    CANDIDATE_AGREEMENT_PROTOCOL,
    CandidateAgreementDetector,
    CandidateAgreementEmptyCandidateError,
    GatewayRequestError,
    SemanticEntropyDetector,
    SemanticEntropyTruncatedError,
    SemanticUsageLogger,
    semantic_runtime_metadata,
)


PROTOCOL = "ragtruth-qa-candidate-agreement-evaluation-v1"
SELECTION_PROTOCOL = "ragtruth-qa-source-balanced-v1"
QUARANTINED_SOURCE_IDS = ("12448",)
HISTORICAL_750_MANIFEST_SHA256 = "19cb9472e1662ac029dab7e144e07267c9e43f7ca50556aa92123a5e268e4f86"
EXPECTED_ELIGIBLE_SOURCES = 749
# The cache inventory is verified against the immutable semantic-entropy
# namespace rather than estimated from an earlier manifest-overlap note.  All
# 559 historically completed eligible sources have cache-compatible sample
# keys under the pinned runtime; the remaining 190 require a live fill.
EXPECTED_REUSED_SOURCES = 559
EXPECTED_COLD_SOURCES = 190
SAMPLES_PER_SOURCE = 15


class WallClockBudgetExceeded(RuntimeError):
    """The caller-set deadline was reached before another source."""


class ProgressReporter:
    """Append-only, redacted liveness for a long serial candidate run."""

    def __init__(self, output_dir: Path, usage: SemanticUsageLogger, total: int):
        self.output_dir = output_dir
        self.usage = usage
        self.total = int(total)
        self.completed = 0
        self.phase = "initializing"
        self.sample_index = 0

    def set_source(self, completed: int, phase: str) -> None:
        self.completed = int(completed)
        self.phase = phase
        self.sample_index = 0

    def callback(self, payload: dict[str, Any]) -> None:
        event = payload.get("event")
        if event == "sample_completed":
            self.sample_index = int(payload.get("sample_index", -1)) + 1
            self.emit("generation_progress")
        elif event == "llm_retry_wait":
            self.emit("retry_heartbeat", **{
                key: payload[key]
                for key in ("component", "reason", "attempt", "sleep_seconds", "retry_seconds", "continuous_429_seconds")
                if key in payload
            })

    def emit(self, event: str, **extra: Any) -> None:
        usage = self.usage.summary()
        payload = {
            "protocol": "ragtruth-candidate-agreement-progress-v1",
            "at_utc": _utc(), "event": event, "phase": self.phase,
            "sources_completed": self.completed, "sources_total": self.total,
            "current_source_samples_completed": self.sample_index,
            "gemini_api_calls": int(usage["api_calls"]),
            "sample_cache_hits": int(usage["cache_hits"]), "retries": int(usage["retries"]),
        }
        payload.update(extra)
        _atomic_json(self.output_dir / "progress.json", payload)
        with (self.output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        print("[candidate-agreement-progress] " + json.dumps(payload, sort_keys=True), flush=True)


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
    return _canonical_hash({
        key: payload[key] for key in ("version", "task", "seed", "quotas", "records") if key in payload
    })


def _make_or_load_manifest(path: Path, instances: Iterable[Instance]) -> tuple[list[Instance], str]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("selection_protocol") != SELECTION_PROTOCOL
            or tuple(payload.get("analysis_quarantined_source_ids", [])) != QUARANTINED_SOURCE_IDS
            or payload.get("manifest_sha256") != _manifest_hash(payload)
            or payload.get("manifest_sha256") != HISTORICAL_750_MANIFEST_SHA256
        ):
            raise ValueError("candidate-agreement manifest is not the verified historical 750-row contract")
        selected = load_manifest_instances(path, instances)
        if len(selected) != 750:
            raise ValueError("candidate-agreement manifest must contain exactly 750 rows")
        return selected, str(payload["manifest_sha256"])
    train_sources, test_sources = qa_sample_quotas(750, "0.2")
    selected = select_qa_sample(instances, seed=42, train_sources=train_sources, test_sources=test_sources)
    write_manifest(path, selected, seed=42, train_sources=train_sources, test_sources=test_sources)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update({
        "selection_protocol": SELECTION_PROTOCOL,
        "analysis_quarantined_source_ids": list(QUARANTINED_SOURCE_IDS),
        "source_level": True,
    })
    payload["manifest_sha256"] = _manifest_hash(payload)
    _atomic_json(path, payload)
    if payload["manifest_sha256"] != HISTORICAL_750_MANIFEST_SHA256:
        raise ValueError("deterministic selection does not match the verified 750-row RAGTruth manifest")
    return selected, str(payload["manifest_sha256"])


def _checkpoint_path(output_dir: Path, instance: Instance) -> Path:
    digest = hashlib.sha256(instance.response_id.encode("utf-8")).hexdigest()
    return output_dir / "score-checkpoints" / f"{digest}.json"


def _record_base(instance: Instance, manifest_sha256: str) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "manifest_sha256": manifest_sha256,
        "response_id": instance.response_id,
        "source_id": instance.source_id,
        "split": instance.split,
        "y": int(instance.y),
        "gen_model": instance.gen_model,
    }


def _valid_checkpoint(path: Path, instance: Instance, manifest_sha256: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        base = _record_base(instance, manifest_sha256)
        if any(payload.get(key) != value for key, value in base.items()):
            return None
        state = payload.get("state")
        if state in {"unscorable_output_length", "unscorable_empty_candidate"}:
            return payload
        if (
            state != "scored"
            or not math.isfinite(float(payload["candidate_disagreement"]))
            or not 0 <= float(payload["candidate_disagreement"]) <= 1
            or not math.isfinite(float(payload["agreement_mass"]))
        ):
            return None
        return payload
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _prior_nli_evaluations(path: Path, manifest_sha256: str) -> int:
    """Carry local NLI work across a cache-compatible serial resume."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol") != PROTOCOL or payload.get("manifest_sha256") != manifest_sha256:
            return 0
        value = int(payload.get("nli_pair_evaluations", 0))
        return value if value >= 0 else 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0


def _score_record(instance: Instance, manifest_sha256: str, score: Any, elapsed_s: float) -> dict[str, Any]:
    payload = _record_base(instance, manifest_sha256)
    payload.update({
        "state": "scored",
        **score.to_dict(),
        "seconds": round(float(elapsed_s), 6),
    })
    return payload


def _unscorable_record(instance: Instance, manifest_sha256: str, state: str, elapsed_s: float) -> dict[str, Any]:
    if state not in {"unscorable_output_length", "unscorable_empty_candidate"}:
        raise ValueError("invalid explicit candidate-agreement unscorable state")
    payload = _record_base(instance, manifest_sha256)
    payload.update({"state": state, "seconds": round(float(elapsed_s), 6)})
    return payload


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            # Execution timing is useful in a checkpoint but must not make a
            # scientific cache-only replay differ byte-for-byte.
            public = {key: value for key, value in row.items() if key != "seconds"}
            handle.write(json.dumps(public, sort_keys=True) + "\n")


def _preflight_sample_inventory(detector: SemanticEntropyDetector, selected: list[Instance]) -> dict[str, int]:
    """Read every expected sample key without sending Gemini or loading NLI."""

    hits = 0
    misses = 0
    for instance in selected:
        for index in range(detector.n_samples):
            try:
                detector._sample(instance.prompt, index)  # noqa: SLF001 - key-compatible offline inventory
                hits += 1
            except CacheOnlyMissError:
                misses += 1
    return {"sample_hits": hits, "sample_misses": misses, "samples_total": hits + misses}


def _select_cached_training_smoke_sources(cfg: Any, selected: list[Instance], count: int) -> list[Instance]:
    """Choose deterministic all-hot training sources without any gateway call."""

    probe = SemanticEntropyDetector(cfg, cache_only=True)
    chosen: list[Instance] = []
    for instance in selected:
        if instance.split != "train":
            continue
        try:
            for index in range(probe.n_samples):
                probe._sample(instance.prompt, index)  # noqa: SLF001 - sample cache contract probe
        except CacheOnlyMissError:
            continue
        chosen.append(instance)
        if len(chosen) == count:
            return chosen
    raise ValueError("fewer than the required cached training sources are available for offline smoke")


def _is_retryable_gateway_failure(error: GatewayRequestError) -> bool:
    return error.status_code is None or error.status_code == 429 or 500 <= error.status_code <= 599


def _run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cfg = load_config(args.config)
    gateway = getattr(cfg, "vertex_gateway", None)
    configured_manifest = config_value(gateway, "manifest_sha256") if gateway is not None else None
    if configured_manifest != args.required_gateway_manifest_sha256:
        raise ValueError("runtime gateway manifest differs from the cache-compatible candidate-agreement contract")
    if int(config_value(cfg.semantic_entropy, "n_samples", 0)) != SAMPLES_PER_SOURCE:
        raise ValueError("candidate agreement requires exactly 15 Gemini samples per source")
    if int(config_value(cfg.semantic_entropy, "max_tokens", 0)) != 65535:
        raise ValueError("candidate agreement requires fixed semantic_entropy.max_tokens=65535")
    if list(config_value(cfg.semantic_entropy, "max_tokens_read_through", []) or []) != [4096, 8192]:
        raise ValueError("candidate agreement requires max_tokens_read_through=[4096, 8192]")
    instances = load_instances(args.data_dir, exclude_implicit_true=bool(cfg.data.exclude_implicit_true))
    manifest_selected, manifest_sha256 = _make_or_load_manifest(Path(args.manifest).resolve(), instances)
    selected = [row for row in manifest_selected if row.source_id not in QUARANTINED_SOURCE_IDS]
    if len(selected) != EXPECTED_ELIGIBLE_SOURCES:
        raise ValueError("historical candidate-agreement analysis must contain 749 eligible sources")
    reference = load_graph_reference(args.graph_reference, manifest_sha256=manifest_sha256)
    validate_reference_against_candidate_rows(reference, [
        {"response_id": row.response_id, "source_id": row.source_id, "split": row.split, "y": row.y}
        for row in selected
    ])

    # This mode deliberately validates only the existing Gemini sample cache.
    # It cannot instantiate NLI or create candidate-comparison cache entries.
    if args.preflight_sample_inventory:
        sampler = SemanticEntropyDetector(cfg, cache_only=True)
        inventory = _preflight_sample_inventory(sampler, selected)
        expected_hits = EXPECTED_REUSED_SOURCES * SAMPLES_PER_SOURCE
        expected_misses = EXPECTED_COLD_SOURCES * SAMPLES_PER_SOURCE
        inventory.update({
            "protocol": "ragtruth-candidate-agreement-sample-preflight-v1",
            "manifest_sha256": manifest_sha256,
            "expected_sample_hits": expected_hits,
            "expected_sample_misses": expected_misses,
            "nli_pair_evaluations": 0,
            "gemini_api_calls": 0,
        })
        _atomic_json(output_dir / "sample_cache_preflight.json", inventory)
        if inventory["sample_hits"] != expected_hits or inventory["sample_misses"] != expected_misses:
            raise ValueError("sample cache inventory does not match the verified 8,385-hit / 2,850-cold contract")
        return 0

    usage = SemanticUsageLogger(output_dir / "usage.jsonl")
    progress = ProgressReporter(output_dir, usage, EXPECTED_ELIGIBLE_SOURCES)
    sampler = SemanticEntropyDetector(
        cfg, usage=usage, cache_only=bool(args.cache_only), progress_callback=progress.callback
    )
    detector = CandidateAgreementDetector(
        cfg, sample_detector=sampler, cache_only=bool(args.cache_only)
    )
    prior_nli_evaluations = _prior_nli_evaluations(output_dir / "run_metadata.json", manifest_sha256)

    def total_nli_evaluations() -> int:
        return prior_nli_evaluations + detector.nli_pair_evaluations

    metadata = {
        "protocol": PROTOCOL,
        "state": "running",
        "started_at_utc": _utc(),
        "cache_only": bool(args.cache_only),
        "offline_sample_smoke": int(args.offline_sample_smoke or 0),
        "manifest_sha256": manifest_sha256,
        "manifest_rows": len(manifest_selected),
        "sources_total": len(selected),
        "quarantined_source_ids": list(QUARANTINED_SOURCE_IDS),
        "graph_reference_sha256": _sha256_file(Path(args.graph_reference)),
        "graph_reference_archive_sha256": reference.archive_sha256,
        "sample_cache_contract": {
            "required_reused_samples": EXPECTED_REUSED_SOURCES * SAMPLES_PER_SOURCE,
            "required_cold_samples": EXPECTED_COLD_SOURCES * SAMPLES_PER_SOURCE,
            "n_samples_per_source": sampler.n_samples,
        },
        "runtime": {
            "semantic_sampling": semantic_runtime_metadata(cfg),
            "candidate_agreement_protocol": CANDIDATE_AGREEMENT_PROTOCOL,
        },
        "data": {
            "source_info_sha256": _sha256_file(Path(args.data_dir) / "source_info.jsonl"),
            "response_sha256": _sha256_file(Path(args.data_dir) / "response.jsonl"),
        },
        "wall_clock_budget_seconds": args.wall_clock_budget_s,
    }
    _atomic_json(output_dir / "run_metadata.json", metadata)

    run_selected = selected
    if args.offline_sample_smoke:
        run_selected = _select_cached_training_smoke_sources(
            cfg, selected, args.offline_sample_smoke
        )
    rows: list[dict[str, Any]] = []
    unscorable: list[dict[str, Any]] = []
    try:
        for instance in run_selected:
            progress.set_source(len(rows) + len(unscorable), "cache_replay" if args.cache_only else "live")
            path = _checkpoint_path(output_dir, instance)
            if args.cache_only:
                # Explicit unscorable markers are protocol outcomes, not
                # substitutes for a cached scientific score.  Every scored
                # response must still exercise the candidate-comparison cache.
                marker_path = (
                    Path(args.marker_checkpoint_dir).resolve()
                    / "score-checkpoints"
                    / path.name
                    if args.marker_checkpoint_dir else path
                )
                existing = _valid_checkpoint(marker_path, instance, manifest_sha256)
                checkpoint = existing if existing and existing["state"] != "scored" else None
            else:
                checkpoint = _valid_checkpoint(path, instance, manifest_sha256) if not args.recompute_from_cache else None
            if checkpoint is not None:
                (unscorable if checkpoint["state"] != "scored" else rows).append(checkpoint)
                progress.set_source(len(rows) + len(unscorable), "checkpoint_reused")
                progress.emit("source_checkpoint_reused")
                continue
            if args.wall_clock_budget_s is not None and time.monotonic() - started >= args.wall_clock_budget_s:
                raise WallClockBudgetExceeded()
            source_started = time.monotonic()
            progress.emit("source_started")
            try:
                score = detector.score_candidate(instance.prompt, instance.response)
                record = _score_record(instance, manifest_sha256, score, time.monotonic() - source_started)
            except SemanticEntropyTruncatedError:
                if args.cache_only:
                    raise
                record = _unscorable_record(
                    instance, manifest_sha256, "unscorable_output_length", time.monotonic() - source_started
                )
            except CandidateAgreementEmptyCandidateError:
                record = _unscorable_record(
                    instance, manifest_sha256, "unscorable_empty_candidate", time.monotonic() - source_started
                )
            if not args.cache_only:
                _atomic_json(path, record)
            (unscorable if record["state"] != "scored" else rows).append(record)
            progress.set_source(len(rows) + len(unscorable), record["state"])
            progress.emit("source_completed")
    except WallClockBudgetExceeded:
        metadata.update({
            "state": "wall_clock_budget_exhausted", "elapsed_seconds": round(time.monotonic() - started, 4),
            "sources_completed": len(rows) + len(unscorable), "usage": usage.summary(),
            "nli_pair_evaluations": total_nli_evaluations(), "finished_at_utc": _utc(),
        })
        _atomic_json(output_dir / "run_metadata.json", metadata)
        progress.emit("wall_clock_budget_exhausted")
        return 76
    except (RateLimitRetryDeadlineExceeded, RetryDeadlineExceeded) as exc:
        metadata.update({
            "state": "retryable_gateway_pause", "error_type": type(exc).__name__,
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "sources_completed": len(rows) + len(unscorable), "usage": usage.summary(),
            "nli_pair_evaluations": total_nli_evaluations(), "finished_at_utc": _utc(),
        })
        _atomic_json(output_dir / "run_metadata.json", metadata)
        progress.emit("retryable_gateway_pause")
        return 75
    except GatewayRequestError as exc:
        state = "retryable_gateway_pause" if _is_retryable_gateway_failure(exc) else "error"
        metadata.update({
            "state": state, "error_type": type(exc).__name__,
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "sources_completed": len(rows) + len(unscorable), "usage": usage.summary(),
            "nli_pair_evaluations": total_nli_evaluations(), "finished_at_utc": _utc(),
        })
        _atomic_json(output_dir / "run_metadata.json", metadata)
        if state == "retryable_gateway_pause":
            progress.emit("retryable_gateway_pause")
            return 75
        raise
    except Exception as exc:  # noqa: BLE001 - terminal diagnostics must stay redacted
        metadata.update({
            "state": "error", "error_type": type(exc).__name__,
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "sources_completed": len(rows) + len(unscorable), "usage": usage.summary(),
            "nli_pair_evaluations": total_nli_evaluations(), "finished_at_utc": _utc(),
        })
        _atomic_json(output_dir / "run_metadata.json", metadata)
        progress.emit("error")
        raise

    rows.sort(key=lambda row: (str(row["split"]), str(row["source_id"]), str(row["response_id"])))
    unscorable.sort(key=lambda row: (str(row["split"]), str(row["source_id"]), str(row["response_id"])))
    _write_jsonl(output_dir / "candidate_scores.jsonl", rows)
    _write_jsonl(output_dir / "unscorable.jsonl", unscorable)
    if args.offline_sample_smoke:
        metadata.update({
            "state": "completed_offline_sample_smoke", "elapsed_seconds": round(time.monotonic() - started, 4),
            "sources_completed": len(rows) + len(unscorable), "usage": usage.summary(),
            "nli_pair_evaluations": total_nli_evaluations(), "finished_at_utc": _utc(),
        })
        _atomic_json(output_dir / "run_metadata.json", metadata)
        progress.emit("completed_offline_sample_smoke")
        return 0
    if len(rows) + len(unscorable) != EXPECTED_ELIGIBLE_SOURCES:
        raise ValueError("candidate-agreement run ended without all 749 explicit source outcomes")
    try:
        paired = evaluate_paired_candidate_agreement(rows, reference, n_bootstrap=args.n_bootstrap, seed=42)
    except GraphReferenceError as exc:
        raise ValueError("candidate-agreement graph pairing coverage shortfall") from exc
    metrics = {
        "protocol": PROTOCOL,
        "score": "likelihood_weighted_candidate_semantic_disagreement",
        "higher_score_means": "candidate_has_less_semantic_support_from_likely_generator_answers",
        "manifest_sha256": manifest_sha256,
        "coverage": {
            "eligible_sources": EXPECTED_ELIGIBLE_SOURCES,
            "scored": len(rows),
            "unscorable_output_length": sum(row["state"] == "unscorable_output_length" for row in unscorable),
            "unscorable_empty_candidate": sum(row["state"] == "unscorable_empty_candidate" for row in unscorable),
        },
        "paired_evaluation": paired,
        "usage": usage.summary(),
        "nli_pair_evaluations": total_nli_evaluations(),
    }
    _atomic_json(output_dir / "metrics.json", metrics)
    scientific_metrics = {
        "protocol": PROTOCOL,
        "score": metrics["score"],
        "higher_score_means": metrics["higher_score_means"],
        "manifest_sha256": manifest_sha256,
        "coverage": metrics["coverage"],
        "paired_evaluation": paired,
    }
    _atomic_json(output_dir / "scientific_metrics.json", scientific_metrics)
    metadata.update({
        "state": "completed_cache_replay" if args.cache_only else "completed",
        "elapsed_seconds": round(time.monotonic() - started, 4),
        "sources_completed": len(rows) + len(unscorable), "sources_scored": len(rows),
        "usage": usage.summary(), "nli_pair_evaluations": total_nli_evaluations(),
        "finished_at_utc": _utc(),
    })
    _atomic_json(output_dir / "run_metadata.json", metadata)
    progress.emit("cache_replay_completed" if args.cache_only else "completed")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--graph-reference", required=True)
    parser.add_argument("--required-gateway-manifest-sha256", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--wall-clock-budget-s", type=float)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--recompute-from-cache", action="store_true")
    parser.add_argument("--marker-checkpoint-dir")
    parser.add_argument("--preflight-sample-inventory", action="store_true")
    parser.add_argument("--offline-sample-smoke", type=int)
    args = parser.parse_args()
    if args.n_bootstrap < 1:
        raise SystemExit("n-bootstrap must be positive")
    if args.wall_clock_budget_s is not None and args.wall_clock_budget_s <= 0:
        raise SystemExit("wall-clock-budget-s must be positive")
    if args.recompute_from_cache and not args.cache_only:
        raise SystemExit("--recompute-from-cache requires --cache-only")
    if args.preflight_sample_inventory and (args.cache_only or args.offline_sample_smoke):
        raise SystemExit("sample preflight cannot be combined with scoring modes")
    if args.offline_sample_smoke is not None and args.offline_sample_smoke <= 0:
        raise SystemExit("offline sample smoke must be positive")
    raise SystemExit(_run(args))


if __name__ == "__main__":
    main()
