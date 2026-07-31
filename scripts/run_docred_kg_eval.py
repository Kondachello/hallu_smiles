#!/usr/bin/env python3
"""Run one deterministic, cache-backed DocRED KG extraction evaluation stage."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.docred import (
    BudgetExceeded,
    BudgetGuard,
    DOCRED_HF_REPO,
    DOCRED_HF_REVISION,
    PriceSnapshot,
    RelationAligner,
    documents_from_manifest,
    evaluate_documents,
    load_docred_documents,
    load_relation_info,
    make_manifest,
    sha256_file,
    select_relation_threshold,
    write_json_atomic,
)
from src.extract import Graph, KGExtractor, UsageLogger
from src.matching import SBERTEmbedder


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ProgressReporter:
    """Persist and print only redacted liveness information."""

    def __init__(self, output_dir: Path, usage: UsageLogger, budget: BudgetGuard):
        self.output_dir = output_dir
        self.usage = usage
        self.budget = budget
        self.phase = "initializing"
        self.outer_completed = 0
        self.outer_total = 0
        self._path = output_dir / "progress.json"
        self._journal = output_dir / "progress.jsonl"

    def update_context(self, phase: str, completed: int, total: int) -> None:
        self.phase = phase
        self.outer_completed = int(completed)
        self.outer_total = int(total)

    def emit(self, event: str, **inner: Any) -> None:
        usage = self.usage.summary()
        payload: dict[str, Any] = {
            "protocol": "docred-progress-v1",
            "at_utc": _utc(),
            "event": event,
            "phase": self.phase,
            "outer_completed": self.outer_completed,
            "outer_total": self.outer_total,
            "api_calls": int(usage["api_calls"]),
            "cache_hits": int(usage["cache_hits"]),
            "retries": int(usage["retries"]),
            "prompt_tokens": int(usage["prompt_tokens"]),
            "completion_tokens": int(usage["completion_tokens"]),
            "estimated_spend_eur": self.budget.estimate_eur(usage),
            "estimated_remaining_eur": self.budget.remaining_eur(usage),
        }
        payload.update({key: value for key, value in inner.items() if value is not None})
        write_json_atomic(self._path, payload)
        self._journal.parent.mkdir(parents=True, exist_ok=True)
        with self._journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        print("[docred-progress] " + json.dumps(payload, sort_keys=True), flush=True)

    def kg_callback(self, payload: dict[str, Any]) -> None:
        # KGExtractor emits only chunk counts and named phases. Filter its
        # payload explicitly so future instrumentation cannot leak text/keys.
        if payload.get("event") == "llm_retry_wait":
            safe = {
                key: payload[key]
                for key in (
                    "component", "reason", "attempt", "sleep_seconds",
                    "retry_seconds", "continuous_429_seconds",
                )
                if key in payload
            }
            self.emit("retry_heartbeat", **safe)
            return
        safe = {
            key: payload[key]
            for key in ("event", "kind", "completed", "total", "phase")
            if key in payload
        }
        self.emit("inner_progress", **safe)


def _graph_path(root: Path, document_id: str) -> Path:
    return root / "graphs" / f"{document_id}.json"


def _read_graph_checkpoint(path: Path) -> Graph | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        graph_payload = payload["graph"]
        graph = Graph.from_dict(graph_payload)
        canonical = json.dumps(graph.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if payload.get("graph_sha256") != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
            return None
        return graph
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_graph_checkpoint(path: Path, graph: Graph) -> None:
    graph_payload = graph.to_dict()
    canonical = json.dumps(graph_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    write_json_atomic(path, {
        "protocol": "docred-graph-checkpoint-v1",
        "graph": graph_payload,
        "graph_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    })


def _cache_available(extractor: KGExtractor, text: str) -> bool:
    # The cache key remains internal; no key is sent to progress/log output.
    return extractor.cache_location(extractor._cache_key(text)) is not None  # noqa: SLF001


def _process_documents(
    documents, *, phase: str, output_dir: Path, extractor: KGExtractor,
    progress: ProgressReporter, budget: BudgetGuard, cache_only: bool,
) -> dict[str, Graph]:
    graphs: dict[str, Graph] = {}
    for index, document in enumerate(documents, start=1):
        progress.update_context(phase, index - 1, len(documents))
        checkpoint = _graph_path(output_dir, document.document_id)
        checkpoint_graph = None if cache_only else _read_graph_checkpoint(checkpoint)
        if checkpoint_graph is not None:
            graphs[document.document_id] = checkpoint_graph
            progress.update_context(phase, index, len(documents))
            progress.emit("outer_checkpoint_reused")
            continue
        if not cache_only and not _cache_available(extractor, document.text):
            budget.assert_can_start_document(extractor.usage.summary())
        progress.emit("outer_document_started", cache_only=cache_only)
        graph = extractor.extract(document.text, kind="docred_kg")
        if not cache_only:
            _write_graph_checkpoint(checkpoint, graph)
        graphs[document.document_id] = graph
        progress.update_context(phase, index, len(documents))
        progress.emit("outer_document_completed", cache_only=cache_only)
    return graphs


def _uncached_document_count(documents, *, output_dir: Path, extractor: KGExtractor) -> int:
    """Count future live documents without emitting cache identities."""
    return sum(
        _read_graph_checkpoint(_graph_path(output_dir, document.document_id)) is None
        and not _cache_available(extractor, document.text)
        for document in documents
    )


def _write_scores(path: Path, scores: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for score in scores:
            handle.write(json.dumps(score, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stage", choices=("all", "replay"), default="all")
    parser.add_argument("--train-count", type=int, default=50)
    parser.add_argument("--dev-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-eur", type=float, default=10.5)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.stage == "replay" and not args.cache_only:
        raise SystemExit("--stage replay requires --cache-only")
    if args.train_count != 50 or args.dev_count != 200:
        raise SystemExit("this preregistered pilot requires --train-count 50 and --dev-count 200")
    if args.n_bootstrap < 1:
        raise SystemExit("--n-bootstrap must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    train_all = load_docred_documents(args.data_dir, "train_annotated")
    dev_all = load_docred_documents(args.data_dir, "dev")
    relation_info = load_relation_info(args.data_dir)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = make_manifest(
            train_documents=train_all, dev_documents=dev_all,
            train_count=args.train_count, dev_count=args.dev_count,
            seed=args.seed, data_dir=args.data_dir,
        )
        write_json_atomic(manifest_path, manifest)
    if manifest.get("dataset", {}).get("repository") != DOCRED_HF_REPO or manifest.get("dataset", {}).get("revision") != DOCRED_HF_REVISION:
        raise SystemExit("DocRED manifest does not use the pinned dataset revision")
    expected_hashes = manifest["dataset"].get("files_sha256")
    if not isinstance(expected_hashes, dict):
        raise SystemExit("DocRED manifest has no file checksum map")
    for name, filename in (("train_annotated", "train_annotated.json.gz"), ("dev", "dev.json.gz"), ("rel_info", "rel_info.json.gz")):
        if expected_hashes.get(name) != sha256_file(Path(args.data_dir) / filename):
            raise SystemExit(f"DocRED materialized file checksum mismatch: {name}")
    train_documents, dev_documents = documents_from_manifest(manifest, train_all, dev_all)

    cfg = load_config(args.config)
    usage = UsageLogger(output_dir / "usage.jsonl")
    budget = BudgetGuard(args.budget_eur, PriceSnapshot())
    progress = ProgressReporter(output_dir, usage, budget)
    extractor = KGExtractor(
        cfg, usage=usage, cache_only=args.cache_only,
        progress_callback=progress.kg_callback,
    )
    embedder = SBERTEmbedder(
        cfg.matching.embedding_model,
        model_revision=getattr(cfg.matching, "embedding_model_revision", None),
        model_path=getattr(cfg.matching, "embedding_model_path", None),
        device=cfg.matching.embedding_device,
        local_files_only=bool(cfg.matching.local_files_only),
    )
    aligner = RelationAligner(relation_info, embedder)
    metadata = {
        "protocol": "docred-kg-evaluation-v1",
        "state": "running",
        "stage": args.stage,
        "cache_only": bool(args.cache_only),
        "manifest_sha256": manifest["manifest_sha256"],
        "budget": budget.manifest(),
        "started_at_utc": _utc(),
    }
    write_json_atomic(output_dir / "run_metadata.json", metadata)
    progress.emit("run_started", cache_only=bool(args.cache_only))
    try:
        if args.stage == "all":
            _process_documents(
                train_documents[:10], phase="smoke", output_dir=output_dir,
                extractor=extractor, progress=progress, budget=budget, cache_only=False,
            )
            progress.emit("smoke_completed")
            # The smoke documents are part of calibration. Before permitting
            # the remaining calibration/held-out work, reserve the configured
            # worst-case document allowance for every still-cold item.
            remaining_live = _uncached_document_count(
                [*train_documents[10:], *dev_documents],
                output_dir=output_dir,
                extractor=extractor,
            )
            budget.assert_can_reserve_documents(extractor.usage.summary(), remaining_live)
            progress.emit("budget_reserve_confirmed", remaining_live_documents=remaining_live)
            train_graphs = _process_documents(
                train_documents, phase="calibration", output_dir=output_dir,
                extractor=extractor, progress=progress, budget=budget, cache_only=False,
            )
        else:
            train_graphs = _process_documents(
                train_documents, phase="replay_calibration", output_dir=output_dir,
                extractor=extractor, progress=progress, budget=budget, cache_only=True,
            )
        threshold, tuning = select_relation_threshold(
            train_documents, train_graphs, {}, aligner,
        )
        write_json_atomic(output_dir / "relation_alignment_tuning.json", {
            "protocol": "docred-relation-alignment-v1",
            "selected_on_split": "train_annotated",
            "selected_threshold": threshold,
            **tuning,
        })
        progress.emit("relation_threshold_frozen", selected_threshold=threshold)
        dev_graphs = _process_documents(
            dev_documents,
            phase="heldout" if args.stage == "all" else "replay_heldout",
            output_dir=output_dir, extractor=extractor, progress=progress,
            budget=budget, cache_only=args.cache_only,
        )
        summary, scores = evaluate_documents(
            dev_documents, dev_graphs, {}, aligner, threshold,
            bootstrap_seed=args.seed, n_bootstrap=args.n_bootstrap,
        )
        summary.update({
            "protocol": "docred-kg-evaluation-v1",
            "evaluation_split": "held-out-development",
            "selected_relation_threshold": threshold,
            "manifest_sha256": manifest["manifest_sha256"],
            "usage": usage.summary(),
            "budget": {**budget.manifest(), "estimated_spend_eur": budget.estimate_eur(usage.summary())},
        })
        write_json_atomic(output_dir / "metrics.json", summary)
        _write_scores(output_dir / "document_scores.jsonl", scores)
        metadata.update({"state": "completed", "finished_at_utc": _utc(), "usage": usage.summary()})
        write_json_atomic(output_dir / "run_metadata.json", metadata)
        progress.emit("run_completed", selected_threshold=threshold)
    except BudgetExceeded as exc:
        metadata.update({"state": "budget_exhausted", "finished_at_utc": _utc(), "error": str(exc), "usage": usage.summary()})
        write_json_atomic(output_dir / "run_metadata.json", metadata)
        progress.emit("budget_exhausted")
        raise SystemExit(75) from exc
    except Exception as exc:
        metadata.update({"state": "error", "finished_at_utc": _utc(), "error_type": type(exc).__name__, "usage": usage.summary()})
        write_json_atomic(output_dir / "run_metadata.json", metadata)
        progress.emit("run_error", error_type=type(exc).__name__)
        raise


if __name__ == "__main__":
    main()
