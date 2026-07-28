"""Typed-vertex metric pass over a historical QA graph cache.

Reuses the exact cache resolution the cache-replay driver already performs (the
bash driver hands us the resolved ``historical_cache_root`` + reconstructed
HalluGraph runtime config), builds the same shared-graph provider, then scores
every selected record with :class:`TypedVertexDetector`:

    EG_type (answer vertices grounded by assigned type) + RP (edge grounding)
    -> CFI_type -> raw_score, written to ``typed_metrics.jsonl``.

Graphs are read cache-only (no extraction LLM calls); only the typing agent makes
gateway LLM calls (plus local HHEM NLI). No other experiment's results are read;
the strict/support numbers are compared post-hoc, not here.
"""
from __future__ import annotations

import concurrent.futures as _futures
import json
import sys
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import make_detection_input
from .datasets.historical_qa import materialize_historical_qa_no_gold
from .detectors import build_controlled_shared_kggen_detectors
from .historical_qa_cache_replay import _fully_cached, _select_replay_records
from .live_one_instance import _load_yaml_mapping
from .shared_graphs import GraphCacheSource
from .typed_vertex_detector import TypedVertexDetector
from .typed_vertex_typer import AgentTyper
from src.extract import CACHE_KEY_SCHEMA_V11_PRE_LENGTH_RETRY


def _build_embedder(hallu: Mapping[str, Any]) -> Any | None:
    """S-BERT embedder for the RP edge component, from the HalluGraph config."""
    emb = dict(hallu.get("embedding") or hallu.get("embedder") or {})
    model_path = emb.get("model_path") or emb.get("path")
    model_name = emb.get("model") or emb.get("model_name")
    if not (model_path or model_name):
        return None
    from src.matching import SBERTEmbedder

    return SBERTEmbedder(
        model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2",
        model_path=model_path,
        model_revision=emb.get("revision"),
    )


_PROGRESS_LOCK = threading.Lock()


class _FailedResult:
    """Stand-in DetectionResult for the defensive case where a worker's
    detector.predict() raises outside its own guard (keeps the run going)."""

    status = "failed"
    raw_score = None
    components: dict[str, Any] = {}

    def __init__(self, response_id: str, source_id: str, method: str, exc: BaseException) -> None:
        self.response_id = response_id
        self.source_id = source_id
        self.method = method
        self.failure = {"error": f"predict_exception: {exc!r}"}


def _progress(payload: Mapping[str, Any]) -> None:
    # One atomic write per event so concurrent worker threads cannot interleave
    # partial lines on stdout.
    line = "TYPED_METRIC_PROGRESS " + json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    with _PROGRESS_LOCK:
        sys.stdout.write(line)
        sys.stdout.flush()


def run_typed_metric_pass(
    *,
    data_dir: str | Path,
    output_root: str | Path,
    hallugraph_config: str | Path,
    grapheval_config: str | Path,
    typing_config: str | Path,
    historical_cache_root: str | Path,
    additional_cache_roots: Sequence[str | Path] = (),
    gateway_manifest_sha256: str | None = None,
    qa_sample_size: int = 100,
    qa_test_fraction: str = "0.2",
    sample_seed: int = 42,
    replay_count: int | str = "all",
    replay_selection_seed: int = 20260722,
    alpha: float = 0.5,
    batch_size: int = 15,
    live_mirror_dir: str | Path | None = None,
    max_workers: int = 1,
) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    hallu = _load_yaml_mapping(hallugraph_config, label="typed pass HalluGraph config")
    graph = _load_yaml_mapping(grapheval_config, label="typed pass GraphEval config")
    typing = _load_yaml_mapping(typing_config, label="typed pass typing config")

    records = materialize_historical_qa_no_gold(
        data_dir, qa_sample_size=qa_sample_size, qa_test_fraction=qa_test_fraction, sample_seed=sample_seed,
    )
    sources = [GraphCacheSource(
        "historical_primary", Path(historical_cache_root), read_only=True,
        priority=len(additional_cache_roots) + 1,
        cache_key_compatibility=(CACHE_KEY_SCHEMA_V11_PRE_LENGTH_RETRY,),
    )]
    for index, extra in enumerate(additional_cache_roots):
        sources.append(GraphCacheSource(
            f"historical_lineage_{index}", Path(extra), read_only=True,
            priority=len(additional_cache_roots) - index,
            cache_key_compatibility=(CACHE_KEY_SCHEMA_V11_PRE_LENGTH_RETRY,),
        ))
    # Reuse the controlled provider; we discard its detectors and score with ours.
    _detectors, provider = build_controlled_shared_kggen_detectors(
        hallugraph_config=hallugraph_config, grapheval_config=graph,
        gateway_manifest_sha256=gateway_manifest_sha256,
        cache_sources=tuple(sources), cache_mode="cache_only",
    )
    coverage = provider.preflight(records, roles=("response", "context", "query"), require_complete=False)
    if not _fully_cached(records, coverage):
        # Cache is proven intact by the read-only diagnostic (2247/2250 legacy
        # hits) yet preflight sees nothing: dump exactly what the provider
        # computed so the mismatch (fingerprint / key schema / source) is visible.
        from collections import Counter
        rows = list(coverage.get("rows", []))
        status_counts = Counter(str(r.get("status")) for r in rows)
        try:
            cfg = getattr(provider.extractor, "cfg", None)
            fp = getattr(getattr(cfg, "llm", None), "runtime_fingerprint", None)
            rev = getattr(getattr(cfg, "llm", None), "model_revision", None)
        except Exception:  # pragma: no cover - diagnostic only
            fp = rev = "<unavailable>"
        sample = [r for r in rows if str(r.get("response_id")) == "16121"][:3] or rows[:3]
        import hashlib
        try:
            key_params = provider.extractor._cache_key_params()
        except Exception as exc:  # pragma: no cover - diagnostic only
            key_params = {"error": repr(exc)}
        rec16121 = next((r for r in records if str(r.get("response_id")) == "16121"), None)
        text_probe = {}
        if rec16121 is not None:
            for role, field in (("response", "response_raw"), ("context", "context_raw"), ("query", "query_raw")):
                t = rec16121.get(field) or rec16121.get(role) or ""
                text_probe[role] = {"len": len(t), "sha8": hashlib.sha256(t.encode("utf-8")).hexdigest()[:8]}
        _progress({
            "event": "coverage_debug", "record_count": len(records),
            "row_count": len(rows), "status_counts": dict(status_counts),
            "extractor_runtime_fingerprint": fp, "extractor_model_revision": rev,
            "cache_key_params": key_params,
            "record_16121_fields": sorted(rec16121.keys()) if rec16121 else None,
            "record_16121_text": text_probe,
            "sample_rows": sample,
            "source_ids": [getattr(s, "source_id", None) for s in sources],
        })
    selected = _select_replay_records(
        records, coverage, replay_count=replay_count, replay_selection_seed=replay_selection_seed,
    )
    _progress({"event": "selection_complete", "selected": len(selected),
               "available_complete": len(_fully_cached(records, coverage))})

    typer = AgentTyper.from_config(
        typing_config, cache_root=str(output_root / "typing-cache"),
        artifacts_root=str(output_root / "typing-runs"),
    )
    detector = TypedVertexDetector(
        shared_graph_provider=provider, typer=typer, embedder=_build_embedder(hallu),
        matching_config=hallu.get("matching"), alpha=alpha,
    )

    # Persist incrementally in batches of ``batch_size`` so partial results survive a
    # crash/timeout (the job's exit trap archives RUN_ROOT even on failure) and can be
    # collected batch by batch. Each batch also appends to the cumulative jsonl.
    batch_size = max(1, int(batch_size))
    metrics_path = output_root / "typed_metrics.jsonl"
    batch_dir = output_root / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    # Optional live mirror onto the (writable) Project disk so a read-only probe
    # can tail progress while the job runs -- only the small summaries + a
    # cumulative progress.json are mirrored, never the heavy per-record jsonl.
    live_dir: Path | None = None
    if live_mirror_dir:
        live_dir = Path(live_mirror_dir)
        try:
            live_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            live_dir = None
    ok = failed = empty = 0
    all_egs: list[float] = []
    batch_no = 0
    current: list[dict[str, Any]] = []

    def _mirror_progress() -> None:
        if live_dir is None:
            return
        try:
            (live_dir / "progress.json").write_text(
                json.dumps({
                    "protocol": "typed-vertex-metric-pass-live-v1", "run": output_root.name,
                    "selected": len(selected), "done": ok + failed + empty,
                    "ok": ok, "failed": failed, "empty_graph": empty, "batches": batch_no,
                    "mean_eg_type": (sum(all_egs) / len(all_egs)) if all_egs else None,
                    "updated_utc": utc_now(),
                }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass

    def _flush_batch() -> None:
        nonlocal batch_no, current
        if not current:
            return
        batch_no += 1
        first_idx = (batch_no - 1) * batch_size + 1
        b_ok = sum(r["status"] == "ok" for r in current)
        b_failed = sum(r["status"] == "failed" for r in current)
        b_empty = sum(r["status"] == "empty_graph" for r in current)
        b_egs = [r["components"].get("eg_type") for r in current
                 if r["status"] == "ok" and r.get("components")]
        tag = f"{batch_no:04d}"
        (batch_dir / f"batch-{tag}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in current),
            encoding="utf-8",
        )
        b_summary = {
            "batch_index": batch_no, "batch_size": batch_size, "count": len(current),
            "first_record_index": first_idx, "last_record_index": first_idx + len(current) - 1,
            "ok": b_ok, "failed": b_failed, "empty_graph": b_empty,
            "mean_eg_type": (sum(b_egs) / len(b_egs)) if b_egs else None,
            "response_ids": [r["response_id"] for r in current],
        }
        (batch_dir / f"batch-{tag}.summary.json").write_text(
            json.dumps(b_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        all_egs.extend(e for e in b_egs if e is not None)
        if live_dir is not None:
            try:
                (live_dir / f"batch-{tag}.summary.json").write_text(
                    json.dumps(b_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            except OSError:
                pass
        _mirror_progress()
        _progress({"event": "batch_flushed", **b_summary})
        current = []

    method_name = getattr(detector, "method_name", "typed_vertex")
    total = len(selected)
    workers = max(1, int(max_workers))
    if workers > 1:
        # Record-level threads carry the concurrency; keep each HHEM forward pass
        # single-threaded so torch intra-op parallelism does not oversubscribe cores.
        try:
            import torch

            torch.set_num_threads(1)
        except Exception:
            pass

    def _row(result: Any) -> dict[str, Any]:
        return {
            "response_id": result.response_id, "source_id": result.source_id,
            "method": result.method, "status": result.status,
            "raw_score": result.raw_score, "components": dict(result.components),
            "failure": dict(result.failure) if result.failure else None,
        }

    with metrics_path.open("w", encoding="utf-8") as handle:
        # Emission is confined to the main thread and driven in strict record order,
        # so metrics.jsonl, batch files and counters need no locking -- only
        # detector.predict() (gateway + HHEM, the slow part) runs on worker threads.
        def _emit(index: int, item: Any, result: Any) -> None:
            nonlocal ok, failed, empty
            row = _row(result)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            current.append(row)
            ok += result.status == "ok"
            failed += result.status == "failed"
            empty += result.status == "empty_graph"
            _progress({"event": "record_finished", "index": index, "total": total,
                       "response_id": item.response_id, "status": result.status,
                       "raw_score": result.raw_score,
                       "eg_type": result.components.get("eg_type") if result.components else None})
            if len(current) >= batch_size:
                _flush_batch()

        def _predict(index: int, item: Any) -> Any:
            _progress({"event": "record_started", "index": index, "total": total,
                       "response_id": item.response_id})
            try:
                return detector.predict(item)
            except Exception as exc:  # pragma: no cover - defensive; predict is self-guarding
                return _FailedResult(item.response_id, item.source_id, method_name, exc)

        items = [(i, make_detection_input(rec)) for i, rec in enumerate(selected, start=1)]
        if workers <= 1:
            for index, item in items:
                _emit(index, item, _predict(index, item))
        else:
            # Bounded out-of-order window: submit all, buffer completed results and
            # emit them in index order so batches stay deterministic and contiguous.
            with _futures.ThreadPoolExecutor(max_workers=workers) as pool:
                fut_map = {pool.submit(_predict, index, item): (index, item)
                           for index, item in items}
                pending: dict[int, tuple[Any, Any]] = {}
                next_index = 1
                for fut in _futures.as_completed(fut_map):
                    index, item = fut_map[fut]
                    pending[index] = (item, fut.result())
                    while next_index in pending:
                        emit_item, emit_result = pending.pop(next_index)
                        _emit(next_index, emit_item, emit_result)
                        next_index += 1
        _flush_batch()  # remaining partial batch

    summary = {
        "protocol": "typed-vertex-metric-pass-v1", "alpha": alpha,
        "selected": len(selected), "ok": ok, "failed": failed, "empty_graph": empty,
        "batch_size": batch_size, "batches": batch_no,
        "historical_cache_root": str(historical_cache_root),
        "metrics_path": str(metrics_path), "batch_dir": str(batch_dir),
    }
    (output_root / "typed_metric_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _progress({"event": "pass_complete", **summary})
    return summary
