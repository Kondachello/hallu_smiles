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

import json
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


def _progress(payload: Mapping[str, Any]) -> None:
    print("TYPED_METRIC_PROGRESS " + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


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
    selected = _select_replay_records(
        records, coverage, replay_count=replay_count, replay_selection_seed=replay_selection_seed,
    )
    _progress({"event": "selection_complete", "selected": len(selected),
               "available_complete": len(_fully_cached(records, coverage))})

    typer = AgentTyper.from_config(
        typing, cache_root=str(output_root / "typing-cache"),
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
    ok = failed = empty = 0
    batch_no = 0
    current: list[dict[str, Any]] = []

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
        _progress({"event": "batch_flushed", **b_summary})
        current = []

    with metrics_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(selected, start=1):
            item = make_detection_input(record)
            _progress({"event": "record_started", "index": index, "total": len(selected),
                       "response_id": item.response_id})
            result = detector.predict(item)
            row = {
                "response_id": result.response_id, "source_id": result.source_id,
                "method": result.method, "status": result.status,
                "raw_score": result.raw_score, "components": dict(result.components),
                "failure": dict(result.failure) if result.failure else None,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            current.append(row)
            ok += result.status == "ok"
            failed += result.status == "failed"
            empty += result.status == "empty_graph"
            _progress({"event": "record_finished", "index": index, "total": len(selected),
                       "response_id": item.response_id, "status": result.status,
                       "raw_score": result.raw_score,
                       "eg_type": result.components.get("eg_type") if result.components else None})
            if len(current) >= batch_size:
                _flush_batch()
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
