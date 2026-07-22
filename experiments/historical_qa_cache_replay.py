"""Run both detectors on one fully compatible historical 100-QA graph set."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .artifacts import RunArchive, atomic_write_json, atomic_write_jsonl
from .datasets.historical_qa import materialize_historical_qa_no_gold
from .detectors import build_controlled_shared_kggen_detectors
from .live_one_instance import _load_yaml_mapping
from .one_instance import _validated_run_id
from .runner import run_paired, seal_run
from .shared_graphs import GraphCacheSource
from src.extract import CACHE_KEY_SCHEMA_V11_PRE_LENGTH_RETRY

PROBE_VERSION = "historical-qa-cache-controlled-replay-v2"


def _statuses(archive: RunArchive) -> dict[str, str]:
    return {str(row["method"]): str(row["status"]) for row in archive.read_jsonl("predictions/raw_predictions.jsonl")}


def _first_fully_cached(records: list[dict[str, Any]], coverage: Mapping[str, Any]) -> dict[str, Any]:
    by_id: dict[str, set[str]] = {}
    for row in coverage.get("rows", []):
        if row.get("status") == "compatible_hit":
            by_id.setdefault(str(row.get("response_id")), set()).add(str(row.get("role")))
    for record in records:
        if by_id.get(str(record["response_id"]), set()) == {"response", "context", "query"}:
            return record
    raise RuntimeError("historical cache has no selected QA record with response, context, and query graphs")


def run_historical_qa_cache_controlled_replay(
    *,
    data_dir: str | Path,
    output_root: str | Path,
    hallugraph_config: str | Path,
    grapheval_config: str | Path,
    historical_cache_root: str | Path,
    lineage_path: str | Path,
    run_id: str,
    qa_sample_size: int = 100,
    qa_test_fraction: str = "0.2",
    sample_seed: int = 42,
    detector_factory: Callable[..., Any] | None = None,
) -> tuple[RunArchive, dict[str, Any]]:
    """Select the first fully warm historical QA input and run cache-only.

    This function never reads an API key and passes ``cache_only`` to the controlled
    factory.  A cache miss fails before a KGGen backend can be constructed.
    """
    hallu = _load_yaml_mapping(hallugraph_config, label="historical replay HalluGraph config")
    graph = _load_yaml_mapping(grapheval_config, label="historical replay GraphEval config")
    if (graph.get("extractor") or {}).get("backend") != "shared_kggen":
        raise ValueError("historical replay requires GraphEval extractor.backend='shared_kggen'")
    if (graph.get("nli") or {}).get("backend") != "hhem":
        raise ValueError("historical replay requires GraphEval nli.backend='hhem'")
    if not (Path(str((graph.get("nli") or {}).get("model", ""))) / "config.json").is_file():
        raise RuntimeError("pinned local HHEM snapshot is unavailable")
    lineage = json.loads(Path(lineage_path).read_text(encoding="utf-8"))
    if not isinstance(lineage, dict) or not isinstance(lineage.get("llm_runtime_fingerprint"), str):
        raise ValueError("historical cache lineage is missing its LLM runtime fingerprint")
    selected_id = _validated_run_id(run_id)
    records = materialize_historical_qa_no_gold(
        data_dir, qa_sample_size=qa_sample_size, qa_test_fraction=qa_test_fraction, sample_seed=sample_seed,
    )
    source = GraphCacheSource(
        "historical_100qa",
        Path(historical_cache_root),
        read_only=True,
        cache_key_compatibility=(CACHE_KEY_SCHEMA_V11_PRE_LENGTH_RETRY,),
    )
    factory = detector_factory or build_controlled_shared_kggen_detectors
    detectors, provider = factory(
        hallugraph_config=hallugraph_config,
        grapheval_config=graph,
        gateway_manifest_sha256=str(lineage.get("gateway_manifest_sha256", "")) or None,
        cache_sources=(source,),
        cache_mode="cache_only",
    )
    inspection = provider.inspection()
    if not inspection["valid"]:
        raise RuntimeError(f"historical cache structure is invalid: {inspection}")
    coverage = provider.preflight(
        records,
        roles=("response", "context", "query"),
        require_complete=False,
    )
    archive = RunArchive.create(
        output_root,
        run_id=selected_id,
        manifest={
            "run_purpose": "historical_qa_cache_controlled_replay",
            "probe_version": PROBE_VERSION,
            "comparison_track": "controlled_shared_kggen_response_v1",
            "cache_mode": "cache_only",
            "historical_cache_root": str(source.root),
            "historical_lineage_id": lineage.get("lineage_id"),
            "historical_llm_runtime_fingerprint": lineage["llm_runtime_fingerprint"],
            "selected_response_id": None,
            "selected_source_id": None,
            "gold_passed_to_detectors": False,
            "llm_gateway_calls_permitted": 0,
        },
    )
    atomic_write_json(archive.path / "reports/historical_cache_coverage.json", coverage)
    try:
        record = _first_fully_cached(records, coverage)
    except RuntimeError as exc:
        archive.update_status("cache_preflight_failed", failure=str(exc))
        raise
    archive.update_status(
        "cache_record_selected",
        selected_response_id=record["response_id"],
        selected_source_id=record["source_id"],
    )
    instances = archive.path / "instances.no_gold.jsonl"
    atomic_write_jsonl(instances, [record])
    summary = run_paired(archive, instances_path=instances, detectors=detectors, shared_graph_provider=provider)
    seal = seal_run(archive, instances)
    validation = archive.validate()
    statuses = _statuses(archive)
    usage = provider.extractor.usage.summary()
    prediction_rows = archive.read_jsonl("predictions/raw_predictions.jsonl")
    graph_eval_extractor_calls = sum(
        int((row.get("usage") or {}).get("extractor_calls", 0))
        for row in prediction_rows
        if row.get("method") == "grapheval"
    )
    sources = {row.get("shared_graph_source") for row in provider.artifact_records()}
    report = {
        "probe_version": PROBE_VERSION,
        "selected_response_id": record["response_id"],
        "selected_source_id": record["source_id"],
        "historical_cache_root": str(source.root),
        "historical_lineage_id": lineage.get("lineage_id"),
        "cache_preflight": coverage,
        "detector_statuses": statuses,
        "summary": summary,
        "seal": seal,
        "validation": validation,
        "kggen_api_calls": int(usage["api_calls"]),
        "gateway_llm_calls": 0,
        "grapheval_extractor_calls": graph_eval_extractor_calls,
        "graph_sources": sorted(str(value) for value in sources),
        "gold_access_state": "hidden",
    }
    archive.write_json("reports/historical_qa_cache_replay_report.json", report)
    if (
        statuses != {"hallugraph": "ok", "grapheval": "ok"}
        or not validation["valid"]
        or report["kggen_api_calls"] != 0
        or report["grapheval_extractor_calls"] != 0
        or sources != {"historical_100qa"}
    ):
        raise RuntimeError(f"historical cache replay invariants failed: {report}")
    archive.update_status("completed")
    return archive, report


def render_historical_qa_cache_replay_summary(report: Mapping[str, Any]) -> str:
    return "\n".join((
        "HISTORICAL 100-QA CACHE REPLAY (NO LLM CALLS)",
        f"  response_id       : {report['selected_response_id']}",
        f"  HalluGraph        : {report['detector_statuses'].get('hallugraph')}",
        f"  GraphEval         : {report['detector_statuses'].get('grapheval')}",
        f"  KGGen API calls   : {report['kggen_api_calls']}",
        f"  graph source      : {', '.join(report['graph_sources'])}",
    ))
