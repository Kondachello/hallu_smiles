"""Run both detectors on reproducibly selected historical 100-QA graph sets."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Mapping

from .artifacts import RunArchive, atomic_write_json, atomic_write_jsonl, utc_now
from .datasets.historical_qa import materialize_historical_qa_no_gold
from .detectors import build_controlled_shared_kggen_detectors
from .live_one_instance import _load_yaml_mapping
from .one_instance import _validated_run_id
from .runner import run_paired, seal_run
from .shared_graphs import GraphCacheSource
from src.extract import CACHE_KEY_SCHEMA_V11_PRE_LENGTH_RETRY

PROBE_VERSION = "historical-qa-cache-controlled-replay-v3"


def _statuses(archive: RunArchive) -> dict[str, str]:
    return {str(row["method"]): str(row["status"]) for row in archive.read_jsonl("predictions/raw_predictions.jsonl")}


def _fully_cached(records: list[dict[str, Any]], coverage: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_id: dict[str, set[str]] = {}
    for row in coverage.get("rows", []):
        if row.get("status") == "compatible_hit":
            by_id.setdefault(str(row.get("response_id")), set()).add(str(row.get("role")))
    return [
        record for record in records
        if by_id.get(str(record["response_id"]), set()) == {"response", "context", "query"}
    ]


def _select_replay_records(
    records: list[dict[str, Any]], coverage: Mapping[str, Any], *, replay_count: int, replay_selection_seed: int,
) -> list[dict[str, Any]]:
    if replay_count < 1:
        raise ValueError("replay_count must be positive")
    compatible = _fully_cached(records, coverage)
    if len(compatible) < replay_count:
        raise RuntimeError(
            "historical cache has no selected QA record with response, context, and query graphs: "
            f"requested={replay_count}, available={len(compatible)}"
        )
    return random.Random(replay_selection_seed).sample(compatible, replay_count)


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
    replay_count: int = 1,
    replay_selection_seed: int = 20260722,
    detector_factory: Callable[..., Any] | None = None,
) -> tuple[RunArchive, dict[str, Any]]:
    """Select complete historical QA inputs and run a strict cache-only replay.

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
            "selected_response_ids": [],
            "replay_count": replay_count,
            "replay_selection_seed": replay_selection_seed,
            "gold_passed_to_detectors": False,
            "llm_gateway_calls_permitted": 0,
        },
    )
    atomic_write_json(archive.path / "reports/historical_cache_coverage.json", coverage)
    try:
        selected_records = _select_replay_records(
            records, coverage, replay_count=replay_count, replay_selection_seed=replay_selection_seed,
        )
    except RuntimeError as exc:
        archive.update_status("cache_preflight_failed", failure=str(exc))
        raise
    selected_ids = [str(record["response_id"]) for record in selected_records]
    progress_path = archive.path / "reports/progress.jsonl"

    def progress(event: Mapping[str, Any]) -> None:
        payload = {
            "event_index": len(progress_path.read_text(encoding="utf-8").splitlines()) + 1 if progress_path.exists() else 1,
            "at_utc": utc_now(),
            **dict(event),
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
        print("HISTORICAL_REPLAY_PROGRESS " + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)

    progress({
        "event": "selection_complete", "requested_count": replay_count,
        "available_complete_count": len(_fully_cached(records, coverage)),
        "selection_seed": replay_selection_seed, "response_ids": selected_ids,
    })
    archive.update_status(
        "cache_record_selected",
        selected_response_id=selected_records[0]["response_id"],
        selected_source_id=selected_records[0]["source_id"],
        selected_response_ids=selected_ids,
    )
    instances = archive.path / "instances.no_gold.jsonl"
    atomic_write_jsonl(instances, selected_records)
    summary = run_paired(
        archive, instances_path=instances, detectors=detectors, shared_graph_provider=provider,
        progress_callback=progress,
    )
    seal = seal_run(archive, instances)
    validation = archive.validate()
    prediction_rows = archive.read_jsonl("predictions/raw_predictions.jsonl")
    per_response_statuses: dict[str, dict[str, str]] = {}
    for row in prediction_rows:
        per_response_statuses.setdefault(str(row["response_id"]), {})[str(row["method"])] = str(row["status"])
    statuses = {
        method: "ok" if all(per_response_statuses.get(response_id, {}).get(method) == "ok" for response_id in selected_ids) else "failed"
        for method in ("hallugraph", "grapheval")
    }
    usage = provider.extractor.usage.summary()
    graph_eval_extractor_calls = sum(
        int((row.get("usage") or {}).get("extractor_calls", 0))
        for row in prediction_rows
        if row.get("method") == "grapheval"
    )
    sources = {row.get("shared_graph_source") for row in provider.artifact_records()}
    report = {
        "probe_version": PROBE_VERSION,
        "selected_response_id": selected_records[0]["response_id"],
        "selected_source_id": selected_records[0]["source_id"],
        "selected_response_ids": selected_ids,
        "replay_count": replay_count,
        "replay_selection_seed": replay_selection_seed,
        "historical_cache_root": str(source.root),
        "historical_lineage_id": lineage.get("lineage_id"),
        "cache_preflight": coverage,
        "detector_statuses": statuses,
        "per_response_statuses": per_response_statuses,
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
        f"  response_ids      : {', '.join(report['selected_response_ids'])}",
        f"  HalluGraph        : {report['detector_statuses'].get('hallugraph')}",
        f"  GraphEval         : {report['detector_statuses'].get('grapheval')}",
        f"  KGGen API calls   : {report['kggen_api_calls']}",
        f"  graph source      : {', '.join(report['graph_sources'])}",
    ))
