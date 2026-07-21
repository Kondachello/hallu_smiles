"""One-response, two-pass real controlled shared-KGGen DataSphere probe.

This is intentionally separate from ``live_one_instance``: the older probe keeps its
native GraphEval gateway extractor, whereas this module injects one KGGen response graph
into both detectors and proves cache-only replay.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .artifacts import RunArchive, atomic_write_jsonl
from .datasets.ragtruth import materialize_one_response_no_gold
from .detectors import build_controlled_shared_kggen_detectors
from .live_one_instance import _gateway_origin_from_environment, _load_yaml_mapping, _redact, validate_gateway_manifest
from .one_instance import _validated_run_id, default_run_id
from .runner import run_paired, seal_run

PROBE_VERSION = "ragtruth-one-instance-controlled-live-shared-kggen-v1"


def _assert_secret_absent(archive: RunArchive, secret: str) -> None:
    for path in archive.path.rglob("*"):
        if path.is_file() and secret in path.read_text(encoding="utf-8", errors="ignore"):
            raise RuntimeError(f"secret leaked into artifact: {path.relative_to(archive.path)}")


def _statuses(archive: RunArchive) -> dict[str, str]:
    return {str(row["method"]): str(row["status"]) for row in archive.read_jsonl("predictions/raw_predictions.jsonl")}


def run_ragtruth_one_instance_controlled_live_probe(
    *, source_info_path: str | Path, response_path: str | Path, response_id: str,
    output_root: str | Path, hallugraph_config: str | Path, grapheval_config: str | Path,
    gateway_manifest_path: str | Path, cache_root: str | Path, run_id: str | None = None,
    environ: Mapping[str, str] | None = None, detector_factory: Callable[..., Any] | None = None,
) -> tuple[RunArchive, RunArchive, dict[str, Any]]:
    """Execute real ``read_write`` then fail-closed ``cache_only`` on one input.

    ``detector_factory`` exists only for offline dependency-injection tests. Production
    resolves ``build_controlled_shared_kggen_detectors`` at call time.
    """
    import os

    environment = os.environ if environ is None else environ
    origin, secret = _gateway_origin_from_environment(environment)
    hallu = _load_yaml_mapping(hallugraph_config, label="HalluGraph runtime config")
    graph = _load_yaml_mapping(grapheval_config, label="GraphEval controlled runtime config")
    logical_model = str((hallu.get("llm") or {}).get("model", ""))
    if not logical_model:
        raise ValueError("HalluGraph runtime config has no llm.model")
    extractor_cfg, nli_cfg = graph.get("extractor") or {}, graph.get("nli") or {}
    if extractor_cfg.get("backend") != "shared_kggen" or nli_cfg.get("backend") != "hhem":
        raise ValueError("controlled live probe requires GraphEval shared_kggen extractor and hhem NLI")
    if not (Path(str(nli_cfg.get("model", ""))) / "config.json").is_file():
        raise RuntimeError("pinned local HHEM snapshot is unavailable")
    manifest = json.loads(Path(gateway_manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("gateway manifest must be an object")
    manifest_sha = validate_gateway_manifest(manifest, logical_model=logical_model)
    record = materialize_one_response_no_gold(source_info_path, response_path, response_id=response_id)
    base_id = _validated_run_id(run_id) if run_id else default_run_id(record["response_id"])
    selected_cache = Path(cache_root)
    factory = detector_factory or build_controlled_shared_kggen_detectors

    def execute(mode: str, suffix: str) -> tuple[RunArchive, Any, dict[str, str], dict[str, Any]]:
        archive = RunArchive.create(output_root, run_id=f"{base_id}-{suffix}", manifest={
            "run_purpose": "ragtruth_one_instance_controlled_live_shared_kggen",
            "probe_version": PROBE_VERSION, "comparison_track": "controlled_shared_kggen_response_v1",
            "cache_mode": mode, "cache_root": str(selected_cache), "gateway_manifest_sha256": manifest_sha,
            "gateway_origin": origin, "selected_response_id": record["response_id"],
            "selected_source_id": record["source_id"], "gold_passed_to_detectors": False,
        })
        instances = archive.path / "instances.no_gold.jsonl"
        atomic_write_jsonl(instances, [record])
        detectors, provider = factory(
            hallugraph_config=hallugraph_config, grapheval_config=graph,
            gateway_manifest_sha256=manifest_sha, cache_mode=mode,
            hallugraph_usage_path=archive.path / "audit/hallugraph_extraction_attempts.jsonl",
        )
        summary = run_paired(archive, instances_path=instances, detectors=detectors, shared_graph_provider=provider)
        seal_run(archive, instances)
        validation = archive.validate()
        statuses = _statuses(archive)
        _assert_secret_absent(archive, secret)
        if not validation["valid"] or statuses != {"hallugraph": "ok", "grapheval": "ok"}:
            raise RuntimeError(f"{mode} controlled pass failed: statuses={statuses}, validation={validation}")
        return archive, provider, statuses, summary

    try:
        materialize, materialize_provider, materialize_statuses, materialize_summary = execute("read_write", "materialize")
        replay, replay_provider, replay_statuses, replay_summary = execute("cache_only", "cache-replay")
        hashes = {
            row.get("shared_graph_sha256")
            for archive in (materialize, replay)
            for row in archive.read_jsonl("predictions/raw_predictions.jsonl")
        }
        hashes.discard(None)
        materialize_calls = materialize_provider.extractor.usage.summary()["api_calls"]
        replay_calls = replay_provider.extractor.usage.summary()["api_calls"]
        report = {
            "probe_version": PROBE_VERSION, "response_id": record["response_id"], "cache_root": str(selected_cache),
            "materialize_archive": str(materialize.path), "cache_replay_archive": str(replay.path),
            "materialize_statuses": materialize_statuses, "cache_replay_statuses": replay_statuses,
            "materialize_real_kggen_calls": materialize_calls, "cache_replay_real_kggen_calls": replay_calls,
            "cache_replay_gateway_calls": 0, "shared_response_graph_sha256": next(iter(hashes)) if len(hashes) == 1 else None,
            "shared_graph_consistent_across_methods_and_passes": len(hashes) == 1,
            "materialize_summary": materialize_summary, "cache_replay_summary": replay_summary,
            "gold_access_state": "hidden",
        }
        if materialize_calls <= 0 or replay_calls != 0 or len(hashes) != 1:
            raise RuntimeError("controlled live two-pass invariants failed")
        for archive in (materialize, replay):
            archive.write_json("reports/controlled_live_two_pass_report.json", _redact(report, secret))
        return materialize, replay, report
    except Exception as exc:
        raise RuntimeError(_redact(repr(exc), secret)) from None


def render_controlled_live_probe_summary(report: Mapping[str, Any]) -> str:
    return "\n".join((
        "RAGTRUTH ONE-INSTANCE CONTROLLED LIVE SHARED-KGGEN PROBE",
        f"  response_id                 : {report['response_id']}",
        f"  materialize real KGGen calls : {report['materialize_real_kggen_calls']}",
        f"  cache replay KGGen calls     : {report['cache_replay_real_kggen_calls']}",
        f"  cache replay gateway calls   : {report['cache_replay_gateway_calls']}",
        f"  same shared graph            : {report['shared_graph_consistent_across_methods_and_passes']}",
    ))
