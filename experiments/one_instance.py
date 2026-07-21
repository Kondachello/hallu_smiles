"""A deliberately bounded, offline RAGTruth paired-detector probe.

The probe exercises the actual HalluGraph adapter and GraphEval facade, while their
model-facing dependencies are replaced by the deterministic fake backends.  It is an
integration check for one explicitly named RAGTruth response, never a scientific run.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from .artifacts import RunArchive, atomic_write_jsonl
from .datasets.ragtruth import materialize_one_response_no_gold
from .detectors import (
    build_controlled_shared_kggen_fake,
    build_grapheval_fake,
    build_hallugraph_fake,
)
from .runner import run_paired, seal_run

PROBE_VERSION = "ragtruth-one-instance-paired-probe-v1"


def default_run_id(response_id: str) -> str:
    """Produce a readable filesystem-safe run id without hiding the response id."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(response_id).strip())
    normalized = normalized.strip(".-") or "response"
    return f"ragtruth-one-{normalized[:96]}"


def _validated_run_id(run_id: str) -> str:
    candidate = str(run_id).strip()
    if not candidate or candidate in {".", ".."} or Path(candidate).name != candidate:
        raise ValueError("run_id must be a non-empty directory name, not a path")
    return candidate


def run_ragtruth_one_instance_probe(
    *,
    source_info_path: str | Path,
    response_path: str | Path,
    response_id: str,
    output_root: str | Path,
    run_id: str | None = None,
    hallugraph_config: str | Path = "config.yaml",
) -> tuple[RunArchive, dict[str, Any]]:
    """Run exactly one no-gold RAGTruth record through both real adapters on fakes.

    This function deliberately has no ``live`` switch.  Adding a live execution path
    must remain a separately reviewed DataSphere operation with its own preflight.
    """
    record = materialize_one_response_no_gold(
        source_info_path,
        response_path,
        response_id=response_id,
    )
    selected_run_id = _validated_run_id(run_id) if run_id is not None else default_run_id(record["response_id"])
    archive = RunArchive.create(
        output_root,
        run_id=selected_run_id,
        manifest={
            "run_purpose": "ragtruth_one_instance_offline_probe",
            "probe_version": PROBE_VERSION,
            "comparison_track": "kggen_untyped_adaptation",
            "data_source": "local_ragtruth_files",
            "source_info_path": str(Path(source_info_path)),
            "response_path": str(Path(response_path)),
            "selected_response_id": record["response_id"],
            "selected_source_id": record["source_id"],
            "selected_split": record["split"],
            "selected_task": record["metadata"]["task"],
            "source_record_sha256": record["source_record_sha256"],
            "response_record_sha256": record["response_record_sha256"],
            "network_access": False,
            "backend_mode": "fake_offline_v1",
            "gold_used_for_selection": False,
            "gold_passed_to_detectors": False,
        },
    )
    instances_path = archive.path / "instances.no_gold.jsonl"
    atomic_write_jsonl(instances_path, [record])
    archive.write_json(
        "probe_manifest.json",
        {
            "probe_version": PROBE_VERSION,
            "scope": "one explicit RAGTruth response",
            "response_id": record["response_id"],
            "source_id": record["source_id"],
            "split": record["split"],
            "task": record["metadata"]["task"],
            "source_line_number": record["source_line_number"],
            "response_line_number": record["response_line_number"],
            "network_access": False,
            "backend_mode": "fake_offline_v1",
            "gold_used_for_selection": False,
            "gold_passed_to_detectors": False,
            "evaluation_performed": False,
        },
    )
    summary = run_paired(
        archive,
        instances_path=instances_path,
        detectors={
            "hallugraph": build_hallugraph_fake(hallugraph_config),
            "grapheval": build_grapheval_fake(),
        },
    )
    seal = seal_run(archive, instances_path)
    validation = archive.validate()
    if not validation["valid"]:
        raise RuntimeError(f"invalid one-instance prediction archive: {validation['errors']}")
    report = {
        "probe_version": PROBE_VERSION,
        "archive_path": str(archive.path),
        "summary": summary,
        "seal": seal,
        "validation": validation,
    }
    archive.write_json("reports/one_instance_probe_report.json", report)
    return archive, report


def render_probe_summary(archive: RunArchive) -> str:
    """Return a compact human-readable result without exposing RAGTruth gold."""
    manifest = archive.path / "run_manifest.json"
    run_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    predictions = {row["method"]: row for row in archive.read_jsonl("predictions/raw_predictions.jsonl")}

    def score(row: dict[str, Any]) -> str:
        value = row.get("raw_score")
        return "n/a" if value is None else f"{float(value):.4f}"

    hallu = predictions["hallugraph"]
    graph = predictions["grapheval"]
    hallu_components = hallu.get("components", {})
    graph_components = graph.get("components", {})
    paired = archive.read_jsonl("predictions/paired_predictions.jsonl")[0]
    rows = [
        ("response_id", str(run_manifest["selected_response_id"])),
        ("source_id", str(run_manifest["selected_source_id"])),
        ("task / split", f"{run_manifest['selected_task']} / {run_manifest['selected_split']}"),
        ("HalluGraph", f"status={hallu['status']}; H={score(hallu)}; EG={hallu_components.get('EG', 'n/a')}; RP={hallu_components.get('RP', 'n/a')}; CFI={hallu_components.get('CFI', 'n/a')}"),
        ("GraphEval", f"status={graph['status']}; H={score(graph)}; valid triples={graph_components.get('n_triples_valid', 'n/a')}"),
        ("paired", f"both_ok={paired['both_status_ok']}; |ΔH|={paired['absolute_score_difference']}"),
        ("safety", "offline fake backends; no network; gold passed to detectors: no"),
        ("archive", f"sealed and checksum-valid: {archive.validate()['valid']}"),
    ]
    width = max(len(name) for name, _ in rows)
    body = ["RAGTRUTH ONE-INSTANCE PAIRED PROBE (OFFLINE)"]
    body.extend(f"  {name:<{width}} : {value}" for name, value in rows)
    border = "=" * max(len(line) for line in body)
    return "\n".join((border, *body, border, f"archive: {archive.path}"))


def run_ragtruth_one_instance_shared_kggen_mock_probe(
    *,
    source_info_path: str | Path,
    response_path: str | Path,
    response_id: str,
    output_root: str | Path,
    run_id: str | None = None,
    hallugraph_config: str | Path = "config.yaml",
    cache_root: str | Path | None = None,
) -> tuple[RunArchive, RunArchive, dict[str, Any]]:
    """Run one response twice: materialize shared KGGen graph, then cache-only replay.

    Both passes use the real HalluGraph and GraphEval adapters, but FakeKGGen/FakeNLI.
    It is a deterministic integration probe, not a live detector-quality experiment.
    ``cache_root`` is the only storage contract: locally it may be a temporary path;
    on DataSphere it should point at a Project-storage directory supplied by the job.
    """
    record = materialize_one_response_no_gold(source_info_path, response_path, response_id=response_id)
    base_id = _validated_run_id(run_id) if run_id is not None else default_run_id(record["response_id"])
    root = Path(output_root)
    selected_cache_root = Path(cache_root) if cache_root is not None else root / f"{base_id}-shared-kg-cache"

    def execute(*, suffix: str, cache_mode: str) -> tuple[RunArchive, Any, dict[str, Any]]:
        archive = RunArchive.create(
            root,
            run_id=f"{base_id}-{suffix}",
            manifest={
                "run_purpose": "ragtruth_one_instance_shared_kggen_mock_probe",
                "probe_version": "ragtruth-one-instance-shared-kggen-cache-v1",
                "comparison_track": "controlled_shared_kggen_response_v1",
                "selected_response_id": record["response_id"],
                "selected_source_id": record["source_id"],
                "cache_mode": cache_mode,
                "cache_root": str(selected_cache_root),
                "network_access": False,
                "backend_mode": "fake_shared_kggen_v1",
                "gold_passed_to_detectors": False,
            },
        )
        instances_path = archive.path / "instances.no_gold.jsonl"
        atomic_write_jsonl(instances_path, [record])
        detectors, provider = build_controlled_shared_kggen_fake(
            hallugraph_config, cache_mode=cache_mode, cache_root=selected_cache_root
        )
        summary = run_paired(
            archive, instances_path=instances_path, detectors=detectors,
            shared_graph_provider=provider,
        )
        seal_run(archive, instances_path)
        validation = archive.validate()
        if not validation["valid"]:
            raise RuntimeError(f"invalid shared-KGGen probe archive: {validation['errors']}")
        return archive, provider, summary

    cold, cold_provider, cold_summary = execute(suffix="materialize", cache_mode="read_write")
    replay, replay_provider, replay_summary = execute(suffix="cache-replay", cache_mode="cache_only")
    cold_predictions = {row["method"]: row for row in cold.read_jsonl("predictions/raw_predictions.jsonl")}
    replay_predictions = {row["method"]: row for row in replay.read_jsonl("predictions/raw_predictions.jsonl")}
    shared_hashes = {
        row["shared_graph_sha256"]
        for row in (*cold_predictions.values(), *replay_predictions.values())
    }
    report = {
        "probe_version": "ragtruth-one-instance-shared-kggen-cache-v1",
        "response_id": record["response_id"],
        "source_id": record["source_id"],
        "cache_root": str(selected_cache_root),
        "materialize_archive": str(cold.path),
        "cache_replay_archive": str(replay.path),
        "materialize_summary": cold_summary,
        "cache_replay_summary": replay_summary,
        "materialize_kggen_api_calls": cold_provider.extractor.usage.summary()["api_calls"],
        "cache_replay_kggen_api_calls": replay_provider.extractor.usage.summary()["api_calls"],
        "shared_graph_sha256": next(iter(shared_hashes)) if len(shared_hashes) == 1 else None,
        "shared_graph_consistent_across_passes": len(shared_hashes) == 1,
        "gold_access_state": "hidden",
    }
    cold.write_json("reports/shared_kggen_two_pass_report.json", report)
    replay.write_json("reports/shared_kggen_two_pass_report.json", report)
    if report["cache_replay_kggen_api_calls"] != 0 or not report["shared_graph_consistent_across_passes"]:
        raise RuntimeError("shared KGGen cache replay invariant failed")
    return cold, replay, report


def render_shared_kggen_mock_probe_summary(report: Mapping[str, Any]) -> str:
    """Human-readable, no-gold handoff for the two-pass offline probe."""
    rows = [
        ("response_id", str(report["response_id"])),
        ("materialize KGGen calls", str(report["materialize_kggen_api_calls"])),
        ("cache replay KGGen calls", str(report["cache_replay_kggen_api_calls"])),
        ("same shared graph", str(report["shared_graph_consistent_across_passes"])),
        ("cache root", str(report["cache_root"])),
    ]
    width = max(len(name) for name, _ in rows)
    lines = ["RAGTRUTH ONE-INSTANCE SHARED-KGGEN CACHE PROBE (OFFLINE)"]
    lines.extend(f"  {name:<{width}} : {value}" for name, value in rows)
    return "\n".join(lines)
