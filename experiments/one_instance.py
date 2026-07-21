"""A deliberately bounded, offline RAGTruth paired-detector probe.

The probe exercises the actual HalluGraph adapter and GraphEval facade, while their
model-facing dependencies are replaced by the deterministic fake backends.  It is an
integration check for one explicitly named RAGTruth response, never a scientific run.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .artifacts import RunArchive, atomic_write_jsonl
from .datasets.ragtruth import materialize_one_response_no_gold
from .detectors import build_grapheval_fake, build_hallugraph_fake
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
