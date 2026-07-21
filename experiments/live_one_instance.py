"""Auditable, bounded live RAGTruth probe for a DataSphere Job.

This module never knows a credential value beyond reading the Project-injected
environment variable needed by the existing detector clients.  It is deliberately
limited to one explicit response and has no sampling, evaluation, or tuning path.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

import yaml

from gateway.core import API_PATH, GATEWAY_PROTOCOL, canonical_manifest_sha256, vertex_model_from_logical

from .artifacts import RunArchive, atomic_write_json, atomic_write_jsonl, read_jsonl
from .datasets.ragtruth import materialize_one_response_no_gold
from .detectors import build_real_detectors
from .one_instance import _validated_run_id, default_run_id
from .runner import run_paired, seal_run

LIVE_PROBE_VERSION = "ragtruth-one-instance-live-probe-v1"
_ENV_SECRET = "HALLU_GATEWAY_API_KEY"
_ENV_GATEWAY = "HALLU_GATEWAY_URL"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gateway_origin_from_environment(environ: Mapping[str, str]) -> tuple[str, str]:
    """Read required Job environment without ever returning it for logging."""
    secret = str(environ.get(_ENV_SECRET, "")).strip()
    if not secret:
        raise RuntimeError(f"{_ENV_SECRET} is missing; configure the DataSphere Project secret")
    origin = str(environ.get(_ENV_GATEWAY, "")).strip().rstrip("/")
    parsed = urlparse(origin)
    if parsed.scheme != "https" or not parsed.netloc or parsed.path or parsed.query or parsed.fragment:
        raise RuntimeError(f"{_ENV_GATEWAY} must be an HTTPS Cloud Run origin without a path")
    return origin, secret


def validate_gateway_manifest(manifest: Mapping[str, Any], *, logical_model: str) -> str:
    """Validate cache-critical gateway identity and return its canonical SHA-256."""
    expected = {
        "protocol": GATEWAY_PROTOCOL,
        "api_path": API_PATH,
        "logical_model": logical_model,
        "vertex_model": vertex_model_from_logical(logical_model),
        "vertex_location": "europe-west4",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"gateway manifest {key!r} mismatch")
    for key in ("gateway_release", "cloud_run_revision"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"gateway manifest is missing {key!r}")
    return canonical_manifest_sha256(manifest)


def _load_yaml_mapping(path: str | Path, *, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a YAML mapping")
    return value


def _redact(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]")
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact(item, secret) for key, item in value.items()}
    return value


class _AuditTrail:
    """Small append-by-rewrite JSONL audit log; all details are secret-redacted."""

    def __init__(self, archive: RunArchive, *, secret: str):
        self.archive = archive
        self.secret = secret
        self.rows: list[dict[str, Any]] = []

    def event(self, stage: str, status: str, **details: Any) -> None:
        row = _redact(
            {
                "event_id": f"{self.archive.run_id}:{len(self.rows) + 1:03d}",
                "at_utc": _utc_now(),
                "stage": stage,
                "status": status,
                "details": details,
                "gold_access_state": "hidden",
            },
            self.secret,
        )
        self.rows.append(row)
        self.archive.write_jsonl("audit/live_one_instance_events.jsonl", self.rows)


def _sanitize_prediction_artifacts(archive: RunArchive, secret: str) -> None:
    """Defence in depth: redact an unlikely credential echo in an error payload."""
    for relative in ("predictions/raw_predictions.jsonl", "predictions/paired_predictions.jsonl", "stages/stage_calls.jsonl"):
        target = archive.path / relative
        if target.exists():
            archive.write_jsonl(relative, [_redact(row, secret) for row in read_jsonl(target)])


def _detector_statuses(archive: RunArchive) -> dict[str, str]:
    return {str(row["method"]): str(row["status"]) for row in archive.read_jsonl("predictions/raw_predictions.jsonl")}


def run_ragtruth_one_instance_live_probe(
    *,
    source_info_path: str | Path,
    response_path: str | Path,
    response_id: str,
    output_root: str | Path,
    hallugraph_config: str | Path,
    grapheval_config: str | Path,
    gateway_manifest_path: str | Path,
    run_id: str | None = None,
    environ: Mapping[str, str] | None = None,
    detector_factory: Callable[..., Mapping[str, Any]] = build_real_detectors,
) -> tuple[RunArchive, dict[str, Any]]:
    """Execute one real paired probe after the Job wrapper completed network preflight.

    ``detector_factory`` is injectable solely for offline unit tests. Production calls
    use ``build_real_detectors`` and therefore the real Cloud Run extractor, KGGen,
    local embedding model, and local pinned HHEM NLI.
    """
    environment = os.environ if environ is None else environ
    origin, secret = _gateway_origin_from_environment(environment)
    hallu_payload = _load_yaml_mapping(hallugraph_config, label="HalluGraph runtime config")
    graph_payload = _load_yaml_mapping(grapheval_config, label="GraphEval runtime config")
    logical_model = str((hallu_payload.get("llm") or {}).get("model", ""))
    if not logical_model:
        raise ValueError("HalluGraph runtime config has no llm.model")
    hallu_llm = hallu_payload.get("llm") or {}
    manifest = json.loads(Path(gateway_manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("gateway manifest must be a JSON object")
    manifest_sha256 = validate_gateway_manifest(manifest, logical_model=logical_model)

    extractor_cfg = graph_payload.get("extractor") or {}
    nli_cfg = graph_payload.get("nli") or {}
    if extractor_cfg.get("backend") != "gateway" or nli_cfg.get("backend") != "hhem":
        raise ValueError("live probe requires GraphEval extractor.backend='gateway' and nli.backend='hhem'")
    model_path = Path(str(nli_cfg.get("model", "")))
    if not (model_path / "config.json").is_file():
        raise RuntimeError(f"pinned HHEM snapshot is unavailable: {model_path}")

    record = materialize_one_response_no_gold(source_info_path, response_path, response_id=response_id)
    selected_run_id = _validated_run_id(run_id) if run_id is not None else default_run_id(record["response_id"])
    archive = RunArchive.create(
        output_root,
        run_id=selected_run_id,
        manifest={
            "run_purpose": "ragtruth_one_instance_live_probe",
            "probe_version": LIVE_PROBE_VERSION,
            "comparison_track": "kggen_untyped_adaptation",
            "data_source": "local_ragtruth_files",
            "selected_response_id": record["response_id"],
            "selected_source_id": record["source_id"],
            "selected_split": record["split"],
            "selected_task": record["metadata"]["task"],
            "network_access": True,
            "backend_mode": "live_datasphere_v1",
            "gateway_origin": origin,
            "gateway_manifest_sha256": manifest_sha256,
            "gold_used_for_selection": False,
            "gold_passed_to_detectors": False,
        },
    )
    audit = _AuditTrail(archive, secret=secret)
    try:
        audit.event("environment", "ok", gateway_secret_present=True, gateway_origin=origin)
        audit.event(
            "gateway_manifest", "ok", sha256=manifest_sha256,
            logical_model=logical_model, gateway_release=manifest["gateway_release"],
            cloud_run_revision=manifest["cloud_run_revision"],
        )
        audit.event(
            "hallugraph_token_policy", "ok",
            max_tokens=hallu_llm.get("max_tokens"),
            length_retry_attempts=hallu_llm.get("length_retry_attempts", 0),
            length_retry_max_tokens=hallu_llm.get("length_retry_max_tokens"),
        )
        audit.event(
            "input_materialization", "ok", response_id=record["response_id"], source_id=record["source_id"],
            source_record_sha256=record["source_record_sha256"], response_record_sha256=record["response_record_sha256"],
            context_sha256=record["context_hash"], query_sha256=record["query_hash"], response_sha256=record["response_hash"],
        )
        archive.write_jsonl("instances.no_gold.jsonl", [record])
        archive.write_json(
            "audit/live_probe_inputs.json",
            {
                "hallugraph_runtime_config_sha256": _sha256_file(hallugraph_config),
                "grapheval_runtime_config_sha256": _sha256_file(grapheval_config),
                "gateway_manifest_sha256": manifest_sha256,
                "gateway_origin": origin,
                "hhem_model_path": str(model_path),
                "hhem_revision": nli_cfg.get("revision"),
                "hallugraph_max_tokens": hallu_llm.get("max_tokens"),
                "hallugraph_length_retry_attempts": hallu_llm.get("length_retry_attempts", 0),
                "hallugraph_length_retry_max_tokens": hallu_llm.get("length_retry_max_tokens"),
                "gold_access_state": "hidden",
            },
        )
        detectors = detector_factory(
            hallugraph_config=hallugraph_config,
            grapheval_config=graph_payload,
            gateway_manifest_sha256=manifest_sha256,
            hallugraph_usage_path=archive.path / "audit/hallugraph_extraction_attempts.jsonl",
        )
        audit.event("detector_construction", "ok", methods=sorted(detectors))
        summary = run_paired(archive, instances_path=archive.path / "instances.no_gold.jsonl", detectors=detectors)
        _sanitize_prediction_artifacts(archive, secret)
        statuses = _detector_statuses(archive)
        audit.event("paired_inference", "ok" if all(value == "ok" for value in statuses.values()) else "failed", statuses=statuses, summary=summary)
        seal = seal_run(archive, archive.path / "instances.no_gold.jsonl")
        validation = archive.validate()
        audit.event("archive_seal", "ok" if validation["valid"] else "failed", validation=validation, prediction_count=seal["prediction_count"])
        report = {
            "probe_version": LIVE_PROBE_VERSION,
            "archive_path": str(archive.path),
            "gateway_manifest_sha256": manifest_sha256,
            "summary": summary,
            "detector_statuses": statuses,
            "seal": seal,
            "validation": validation,
            "gold_access_state": "hidden",
        }
        archive.write_json("reports/live_one_instance_probe_report.json", report)
        if not validation["valid"] or set(statuses) != {"hallugraph", "grapheval"} or any(value != "ok" for value in statuses.values()):
            archive.update_status("failed", failure_reason="one or more live detectors did not complete")
            raise RuntimeError(f"live one-instance probe failed: statuses={statuses}, archive_valid={validation['valid']}")
        archive.update_status("completed", completed_at_utc=_utc_now())
        return archive, report
    except Exception as exc:
        audit.event("probe", "failed", error=repr(exc))
        archive.update_status("failed", failure_reason=_redact(repr(exc), secret))
        raise


def render_live_probe_summary(archive: RunArchive) -> str:
    """Render a redacted terminal handoff summary for the DataSphere Job log."""
    report = json.loads((archive.path / "reports/live_one_instance_probe_report.json").read_text(encoding="utf-8"))
    rows = [
        ("response_id", archive.run_id),
        ("gateway manifest", report["gateway_manifest_sha256"]),
        ("HalluGraph", report["detector_statuses"].get("hallugraph", "missing")),
        ("GraphEval", report["detector_statuses"].get("grapheval", "missing")),
        ("archive valid", str(report["validation"]["valid"])),
        ("gold", "not passed to detectors"),
    ]
    width = max(len(label) for label, _ in rows)
    lines = ["RAGTRUTH ONE-INSTANCE PAIRED PROBE (LIVE DATASPHERE)"]
    lines.extend(f"  {label:<{width}} : {value}" for label, value in rows)
    border = "=" * max(len(line) for line in lines)
    return "\n".join((border, *lines, border, f"archive: {archive.path}"))
