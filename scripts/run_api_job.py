#!/usr/bin/env python3
"""Job-local orchestration for bounded API probe and full QA pilot runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_api_contract import run_contract_probe
from scripts.import_probe_cache import import_probe_cache


PROBE_TEXT = (
    "Ada Lovelace wrote notes about Charles Babbage's Analytical Engine. "
    "Marie Curie was born in Warsaw."
)
SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
PROVIDER_FIELDS = {
    "outcome",
    "request_id",
    "latency_s",
    "http_status",
    "retry_index",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "error_type",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_id_from_directory(run_dir: Path, mode: str) -> str | None:
    prefix = f"api-{mode}-artifacts-"
    return run_dir.name[len(prefix) :] if run_dir.name.startswith(prefix) else None


def atomic_json(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _redact(value: str, secret: str | None) -> str:
    return value.replace(secret, "[REDACTED]") if secret else value


def _source_commit(expected: str) -> str:
    if not SHA40_RE.fullmatch(expected):
        raise ValueError("EXPECTED_SOURCE_COMMIT must be a full lowercase 40-character Git SHA")
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected:
        raise ValueError(f"checked-out source commit differs from expected commit: {actual}")
    return actual


def _requirements_pins(path: Path | None = None) -> dict[str, str]:
    path = path or ROOT / "requirements.datasphere.api.txt"
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("--"):
            continue
        if "==" not in line:
            raise RuntimeError(f"DataSphere API runtime dependency is not pinned: {line}")
        distribution, version = line.split("==", 1)
        if not distribution or not version or distribution in pins:
            raise RuntimeError(f"invalid or duplicate DataSphere runtime pin: {line}")
        pins[distribution] = version
    if not pins:
        raise RuntimeError(f"DataSphere API requirements contain no pins: {path}")
    return pins


def _runtime_versions(*, enforce: bool) -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for distribution, expected in _requirements_pins().items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            if enforce:
                raise RuntimeError(f"required distribution is absent: {distribution}") from None
            actual = "absent"
        versions[distribution] = actual
        if enforce and actual != expected:
            raise RuntimeError(
                f"runtime version drift for {distribution}: expected {expected}, found {actual}"
            )
    if enforce and sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"DataSphere API jobs require Python 3.11, found {versions['python']}")
    return versions


def _config_for_job(config_path: Path, data_dir: Path, run_dir: Path):
    from src.config import load_config, resolve_api_key

    cfg = load_config(config_path)
    model = str(cfg.llm.model).strip()
    if not model or "PLACEHOLDER" in model:
        raise ValueError("config llm.model must name one live provider model")
    api_base = str(cfg.llm.api_base).strip()
    parsed_base = urlparse(api_base)
    if parsed_base.scheme != "https" or not parsed_base.netloc:
        raise ValueError("config llm.api_base must be an absolute HTTPS endpoint")
    if not str(cfg.llm.api_key_env).strip():
        raise ValueError("config llm.api_key_env must name the project secret")
    if float(cfg.llm.temperature) != 0.0:
        raise ValueError("config llm.temperature must be zero for deterministic extraction")
    if int(cfg.llm.max_tokens) <= 0 or float(cfg.llm.request_timeout_s) <= 0:
        raise ValueError("config token and request timeout bounds must be positive")
    if int(cfg.llm.max_retries) <= 0:
        raise ValueError("config llm.max_retries must be positive")
    if int(cfg.llm.concurrency) != 1:
        raise ValueError("config llm.concurrency must be one for the bounded API runtime")
    if str(cfg.llm.structured_output_transport) != "json_object":
        raise ValueError("config must use the json_object structured-output transport")
    if cfg.llm.extra_body.enable_thinking is not False:
        raise ValueError("config must explicitly disable provider-side thinking")
    if cfg.extraction.cluster is not True:
        raise ValueError("official KGGen LLM clustering must remain enabled")
    secret = resolve_api_key(cfg)
    if not secret:
        raise RuntimeError(
            f"required DataSphere secret {str(cfg.llm.api_key_env)!r} is absent or empty"
        )
    cfg.data._data["dir"] = str(data_dir)  # noqa: SLF001
    cfg.data.dir = str(data_dir)
    kg_cache = run_dir / ".cache" / "kg"
    verdict_cache = run_dir / ".cache" / "verdicts"
    cfg._data["cache_dir"] = str(kg_cache)  # noqa: SLF001
    cfg.cache_dir = str(kg_cache)
    cfg.relation_verifier._data["cache_dir"] = str(verdict_cache)  # noqa: SLF001
    cfg.relation_verifier.cache_dir = str(verdict_cache)
    return cfg, secret


def _validate_data_dir(data_dir: Path, run_dir: Path) -> None:
    data_dir = data_dir.resolve()
    run_dir = run_dir.resolve()
    if data_dir == run_dir or data_dir in run_dir.parents or run_dir in data_dir.parents:
        raise ValueError("run directory and read-only RAGTruth directory must be disjoint")
    for filename in ("source_info.jsonl", "response.jsonl"):
        path = data_dir / filename
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"RAGTruth asset is absent or not a regular file: {path}")


def _graph_digest(graph: Any) -> str:
    payload = json.dumps(graph.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _graph_summary(graph: Any) -> dict[str, Any]:
    return {
        "entities": len(graph.entities),
        "relations": len(graph.relations),
        "sha256": _graph_digest(graph),
    }


def _identity(instance: Any) -> dict[str, Any]:
    return {
        "source_id": str(instance.source_id),
        "response_id": str(instance.response_id),
        "split": instance.split,
        "y": int(instance.y),
    }


def _has_endpoints(relations: Any, left: str, right: str) -> bool:
    left, right = left.casefold(), right.casefold()
    for subject, _predicate, obj in relations:
        subject, obj = str(subject).casefold(), str(obj).casefold()
        if (left in subject and right in obj) or (right in subject and left in obj):
            return True
    return False


def _run_synthetic_probe(extractor: Any) -> dict[str, Any]:
    started = time.perf_counter()
    graph = extractor.extract(PROBE_TEXT, kind="synthetic_probe")
    if len(graph.entities) < 4 or len(graph.relations) < 2:
        raise RuntimeError("synthetic KGGen probe returned an incomplete graph")
    ada = _has_endpoints(graph.relations, "Ada Lovelace", "Analytical Engine") or _has_endpoints(
        graph.relations, "Ada Lovelace", "Charles Babbage"
    )
    marie = _has_endpoints(graph.relations, "Marie Curie", "Warsaw")
    if not ada or not marie:
        raise RuntimeError("synthetic KGGen graph omitted a required semantic anchor")
    return {
        "protocol": "hallu-api-synthetic-kggen-v1",
        "status": "ready",
        "cluster": True,
        "official_kggen_clustering": True,
        "entities": len(graph.entities),
        "relations": len(graph.relations),
        "semantic_anchors": {"ada_lovelace": ada, "marie_curie_warsaw": marie},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _run_verifier_probe(verifier: Any) -> dict[str, Any]:
    started = time.perf_counter()
    verdict = verifier.verify(
        ("Alice", "works at", "Acme"),
        "Alice works at Acme.",
        None,
        matching_params={"probe": "verifier-contract-v1"},
    )
    if verdict.verdict != "entailed":
        raise RuntimeError(f"support verifier probe expected entailed, found {verdict.verdict!r}")
    return {
        "protocol": "hallu-api-verifier-probe-v1",
        "status": "ready",
        "verdict": verdict.verdict,
        "evidence": [span.to_dict() for span in verdict.evidence],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _write_audits(
    instances: list[Any], results: dict[str, Any], out_dir: Path, relation_mode: str
) -> None:
    from src.audit import build_audit_record, write_audit

    diagnostic_alpha = 0.7
    by_id = {instance.response_id: instance for instance in instances}
    for response_id, result in results.items():
        record = build_audit_record(
            by_id[response_id],
            result,
            diagnostic_alpha,
            alpha_support=diagnostic_alpha,
            relation_mode=relation_mode,
        )
        record["probe_diagnostic_alpha_not_tuned"] = True
        write_audit(record, out_dir / "audit")


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _merge_provider_telemetry(paths: list[Path], destination: Path) -> dict[str, int]:
    records: list[dict[str, Any]] = []
    totals = {
        "provider_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for path in paths:
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or set(record) - PROVIDER_FIELDS:
                raise RuntimeError(f"provider telemetry is not allowlist-only: {path}:{number}")
            records.append(record)
            totals["provider_calls"] += 1
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = record.get(field)
                if not isinstance(value, int) or value < 0:
                    raise RuntimeError(f"invalid provider telemetry {field}: {path}:{number}")
                totals[field] += value
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return totals


def _csv_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_comparison(run_dir: Path, manifest_sha256: str) -> dict[str, Any]:
    modes: dict[str, dict[str, Any]] = {}
    identities: dict[str, list[dict[str, Any]]] = {}
    for mode in ("strict", "support"):
        directory = run_dir / mode
        metric_rows = _csv_records(directory / "metrics.csv")
        if len(metric_rows) != 20:
            raise RuntimeError(f"{mode} metrics.csv must contain exactly 20 rows")
        identities[mode] = [
            {
                "source_id": row["source_id"],
                "response_id": row["response_id"],
                "split": row["split"],
                "y": int(row["y"]),
            }
            for row in metric_rows
        ]
        summary_rows = _csv_records(directory / "summary_metrics.csv")
        if len(summary_rows) != 1:
            raise RuntimeError(f"{mode} summary_metrics.csv must contain one row")
        summary = summary_rows[0]
        tuning = json.loads((directory / "tuning.json").read_text(encoding="utf-8"))
        modes[mode] = {
            "relation_mode": mode,
            "alpha": float(tuning["alpha"]),
            "theta": float(tuning["theta"]),
            "tau_e": float(tuning["tau_e"]),
            "tau_r": float(tuning["tau_r"]),
            "test_auc": float(summary["overall_AUC_exclude_unscorable"]),
            "test_f1": float(summary["overall_F1"]),
        }
        if not all(
            math.isfinite(float(modes[mode][field]))
            for field in ("alpha", "theta", "tau_e", "tau_r", "test_auc", "test_f1")
        ):
            raise RuntimeError(f"{mode} comparison contains a non-finite scientific metric")
    if identities["strict"] != identities["support"]:
        raise RuntimeError("strict and support metrics do not describe the same 20 QA records")
    comparison = {
        "protocol": "hallu-strict-support-comparison-v1",
        "status": "ready",
        "manifest_sha256": manifest_sha256,
        "records": 20,
        "strict": modes["strict"],
        "support": modes["support"],
        "delta_support_minus_strict": {
            "test_auc": modes["support"]["test_auc"] - modes["strict"]["test_auc"],
            "test_f1": modes["support"]["test_f1"] - modes["strict"]["test_f1"],
        },
    }
    atomic_json(run_dir / "comparison.json", comparison)
    return comparison


def _write_cache_inventory(run_dir: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for kind in ("kg", "verdicts"):
        directory = run_dir / ".cache" / kind
        for path in sorted(directory.glob("*.json")):
            if path.is_symlink() or not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}\.json", path.name):
                raise RuntimeError(f"unexpected cache entry: {path}")
            data = path.read_bytes()
            value = json.loads(data)
            if not isinstance(value, dict):
                raise RuntimeError(f"cache entry is not a JSON object: {path}")
            if kind == "kg":
                if not isinstance(value.get("entities"), list) or not isinstance(
                    value.get("relations"), list
                ):
                    raise RuntimeError(f"KG cache entry has an invalid graph shape: {path}")
            elif value.get("verdict") not in {"entailed", "contradicted", "unknown"}:
                raise RuntimeError(f"verifier cache entry has an invalid verdict: {path}")
            entries.append({
                "path": str(path.relative_to(run_dir)),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            })
    if not any(entry["path"].startswith(".cache/kg/") for entry in entries):
        raise RuntimeError("probe produced no KG cache entries")
    inventory = {
        "protocol": "hallu-api-cache-inventory-v1",
        "status": "ready",
        "entries": entries,
    }
    atomic_json(run_dir / "cache_inventory.json", inventory)
    return inventory


def _write_extraction_summary(
    path: Path,
    instances: list[Any],
    refs: dict[str, tuple[Any, Any]],
    responses: dict[str, Any],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    completed = [
        instance
        for instance in instances
        if instance.source_id in refs and instance.response_id in responses
    ]
    graph_records = []
    for instance in completed:
        context, query = refs[instance.source_id]
        graph_records.append({
            **_identity(instance),
            "context": _graph_summary(context),
            "query": _graph_summary(query),
            "answer": _graph_summary(responses[instance.response_id]),
        })
    report = {
        "protocol": "hallu-api-extraction-summary-v1",
        "status": "ready" if len(completed) == len(instances) and not failures else "error",
        "expected_records": [_identity(instance) for instance in instances],
        "completed_records": [_identity(instance) for instance in completed],
        "pairs_completed": len(completed),
        "failures": failures,
        "graph_records": graph_records,
    }
    atomic_json(path, report)
    if report["status"] != "ready":
        raise RuntimeError("3-QA extraction did not produce three complete graph pairs")
    return report


def _require_scores(results: dict[str, Any], instances: list[Any], label: str) -> None:
    expected = [instance.response_id for instance in instances]
    if list(results) != expected:
        # score_all preserves instance order; require it so artifacts match the manifest prefix.
        raise RuntimeError(f"{label} scores are incomplete or out of order")


def run_probe(cfg: Any, run_dir: Path, data_dir: Path, usage: Any) -> dict[str, Any]:
    from run import extract_all, get_embedder, persist_scored, score_all
    from src.data import load_instances
    from src.extract import KGExtractor, UsageLogger
    from src.sampling import select_qa_pilot, write_manifest
    from src.verifier import RelationVerifier

    extractor = KGExtractor(cfg, usage=usage)
    try:
        contract = run_contract_probe(extractor, data_dir, repeat=3)
    except Exception as exc:
        atomic_json(run_dir / "contract_probe.json", getattr(exc, "report", {
            "protocol": "hallu-api-json-object-contract-v1",
            "status": "error",
            "passed": 0,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }))
        raise
    atomic_json(run_dir / "contract_probe.json", contract)
    try:
        synthetic = _run_synthetic_probe(extractor)
    except Exception as exc:
        atomic_json(run_dir / "synthetic_probe.json", {
            "protocol": "hallu-api-synthetic-kggen-v1",
            "status": "error",
            "cluster": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise
    atomic_json(run_dir / "synthetic_probe.json", synthetic)
    verifier = RelationVerifier(cfg, usage=usage)
    try:
        verifier_probe = _run_verifier_probe(verifier)
    except Exception as exc:
        atomic_json(run_dir / "verifier_probe.json", {
            "protocol": "hallu-api-verifier-probe-v1",
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise
    atomic_json(run_dir / "verifier_probe.json", verifier_probe)

    all_instances = load_instances(
        data_dir, exclude_implicit_true=bool(cfg.data.exclude_implicit_true)
    )
    selected = select_qa_pilot(all_instances, seed=42, train_sources=16, test_sources=4)
    manifest_path = write_manifest(
        run_dir / "qa_pilot_manifest.json",
        selected,
        seed=42,
        train_sources=16,
        test_sources=4,
    )
    instances = selected[:3]
    refs, responses, failures = extract_all(cfg, instances, extractor, run_dir)
    extraction = _write_extraction_summary(
        run_dir / "extraction_summary.json", instances, refs, responses, failures
    )
    embedder = get_embedder(cfg, fake=False)

    strict = score_all(
        cfg, instances, refs, responses, embedder, relation_mode="strict", verifier=None
    )
    _require_scores(strict, instances, "strict")
    strict_dir = run_dir / "strict"
    strict_dir.mkdir(parents=True, exist_ok=True)
    persist_scored(strict_dir / "scored.jsonl", instances, strict, "strict")
    _write_audits(instances, strict, strict_dir, "strict")

    support = score_all(
        cfg, instances, refs, responses, embedder, relation_mode="support", verifier=verifier
    )
    _require_scores(support, instances, "support")
    support_dir = run_dir / "support"
    support_dir.mkdir(parents=True, exist_ok=True)
    persist_scored(support_dir / "scored.jsonl", instances, support, "support")
    _write_audits(instances, support, support_dir, "support")

    replay_usage = UsageLogger(
        run_dir / "cache_replay_usage.jsonl",
        provider_calls_path=run_dir / "cache_replay_provider_calls.jsonl",
    )
    replay_extractor = KGExtractor(cfg, usage=replay_usage, cache_only=True)
    replay_verifier = RelationVerifier(cfg, usage=replay_usage, cache_only=True)
    replay_dir = run_dir / "cache_replay_work"
    replay_refs, replay_responses, replay_failures = extract_all(
        cfg, instances, replay_extractor, replay_dir
    )
    # ``extract_all`` always materializes this file.  The top-level live
    # failure log is the canonical gate artifact; replay status is captured in
    # cache_replay.json so the tar contains no ambiguous duplicate suffix.
    (replay_dir / "failed_extractions.jsonl").unlink(missing_ok=True)
    replay_strict = score_all(
        cfg,
        instances,
        replay_refs,
        replay_responses,
        embedder,
        relation_mode="strict",
        verifier=None,
    )
    replay_support = score_all(
        cfg,
        instances,
        replay_refs,
        replay_responses,
        embedder,
        relation_mode="support",
        verifier=replay_verifier,
    )
    _require_scores(replay_strict, instances, "cache-only strict")
    _require_scores(replay_support, instances, "cache-only support")
    replay_summary = replay_usage.summary()
    replay = {
        "protocol": "hallu-api-cache-replay-v1",
        "status": "ready",
        "qa_completed": len(replay_support),
        "failures": len(replay_failures),
        **replay_summary,
    }
    if replay_failures or replay_summary["provider_calls"] != 0:
        raise RuntimeError("cache-only replay was incomplete or reached the API provider")
    atomic_json(run_dir / "cache_replay.json", replay)
    _write_cache_inventory(run_dir)
    provider_totals = _merge_provider_telemetry(
        [run_dir / "provider_calls.jsonl"], run_dir / "provider_calls.jsonl"
    )
    usage_summary = {**usage.summary(), **provider_totals}
    atomic_json(run_dir / "usage_summary.json", usage_summary)
    return {
        "kind": "api-probe-c1",
        "contract_passed": int(contract["passed"]),
        "qa_completed": int(extraction["pairs_completed"]),
        "failed_extractions": len(failures),
        "cache_replay_provider_calls": int(replay_summary["provider_calls"]),
        "manifest_sha256": _manifest_sha256(Path(manifest_path)),
        **provider_totals,
    }


def _run_pipeline_command(command: list[str], label: str) -> None:
    print(f"[pilot] starting {label}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"[pilot] completed {label}", flush=True)


def _jsonl_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def run_pilot(
    cfg: Any,
    config_path: Path,
    run_dir: Path,
    data_dir: Path,
    probe_artifact: Path,
    source_commit: str,
    secret: str,
) -> dict[str, Any]:
    import_report = import_probe_cache(
        probe_artifact,
        run_dir,
        expected_commit=source_commit,
        expected_model=str(cfg.llm.model),
        expected_api_base=str(cfg.llm.api_base),
        secret=secret,
    )
    manifest = run_dir / "qa_pilot_manifest.json"
    common = [
        sys.executable,
        str(ROOT / "run.py"),
        "--config",
        str(config_path),
        "--stage",
        "all",
        "--data-dir",
        str(data_dir),
        "--cache-dir",
        str(run_dir / ".cache" / "kg"),
        "--qa-pilot-manifest",
        str(manifest),
    ]
    for relation_mode in ("strict", "support"):
        output = run_dir / relation_mode
        command = common + [
            "--relation-mode",
            relation_mode,
            "--output-dir",
            str(output),
        ]
        if relation_mode == "support":
            command.append("--kg-cache-only")
        _run_pipeline_command(command, relation_mode)
        if _jsonl_count(output / "scored.jsonl") != 20:
            raise RuntimeError(f"full {relation_mode} pilot did not score exactly 20 QA rows")
        if (output / "failed_extractions.jsonl").read_text(encoding="utf-8").strip():
            raise RuntimeError(f"full {relation_mode} pilot contains failed extractions")
        if len(list((output / "audit").glob("*.json"))) != 20:
            raise RuntimeError(f"full {relation_mode} pilot did not write 20 audits")
        for required in ("metrics.csv", "report.md", "tuning.json"):
            if not (output / required).is_file():
                raise RuntimeError(f"full {relation_mode} pilot is missing {required}")
    usage = _merge_provider_telemetry(
        [
            run_dir / "strict" / "provider_calls.jsonl",
            run_dir / "support" / "provider_calls.jsonl",
        ],
        run_dir / "provider_calls.jsonl",
    )
    atomic_json(run_dir / "usage_summary.json", usage)
    comparison = _write_comparison(run_dir, import_report["manifest_sha256"])
    return {
        "kind": "api-pilot-c1",
        "qa_completed": 20,
        "failed_extractions": 0,
        "manifest_sha256": import_report["manifest_sha256"],
        "comparison_status": comparison["status"],
        **usage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("probe", "pilot"))
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--probe-artifact")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    config_path = Path(args.config).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_commit = os.environ.get("EXPECTED_SOURCE_COMMIT", "")
    secret: str | None = None
    metadata: dict[str, Any] = {
        "protocol": "hallu-api-job-v1",
        "mode": args.mode,
        "state": "started",
        "status": "running",
        "started_at_utc": utc_now(),
        "source_commit": source_commit,
    }
    atomic_json(run_dir / "run_metadata.json", metadata)
    try:
        source_commit = _source_commit(source_commit)
        _validate_data_dir(data_dir, run_dir)
        versions = _runtime_versions(enforce=True)
        cfg, secret = _config_for_job(config_path, data_dir, run_dir)
        from src.api_runtime import llm_runtime_fingerprint

        provider_fingerprint = llm_runtime_fingerprint(cfg)
        # Record every dependency pinned by the DataSphere runtime file, not
        # merely the smaller subset that changes structured-output semantics.
        provider_fingerprint["runtime_versions"] = versions
        metadata.update({
            "source_commit": source_commit,
            "model": str(cfg.llm.model),
            "api_base": str(cfg.llm.api_base),
            "api_key_env": str(cfg.llm.api_key_env),
            "runtime_versions": versions,
            "llm_runtime_fingerprint": provider_fingerprint,
            "data_dir": str(data_dir),
            "run_dir": str(run_dir),
        })
        atomic_json(run_dir / "run_metadata.json", metadata)
        if args.mode == "probe":
            if args.probe_artifact:
                raise ValueError("probe mode must not receive --probe-artifact")
            from src.extract import UsageLogger

            usage = UsageLogger(
                run_dir / "usage.jsonl",
                provider_calls_path=run_dir / "provider_calls.jsonl",
            )
            result = run_probe(cfg, run_dir, data_dir, usage)
        else:
            if not args.probe_artifact:
                raise ValueError("pilot mode requires --probe-artifact")
            result = run_pilot(
                cfg,
                config_path,
                run_dir,
                data_dir,
                Path(args.probe_artifact),
                source_commit,
                secret,
            )
        metadata.update({
            "state": "completed",
            "status": "success",
            "kind": result["kind"],
            "completed_at_utc": utc_now(),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "result": result,
        })
        atomic_json(run_dir / "run_metadata.json", metadata)
        gate = {
            "protocol": "hallu-api-probe-v1" if args.mode == "probe" else "hallu-api-pilot-v1",
            "kind": result["kind"],
            "status": "success",
            "source_commit": source_commit,
            "model": str(cfg.llm.model),
            "api_base": str(cfg.llm.api_base),
            **{key: value for key, value in result.items() if key != "kind"},
        }
        run_id = _run_id_from_directory(run_dir, args.mode)
        if run_id is not None:
            metadata["run_id"] = run_id
            gate["run_id"] = run_id
            atomic_json(run_dir / "run_metadata.json", metadata)
        atomic_json(run_dir / "gate_metadata.json", gate)
        print(json.dumps({"status": "success", "kind": result["kind"]}, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - preserve every failure in the archive
        message = _redact(str(exc), secret)
        metadata.update({
            "state": "error",
            "status": "error",
            "completed_at_utc": utc_now(),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error_type": type(exc).__name__,
            "error": message,
        })
        atomic_json(run_dir / "run_metadata.json", metadata)
        atomic_json(run_dir / "gate_metadata.json", {
            "protocol": "hallu-api-probe-v1" if args.mode == "probe" else "hallu-api-pilot-v1",
            "kind": "api-probe-c1" if args.mode == "probe" else "api-pilot-c1",
            "status": "error",
            "source_commit": source_commit,
            "error_type": type(exc).__name__,
            "error": message,
        })
        print(f"[api-job:error] {type(exc).__name__}: {message}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
