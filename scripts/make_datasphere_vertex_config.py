#!/usr/bin/env python3
"""Create a CPU DataSphere config bound to an authenticated Vertex gateway manifest.

The base ``config.yaml`` remains the one logical source of ``llm.model``. This
script refuses manifests that name another model and derives every mutable
endpoint/runtime value into the Job-local YAML only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.core import GATEWAY_PROTOCOL, canonical_manifest_sha256, vertex_model_from_logical


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid gateway manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("gateway manifest must be a JSON object")
    return payload


def _normalise_gateway_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("--gateway-url must be an https URL without query or fragment")
    path = parsed.path.rstrip("/")
    if path:
        raise ValueError("--gateway-url must be the Cloud Run origin, without a path")
    return f"https://{parsed.netloc}"


def _validate_manifest(manifest: dict, logical_model: str) -> None:
    required = {
        "protocol": GATEWAY_PROTOCOL,
        "api_path": "/v1",
        "logical_model": logical_model,
        "vertex_model": vertex_model_from_logical(logical_model),
        "vertex_location": "europe-west4",
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"gateway manifest {key!r} mismatch: expected {expected!r}, "
                f"found {manifest.get(key)!r}"
            )
    for key in ("gateway_release", "cloud_run_revision"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"gateway manifest has no non-empty {key!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--gateway-manifest", required=True)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--datasphere-runtime-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument(
        "--cache-root",
        default=None,
        help="Optional durable root containing kg/ and verdicts/ caches; defaults to --work-dir/cache.",
    )
    parser.add_argument(
        "--kg-cache-read-dir",
        action="append",
        default=[],
        help="Immutable historical KG cache directory to search after the primary cache (repeatable).",
    )
    parser.add_argument(
        "--relation-cache-read-dir",
        action="append",
        default=[],
        help="Immutable historical support-verdict cache to search after the primary cache (repeatable).",
    )
    parser.add_argument(
        "--critical-cache-read-root",
        action="append",
        default=[],
        help=(
            "Immutable prior support-critical cache root. Its component namespaces are read-through only "
            "and may be repeated."
        ),
    )
    parser.add_argument(
        "--llm-runtime-fingerprint-override",
        default=None,
        help=(
            "Exact, recorded LLM cache identity for a validated historical checkpoint. "
            "This affects cache lookup only; the current DataSphere runtime is still recorded "
            "under vertex_gateway."
        ),
    )
    # Vertex's 2.5 Flash can spend part of a structured extraction on internal
    # reasoning.  1024 truncated real RAGTruth relation lists in the first
    # probe, while the provider bills actual generated tokens rather than this
    # ceiling.  Use a safe ceiling and one in-flight source for the bounded
    # on-demand capacity probe.
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=1)
    # Public/on-demand Vertex capacity can briefly return HTTP 429 even for a
    # single in-flight request.  ``0`` is deliberate: retry transient
    # provider failures until the enclosing DataSphere Job wall-time limit.
    # Positive values retain a finite local attempt budget for short probes.
    parser.add_argument("--max-retries", type=int, default=7)
    parser.add_argument("--retry-backoff-base-s", type=float, default=5.0)
    parser.add_argument("--retry-backoff-max-s", type=float, default=60.0)
    parser.add_argument(
        "--rate-limit-cooldown-max-s",
        type=float,
        default=900.0,
        help="maximum cooldown after an HTTP 429; must be at least --retry-backoff-max-s",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="stratified folds used only for train-only alpha/tau selection",
    )
    args = parser.parse_args()
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be non-negative (0 retries until Job timeout)")
    if args.retry_backoff_base_s <= 0:
        raise ValueError("--retry-backoff-base-s must be positive")
    if args.retry_backoff_max_s <= 0:
        raise ValueError("--retry-backoff-max-s must be positive")
    if args.retry_backoff_max_s < args.retry_backoff_base_s:
        raise ValueError("--retry-backoff-max-s must be at least --retry-backoff-base-s")
    if args.rate_limit_cooldown_max_s < args.retry_backoff_max_s:
        raise ValueError(
            "--rate-limit-cooldown-max-s must be at least --retry-backoff-max-s"
        )
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2")
    if args.llm_runtime_fingerprint_override is not None:
        override = str(args.llm_runtime_fingerprint_override).strip()
        if not override.startswith("vertex-gateway:") or len(override) <= len("vertex-gateway:"):
            raise ValueError(
                "--llm-runtime-fingerprint-override must be a non-empty vertex-gateway identity"
            )

    with open(args.base_config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("llm"), dict):
        raise ValueError("base config has no llm mapping")
    try:
        rate_limit_retry_deadline_s = float(
            config["llm"].get("rate_limit_retry_deadline_s", 1800)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("llm.rate_limit_retry_deadline_s must be numeric") from exc
    if rate_limit_retry_deadline_s <= 0:
        raise ValueError("llm.rate_limit_retry_deadline_s must be positive")
    config["llm"]["rate_limit_retry_deadline_s"] = rate_limit_retry_deadline_s
    try:
        retry_deadline_s = float(config["llm"].get("retry_deadline_s", 1800))
    except (TypeError, ValueError) as exc:
        raise ValueError("llm.retry_deadline_s must be numeric") from exc
    if retry_deadline_s <= 0:
        raise ValueError("llm.retry_deadline_s must be positive")
    config["llm"]["retry_deadline_s"] = retry_deadline_s
    try:
        request_min_interval_s = float(config["llm"].get("request_min_interval_s", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("llm.request_min_interval_s must be numeric") from exc
    if request_min_interval_s < 0:
        raise ValueError("llm.request_min_interval_s must be non-negative")
    config["llm"]["request_min_interval_s"] = request_min_interval_s
    logical_model = str(config["llm"].get("model", ""))
    manifest = _read_json(Path(args.gateway_manifest))
    _validate_manifest(manifest, logical_model)
    runtime_manifest = _read_json(Path(args.datasphere_runtime_manifest))
    runtime_fingerprint = runtime_manifest.get("runtime_fingerprint")
    if not isinstance(runtime_fingerprint, str) or not runtime_fingerprint:
        raise ValueError("DataSphere runtime manifest has no runtime_fingerprint")

    gateway_url = _normalise_gateway_url(args.gateway_url)
    manifest_hash = canonical_manifest_sha256(manifest)
    combined = {
        "datasphere_runtime_fingerprint": runtime_fingerprint,
        "gateway_manifest_sha256": manifest_hash,
        "gateway_url": gateway_url,
    }
    combined_hash = hashlib.sha256(
        json.dumps(combined, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    work_dir = Path(args.work_dir)
    cache_root = Path(args.cache_root) if args.cache_root else work_dir / "cache"
    llm = config["llm"]
    llm["api_base"] = f"{gateway_url}/v1"
    llm["api_key_env"] = "HALLU_GATEWAY_API_KEY"
    llm["model_revision"] = (
        f"{manifest['vertex_model']}:{manifest['gateway_release']}:"
        f"{manifest['cloud_run_revision']}"
    )
    computed_llm_fingerprint = f"vertex-gateway:{combined_hash}"
    llm["runtime_fingerprint"] = (
        str(args.llm_runtime_fingerprint_override).strip()
        if args.llm_runtime_fingerprint_override is not None
        else computed_llm_fingerprint
    )
    llm["max_tokens"] = args.max_tokens
    llm["concurrency"] = args.concurrency
    llm["max_retries"] = args.max_retries
    llm["retry_backoff_base_s"] = args.retry_backoff_base_s
    llm["retry_backoff_max_s"] = args.retry_backoff_max_s
    llm["rate_limit_cooldown_max_s"] = args.rate_limit_cooldown_max_s
    llm["structured_output_transport"] = "response_format"
    llm["structured_output_backend"] = "vertex"
    llm["structured_output_request_backend"] = None
    llm["vllm_guided_json"] = False
    config["extraction"]["serial_chunking"] = False
    config["extraction"]["cluster_context_mode"] = "source_text"
    config["matching"]["embedding_model_path"] = "/opt/hallu/models/all-MiniLM-L6-v2"
    config["matching"]["embedding_device"] = "cpu"
    config["matching"]["local_files_only"] = True
    config["data"]["dir"] = args.data_dir
    config["cache_dir"] = str(cache_root / "kg")
    config["cache_read_dirs"] = [str(Path(path)) for path in args.kg_cache_read_dir]
    relation_verifier = config.setdefault("relation_verifier", {})
    relation_verifier["cache_dir"] = str(cache_root / "verdicts")
    relation_verifier["cache_read_dirs"] = [
        str(Path(path)) for path in args.relation_cache_read_dir
    ]
    critical = config.setdefault("support_critical", {})
    for section_name, namespace in (
        ("claim_extractor", "critical_claims"),
        ("coverage_reviewer", "critical_coverage"),
        ("claim_verifier", "critical_verdicts"),
    ):
        section = critical.setdefault(section_name, {})
        section["cache_dir"] = str(cache_root / namespace)
        section["cache_read_dirs"] = [
            str(Path(root) / namespace) for root in args.critical_cache_read_root
        ]
    config["output_dir"] = str(work_dir / "results")
    config["eval"]["alpha_cv_folds"] = args.cv_folds
    config.setdefault("vertex_gateway", {}).update(
        {
            "manifest_sha256": manifest_hash,
            "gateway_manifest": manifest,
            "gateway_url": gateway_url,
            "datasphere_runtime_fingerprint": runtime_fingerprint,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(json.dumps({
        "gateway_manifest_sha256": manifest_hash,
        "runtime_fingerprint": llm["runtime_fingerprint"],
        "computed_runtime_fingerprint": computed_llm_fingerprint,
        "historical_cache_identity": args.llm_runtime_fingerprint_override is not None,
    }))


if __name__ == "__main__":
    main()
