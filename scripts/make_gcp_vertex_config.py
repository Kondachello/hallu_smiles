#!/usr/bin/env python3
"""Create a GCP Compute Engine config bound to a pinned gateway manifest.

The output has no bearer token, no cache keys and no historical cache
read-through roots.  Historical C/Q graphs enter only through the separate
frozen-reference artifact accepted by ``run.py``.
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

from gateway.core import (  # noqa: E402
    API_PATH,
    GATEWAY_PROTOCOL,
    canonical_manifest_sha256,
    vertex_model_from_logical,
)


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
    if parsed.path.rstrip("/"):
        raise ValueError("--gateway-url must be the Cloud Run origin, without a path")
    return f"https://{parsed.netloc}"


def _validate_manifest(manifest: dict, logical_model: str) -> None:
    required = {
        "protocol": GATEWAY_PROTOCOL,
        "api_path": API_PATH,
        "logical_model": logical_model,
        "vertex_model": vertex_model_from_logical(logical_model),
        "vertex_location": "europe-west4",
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"gateway manifest {key!r} mismatch: expected {expected!r}, found {manifest.get(key)!r}"
            )
    for key in ("gateway_release", "cloud_run_revision"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"gateway manifest has no non-empty {key!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--gateway-manifest", required=True)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--gcp-runtime-manifest", required=True)
    parser.add_argument("--embedding-model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--identity-output", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--extraction-max-tokens-ceiling", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if args.concurrency != 1:
        raise SystemExit("controlled GCP evaluation requires --concurrency 1")
    if args.max_tokens <= 0 or args.extraction_max_tokens_ceiling < args.max_tokens:
        raise SystemExit("invalid GCP token limits")

    with Path(args.base_config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("llm"), dict):
        raise SystemExit("base config has no llm mapping")
    manifest = _read_json(Path(args.gateway_manifest))
    _validate_manifest(manifest, str(config["llm"].get("model", "")))
    runtime = _read_json(Path(args.gcp_runtime_manifest))
    runtime_fingerprint = runtime.get("runtime_fingerprint")
    if (
        runtime.get("protocol") != "hallu-gcp-compute-cpu-runtime-v1"
        or not isinstance(runtime_fingerprint, str)
        or not runtime_fingerprint.startswith("gcp-compute-cpu:")
    ):
        raise SystemExit("GCP runtime manifest has no valid controlled CPU runtime identity")
    embedding = Path(args.embedding_model_path).resolve()
    if not (embedding / "config.json").is_file():
        raise SystemExit("embedding model path is not an offline S-BERT snapshot")

    gateway_url = _normalise_gateway_url(args.gateway_url)
    manifest_hash = canonical_manifest_sha256(manifest)
    combined = {
        "gcp_runtime_fingerprint": runtime_fingerprint,
        "gateway_manifest_sha256": manifest_hash,
        "gateway_url": gateway_url,
    }
    runtime_hash = hashlib.sha256(
        json.dumps(combined, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    llm = config["llm"]
    llm.update({
        "api_base": f"{gateway_url}/v1",
        "api_key_env": "HALLU_GATEWAY_API_KEY",
        "model_revision": (
            f"{manifest['vertex_model']}:{manifest['gateway_release']}:"
            f"{manifest['cloud_run_revision']}"
        ),
        "runtime_fingerprint": f"vertex-gateway:{runtime_hash}",
        "max_tokens": args.max_tokens,
        "concurrency": 1,
        "structured_output_transport": "response_format",
        "structured_output_backend": "vertex",
        "structured_output_request_backend": None,
    })
    extraction = config.setdefault("extraction", {})
    extraction.update({
        "serial_chunking": True,
        "explicit_clustering": True,
        "cluster_context_mode": "source_text",
        "max_tokens_ceiling": args.extraction_max_tokens_ceiling,
    })
    config.setdefault("data", {})["dir"] = str(Path(args.data_dir))
    config.setdefault("matching", {}).update({
        "embedding_model_path": str(embedding),
        "embedding_device": "cpu",
        "local_files_only": True,
    })
    cache_root = Path(args.cache_root)
    config["cache_dir"] = str(cache_root / "kg")
    config["cache_read_dirs"] = []
    config.setdefault("relation_verifier", {})["cache_dir"] = str(cache_root / "verdicts")
    config["relation_verifier"]["cache_read_dirs"] = []
    for name, namespace in (
        ("claim_extractor", "critical_claims"),
        ("coverage_reviewer", "critical_coverage"),
        ("claim_verifier", "critical_verdicts"),
    ):
        section = config.setdefault("support_critical", {}).setdefault(name, {})
        section["cache_dir"] = str(cache_root / namespace)
        section["cache_read_dirs"] = []
    config["output_dir"] = str(Path(args.work_dir) / "unused-default-output")
    config.setdefault("vertex_gateway", {}).update({
        "manifest_sha256": manifest_hash,
        "gateway_manifest": manifest,
        "gateway_url": gateway_url,
        "gcp_runtime_fingerprint": runtime_fingerprint,
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    identity = {
        "protocol": "hallu-gcp-ragtruth-llama31-config-identity-v1",
        "gateway_manifest_sha256": manifest_hash,
        "gateway_manifest": manifest,
        "gateway_url": gateway_url,
        "gcp_runtime_fingerprint": runtime_fingerprint,
        "runtime_fingerprint": llm["runtime_fingerprint"],
        "cache_read_dirs": [],
        "reference_graph_mode": "frozen_historical_artifact",
    }
    identity_path = Path(args.identity_output)
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "gateway_manifest_sha256": manifest_hash,
        "runtime_fingerprint": llm["runtime_fingerprint"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
