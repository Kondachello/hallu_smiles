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
    # Vertex's 2.5 Flash can spend part of a structured extraction on internal
    # reasoning.  1024 truncated real RAGTruth relation lists in the first
    # probe, while the provider bills actual generated tokens rather than this
    # ceiling.  Use a safe ceiling and one in-flight source for the bounded
    # on-demand capacity probe.
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")

    with open(args.base_config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("llm"), dict):
        raise ValueError("base config has no llm mapping")
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
    llm = config["llm"]
    llm["api_base"] = f"{gateway_url}/v1"
    llm["api_key_env"] = "HALLU_GATEWAY_API_KEY"
    llm["model_revision"] = (
        f"{manifest['vertex_model']}:{manifest['gateway_release']}:"
        f"{manifest['cloud_run_revision']}"
    )
    llm["runtime_fingerprint"] = f"vertex-gateway:{combined_hash}"
    llm["max_tokens"] = args.max_tokens
    llm["concurrency"] = args.concurrency
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
    config["cache_dir"] = str(work_dir / "cache" / "kg")
    config.setdefault("relation_verifier", {})["cache_dir"] = str(work_dir / "cache" / "verdicts")
    config["output_dir"] = str(work_dir / "results")
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
    print(json.dumps({"gateway_manifest_sha256": manifest_hash, "runtime_fingerprint": llm["runtime_fingerprint"]}))


if __name__ == "__main__":
    main()
