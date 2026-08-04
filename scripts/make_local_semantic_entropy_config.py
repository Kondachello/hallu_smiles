#!/usr/bin/env python3
"""Bind SemanticEntropy's local runtime to an authenticated Vertex gateway."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.core import canonical_manifest_sha256
from scripts.make_datasphere_vertex_config import (  # noqa: E402
    _normalise_gateway_url,
    _read_json,
    _validate_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--gateway-manifest", required=True)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--nli-model-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--request-min-interval-s", type=float, default=4.0)
    parser.add_argument("--rate-limit-deadline-s", type=float, default=1800.0)
    args = parser.parse_args()
    if args.request_min_interval_s < 0 or args.rate_limit_deadline_s <= 0:
        raise SystemExit("invalid retry/pacing bounds")

    with Path(args.base_config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("llm"), dict):
        raise SystemExit("base config has no llm mapping")
    manifest = _read_json(Path(args.gateway_manifest))
    _validate_manifest(manifest, str(config["llm"].get("model", "")))
    nli_path = Path(args.nli_model_path).resolve()
    if not (nli_path / "config.json").is_file():
        raise SystemExit("NLI model path is not a complete local transformers snapshot")

    gateway_url = _normalise_gateway_url(args.gateway_url)
    manifest_hash = canonical_manifest_sha256(manifest)
    runtime_inputs = {
        "gateway_manifest_sha256": manifest_hash,
        "gateway_url": gateway_url,
        "semantic_entropy_protocol": config.get("semantic_entropy", {}).get("protocol"),
        "nli_model": config.get("semantic_entropy", {}).get("nli_model"),
        "nli_model_path": str(nli_path),
    }
    runtime_hash = hashlib.sha256(
        json.dumps(runtime_inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
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
        "concurrency": 1,
        "max_retries": 0,
        "request_min_interval_s": args.request_min_interval_s,
        "rate_limit_retry_deadline_s": args.rate_limit_deadline_s,
        "retry_deadline_s": args.rate_limit_deadline_s,
    })
    semantic = config.setdefault("semantic_entropy", {})
    semantic.update({
        "cache_dir": str(Path(args.cache_root).resolve() / "semantic_entropy"),
        "cache_read_dirs": [],
        "nli_model_path": str(nli_path),
    })
    config.setdefault("data", {})["dir"] = str(Path(args.data_dir).resolve())
    config["output_dir"] = str(Path(args.output).resolve().parent)
    config.setdefault("vertex_gateway", {}).update({
        "manifest_sha256": manifest_hash,
        "gateway_manifest": manifest,
        "gateway_url": gateway_url,
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(json.dumps({
        "gateway_manifest_sha256": manifest_hash,
        "runtime_fingerprint": llm["runtime_fingerprint"],
        "nli_model_path": str(nli_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
