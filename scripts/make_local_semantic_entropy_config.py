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

from gateway.core import SELECTED_TOKEN_LOGPROBS_PROTOCOL, canonical_manifest_sha256
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
    parser.add_argument("--cache-read-dir", action="append", default=[])
    parser.add_argument("--legacy-sample-cache-dir", action="append", default=[])
    parser.add_argument("--legacy-run-metadata", action="append", default=[])
    parser.add_argument("--legacy-runtime-config", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--request-min-interval-s", type=float, default=4.0)
    parser.add_argument("--rate-limit-deadline-s", type=float, default=1800.0)
    args = parser.parse_args()
    if args.request_min_interval_s < 0 or args.rate_limit_deadline_s <= 0:
        raise SystemExit("invalid retry/pacing bounds")
    legacy_counts = {
        len(args.legacy_sample_cache_dir),
        len(args.legacy_run_metadata),
        len(args.legacy_runtime_config),
    }
    if len(legacy_counts) != 1:
        raise SystemExit(
            "each legacy sample-cache directory needs matching run metadata and runtime config"
        )

    with Path(args.base_config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("llm"), dict):
        raise SystemExit("base config has no llm mapping")
    manifest = _read_json(Path(args.gateway_manifest))
    _validate_manifest(manifest, str(config["llm"].get("model", "")))
    if manifest.get("selected_token_logprobs") != SELECTED_TOKEN_LOGPROBS_PROTOCOL:
        raise SystemExit(
            "gateway manifest does not advertise selected-token logprobs; deploy the entropy gateway revision"
        )
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
    legacy_contracts = []
    for cache_dir_arg, metadata_arg, runtime_config_arg in zip(
        args.legacy_sample_cache_dir,
        args.legacy_run_metadata,
        args.legacy_runtime_config,
        strict=True,
    ):
        cache_dir = Path(cache_dir_arg).resolve()
        if not cache_dir.is_dir():
            raise SystemExit(f"legacy sample cache directory does not exist: {cache_dir}")
        metadata = _read_json(Path(metadata_arg))
        runtime = metadata.get("runtime") if isinstance(metadata, dict) else None
        identity = runtime.get("llm") if isinstance(runtime, dict) else None
        if not isinstance(identity, dict):
            raise SystemExit("legacy run metadata has no recorded LLM cache identity")
        with Path(runtime_config_arg).open(encoding="utf-8") as handle:
            legacy_config = yaml.safe_load(handle)
        legacy_llm = legacy_config.get("llm") if isinstance(legacy_config, dict) else None
        legacy_api_base = legacy_llm.get("api_base") if isinstance(legacy_llm, dict) else None
        if legacy_api_base != f"{gateway_url}/v1":
            raise SystemExit("legacy cache endpoint differs from the authenticated gateway")
        legacy_semantic = legacy_config.get("semantic_entropy") if isinstance(legacy_config, dict) else None
        legacy_cap = legacy_semantic.get("max_tokens") if isinstance(legacy_semantic, dict) else None
        if not isinstance(legacy_cap, int) or isinstance(legacy_cap, bool) or legacy_cap <= 0:
            raise SystemExit("legacy runtime config has no valid semantic-entropy output cap")
        legacy_contracts.append({
            "cache_dir": str(cache_dir),
            "llm_identity": identity,
            "api_base": legacy_api_base,
            "max_tokens": [legacy_cap],
        })
    semantic.update({
        "cache_dir": str(Path(args.cache_root).resolve() / "semantic_entropy"),
        "cache_read_dirs": [str(Path(path).resolve()) for path in args.cache_read_dir],
        "legacy_sample_cache_contracts": legacy_contracts,
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
        "legacy_sample_cache_contracts": len(legacy_contracts),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
