#!/usr/bin/env python3
"""Create a local CPU KGGen config bound to an authenticated Vertex gateway."""
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
    parser.add_argument("--local-runtime-manifest", required=True)
    parser.add_argument("--embedding-model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--extraction-max-tokens-ceiling", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--retry-backoff-base-s", type=float, default=5.0)
    parser.add_argument("--retry-backoff-max-s", type=float, default=60.0)
    parser.add_argument("--rate-limit-cooldown-max-s", type=float, default=900.0)
    args = parser.parse_args()

    if args.max_tokens <= 0 or args.extraction_max_tokens_ceiling < args.max_tokens:
        raise SystemExit("token limits must be positive and ceiling must be at least max-tokens")
    if args.concurrency != 1:
        raise SystemExit("the fixed DocRED protocol requires --concurrency 1")
    if args.max_retries != 0:
        raise SystemExit("the local DocRED runner requires --max-retries 0 with bounded deadlines")
    if args.retry_backoff_base_s <= 0 or args.retry_backoff_max_s < args.retry_backoff_base_s:
        raise SystemExit("invalid retry backoff bounds")
    if args.rate_limit_cooldown_max_s < args.retry_backoff_max_s:
        raise SystemExit("rate-limit cooldown must be at least retry-backoff-max-s")

    with Path(args.base_config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("llm"), dict):
        raise SystemExit("base config has no llm mapping")
    manifest = _read_json(Path(args.gateway_manifest))
    logical_model = str(config["llm"].get("model", ""))
    _validate_manifest(manifest, logical_model)
    runtime = _read_json(Path(args.local_runtime_manifest))
    runtime_fingerprint = runtime.get("runtime_fingerprint")
    if not isinstance(runtime_fingerprint, str) or not runtime_fingerprint.startswith("local-cpu:"):
        raise SystemExit("local runtime manifest has no valid runtime_fingerprint")
    embedding = Path(args.embedding_model_path).resolve()
    if not (embedding / "config.json").is_file():
        raise SystemExit("embedding model path is not an offline S-BERT snapshot")

    gateway_url = _normalise_gateway_url(args.gateway_url)
    manifest_hash = canonical_manifest_sha256(manifest)
    combined = {
        "local_runtime_fingerprint": runtime_fingerprint,
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
        "max_retries": 0,
        "retry_backoff_base_s": args.retry_backoff_base_s,
        "retry_backoff_max_s": args.retry_backoff_max_s,
        "rate_limit_cooldown_max_s": args.rate_limit_cooldown_max_s,
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
    matching = config.setdefault("matching", {})
    matching.update({
        "embedding_model_path": str(embedding),
        "embedding_device": "cpu",
        "local_files_only": True,
    })
    cache_root = Path(args.cache_root)
    config.setdefault("data", {})["dir"] = str(Path(args.data_dir))
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
    config["output_dir"] = str(Path(args.work_dir) / "results")
    config.setdefault("vertex_gateway", {}).update({
        "manifest_sha256": manifest_hash,
        "gateway_manifest": manifest,
        "gateway_url": gateway_url,
        "local_runtime_fingerprint": runtime_fingerprint,
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(json.dumps({
        "gateway_manifest_sha256": manifest_hash,
        "runtime_fingerprint": llm["runtime_fingerprint"],
        "local_runtime_fingerprint": runtime_fingerprint,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
