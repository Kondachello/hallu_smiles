#!/usr/bin/env python3
"""Add the isolated candidate-comparison cache to a validated entropy runtime."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.candidate_agreement import canonical_hash


CANDIDATE_AGREEMENT_PROTOCOL = "ragtruth-candidate-agreement-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-runtime-config", required=True)
    parser.add_argument("--candidate-cache-root", required=True)
    parser.add_argument("--expected-gateway-manifest-sha256", required=True)
    parser.add_argument("--nli-batch-size", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.semantic_runtime_config).resolve()
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise SystemExit("validated semantic runtime config is not a YAML mapping")
    gateway = payload.get("vertex_gateway")
    if not isinstance(gateway, dict) or gateway.get("manifest_sha256") != args.expected_gateway_manifest_sha256:
        raise SystemExit("semantic runtime gateway manifest differs from the required cache-compatible revision")
    semantic = payload.get("semantic_entropy")
    if not isinstance(semantic, dict):
        raise SystemExit("validated semantic runtime has no semantic_entropy section")
    if semantic.get("max_tokens") != 65535 or semantic.get("max_tokens_read_through") != [4096, 8192]:
        raise SystemExit("candidate agreement requires fixed 65535-token semantic sampling and [4096, 8192] read-through")
    if semantic.get("n_samples") != 15 or semantic.get("temperature") != 1.0:
        raise SystemExit("candidate agreement requires 15 samples at temperature 1.0")
    if args.nli_batch_size is not None:
        if args.nli_batch_size <= 0:
            raise SystemExit("--nli-batch-size must be positive")
        # Batch size is an execution-only setting: the tokenizer, model,
        # max-length, labels, and cache-keyed NLI identity are unchanged.
        semantic["nli_batch_size"] = int(args.nli_batch_size)
    cache_root = Path(args.candidate_cache_root).resolve()
    payload["candidate_agreement"] = {
        "protocol": CANDIDATE_AGREEMENT_PROTOCOL,
        "cache_dir": str(cache_root / "candidate_agreement"),
        "cache_read_dirs": [],
        "sample_cache_contract_sha256": canonical_hash({
            "semantic_entropy_protocol": semantic.get("protocol"),
            "cache_dir": semantic.get("cache_dir"),
            "gateway_manifest_sha256": gateway.get("manifest_sha256"),
            "n_samples": semantic.get("n_samples"),
            "temperature": semantic.get("temperature"),
            "max_tokens": semantic.get("max_tokens"),
            "likelihood_normalization": semantic.get("likelihood_normalization"),
        }),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(json.dumps({
        "candidate_agreement_protocol": CANDIDATE_AGREEMENT_PROTOCOL,
        "candidate_cache_dir": payload["candidate_agreement"]["cache_dir"],
        "gateway_manifest_sha256": gateway["manifest_sha256"],
        "nli_batch_size": semantic.get("nli_batch_size"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
