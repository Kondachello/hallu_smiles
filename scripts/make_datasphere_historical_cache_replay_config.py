#!/usr/bin/env python3
"""Write a cache-only HalluGraph configuration for a recorded historical lineage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--gateway-url", required=True, help="Recorded gateway origin; never contacted by this script")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--cache-read-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=16384)
    args = parser.parse_args()
    parsed = urlparse(args.gateway_url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.path.rstrip("/") or parsed.query or parsed.fragment:
        raise ValueError("--gateway-url must be an HTTPS origin")
    lineage = json.loads(Path(args.lineage).read_text(encoding="utf-8"))
    fingerprint = lineage.get("llm_runtime_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint.startswith("vertex-gateway:"):
        raise ValueError("lineage has no valid historical LLM runtime fingerprint")
    config = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("llm"), dict):
        raise ValueError("base config has no llm mapping")
    config["llm"].update({
        "api_base": f"https://{parsed.netloc}/v1",
        "api_key_env": "HALLU_GATEWAY_API_KEY",
        "runtime_fingerprint": fingerprint,
        "max_tokens": args.max_tokens,
        "concurrency": 1,
        "max_retries": 0,
        "length_retry_attempts": 0,
        "length_retry_max_tokens": None,
    })
    config["cache_dir"] = str(Path(args.cache_root))
    config["cache_read_dirs"] = [str(Path(args.cache_read_root))]
    config["data"]["dir"] = str(Path(args.data_dir))
    config["matching"]["embedding_model_path"] = "/opt/hallu/models/all-MiniLM-L6-v2"
    config["matching"]["embedding_device"] = "cpu"
    config["matching"]["local_files_only"] = True
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(json.dumps({
        "historical_llm_runtime_fingerprint": fingerprint,
        "cache_read_root": str(Path(args.cache_read_root)),
        "gateway_contacted": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
