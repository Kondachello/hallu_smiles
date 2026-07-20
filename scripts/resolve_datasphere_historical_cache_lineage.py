#!/usr/bin/env python3
"""Resolve a conservative historical KG cache identity from durable metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineages", required=True)
    parser.add_argument("--checkpoint-identity", required=True)
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--gateway-manifest-sha256", required=True)
    parser.add_argument("--qa-total", type=int, required=True)
    parser.add_argument("--qa-train", type=int, required=True)
    parser.add_argument("--qa-test", type=int, required=True)
    parser.add_argument("--cv-folds", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = _read(args.lineages)
    if payload.get("protocol") != "hallu-historical-kg-cache-lineages-v1":
        raise ValueError("unsupported historical cache lineage registry")
    checkpoint = _read(args.checkpoint_identity)
    expected_sample = {
        "total": args.qa_total,
        "train": args.qa_train,
        "test": args.qa_test,
        "alpha_cv_folds": args.cv_folds,
    }
    if checkpoint.get("gateway_manifest_sha256") != args.gateway_manifest_sha256:
        raise ValueError("historical checkpoint gateway manifest does not match the authenticated gateway")
    if checkpoint.get("qa_sample") != expected_sample:
        raise ValueError("historical checkpoint has a different QA manifest shape")

    matches = [
        item for item in payload.get("lineages", [])
        if isinstance(item, dict)
        and item.get("checkpoint_protocol") == checkpoint.get("protocol")
        and item.get("source_commit") == checkpoint.get("source_commit")
        and item.get("gateway_manifest_sha256") == args.gateway_manifest_sha256
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one compatible historical cache lineage, found {len(matches)}")
    lineage = matches[0]
    runtime = _read(args.runtime_manifest)
    if runtime.get("client_runtime") != lineage.get("client_runtime"):
        raise ValueError("current CPU client package versions differ from the historical KG cache lineage")
    fingerprint = lineage.get("llm_runtime_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint.startswith("vertex-gateway:"):
        raise ValueError("historical lineage has no valid LLM cache identity")

    result = {
        "protocol": "hallu-historical-kg-cache-lineage-resolution-v1",
        "lineage_id": lineage["id"],
        "historical_checkpoint": checkpoint,
        "llm_runtime_fingerprint": fingerprint,
        "gateway_manifest_sha256": args.gateway_manifest_sha256,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
