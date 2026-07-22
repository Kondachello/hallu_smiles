#!/usr/bin/env python3
"""Resolve one immutable 100-QA cache root without contacting the gateway."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-base", required=True)
    parser.add_argument("--lineages", required=True)
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--qa-total", type=int, default=100)
    parser.add_argument("--qa-train", type=int, default=80)
    parser.add_argument("--qa-test", type=int, default=20)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    registry = _read(Path(args.lineages))
    if registry.get("protocol") != "hallu-historical-kg-cache-lineages-v1":
        raise ValueError("unsupported historical cache lineage registry")
    runtime = _read(Path(args.runtime_manifest))
    expected_sample = {
        "total": args.qa_total,
        "train": args.qa_train,
        "test": args.qa_test,
        "alpha_cv_folds": args.cv_folds,
    }
    matches: list[tuple[Path, dict, dict]] = []
    base = Path(args.checkpoint_base)
    for candidate in sorted(path for path in base.iterdir() if path.is_dir()) if base.is_dir() else []:
        identity_path = candidate / "checkpoint-identity.json"
        kg_root = candidate / "kg"
        if not identity_path.is_file() or not kg_root.is_dir():
            continue
        identity = _read(identity_path)
        for lineage in registry.get("lineages", []):
            if not isinstance(lineage, dict):
                continue
            if (
                identity.get("protocol") == lineage.get("checkpoint_protocol")
                and identity.get("source_commit") == lineage.get("source_commit")
                and identity.get("gateway_manifest_sha256") == lineage.get("gateway_manifest_sha256")
                and identity.get("qa_sample") == expected_sample
                and runtime.get("client_runtime") == lineage.get("client_runtime")
            ):
                matches.append((candidate, identity, lineage))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one compatible historical QA cache, found {len(matches)}")
    candidate, identity, lineage = matches[0]
    result = {
        "protocol": "hallu-historical-qa-cache-replay-resolution-v1",
        "lineage_id": lineage["id"],
        "historical_checkpoint": identity,
        "historical_cache_root": str(candidate / "kg"),
        "llm_runtime_fingerprint": lineage["llm_runtime_fingerprint"],
        "gateway_manifest_sha256": lineage["gateway_manifest_sha256"],
        "gateway_contacted": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
