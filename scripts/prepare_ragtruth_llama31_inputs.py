#!/usr/bin/env python3
"""Validate the annotated Llama-3.1 CSV and write its immutable manifest.

This script never emits answers or prompts.  Its JSON output contains only
checksums, counts, source identifiers, split membership and per-record text
hashes so it is safe to place in a redacted terminal archive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llama31_eval import (
    DEFAULT_RAGTRUTH_COMMIT,
    build_llama31_instances,
    file_sha256,
    write_llama31_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--historical-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ragtruth-commit", default=DEFAULT_RAGTRUTH_COMMIT)
    args = parser.parse_args()
    if args.ragtruth_commit != DEFAULT_RAGTRUTH_COMMIT:
        raise SystemExit(
            "controlled Llama evaluation pins RAGTruth at "
            f"{DEFAULT_RAGTRUTH_COMMIT}, got {args.ragtruth_commit}"
        )
    instances, provenance = build_llama31_instances(
        args.annotations_csv, args.data_dir, args.historical_manifest
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_llama31_manifest(output_dir / "llama31_manifest.json", instances, provenance)
    payload = {
        "protocol": "hallu-ragtruth-llama31-input-provenance-v1",
        "ragtruth_commit": args.ragtruth_commit,
        "manifest": manifest_path.name,
        "manifest_sha256": file_sha256(manifest_path),
        **provenance,
    }
    provenance_path = output_dir / "input_provenance.json"
    provenance_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": payload["manifest_sha256"],
        "records": len(instances),
        "train": sum(inst.split == "train" for inst in instances),
        "test": sum(inst.split == "test" for inst in instances),
        "labels": provenance["label_counts"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
