#!/usr/bin/env python3
"""Build the private frozen C/Q graph artifact for the controlled Llama run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llama31_eval import (
    build_frozen_reference_artifact_from_historical_cache,
    build_llama31_instances,
    file_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--historical-manifest", required=True)
    parser.add_argument("--historical-extraction-summary", required=True)
    parser.add_argument("--historical-run-metadata", required=True)
    parser.add_argument("--historical-runtime-identity", required=True)
    parser.add_argument("--historical-cache-root", required=True)
    parser.add_argument("--historical-cache-export", required=True)
    parser.add_argument("--historical-cache-export-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    instances, _ = build_llama31_instances(
        args.annotations_csv, args.data_dir, args.historical_manifest
    )
    artifact, provenance = build_frozen_reference_artifact_from_historical_cache(
        instances=instances,
        historical_extraction_summary=args.historical_extraction_summary,
        historical_run_metadata=args.historical_run_metadata,
        historical_runtime_identity=args.historical_runtime_identity,
        historical_cache_root=args.historical_cache_root,
        historical_cache_export=args.historical_cache_export,
        historical_cache_export_sha256_file=args.historical_cache_export_sha256,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "artifact": str(output),
        "artifact_sha256": file_sha256(output),
        "sources": provenance["source_count"],
        "historical_gateway_manifest_sha256": provenance["gateway_manifest_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
