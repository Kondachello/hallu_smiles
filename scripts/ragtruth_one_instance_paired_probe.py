#!/usr/bin/env python3
"""Offline integration probe for one explicitly selected local RAGTruth response.

It runs the actual HalluGraph and GraphEval adapters over deterministic fake model
backends, writes a sealed archive, and never downloads data or calls a live model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.one_instance import render_probe_summary, run_ragtruth_one_instance_probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-id", required=True, help="exact id from local response.jsonl")
    parser.add_argument("--data-dir", help="directory containing source_info.jsonl and response.jsonl")
    parser.add_argument("--source-info", help="local source_info.jsonl; overrides --data-dir")
    parser.add_argument("--responses", help="local response.jsonl; overrides --data-dir")
    parser.add_argument("--output-root", default="results/ragtruth_one_instance_probe", help="new archive parent directory")
    parser.add_argument("--run-id", help="optional unique archive directory name")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"), help="HalluGraph config; no secret is read in fake mode")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None
    source_info = Path(args.source_info) if args.source_info else (data_dir / "source_info.jsonl" if data_dir else None)
    responses = Path(args.responses) if args.responses else (data_dir / "response.jsonl" if data_dir else None)
    if source_info is None or responses is None:
        parser.error("provide --data-dir or both --source-info and --responses")
    if not source_info.is_file():
        parser.error(f"source file does not exist: {source_info}")
    if not responses.is_file():
        parser.error(f"response file does not exist: {responses}")

    archive, _report = run_ragtruth_one_instance_probe(
        source_info_path=source_info,
        response_path=responses,
        response_id=args.response_id,
        output_root=args.output_root,
        run_id=args.run_id,
        hallugraph_config=args.config,
    )
    print(render_probe_summary(archive))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
