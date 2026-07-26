#!/usr/bin/env python3
"""Command-line entrypoint for a no-LLM historical QA cache replay."""
from __future__ import annotations

import argparse

from experiments.historical_qa_cache_replay import (
    render_historical_qa_cache_replay_summary,
    run_historical_qa_cache_controlled_replay,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", required=True)
parser.add_argument("--output-root", required=True)
parser.add_argument("--hallugraph-config", required=True)
parser.add_argument("--grapheval-config", required=True)
parser.add_argument("--historical-cache-root", required=True)
parser.add_argument("--lineage", required=True)
parser.add_argument("--run-id", required=True)
parser.add_argument("--qa-sample-size", type=int, default=100)
parser.add_argument("--qa-test-fraction", default="0.2")
parser.add_argument("--sample-seed", type=int, default=42)
parser.add_argument(
    "--replay-count", default="1",
    help="number of fully-cached records to replay, or 'all' for every complete record",
)
parser.add_argument("--replay-selection-seed", type=int, default=20260722)
args = parser.parse_args()

archive, report = run_historical_qa_cache_controlled_replay(
    data_dir=args.data_dir,
    output_root=args.output_root,
    hallugraph_config=args.hallugraph_config,
    grapheval_config=args.grapheval_config,
    historical_cache_root=args.historical_cache_root,
    lineage_path=args.lineage,
    run_id=args.run_id,
    qa_sample_size=args.qa_sample_size,
    qa_test_fraction=args.qa_test_fraction,
    sample_seed=args.sample_seed,
    replay_count=args.replay_count,
    replay_selection_seed=args.replay_selection_seed,
)
print(render_historical_qa_cache_replay_summary(report))
print(archive.path)
