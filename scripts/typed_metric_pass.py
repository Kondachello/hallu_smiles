#!/usr/bin/env python3
"""CLI for the typed-vertex metric pass over a resolved historical QA graph cache.

Reads graphs cache-only; the typing agent makes the (permitted) gateway + local
HHEM calls. Emits ``typed_metrics.jsonl`` + ``typed_metric_summary.json`` under
``--output-root``.
"""
from __future__ import annotations

import argparse

from experiments.typed_metric_pass import run_typed_metric_pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--hallugraph-config", required=True)
    parser.add_argument("--grapheval-config", required=True)
    parser.add_argument("--typing-config", required=True,
                        help="dynamic typing agent YAML (live gateway + HHEM NLI)")
    parser.add_argument("--historical-cache-root", required=True,
                        help="primary (highest-priority) kg/ cache directory to read")
    parser.add_argument("--additional-cache-root", action="append", default=[],
                        help="lower-priority kg/ cache directory read through after the primary")
    parser.add_argument("--gateway-manifest-sha256", default=None)
    parser.add_argument("--qa-sample-size", type=int, default=100)
    parser.add_argument("--qa-test-fraction", default="0.2")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--replay-count", default="all",
                        help="number of fully-cached records to score, or 'all'")
    parser.add_argument("--replay-selection-seed", type=int, default=20260722)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="CFI_type = alpha*EG_type + (1-alpha)*RP")
    parser.add_argument("--batch-size", type=int, default=15,
                        help="flush a batch file every N records (partial results survive crashes)")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="record-level typing concurrency; >1 overlaps gateway+HHEM calls "
                             "(results are identical, emitted in strict record order)")
    args = parser.parse_args()

    summary = run_typed_metric_pass(
        data_dir=args.data_dir,
        output_root=args.output_root,
        hallugraph_config=args.hallugraph_config,
        grapheval_config=args.grapheval_config,
        typing_config=args.typing_config,
        historical_cache_root=args.historical_cache_root,
        additional_cache_roots=args.additional_cache_root,
        gateway_manifest_sha256=args.gateway_manifest_sha256,
        qa_sample_size=args.qa_sample_size,
        qa_test_fraction=args.qa_test_fraction,
        sample_seed=args.sample_seed,
        replay_count=args.replay_count,
        replay_selection_seed=args.replay_selection_seed,
        alpha=args.alpha,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
    )
    import json

    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
