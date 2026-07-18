#!/usr/bin/env python3
"""Run one visible KGGen extraction on a compact RAGTruth QA record."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_instances  # noqa: E402
from src.micro_qa_demo import list_qa_candidates, run_micro_demo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="YAML config containing the concrete llm.model to use.",
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing RAGTruth JSONL files.")
    parser.add_argument(
        "--output-dir",
        default="results/micro_qa_demo",
        help="Directory to open as an Obsidian vault after the run.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=3000,
        help="Select only QA records at or below this context length (default: 3000).",
    )
    parser.add_argument(
        "--response-id",
        default=None,
        help="Optional exact QA response id; otherwise choose a compact record deterministically.",
    )
    parser.add_argument(
        "--list-candidates",
        action="store_true",
        help="List larger QA records for manual --response-id selection; no API call or key needed.",
    )
    parser.add_argument("--candidate-limit", type=int, default=20, help="Rows for --list-candidates.")
    parser.add_argument("--min-context-chars", type=int, default=0, help="C lower bound for --list-candidates.")
    parser.add_argument("--min-query-chars", type=int, default=0, help="Q lower bound for --list-candidates.")
    parser.add_argument("--min-response-chars", type=int, default=0, help="A lower bound for --list-candidates.")
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run the normal local EG/RP matcher and write audit/<response_id>.json.",
    )
    parser.add_argument(
        "--audit-alpha",
        type=float,
        default=0.7,
        help="Illustrative CFI weight for --audit (default: 0.7; not trained on one record).",
    )
    args = parser.parse_args()
    if args.list_candidates:
        candidates = list_qa_candidates(
            load_instances(args.data_dir),
            min_context_chars=args.min_context_chars,
            min_query_chars=args.min_query_chars,
            min_response_chars=args.min_response_chars,
            limit=args.candidate_limit,
        )
        print("response_id\tsource_id\tsplit\tlabel\tC chars\tQ chars\tA chars\tquery")
        for inst in candidates:
            query = " ".join((inst.query or "").split())
            print(
                f"{inst.response_id}\t{inst.source_id}\t{inst.split}\t{inst.y}\t"
                f"{len(inst.context)}\t{len(inst.query or '')}\t{len(inst.response)}\t{query}"
            )
        return
    run_micro_demo(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_context_chars=args.max_context_chars,
        response_id=args.response_id,
        audit=args.audit,
        audit_alpha=args.audit_alpha,
    )


if __name__ == "__main__":
    main()
