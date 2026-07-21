#!/usr/bin/env python3
"""Run one real, audited HalluGraph + GraphEval RAGTruth probe in DataSphere."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.live_one_instance import render_live_probe_summary, run_ragtruth_one_instance_live_probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-id", required=True, help="exact id from the mounted local response.jsonl")
    parser.add_argument("--data-dir", help="directory containing source_info.jsonl and response.jsonl")
    parser.add_argument("--source-info", help="mounted source_info.jsonl; overrides --data-dir")
    parser.add_argument("--responses", help="mounted response.jsonl; overrides --data-dir")
    parser.add_argument("--output-root", required=True, help="writable DataSphere Job artifact directory")
    parser.add_argument("--hallugraph-config", required=True, help="Job-local redacted runtime config")
    parser.add_argument("--grapheval-config", default=str(ROOT / "graph_eval/config.datasphere.one-instance.live.yaml"))
    parser.add_argument("--gateway-manifest", required=True, help="authenticated, validated manifest from the Job preflight")
    parser.add_argument("--run-id", help="optional unique archive directory name")
    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else None
    source_info = Path(args.source_info) if args.source_info else (data_dir / "source_info.jsonl" if data_dir else None)
    responses = Path(args.responses) if args.responses else (data_dir / "response.jsonl" if data_dir else None)
    if source_info is None or responses is None:
        parser.error("provide --data-dir or both --source-info and --responses")
    for label, path in (("source", source_info), ("response", responses), ("HalluGraph config", Path(args.hallugraph_config)), ("GraphEval config", Path(args.grapheval_config)), ("gateway manifest", Path(args.gateway_manifest))):
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")
    archive, _report = run_ragtruth_one_instance_live_probe(
        source_info_path=source_info,
        response_path=responses,
        response_id=args.response_id,
        output_root=args.output_root,
        hallugraph_config=args.hallugraph_config,
        grapheval_config=args.grapheval_config,
        gateway_manifest_path=args.gateway_manifest,
        run_id=args.run_id,
    )
    print(render_live_probe_summary(archive))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
