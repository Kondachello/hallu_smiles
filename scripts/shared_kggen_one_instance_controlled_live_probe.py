#!/usr/bin/env python3
"""Run the real two-pass controlled shared-KGGen probe inside DataSphere."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.controlled_live_one_instance import render_controlled_live_probe_summary, run_ragtruth_one_instance_controlled_live_probe

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", required=True); parser.add_argument("--response-id", required=True)
parser.add_argument("--output-root", required=True); parser.add_argument("--cache-root", required=True)
parser.add_argument("--hallugraph-config", required=True); parser.add_argument("--grapheval-config", required=True)
parser.add_argument("--gateway-manifest", required=True); parser.add_argument("--run-id", default="controlled-live")
args = parser.parse_args()
first, second, report = run_ragtruth_one_instance_controlled_live_probe(
    source_info_path=Path(args.data_dir) / "source_info.jsonl", response_path=Path(args.data_dir) / "response.jsonl",
    response_id=args.response_id, output_root=args.output_root, cache_root=args.cache_root,
    hallugraph_config=args.hallugraph_config, grapheval_config=args.grapheval_config,
    gateway_manifest_path=args.gateway_manifest, run_id=args.run_id,
)
print(render_controlled_live_probe_summary(report))
