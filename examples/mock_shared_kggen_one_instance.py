"""Two-pass, one-response RAGTruth plumbing probe for the controlled KGGen track.

No gateway, secret, real KGGen model or HHEM model is used.  The input files may be
real local RAGTruth JSONL files, but only the selected no-gold record reaches detectors.
"""
from __future__ import annotations

import argparse
import json

from experiments.one_instance import (
    render_shared_kggen_mock_probe_summary,
    run_ragtruth_one_instance_shared_kggen_mock_probe,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="one-response shared-KGGen cache mock probe")
    parser.add_argument("--source-info", required=True, help="local RAGTruth source_info.jsonl")
    parser.add_argument("--responses", required=True, help="local RAGTruth response.jsonl")
    parser.add_argument("--response-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-root", help="shared graph-cache root; use DataSphere Project storage later")
    parser.add_argument("--run-id")
    parser.add_argument("--hallugraph-config", default="config.yaml")
    args = parser.parse_args()
    _cold, _replay, report = run_ragtruth_one_instance_shared_kggen_mock_probe(
        source_info_path=args.source_info,
        response_path=args.responses,
        response_id=args.response_id,
        output_root=args.output_root,
        cache_root=args.cache_root,
        run_id=args.run_id,
        hallugraph_config=args.hallugraph_config,
    )
    print(render_shared_kggen_mock_probe_summary(report))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
