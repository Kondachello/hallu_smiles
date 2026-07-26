#!/usr/bin/env python3
"""Write a deterministic QA manifest and prove its KG cache is fully warm."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache_preflight import manifest_sha256, verify_kg_cache
from src.config import load_config
from src.data import load_instances
from src.sampling import qa_sample_quotas, select_qa_sample, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--qa-sample-size", type=int, required=True)
    parser.add_argument("--qa-test-fraction", required=True)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--exclude-source-id",
        action="append",
        default=[],
        help=(
            "source ID explicitly quarantined from analysis; its graph cache is "
            "not required and no live extraction will be attempted (repeatable)"
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "record cold graph keys without failing; use only before the initial "
            "read-through cache-fill run, never for a cache-only replay"
        ),
    )
    args = parser.parse_args()

    train_sources, test_sources = qa_sample_quotas(
        args.qa_sample_size, args.qa_test_fraction
    )
    cfg = load_config(args.config)
    all_instances = load_instances(
        args.data_dir, exclude_implicit_true=bool(cfg.data.exclude_implicit_true)
    )
    selected = select_qa_sample(
        all_instances,
        seed=args.sample_seed,
        train_sources=train_sources,
        test_sources=test_sources,
    )
    manifest_output = Path(args.manifest_output)
    write_manifest(
        manifest_output,
        selected,
        seed=args.sample_seed,
        train_sources=train_sources,
        test_sources=test_sources,
    )
    manifest = json.loads(manifest_output.read_text(encoding="utf-8"))
    report = verify_kg_cache(
        cfg, selected, excluded_source_ids=args.exclude_source_id
    )
    report.update({
        "qa_sample": {
            "total": args.qa_sample_size,
            "train": train_sources,
            "test": test_sources,
            "seed": args.sample_seed,
        },
        "manifest_sha256": manifest_sha256(manifest),
        "cache_dir": str(cfg.cache_dir),
        "llm_runtime_fingerprint": str(cfg.llm.runtime_fingerprint),
    })
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "ready" and not args.allow_missing:
        raise SystemExit(
            "historical KG cache preflight failed: "
            f"{report['missing_count']} required graph entries are unavailable"
        )


if __name__ == "__main__":
    main()
