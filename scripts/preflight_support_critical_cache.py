#!/usr/bin/env python3
"""Prove a fixed support-critical manifest is completely cache-replayable.

This deliberately performs the same claim extraction, coverage review, and
evidence selection as scoring, with every component in ``cache_only`` mode.
It is a fail-fast gate for a historical sub-study: a missing verdict must be
reported before train-only CV starts, never papered over by a live request.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache import CacheOnlyMissError
from src.config import load_config
from src.critical import CriticalClaimPipeline
from src.data import load_instances
from src.matching import SBERTEmbedder
from src.sampling import load_manifest_instances


def _embedder(cfg):
    matching = cfg.matching
    return SBERTEmbedder(
        str(matching.embedding_model),
        model_revision=str(matching.embedding_model_revision),
        model_path=str(matching.embedding_model_path),
        device=str(getattr(matching, "embedding_device", "cpu")),
        local_files_only=bool(getattr(matching, "local_files_only", True)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--qa-manifest", required=True)
    parser.add_argument("--exclude-source-id", action="append", default=[])
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    instances = load_manifest_instances(args.qa_manifest, load_instances(args.data_dir))
    excluded = {str(source_id) for source_id in args.exclude_source_id}
    selected_sources = {inst.source_id for inst in instances}
    unknown = sorted(excluded - selected_sources)
    if unknown:
        raise SystemExit("cache preflight exclusion is absent from manifest: " + ", ".join(unknown))

    pipeline = CriticalClaimPipeline(cfg, cache_only=True, embedder=_embedder(cfg))
    checked = 0
    misses: list[dict[str, str]] = []
    for inst in instances:
        if inst.source_id in excluded:
            continue
        checked += 1
        try:
            pipeline.assess(inst.response, inst.context, inst.query)
        except CacheOnlyMissError as exc:
            # Preserve the cache diagnosis without storing the raw claim,
            # evidence, prompt, or completion in the redacted archive.
            misses.append({
                "source_id": inst.source_id,
                "response_id": inst.response_id,
                "component": exc.component,
                "cache_key": exc.key,
            })

    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "protocol": "support-critical-cache-only-preflight-v1",
        "cache_only": True,
        "manifest_records": len(instances),
        "explicitly_excluded_sources": sorted(excluded),
        "analysis_records": checked,
        "missing_count": len(misses),
        "missing": misses,
        "status": "ready" if not misses else "cache_miss",
    }
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if misses:
        raise SystemExit("support-critical cache-only preflight found missing artifacts")


if __name__ == "__main__":
    main()
