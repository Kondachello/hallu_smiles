#!/usr/bin/env python3
"""Verify the support verdict path once live and once cache-only."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.extract import UsageLogger
from src.verifier import RelationVerifier


CLAIM = ("Paris", "is capital of", "France")
EVIDENCE = "Paris is the capital of France."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    cfg = load_config(args.config)
    params = {"tau_e": 0.9, "tau_r": 0.75, "allow_substring_match": True, "min_substring_chars": 2, "stopwords": []}
    live_usage = UsageLogger(None)
    replay_usage = UsageLogger(None)
    live = RelationVerifier(cfg, usage=live_usage).verify(
        CLAIM, EVIDENCE, None, matching_params=params
    )
    replay = RelationVerifier(cfg, usage=replay_usage, cache_only=True).verify(
        CLAIM, EVIDENCE, None, matching_params=params
    )
    if live.verdict != "entailed" or replay.verdict != "entailed" or not replay.cache_hit:
        raise RuntimeError("verifier live/cache-only contract failed")
    if replay_usage.calls != 0:
        raise RuntimeError("cache-only verifier replay recorded a live inference call")
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "ready",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "model": cfg.llm.model,
        "structured_output_transport": cfg.llm.structured_output_transport,
        "structured_output_backend": cfg.llm.structured_output_backend,
        "claim": list(CLAIM),
        "live_verdict": live.verdict,
        "cache_only_verdict": replay.verdict,
        "cache_hit": replay.cache_hit,
        "live_usage": live_usage.summary(),
        "cache_only_usage": replay_usage.summary(),
    }
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
