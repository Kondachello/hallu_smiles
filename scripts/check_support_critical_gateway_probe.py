#!/usr/bin/env python3
"""Exercise all three support-critical schemas live, then cache-only.

This is a transport/cache compatibility gate, not a semantic test. It accepts
any of the four closed-world verdicts and does not prescribe an extraction;
it verifies only valid offsets, schemas, bounded retry behaviour, and zero
network calls on replay.
"""
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
from src.critical import (
    CRITICAL_VERDICTS,
    AtomicClaimExtractor,
    CriticalClaimVerifier,
    FullContextReviewer,
)
from src.extract import UsageLogger


CONTEXT = "Mammals are animals."
QUERY = "What are mammals?"
ANSWER = "Mammals are animals."
CLAIM = "Mammals are animals."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    cfg = load_config(args.config)
    live_usage = UsageLogger(None)
    replay_usage = UsageLogger(None)
    failure: Exception | None = None
    atomic_count = coverage_count = 0
    live_verdict = replay_verdict = None
    try:
        atomic = AtomicClaimExtractor(cfg, usage=live_usage).extract(ANSWER)
        reviewed = FullContextReviewer(cfg, usage=live_usage).review(ANSWER, CONTEXT, QUERY, atomic)
        decision = CriticalClaimVerifier(cfg, usage=live_usage).verify_claim(CLAIM, CONTEXT, QUERY)
        atomic_count, coverage_count, live_verdict = len(atomic), len(reviewed), decision.verdict

        replay_atomic = AtomicClaimExtractor(cfg, usage=replay_usage, cache_only=True).extract(ANSWER)
        replay_reviewed = FullContextReviewer(cfg, usage=replay_usage, cache_only=True).review(
            ANSWER, CONTEXT, QUERY, replay_atomic
        )
        replay = CriticalClaimVerifier(cfg, usage=replay_usage, cache_only=True).verify_claim(
            CLAIM, CONTEXT, QUERY
        )
        replay_verdict = replay.verdict
        if replay_atomic != atomic or replay_reviewed != reviewed:
            raise RuntimeError("support-critical claim cache replay differed from the live artifact")
        if live_verdict not in CRITICAL_VERDICTS or replay_verdict != live_verdict or not replay.cache_hit:
            raise RuntimeError("support-critical verdict live/cache-only contract failed")
        if replay_usage.calls != 0:
            raise RuntimeError("support-critical cache-only replay made a live inference call")
    except Exception as exc:  # Keep a redacted diagnostic if a provider gate fails.
        failure = exc
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "failed" if failure else "ready",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "atomic_claim_count": atomic_count,
        "coverage_candidate_count": coverage_count,
        "live_verdict": live_verdict,
        "cache_only_verdict": replay_verdict,
        "live_usage": live_usage.summary(),
        "cache_only_usage": replay_usage.summary(),
    }
    if failure is not None:
        report["failure_class"] = type(failure).__name__
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
