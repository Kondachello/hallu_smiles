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
# Exercise the segmented cache path before a 100-QA Job starts. This text is
# repetitive deliberately: the probe is about exact offset/cache transport,
# not an expected scientific verdict or a model's ability to infer facts.
SEGMENTED_ANSWER = "Mammals are animals. " * 60


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="require previously validated probe entries and forbid live gateway calls",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    cfg = load_config(args.config)
    live_usage = UsageLogger(None)
    replay_usage = UsageLogger(None)
    failure: Exception | None = None
    atomic_count = coverage_count = segmented_atomic_count = segmented_coverage_count = 0
    live_verdict = replay_verdict = None
    try:
        atomic = AtomicClaimExtractor(cfg, usage=live_usage, cache_only=args.cache_only).extract(ANSWER)
        reviewed = FullContextReviewer(cfg, usage=live_usage, cache_only=args.cache_only).review(
            ANSWER, CONTEXT, QUERY, atomic
        )
        decision = CriticalClaimVerifier(cfg, usage=live_usage, cache_only=args.cache_only).verify_claim(
            CLAIM, CONTEXT, QUERY
        )
        atomic_count, coverage_count, live_verdict = len(atomic), len(reviewed), decision.verdict

        if len(SEGMENTED_ANSWER) <= int(cfg.support_critical.claim_extractor.chunk_chars):
            raise RuntimeError("support-critical segmented gateway probe is not longer than its configured chunk")
        segmented_atomic = AtomicClaimExtractor(cfg, usage=live_usage, cache_only=args.cache_only).extract(
            SEGMENTED_ANSWER
        )
        segmented_reviewed = FullContextReviewer(cfg, usage=live_usage, cache_only=args.cache_only).review(
            SEGMENTED_ANSWER, CONTEXT, QUERY, segmented_atomic
        )
        segmented_atomic_count, segmented_coverage_count = len(segmented_atomic), len(segmented_reviewed)

        if args.cache_only:
            replay_verdict = decision.verdict
            if live_usage.calls != 0 or not decision.cache_hit:
                raise RuntimeError("support-critical cache-only probe made a live inference call")
        else:
            replay_atomic = AtomicClaimExtractor(cfg, usage=replay_usage, cache_only=True).extract(ANSWER)
            replay_reviewed = FullContextReviewer(cfg, usage=replay_usage, cache_only=True).review(
                ANSWER, CONTEXT, QUERY, replay_atomic
            )
            replay = CriticalClaimVerifier(cfg, usage=replay_usage, cache_only=True).verify_claim(
                CLAIM, CONTEXT, QUERY
            )
            replay_verdict = replay.verdict
            replay_segmented_atomic = AtomicClaimExtractor(cfg, usage=replay_usage, cache_only=True).extract(
                SEGMENTED_ANSWER
            )
            replay_segmented_reviewed = FullContextReviewer(
                cfg, usage=replay_usage, cache_only=True
            ).review(SEGMENTED_ANSWER, CONTEXT, QUERY, replay_segmented_atomic)
            if replay_atomic != atomic or replay_reviewed != reviewed:
                raise RuntimeError("support-critical claim cache replay differed from the live artifact")
            if replay_segmented_atomic != segmented_atomic or replay_segmented_reviewed != segmented_reviewed:
                raise RuntimeError("support-critical segmented cache replay differed from the live artifact")
            if replay_verdict != live_verdict or not replay.cache_hit:
                raise RuntimeError("support-critical verdict live/cache-only contract failed")
            if replay_usage.calls != 0:
                raise RuntimeError("support-critical cache-only replay made a live inference call")
        if live_verdict not in CRITICAL_VERDICTS:
            raise RuntimeError("support-critical probe returned an invalid verdict")
    except Exception as exc:  # Keep a redacted diagnostic if a provider gate fails.
        failure = exc
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "failed" if failure else "ready",
        "mode": "cache-only" if args.cache_only else "live-then-cache-only",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "atomic_claim_count": atomic_count,
        "coverage_candidate_count": coverage_count,
        "segmented_answer_chars": len(SEGMENTED_ANSWER),
        "segmented_atomic_claim_count": segmented_atomic_count,
        "segmented_coverage_candidate_count": segmented_coverage_count,
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
