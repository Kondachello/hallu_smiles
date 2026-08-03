#!/usr/bin/env python3
"""Summarise redacted DocRED usage and reject live inference in replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _summary(path: Path) -> dict[str, int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    final = rows[-1] if rows else {}
    return {
        "api_calls": int(final.get("cum_calls", 0)),
        "requests_total": int(final.get("cum_requests", 0)),
        "cache_hits": int(final.get("cum_cache_hits", 0)),
        "retries": int(final.get("cum_retries", 0)),
        "prompt_tokens": int(final.get("cum_prompt_tokens", 0)),
        "completion_tokens": int(final.get("cum_completion_tokens", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-usage", required=True)
    parser.add_argument("--replay-usage", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = {
        "protocol": "docred-usage-v1",
        "live": _summary(Path(args.live_usage)),
        "replay": _summary(Path(args.replay_usage)),
    }
    if payload["replay"]["api_calls"] != 0:
        raise SystemExit("DocRED cache-only replay made live inference calls")
    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
