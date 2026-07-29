#!/usr/bin/env python3
"""Verify deterministic replay outputs without comparing cache-observability flags.

``verifier_cache_hit`` truthfully differs between the cache-fill pass (False for
new entries) and its cache-only replay (True).  It is an audit diagnostic, not a
scientific score or cache payload.  All other serialized scored-record fields
remain part of the strict equality check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BYTE_IDENTICAL_FILES = ("metrics.csv", "summary_metrics.csv", "tuning.json")
DIAGNOSTIC_KEYS = frozenset({"verifier_cache_hit"})


def _without_diagnostics(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_diagnostics(item)
            for key, item in value.items()
            if key not in DIAGNOSTIC_KEYS
        }
    if isinstance(value, list):
        return [_without_diagnostics(item) for item in value]
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSONL in {path.name}: line {exc.lineno}") from exc


def verify_replay(live_dir: Path, replay_dir: Path) -> None:
    for filename in BYTE_IDENTICAL_FILES:
        live = live_dir / filename
        replay = replay_dir / filename
        if live.read_bytes() != replay.read_bytes():
            raise SystemExit(f"cache-only replay changed {filename}")

    live_rows = _read_jsonl(live_dir / "scored.jsonl")
    replay_rows = _read_jsonl(replay_dir / "scored.jsonl")
    if len(live_rows) != len(replay_rows):
        raise SystemExit(
            "cache-only replay changed scored.jsonl row count "
            f"({len(live_rows)} != {len(replay_rows)})"
        )
    for index, (live, replay) in enumerate(zip(live_rows, replay_rows), start=1):
        if _without_diagnostics(live) != _without_diagnostics(replay):
            raise SystemExit(
                "cache-only replay changed scored.jsonl scientific content "
                f"at record {index}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    args = parser.parse_args()
    verify_replay(args.live_dir, args.replay_dir)


if __name__ == "__main__":
    main()
