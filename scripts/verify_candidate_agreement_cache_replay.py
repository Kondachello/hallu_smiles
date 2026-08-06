#!/usr/bin/env python3
"""Require a zero-inference, byte-identical candidate-agreement replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SCIENTIFIC_FILES = ("candidate_scores.jsonl", "unscorable.jsonl", "scientific_metrics.json")


def _metadata(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit("cache replay metadata is unreadable") from exc
    if not isinstance(value, dict):
        raise SystemExit("cache replay metadata is malformed")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-dir", required=True, type=Path)
    parser.add_argument("--replay-dir", required=True, type=Path)
    parser.add_argument("--cache-before", required=True, type=Path)
    parser.add_argument("--cache-after", required=True, type=Path)
    args = parser.parse_args()
    for name in SCIENTIFIC_FILES:
        if (args.live_dir / name).read_bytes() != (args.replay_dir / name).read_bytes():
            raise SystemExit(f"cache-only replay changed scientific file: {name}")
    if args.cache_before.read_bytes() != args.cache_after.read_bytes():
        raise SystemExit("cache-only replay changed candidate/sample cache inventory")
    metadata = _metadata(args.replay_dir / "run_metadata.json")
    usage = metadata.get("usage") if isinstance(metadata.get("usage"), dict) else {}
    if int(usage.get("api_calls", -1)) != 0 or int(metadata.get("nli_pair_evaluations", -1)) != 0:
        raise SystemExit("cache-only replay made Gemini or NLI calls")
    if metadata.get("state") != "completed_cache_replay":
        raise SystemExit("cache-only replay did not complete")
    print(json.dumps({"state": "verified", "scientific_files": list(SCIENTIFIC_FILES)}, sort_keys=True))


if __name__ == "__main__":
    main()
