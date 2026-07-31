#!/usr/bin/env python3
"""Reject a DocRED cache replay whose scientific outputs changed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-dir", required=True)
    parser.add_argument("--replay-dir", required=True)
    args = parser.parse_args()
    live, replay = Path(args.live_dir), Path(args.replay_dir)
    for filename in ("relation_alignment_tuning.json", "document_scores.jsonl"):
        if (live / filename).read_bytes() != (replay / filename).read_bytes():
            raise SystemExit(f"cache-only replay changed {filename}")
    live_metrics, replay_metrics = _json(live / "metrics.json"), _json(replay / "metrics.json")
    for payload in (live_metrics, replay_metrics):
        payload.pop("usage", None)
        payload.pop("budget", None)
    if live_metrics != replay_metrics:
        raise SystemExit("cache-only replay changed DocRED scientific metrics")
    for payload in (_json(live / "run_metadata.json"), _json(replay / "run_metadata.json")):
        if payload.get("state") != "completed":
            raise SystemExit("DocRED replay did not complete")
    print("[ok] DocRED cache-only replay is scientifically identical")


if __name__ == "__main__":
    main()
