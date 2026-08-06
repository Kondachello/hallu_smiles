#!/usr/bin/env python3
"""Record redacted snapshots of a local candidate-agreement run; never cancel it."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAFE_METADATA = {
    "state", "error_type", "sources_completed", "sources_scored", "elapsed_seconds",
    "nli_pair_evaluations", "cache_only", "finished_at_utc",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--interval-seconds", type=float, default=900.0)
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        raise SystemExit("interval-seconds must be positive")
    root = args.run_root.resolve()
    snapshots = root / "live-snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    while True:
        entries = []
        for path in sorted(root.glob("**/progress.json")):
            progress = _read_json(path)
            if progress:
                entries.append({"path": str(path.relative_to(root)), "progress": progress})
        metadata = []
        for path in sorted(root.glob("**/run_metadata.json")):
            value = _read_json(path)
            if value:
                metadata.append({
                    "path": str(path.relative_to(root)),
                    "metadata": {key: value[key] for key in SAFE_METADATA if key in value},
                })
        snapshot = {
            "protocol": "local-candidate-agreement-monitor-v1",
            "at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "runner_pid": args.pid,
            "runner_alive": _alive(args.pid),
            "progress": entries,
            "metadata": metadata,
        }
        path = snapshots / f"snapshot-{int(time.time())}.json"
        path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "runner_alive": snapshot["runner_alive"], "progress_records": len(entries),
            "metadata_records": len(metadata),
        }, sort_keys=True), flush=True)
        if not snapshot["runner_alive"]:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
