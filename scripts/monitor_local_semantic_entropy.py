#!/usr/bin/env python3
"""Persist redacted liveness snapshots for an unattended entropy run."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAFE_FIELDS = {
    "protocol", "at_utc", "event", "phase", "sources_completed", "sources_total",
    "current_source_samples_completed", "api_calls", "cache_hits", "retries",
    "prompt_tokens", "completion_tokens", "component", "reason", "attempt",
    "sleep_seconds", "retry_seconds", "continuous_429_seconds", "n_classes", "error_type",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_progress(run_root: Path) -> dict[str, Any] | None:
    path = run_root / "progress.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {"event": "progress_unreadable"}
    return {key: payload[key] for key in SAFE_FIELDS if key in payload} if isinstance(payload, dict) else {"event": "progress_invalid"}


def _write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--interval-seconds", type=int, default=900)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        raise SystemExit("interval must be positive")
    root = Path(args.run_root)
    snapshots = root / "live-snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    while True:
        payload = {
            "protocol": "local-ragtruth-semantic-entropy-monitor-v1",
            "checked_at_utc": _utc(),
            "runner_pid": args.pid,
            "runner_alive": _pid_alive(args.pid),
            "progress": _read_progress(root),
        }
        stamp = payload["checked_at_utc"].replace(":", "-")
        _write(snapshots / f"{stamp}.json", payload)
        _write(snapshots / "latest.json", payload)
        print("[semantic-entropy-monitor] " + json.dumps(payload, sort_keys=True), flush=True)
        if args.once or not payload["runner_alive"]:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
