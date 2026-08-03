#!/usr/bin/env python3
"""Persist redacted 15-minute liveness snapshots for one local DocRED run."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAFE_PROGRESS_FIELDS = {
    "protocol", "at_utc", "event", "phase", "outer_completed", "outer_total",
    "api_calls", "cache_hits", "retries", "prompt_tokens", "completion_tokens",
    "estimated_spend_eur", "estimated_remaining_eur", "reserved_live_requests",
    "component", "reason", "attempt", "sleep_seconds", "retry_seconds",
    "continuous_429_seconds", "kind", "completed", "total",
    "cache_only", "remaining_live_documents", "reserved_requests_per_document",
    "selected_threshold",
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


def _latest_progress(run_root: Path) -> dict[str, Any] | None:
    candidates = [
        run_root / "docred-live" / "progress.json",
        run_root / "docred-replay" / "progress.json",
    ]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        return None
    path = max(existing, key=lambda item: item.stat().st_mtime)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"event": "progress_unreadable"}
    if not isinstance(payload, dict):
        return {"event": "progress_invalid"}
    return {key: payload[key] for key in SAFE_PROGRESS_FIELDS if key in payload}


def snapshot(run_root: Path, pid: int) -> dict[str, Any]:
    return {
        "protocol": "local-docred-monitor-v1",
        "checked_at_utc": _utc(),
        "runner_pid": pid,
        "runner_alive": _pid_alive(pid),
        "progress": _latest_progress(run_root),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _material_key(payload: dict[str, Any]) -> tuple[Any, ...]:
    progress = payload.get("progress") or {}
    return (
        payload["runner_alive"], progress.get("event"), progress.get("phase"),
        progress.get("outer_completed"), progress.get("outer_total"),
        progress.get("completed"), progress.get("total"), progress.get("retries"),
        progress.get("continuous_429_seconds"), progress.get("estimated_spend_eur"),
    )


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
    previous: tuple[Any, ...] | None = None
    while True:
        payload = snapshot(root, args.pid)
        stamp = payload["checked_at_utc"].replace(":", "-")
        _write_json(snapshots / f"{stamp}.json", payload)
        _write_json(snapshots / "latest.json", payload)
        material = _material_key(payload)
        if material != previous:
            print("[local-docred-monitor] " + json.dumps(payload, sort_keys=True), flush=True)
            previous = material
        if args.once or not payload["runner_alive"]:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
