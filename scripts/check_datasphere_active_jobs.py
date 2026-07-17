#!/usr/bin/env python3
"""Fail closed if another HalluGraph GPU Job is non-terminal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


TERMINAL = {"SUCCESS", "ERROR", "CANCELLED"}
GPU_PREFIXES = ("hallu-cluster-probe-g1-", "hallu-qa-g1-")


def active_gpu_jobs(payload: object) -> list[dict]:
    if payload is None:
        rows: list[object] = []
    elif isinstance(payload, dict):
        rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("DataSphere job list must be a JSON object, array, or null")
    active: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("DataSphere job list contains a non-object row")
        name = str(row.get("name", ""))
        if not name.startswith(GPU_PREFIXES):
            continue
        raw_status = str(row.get("status", ""))
        status = raw_status.removeprefix("JOB_STATUS_")
        if status not in TERMINAL:
            active.append({"id": row.get("id"), "name": name, "status": raw_status or "UNKNOWN"})
    return active


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-json", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.jobs_json).read_text(encoding="utf-8"))
    active = active_gpu_jobs(payload)
    if active:
        raise SystemExit(
            "refusing to create a second HalluGraph GPU Job; active jobs: "
            + json.dumps(active, sort_keys=True)
        )
    print(json.dumps({"status": "ready", "active_hallu_gpu_jobs": []}, sort_keys=True))


if __name__ == "__main__":
    main()
