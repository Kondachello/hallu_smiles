#!/usr/bin/env python3
"""Write auditable live/cache-only usage counts for the CPU Vertex probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _usage(path: Path) -> dict[str, Any]:
    """Reduce a UsageLogger JSONL stream without exposing any prompt content."""
    records: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"invalid usage record in {path}")
            records.append(payload)
    if not records:
        return {
            "usage_file": str(path),
            "api_calls": 0,
            "requests_total": 0,
            "cache_hits": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
    final = records[-1]
    return {
        "usage_file": str(path),
        "api_calls": int(final.get("cum_calls", 0)),
        "requests_total": int(final.get("cum_requests", 0)),
        "cache_hits": int(final.get("cum_cache_hits", 0)),
        "prompt_tokens": int(final.get("cum_prompt_tokens", 0)),
        "completion_tokens": int(final.get("cum_completion_tokens", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kggen-probe", required=True)
    parser.add_argument("--verifier-probe", required=True)
    parser.add_argument("--live-usage", required=True)
    parser.add_argument("--replay-usage", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    kggen_probe = json.loads(Path(args.kggen_probe).read_text(encoding="utf-8"))
    verifier_probe = json.loads(Path(args.verifier_probe).read_text(encoding="utf-8"))
    live = _usage(Path(args.live_usage))
    replay = _usage(Path(args.replay_usage))
    verifier_replay = verifier_probe.get("cache_only_usage", {})
    if replay["api_calls"] != 0:
        raise RuntimeError("cache-only extraction replay recorded live inference calls")
    if int(verifier_replay.get("api_calls", -1)) != 0:
        raise RuntimeError("cache-only verifier replay recorded live inference calls")
    payload = {
        "status": "ready",
        "kggen_schema_probe": kggen_probe.get("usage", {}),
        "verifier_live": verifier_probe.get("live_usage", {}),
        "verifier_cache_only": verifier_replay,
        "extraction_live": live,
        "extraction_cache_only": replay,
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
