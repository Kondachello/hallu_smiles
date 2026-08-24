#!/usr/bin/env python3
"""Fetch an authenticated gateway manifest without writing the bearer to disk."""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bearer = os.environ.get("HALLU_GATEWAY_API_KEY", "")
    if not bearer:
        raise SystemExit("HALLU_GATEWAY_API_KEY is not set")
    url = args.gateway_url.rstrip("/") + "/v1/hallu/manifest"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {bearer}"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("authenticated gateway manifest request failed") from exc
    if not isinstance(payload, dict):
        raise SystemExit("gateway manifest is not a JSON object")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
