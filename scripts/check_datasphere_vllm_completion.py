#!/usr/bin/env python3
"""Fail fast unless the local vLLM OpenAI-compatible completion endpoint works."""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.port <= 0 or args.timeout <= 0:
        raise ValueError("--port and --timeout must be positive")

    payload = json.dumps({
        "model": args.model_id,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "temperature": 0.0,
        "max_tokens": 2,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local-datasphere-key"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM completion smoke check returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"vLLM completion smoke check could not reach the server: {exc}") from exc
    if not data.get("choices"):
        raise RuntimeError("vLLM completion smoke check returned no choices")
    print(json.dumps({"status": "ready", "model": args.model_id}))


if __name__ == "__main__":
    main()
