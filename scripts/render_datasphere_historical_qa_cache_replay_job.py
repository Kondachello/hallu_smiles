#!/usr/bin/env python3
"""Render the CPU-only historical 100-QA cache replay Job."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

from datasphere_runtime_image import require_runtime_image

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "datasphere/jobs/historical-qa-cache-replay.template.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True); parser.add_argument("--run-id", required=True)
    parser.add_argument("--gateway-url", required=True); parser.add_argument("--docker-image", required=True)
    parser.add_argument("--qa-sample-size", type=int, default=100,
                        help="total records in the target historical checkpoint (e.g. 100 or 750); "
                             "disambiguates between multiple checkpoints sharing one gateway manifest")
    parser.add_argument("--replay-count", default="5",
                        help="records to replay (1..qa-sample-size), or 'all' for every fully-cached record")
    parser.add_argument("--replay-selection-seed", type=int, default=20260722)
    parser.add_argument("--timeout-seconds", type=int, default=43200,
                        help="Job wall-time ceiling (default: 43200 = 12h; raise for large replay-count)")
    parser.add_argument("--diagnostic-only", action="store_true",
                        help="Skip the real replay; only report how many computed cache_keys "
                             "for the selected QA texts already exist in the historical cache root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", args.run_id):
        raise SystemExit("invalid commit or run id")
    gateway = urlparse(args.gateway_url)
    if gateway.scheme != "https" or not gateway.netloc or gateway.path.rstrip("/") or gateway.query or gateway.fragment:
        raise SystemExit("--gateway-url must be an HTTPS origin")
    image = require_runtime_image(args.docker_image, registry=True)
    if args.qa_sample_size < 1:
        raise SystemExit("--qa-sample-size must be a positive integer")
    if str(args.replay_count) != "all":
        try:
            replay_count_int = int(args.replay_count)
        except ValueError:
            raise SystemExit("--replay-count must be a positive integer or 'all'")
        if not 1 <= replay_count_int <= args.qa_sample_size:
            raise SystemExit("--replay-count must be between 1 and --qa-sample-size")
    if args.timeout_seconds < 60:
        raise SystemExit("--timeout-seconds must be at least 60")
    rendered = TEMPLATE.read_text(encoding="utf-8")
    rendered = (rendered.replace("__GIT_COMMIT__", args.commit).replace("__RUN_ID__", args.run_id)
        .replace("__GATEWAY_URL__", f"https://{gateway.netloc}").replace("__DOCKER_IMAGE__", image)
        .replace("__QA_SAMPLE_SIZE__", str(args.qa_sample_size))
        .replace("__REPLAY_COUNT__", str(args.replay_count)).replace("__REPLAY_SELECTION_SEED__", str(args.replay_selection_seed))  # noqa: E501 replay-count may be 'all'
        .replace("__JOB_TIMEOUT_SECONDS__", str(args.timeout_seconds))
        .replace("__DIAGNOSTIC_ONLY__", "1" if args.diagnostic_only else "0")
        .replace("__DOCKER_ENV_BLOCK__", f"  docker:\n    image: {image}"))
    if "__" in rendered:
        raise RuntimeError("unresolved Job template placeholder")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
