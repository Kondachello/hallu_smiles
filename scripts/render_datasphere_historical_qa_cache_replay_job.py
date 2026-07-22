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
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", args.run_id):
        raise SystemExit("invalid commit or run id")
    gateway = urlparse(args.gateway_url)
    if gateway.scheme != "https" or not gateway.netloc or gateway.path.rstrip("/") or gateway.query or gateway.fragment:
        raise SystemExit("--gateway-url must be an HTTPS origin")
    image = require_runtime_image(args.docker_image, registry=True)
    rendered = TEMPLATE.read_text(encoding="utf-8")
    rendered = (rendered.replace("__GIT_COMMIT__", args.commit).replace("__RUN_ID__", args.run_id)
        .replace("__GATEWAY_URL__", f"https://{gateway.netloc}").replace("__DOCKER_IMAGE__", image)
        .replace("__DOCKER_ENV_BLOCK__", f"  docker:\n    image: {image}"))
    if "__" in rendered:
        raise RuntimeError("unresolved Job template placeholder")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
