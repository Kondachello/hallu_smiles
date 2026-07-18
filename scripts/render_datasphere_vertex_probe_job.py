#!/usr/bin/env python3
"""Render the independent CPU Vertex 3-QA probe Job."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

from datasphere_runtime_image import require_runtime_image


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "datasphere/jobs/vertex-cpu-probe.template.yaml"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")


def _gateway_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.path.rstrip("/") or parsed.query or parsed.fragment:
        raise ValueError("--gateway-url must be an HTTPS Cloud Run origin without a path")
    return f"https://{parsed.netloc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--output", required=True)
    image = parser.add_mutually_exclusive_group(required=True)
    image.add_argument("--docker-image-id")
    image.add_argument("--docker-image")
    args = parser.parse_args()
    if not COMMIT_RE.fullmatch(args.commit):
        raise SystemExit("--commit must be a lowercase full 40-character Git SHA")
    if not RUN_ID_RE.fullmatch(args.run_id):
        raise SystemExit("--run-id must match [a-z0-9][a-z0-9-]{0,47}")
    try:
        gateway_url = _gateway_url(args.gateway_url)
        runtime_image = args.docker_image_id or args.docker_image
        require_runtime_image(runtime_image, registry=args.docker_image is not None)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    docker_block = f"  docker: {runtime_image}" if args.docker_image_id else f"  docker:\n    image: {runtime_image}"
    rendered = TEMPLATE.read_text(encoding="utf-8")
    rendered = (rendered.replace("__GIT_COMMIT__", args.commit).replace("__RUN_ID__", args.run_id)
        .replace("__GATEWAY_URL__", gateway_url).replace("__DOCKER_IMAGE_ID__", runtime_image)
        .replace("__DOCKER_ENV_BLOCK__", docker_block))
    if "__" in rendered:
        raise RuntimeError("unresolved Job template placeholder")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
