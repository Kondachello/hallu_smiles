#!/usr/bin/env python3
"""Render the read-only checkpoint-listing diagnostic Job (no gateway call, no secret)."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from datasphere_runtime_image import require_runtime_image

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "datasphere/jobs/checkpoint-listing-probe.template.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--docker-image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", args.run_id):
        raise SystemExit("invalid run id")
    image = require_runtime_image(args.docker_image, registry=True)
    rendered = TEMPLATE.read_text(encoding="utf-8")
    rendered = (
        rendered.replace("__RUN_ID__", args.run_id)
        .replace("__DOCKER_IMAGE__", image)
        .replace("__DOCKER_ENV_BLOCK__", f"  docker:\n    image: {image}")
    )
    if "__" in rendered:
        raise RuntimeError("unresolved Job template placeholder")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
