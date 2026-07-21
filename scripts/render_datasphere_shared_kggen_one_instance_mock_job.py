#!/usr/bin/env python3
"""Render the CPU-only two-pass shared-KGGen one-response mock Job."""
from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

from datasphere_runtime_image import require_runtime_image


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "datasphere/jobs/shared-kggen-one-instance-mock.template.yaml"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--response-id", required=True)
    parser.add_argument("--docker-image", required=True, help="OCI image pinned by sha256 digest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not COMMIT_RE.fullmatch(args.commit):
        raise SystemExit("--commit must be a lowercase full 40-character Git SHA")
    if not RUN_ID_RE.fullmatch(args.run_id):
        raise SystemExit("--run-id must match [a-z0-9][a-z0-9-]{0,47}")
    response_id = str(args.response_id)
    if not response_id or "\x00" in response_id or len(response_id.encode("utf-8")) > 512:
        raise SystemExit("--response-id must be a non-empty UTF-8 value of at most 512 bytes")
    try:
        image = require_runtime_image(args.docker_image, registry=True)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    cache_namespace = f"controlled-shared-kggen-mock-v1/{args.commit}/{response_id}"
    rendered = TEMPLATE.read_text(encoding="utf-8")
    rendered = (rendered.replace("__GIT_COMMIT__", args.commit)
                .replace("__RUN_ID__", args.run_id)
                .replace("__RESPONSE_ID_B64__", base64.b64encode(response_id.encode("utf-8")).decode("ascii"))
                .replace("__CACHE_NAMESPACE__", cache_namespace)
                .replace("__DOCKER_IMAGE__", image)
                .replace("__DOCKER_ENV_BLOCK__", f"  docker:\n    image: {image}"))
    if "__" in rendered:
        raise RuntimeError("unresolved Job template placeholder")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
