#!/usr/bin/env python3
"""Render a parameterized CPU Vertex strict/support/support-critical QA Job."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from datasphere_runtime_image import require_runtime_image
from render_datasphere_vertex_probe_job import _gateway_url


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TEMPLATE = ROOT / "datasphere/jobs/vertex-cpu-qa-pilot.template.yaml"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

from src.sampling import qa_sample_quotas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--gateway-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qa-sample-size", type=int, default=20)
    parser.add_argument("--qa-test-fraction", default="0.2")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--force-cache-only",
        action="store_true",
        help="forbid all inference cache misses during a recovery-only Job",
    )
    parser.add_argument(
        "--exclude-source-id",
        action="append",
        default=[],
        help=(
            "explicit source-level quarantine recorded in the experiment audit; "
            "no KG or metric calls are made for it (repeatable)"
        ),
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=0,
        help=(
            "optional in-container wall-time ceiling; 0 lets DataSphere own the deadline "
            "so persistent caches can resume after platform interruption"
        ),
    )
    image = parser.add_mutually_exclusive_group(required=True)
    image.add_argument("--docker-image-id")
    image.add_argument("--docker-image")
    args = parser.parse_args()
    if not COMMIT_RE.fullmatch(args.commit):
        raise SystemExit("--commit must be a lowercase full 40-character Git SHA")
    if not RUN_ID_RE.fullmatch(args.run_id):
        raise SystemExit("--run-id must match [a-z0-9][a-z0-9-]{0,47}")
    if not SHA256_RE.fullmatch(args.gateway_manifest_sha256):
        raise SystemExit("--gateway-manifest-sha256 must be a lowercase SHA-256 hex digest")
    try:
        train_sources, _ = qa_sample_quotas(args.qa_sample_size, args.qa_test_fraction)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.cv_folds < 2 or args.cv_folds > train_sources:
        raise SystemExit("--cv-folds must be between 2 and the selected train source count")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be positive")
    excluded_source_ids = [str(source_id) for source_id in args.exclude_source_id]
    if any(not re.fullmatch(r"[A-Za-z0-9._:-]+", source_id) for source_id in excluded_source_ids):
        raise SystemExit("--exclude-source-id contains an invalid source ID")
    if len(set(excluded_source_ids)) != len(excluded_source_ids):
        raise SystemExit("--exclude-source-id cannot be repeated for the same source")
    if args.timeout_seconds < 0 or 0 < args.timeout_seconds < 60:
        raise SystemExit("--timeout-seconds must be 0 or at least 60")
    try:
        gateway_url = _gateway_url(args.gateway_url)
        runtime_image = args.docker_image_id or args.docker_image
        require_runtime_image(runtime_image, registry=args.docker_image is not None)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    docker_block = f"  docker: {runtime_image}" if args.docker_image_id else f"  docker:\n    image: {runtime_image}"
    run_command = (
        "bash source/scripts/run_datasphere_vertex_cpu_qa_pilot.sh"
        if args.timeout_seconds == 0
        else (
            "timeout --signal=TERM --kill-after=60s "
            f"{args.timeout_seconds} bash source/scripts/run_datasphere_vertex_cpu_qa_pilot.sh"
        )
    )
    rendered = TEMPLATE.read_text(encoding="utf-8")
    rendered = (rendered.replace("__GIT_COMMIT__", args.commit).replace("__RUN_ID__", args.run_id)
        .replace("__GATEWAY_URL__", gateway_url).replace("__DOCKER_IMAGE_ID__", runtime_image)
        .replace("__GATEWAY_MANIFEST_SHA256__", args.gateway_manifest_sha256)
        .replace("__QA_SAMPLE_SIZE__", str(args.qa_sample_size))
        .replace("__QA_TEST_FRACTION__", str(args.qa_test_fraction))
        .replace("__QA_CV_FOLDS__", str(args.cv_folds))
        .replace("__LLM_CONCURRENCY__", str(args.concurrency))
        .replace("__QA_FORCE_CACHE_ONLY__", "1" if args.force_cache_only else "0")
        .replace("__QA_EXCLUDE_SOURCE_IDS__", ",".join(excluded_source_ids))
        .replace("__JOB_RUN_COMMAND__", run_command)
        .replace("__DOCKER_ENV_BLOCK__", docker_block))
    if "__" in rendered:
        raise RuntimeError("unresolved Job template placeholder")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
