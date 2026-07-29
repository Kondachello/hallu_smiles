#!/usr/bin/env python3
"""Render the CPU historical QA typed-vertex metric-pass Job.

Same cache resolution as the cache-replay job, but runs the typed-vertex CFI
metric pass (typing agent assigns vertex types via gateway+HHEM; graphs stay
cache-only). Reuses the replay render logic; adds --alpha for CFI_type.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

from datasphere_runtime_image import require_runtime_image

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "datasphere/jobs/historical-qa-typed-metric-pass.template.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True,
                        help="commit to git-checkout (our code, may differ from the image build commit)")
    parser.add_argument("--image-source-commit", required=True,
                        help="commit baked into the pinned runtime image's /opt/hallu/runtime-manifest.json "
                             "(check_datasphere_vertex_cpu_runtime.py requires exact equality); pass --commit "
                             "here only if the image was actually rebuilt from that same commit")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--docker-image", required=True)
    parser.add_argument("--qa-sample-size", type=int, default=750,
                        help="total records in the target historical checkpoint (e.g. 750)")
    parser.add_argument("--replay-count", default="all",
                        help="records to score (1..qa-sample-size), or 'all'")
    parser.add_argument("--replay-selection-seed", type=int, default=20260722)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="CFI_type = alpha*EG_type + (1-alpha)*RP")
    parser.add_argument("--timeout-seconds", type=int, default=43200,
                        help="Job wall-time ceiling (default 12h; raise for typing 750 on CPU)")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="record-level typing concurrency (overlaps gateway+HHEM calls; "
                             "identical results, ~N x faster). 1 = sequential.")
    parser.add_argument("--typing-qps", type=float, default=0.0,
                        help="process-wide gateway request rate cap (req/sec) to stay under the "
                             "Vertex quota and avoid 429 storms. 0 = unlimited.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", args.run_id):
        raise SystemExit("invalid commit or run id")
    if not re.fullmatch(r"[0-9a-f]{40}", args.image_source_commit):
        raise SystemExit("invalid --image-source-commit")
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
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be in [0, 1]")
    if args.timeout_seconds < 60:
        raise SystemExit("--timeout-seconds must be at least 60")
    rendered = TEMPLATE.read_text(encoding="utf-8")
    rendered = (rendered.replace("__GIT_COMMIT__", args.commit)
        .replace("__IMAGE_SOURCE_COMMIT__", args.image_source_commit)
        .replace("__RUN_ID__", args.run_id)
        .replace("__GATEWAY_URL__", f"https://{gateway.netloc}").replace("__DOCKER_IMAGE__", image)
        .replace("__QA_SAMPLE_SIZE__", str(args.qa_sample_size))
        .replace("__REPLAY_COUNT__", str(args.replay_count))
        .replace("__REPLAY_SELECTION_SEED__", str(args.replay_selection_seed))
        .replace("__TYPED_METRIC_ALPHA__", str(args.alpha))
        .replace("__TYPED_METRIC_MAX_WORKERS__", str(args.max_workers))
        .replace("__TYPED_METRIC_QPS__", str(args.typing_qps))
        .replace("__JOB_TIMEOUT_SECONDS__", str(args.timeout_seconds))
        .replace("__DOCKER_ENV_BLOCK__", f"  docker:\n    image: {image}"))
    if "__" in rendered:
        raise RuntimeError("unresolved Job template placeholder")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
