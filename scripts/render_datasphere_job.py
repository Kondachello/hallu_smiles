#!/usr/bin/env python3
"""Render a commit-pinned DataSphere Job YAML from a checked-in template."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = {
    "preflight": ROOT / "datasphere/jobs/preflight-shared-assets.template.yaml",
    "qa-pilot-g1": ROOT / "datasphere/jobs/qa-pilot-g1.template.yaml",
}
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=sorted(TEMPLATES), required=True)
    parser.add_argument("--commit", required=True, help="Lowercase Git SHA to clone and verify.")
    parser.add_argument("--run-id", required=True, help="Lowercase, dash-separated output/job suffix.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model ID selected for this Job.")
    args = parser.parse_args()

    if not COMMIT_RE.fullmatch(args.commit):
        raise SystemExit("--commit must be a lowercase full 40-character hexadecimal Git SHA")
    if not RUN_ID_RE.fullmatch(args.run_id):
        raise SystemExit("--run-id must match [a-z0-9][a-z0-9-]{0,47}")
    if not MODEL_ID_RE.fullmatch(args.model_id):
        raise SystemExit("--model-id must look like an organization/model Hugging Face ID")
    rendered = TEMPLATES[args.kind].read_text(encoding="utf-8")
    rendered = (
        rendered.replace("__GIT_COMMIT__", args.commit)
        .replace("__RUN_ID__", args.run_id)
        .replace("__MODEL_ID__", args.model_id)
    )
    if "__" in rendered:
        raise RuntimeError("unresolved placeholder in Job template")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
