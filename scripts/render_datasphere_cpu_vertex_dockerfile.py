#!/usr/bin/env python3
"""Render the CPU-only DataSphere Vertex image Dockerfile from a pinned commit."""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "datasphere/docker/Dockerfile.cpu-vertex.template"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--branch", default="new-metrics")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--skip-pushed-check", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not COMMIT_RE.fullmatch(args.commit):
        raise SystemExit("--commit must be a lowercase full 40-character Git SHA")
    if not args.skip_pushed_check:
        subprocess.run(["git", "fetch", "--quiet", "origin", f"refs/heads/{args.branch}"], cwd=args.repo_root, check=True)
        if subprocess.run(["git", "merge-base", "--is-ancestor", args.commit, "FETCH_HEAD"], cwd=args.repo_root).returncode:
            raise SystemExit(f"commit {args.commit} is not published in origin/{args.branch}")
    rendered = TEMPLATE.read_text(encoding="utf-8").replace("__GIT_COMMIT__", args.commit)
    if "__GIT_COMMIT__" in rendered:
        raise RuntimeError("unresolved Dockerfile placeholder")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
