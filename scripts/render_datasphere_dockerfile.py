#!/usr/bin/env python3
"""Render the remote DataSphere Docker build from a pushed Git commit."""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "datasphere" / "docker" / "Dockerfile.template"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--branch", default="new-metrics")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument(
        "--skip-pushed-check",
        action="store_true",
        help="tests only: render without proving that origin/BRANCH contains COMMIT",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not COMMIT_RE.fullmatch(args.commit):
        raise SystemExit("--commit must be a lowercase full 40-character Git SHA")
    if not args.skip_pushed_check:
        repo = Path(args.repo_root).resolve()
        subprocess.run(
            ["git", "fetch", "--quiet", "origin", f"refs/heads/{args.branch}"],
            cwd=repo,
            check=True,
        )
        contained = subprocess.run(
            ["git", "merge-base", "--is-ancestor", args.commit, "FETCH_HEAD"],
            cwd=repo,
            check=False,
        )
        if contained.returncode != 0:
            raise SystemExit(
                f"commit {args.commit} is not published in origin/{args.branch}; "
                "push it before rendering the remote image"
            )
    rendered = TEMPLATE.read_text(encoding="utf-8").replace("__GIT_COMMIT__", args.commit)
    if "__GIT_COMMIT__" in rendered:
        raise RuntimeError("unresolved Dockerfile placeholder")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
