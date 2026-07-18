#!/usr/bin/env python3
"""Render one commit-pinned CPU DataSphere API Job from a checked-in template."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = {
    "api-probe-c1": ROOT / "datasphere/jobs/api-probe-c1.template.yaml",
    "api-pilot-c1": ROOT / "datasphere/jobs/api-pilot-c1.template.yaml",
}
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,43}$")


def render_job(
    *,
    kind: str,
    commit: str,
    run_id: str,
    output: Path,
    gate_artifact: Path | None = None,
) -> Path:
    """Render and write a Job, rejecting parameters DataSphere cannot safely parse."""
    if kind not in TEMPLATES:
        raise ValueError(f"unsupported Job kind: {kind}")
    if not COMMIT_RE.fullmatch(commit):
        raise ValueError("commit must be a lowercase full 40-character hexadecimal Git SHA")
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run-id must match [a-z0-9][a-z0-9-]{0,43}")
    if kind == "api-pilot-c1":
        if gate_artifact is None:
            raise ValueError("api-pilot-c1 requires --gate-artifact from a successful API probe")
        gate_artifact = gate_artifact.expanduser().resolve(strict=True)
        if not gate_artifact.is_file():
            raise ValueError("--gate-artifact must be a regular file")
        gate_value = json.dumps(str(gate_artifact), ensure_ascii=False)
    else:
        if gate_artifact is not None:
            raise ValueError("--gate-artifact is only valid for api-pilot-c1")
        gate_value = ""

    rendered = TEMPLATES[kind].read_text(encoding="utf-8")
    rendered = (
        rendered.replace("__GIT_COMMIT__", commit)
        .replace("__RUN_ID__", run_id)
        .replace("__GATE_ARTIFACT__", gate_value)
    )
    unresolved = sorted(set(re.findall(r"__[A-Z][A-Z0-9_]*__", rendered)))
    if unresolved:
        raise RuntimeError(f"unresolved placeholder(s): {', '.join(unresolved)}")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=sorted(TEMPLATES), required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-artifact", type=Path)
    args = parser.parse_args()
    try:
        output = render_job(
            kind=args.kind,
            commit=args.commit,
            run_id=args.run_id,
            output=args.output,
            gate_artifact=args.gate_artifact,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(output)


if __name__ == "__main__":
    main()
