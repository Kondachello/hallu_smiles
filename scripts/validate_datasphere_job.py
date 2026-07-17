#!/usr/bin/env python3
"""Validate a rendered DataSphere Job before submitting it to the service.

The DataSphere CLI parses ``cmd`` before it creates a Job.  Catch shell/YAML
errors and accidental fallback to a paid manual environment on the laptop,
before allocating a CPU or GPU instance.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import yaml


SHELL_PREFIX = "bash -lc '"
SHELL_VAR = re.compile(r"\$\{([^}]+)\}")
ALLOWED_DATASPHERE_VARS = {"ARTIFACT_ARCHIVE"}
DOCKER_IMAGE_ID_RE = re.compile(r"^b[a-z0-9]{19}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_job(path: Path, repo_root: Path) -> dict:  # noqa: ARG001 - stable public helper
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {path}") from exc
    _require(isinstance(document, dict), "Job YAML must be a mapping")

    command = document.get("cmd")
    _require(isinstance(command, str), "Job YAML has no string cmd")
    _require(
        command.startswith(SHELL_PREFIX) and command.endswith("'"),
        "cmd must use a bash -lc wrapper; DataSphere cannot submit a command beginning with set",
    )
    body = command[len(SHELL_PREFIX):-1]
    unknown_vars = sorted(set(SHELL_VAR.findall(body)) - ALLOWED_DATASPHERE_VARS)
    _require(
        not unknown_vars,
        "Use $VAR (not ${VAR}) for shell variables in Job YAML; "
        f"DataSphere would parse these as undeclared variables: {', '.join(unknown_vars)}",
    )
    subprocess.run(["bash", "-n", "-c", body], check=True)

    flags = document.get("flags", [])
    _require("attach-project-disk" in flags, "Job must attach the shared Project storage")
    env = document.get("env")
    _require(isinstance(env, dict), "Job must define env.docker")
    _require("python" not in env, "Docker Job must not also define env.python")
    docker_image_id = env.get("docker")
    _require(
        isinstance(docker_image_id, str) and DOCKER_IMAGE_ID_RE.fullmatch(docker_image_id),
        "env.docker must be an immutable DataSphere project Docker resource ID",
    )
    _require(
        f'DATASPHERE_DOCKER_IMAGE_ID="{docker_image_id}"' in body,
        "Job must record its DataSphere Docker resource ID in the runtime metadata",
    )
    if "g1.1" in document.get("cloud-instance-types", []):
        _require(
            re.search(r'EXPECTED_SOURCE_COMMIT="[0-9a-f]{40}"', body) is not None,
            "GPU Job must require the Docker runtime to match its exact Git commit",
        )

    lowered = body.lower()
    _require("huggingface-cli download" not in lowered, "GPU/CPU Job must not download model weights")
    _require("pip install" not in lowered, "GPU/CPU Job must not install packages at runtime")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Rendered Job YAML, never a template with placeholders.")
    parser.add_argument("--repo-root", default=".", help="Repository root that contains requirements files.")
    args = parser.parse_args()

    path = Path(args.job).resolve()
    repo_root = Path(args.repo_root).resolve()
    document = validate_job(path, repo_root)
    print(f"[ok] {path.name}: {document.get('name')} is safe for DataSphere CLI submission")


if __name__ == "__main__":
    main()
