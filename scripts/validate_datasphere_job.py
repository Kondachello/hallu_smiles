#!/usr/bin/env python3
"""Validate a rendered DataSphere Job before submitting it to the service.

The DataSphere CLI parses ``cmd`` before it creates a Job.  A shell entrypoint
therefore has a few non-obvious constraints: it must start with an executable,
not ``set``; shell variables must not look like DataSphere substitutions; and a
manual Python environment needs explicit local paths.  Catch those errors on a
laptop, before allocating a CPU or GPU instance.
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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_requirements(path: Path) -> None:
    _require(path.is_file(), f"requirements file is absent: {path}")
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        # The DataSphere CLI's manual-environment parser accepts only direct
        # PEP 508 requirements; comments and -r inclusions fail before submit.
        _require(
            not line.startswith("#") and not line.startswith("-r"),
            f"{path}:{number}: DataSphere manual requirements must not use comments or -r includes",
        )


def _validate_gpu_requirements(path: Path) -> None:
    lines = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    _require(
        "vllm==0.6.3.post1" in lines,
        "g1.1 requires the driver-compatible vllm==0.6.3.post1 pin; do not use a floating vLLM range",
    )
    _require(
        "transformers==4.45.2" in lines,
        "vllm==0.6.3.post1 + lm-format-enforcer==0.10.6 require the tested transformers==4.45.2 pin",
    )
    _require(
        "lm-format-enforcer==0.10.6" in lines,
        "g1.1 vLLM runtime requires the pinned lm-format-enforcer guided-decoding backend",
    )
    _require(
        "pydantic==2.10.6" in lines,
        "KGGen/DSPy local-vLLM path requires the tested pydantic==2.10.6 pin",
    )


def _validate_preflight_requirements(path: Path, repo_root: Path) -> None:
    runtime_path = repo_root / "requirements.datasphere.txt"
    runtime_lines = [line.strip() for line in runtime_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    preflight_lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(
        preflight_lines == runtime_lines,
        "CPU preflight requirements must exactly match the GPU runtime requirements",
    )


def validate_job(path: Path, repo_root: Path) -> dict:
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
    python = document.get("env", {}).get("python", {})
    _require(isinstance(python, dict) and python.get("type") == "manual", "Job must use a manual Python environment")
    _require(python.get("local-paths"), "manual shell Job requires non-empty env.python.local-paths")
    requirements = python.get("requirements-file")
    _require(isinstance(requirements, str), "manual Job requires env.python.requirements-file")
    requirements_path = repo_root / requirements
    _validate_requirements(requirements_path)

    lowered = body.lower()
    _require("huggingface-cli download" not in lowered, "GPU/CPU Job must not download model weights")
    _require("pip install" not in lowered, "GPU/CPU Job must not install packages at runtime")
    if document.get("cloud-instance-types") == ["g1.1"]:
        _validate_gpu_requirements(requirements_path)
    if document.get("cloud-instance-types") == ["c1.4"]:
        _validate_preflight_requirements(requirements_path, repo_root)
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
