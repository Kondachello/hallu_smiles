#!/usr/bin/env python3
"""Validate a rendered CPU API Job before DataSphere CLI submission."""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import yaml


SHELL_PREFIX = "bash -lc '"
SHELL_VAR = re.compile(r"\$\{([^}]+)\}")
COMMIT_RE = re.compile(r'EXPECTED_SOURCE_COMMIT="([0-9a-f]{40})"')
REQUIRED_PINS = {
    "torch": "2.6.0",
    "kg-gen": "0.4.0",
    "dspy": "3.2.1",
    "litellm": "1.91.1",
    "sentence-transformers": "5.6.0",
}
FORBIDDEN = (
    "g1.1",
    "vllm",
    "cuda",
    "hf_token",
    "huggingface-cli",
    "shared/models",
    "--model-id",
    "qwen3-8b",
    "dashscope-intl.aliyuncs.com",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_requirements(repo_root: Path) -> dict[str, str]:
    path = repo_root / "requirements.datasphere.api.txt"
    _require(path.is_file(), "requirements.datasphere.api.txt is missing")
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("--"):
            _require(
                line == "--extra-index-url https://download.pytorch.org/whl/cpu",
                f"unapproved pip option in DataSphere requirements: {line}",
            )
            continue
        _require("==" in line, f"DataSphere dependency is not exactly pinned: {line}")
        name, version = line.split("==", 1)
        _require(bool(name and version), f"invalid dependency pin: {line}")
        pins[name.lower()] = version
    for package, version in REQUIRED_PINS.items():
        _require(pins.get(package) == version, f"{package} must be pinned to {version}")
    _require(
        "--extra-index-url https://download.pytorch.org/whl/cpu"
        in path.read_text(encoding="utf-8").splitlines(),
        "DataSphere runtime must use the official PyTorch CPU wheel index",
    )
    return pins


def validate_job(path: Path, repo_root: Path) -> dict:
    try:
        raw = path.read_text(encoding="utf-8")
        document = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {path}") from exc
    _require(isinstance(document, dict), "Job YAML must be a mapping")
    _require("__" not in raw, "rendered Job contains an unresolved placeholder")

    name = document.get("name")
    _require(isinstance(name, str), "Job must have a string name")
    if name.startswith("hallu-api-probe-c1-"):
        kind = "api-probe-c1"
        expected_mode = "probe"
        expected_seconds = "7200"
        _require("inputs" not in document, "probe Job must not accept a gate artifact")
    elif name.startswith("hallu-api-pilot-c1-"):
        kind = "api-pilot-c1"
        expected_mode = "pilot"
        expected_seconds = "43200"
        inputs = document.get("inputs")
        _require(
            isinstance(inputs, list)
            and len(inputs) == 1
            and isinstance(inputs[0], dict)
            and list(inputs[0].values()) == ["PROBE_ARTIFACT"],
            "pilot Job must have exactly one PROBE_ARTIFACT input",
        )
    else:
        raise ValueError("only hallu-api-probe-c1 and hallu-api-pilot-c1 Jobs are allowed")

    command = document.get("cmd")
    _require(isinstance(command, str), "Job YAML has no string cmd")
    _require(
        command.startswith(SHELL_PREFIX) and command.endswith("'"),
        "cmd must use one bash -lc wrapper for DataSphere CLI",
    )
    body = command[len(SHELL_PREFIX) : -1]
    _require("'" not in body, "single quotes inside the bash -lc body are unsafe")
    unknown_vars = sorted(set(SHELL_VAR.findall(body)))
    _require(not unknown_vars, f"use shell $VAR form; undeclared DataSphere variables: {unknown_vars}")
    subprocess.run(["bash", "-n", "-c", body], check=True)

    _require(document.get("cloud-instance-types") == ["c1.4"], "API Jobs must use only c1.4")
    _require(document.get("flags") == ["attach-project-disk"], "Job must attach project storage")
    _require("working-storage" not in document, "API Jobs do not need extended working storage")
    env = document.get("env")
    _require(isinstance(env, dict) and set(env) == {"python"}, "Job must use only env.python")
    python_env = env["python"]
    _require(
        python_env
        == {
            "type": "manual",
            "version": "3.11",
            "requirements-file": "requirements.datasphere.api.txt",
            "local-paths": ["scripts"],
        },
        "Job must use the pinned Python 3.11 API environment",
    )
    _load_requirements(repo_root)

    lowered = raw.lower()
    for token in FORBIDDEN:
        _require(token not in lowered, f"forbidden local-model/GPU setting in API Job: {token}")
    _require("pip install" not in lowered, "Job must not install packages inside cmd")
    _require("https://github.com/kondachello/hallu_smiles.git" in lowered, "Job must fetch the public repository")
    _require("git -c source fetch --depth 1 origin" in lowered, "Job must fetch only the selected commit")
    _require(COMMIT_RE.search(body) is not None, "Job must record a full expected source commit")
    _require('test "$actual_commit" = ' in body, "Job must verify the checked-out commit")
    _require("printenv DASHSCOPE_API_KEY" in body, "Job must fail closed when the project secret is absent")
    _require(
        f"run_datasphere_api_job.sh --mode {expected_mode}" in body,
        f"Job must invoke the API runner in {expected_mode} mode",
    )
    _require(f"timeout --signal=TERM --kill-after=60s {expected_seconds}" in body, "wrong Job timeout")
    _require("$DS_PROJECT_HOME/hallu_smiles/shared/ragtruth" in body, "Job must read shared RAGTruth")
    if kind == "api-pilot-c1":
        _require('--probe-artifact "$PROBE_ARTIFACT"' in body, "pilot must pass its probe artifact")

    outputs = document.get("outputs")
    _require(isinstance(outputs, list) and len(outputs) == 1, "Job must publish exactly one archive")
    _require(list(outputs[0].values()) == ["ARTIFACT_ARCHIVE"], "archive output variable is invalid")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    try:
        document = validate_job(args.job.resolve(), args.repo_root.resolve())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"[ok] {args.job.name}: {document['name']} is safe for CPU API submission")


if __name__ == "__main__":
    main()
