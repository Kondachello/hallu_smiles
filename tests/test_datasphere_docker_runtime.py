from __future__ import annotations

import json
import subprocess

import pytest

from scripts.check_datasphere_docker_runtime import (
    _check_native_build_toolchain,
    _inspection_payload,
)


def _completed(stdout: str, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["python", "-c", "inspection"],
        returncode=0,
        stdout=stdout,
        stderr=stderr,
    )


def test_inspection_payload_ignores_dependency_logs_on_stdout():
    payload = {
        "versions": {"vllm": "0.8.5.post1+cu118"},
        "modules": ["torch", "vllm"],
        "torch_cuda": "11.8",
    }
    completed = _completed(
        "INFO platform detection was written to stdout\n"
        + json.dumps(payload, sort_keys=True)
        + "\n"
    )

    assert _inspection_payload(completed) == payload


def test_inspection_payload_reports_captured_output_when_payload_is_missing():
    completed = _completed("INFO only\n", "warning from imported dependency\n")

    with pytest.raises(RuntimeError, match="no valid JSON payload") as raised:
        _inspection_payload(completed)

    assert "INFO only" in str(raised.value)
    assert "warning from imported dependency" in str(raised.value)


def test_native_build_toolchain_reports_gcc_and_python_header(monkeypatch):
    completed = _completed(
        json.dumps(
            {
                "python_include": "/usr/include/python3.11",
                "python_header": "/usr/include/python3.11/Python.h",
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        "scripts.check_datasphere_docker_runtime.shutil.which",
        lambda executable: "/usr/bin/gcc" if executable == "gcc" else None,
    )
    monkeypatch.setattr(
        "scripts.check_datasphere_docker_runtime.subprocess.run",
        lambda *args, **kwargs: completed,
    )

    assert _check_native_build_toolchain("/opt/hallu/server/bin/python") == {
        "gcc": "/usr/bin/gcc",
        "python_include": "/usr/include/python3.11",
        "python_header": "/usr/include/python3.11/Python.h",
    }


def test_native_build_toolchain_fails_without_gcc(monkeypatch):
    monkeypatch.setattr(
        "scripts.check_datasphere_docker_runtime.shutil.which", lambda executable: None
    )

    with pytest.raises(RuntimeError, match="missing gcc"):
        _check_native_build_toolchain("/opt/hallu/server/bin/python")
