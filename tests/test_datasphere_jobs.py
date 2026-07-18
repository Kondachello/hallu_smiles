from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.render_datasphere_job import render_job
from scripts.validate_datasphere_job import validate_job


ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40


@pytest.mark.parametrize(
    ("kind", "mode", "seconds", "archive"),
    [
        ("api-probe-c1", "probe", "7200", "api-probe-run-one.tar.gz"),
        ("api-pilot-c1", "pilot", "43200", "api-pilot-run-one.tar.gz"),
    ],
)
def test_rendered_api_jobs_are_cpu_only_and_valid(
    tmp_path: Path, kind: str, mode: str, seconds: str, archive: str
) -> None:
    gate = tmp_path / "probe gate.tar.gz"
    gate.write_bytes(b"placeholder")
    output = tmp_path / f"{kind}.yaml"
    render_job(
        kind=kind,
        commit=SHA,
        run_id="run-one",
        output=output,
        gate_artifact=gate if kind == "api-pilot-c1" else None,
    )

    document = validate_job(output, ROOT)
    raw = output.read_text(encoding="utf-8")
    assert document["cloud-instance-types"] == ["c1.4"]
    assert document["env"]["python"]["version"] == "3.11"
    assert document["env"]["python"]["local-paths"] == ["scripts"]
    assert f"--mode {mode}" in document["cmd"]
    assert f"kill-after=60s {seconds}" in document["cmd"]
    assert list(document["outputs"][0]) == [archive]
    assert "DASHSCOPE_API_KEY" in raw
    for forbidden in ("g1.1", "vllm", "cuda", "hf_token", "docker", "--model-id"):
        assert forbidden not in raw.lower()
    assert "qwen" not in raw.lower(), "config.yaml must remain the sole model source"
    if kind == "api-pilot-c1":
        assert document["inputs"] == [{str(gate.resolve()): "PROBE_ARTIFACT"}]


def test_renderer_rejects_gate_on_probe_and_requires_it_on_pilot(tmp_path: Path) -> None:
    gate = tmp_path / "gate.tar.gz"
    gate.write_bytes(b"x")
    with pytest.raises(ValueError, match="only valid"):
        render_job(
            kind="api-probe-c1",
            commit=SHA,
            run_id="probe",
            output=tmp_path / "probe.yaml",
            gate_artifact=gate,
        )
    with pytest.raises(ValueError, match="requires --gate-artifact"):
        render_job(
            kind="api-pilot-c1",
            commit=SHA,
            run_id="pilot",
            output=tmp_path / "pilot.yaml",
        )


def test_validator_rejects_a_model_second_source(tmp_path: Path) -> None:
    output = tmp_path / "probe.yaml"
    render_job(kind="api-probe-c1", commit=SHA, run_id="probe", output=output)
    raw = output.read_text(encoding="utf-8").replace(
        "desc: Validate", "desc: qwen3-8b Validate"
    )
    output.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden"):
        validate_job(output, ROOT)


def test_datasphere_requirements_force_cpu_torch_and_exact_runtime_pins() -> None:
    lines = (ROOT / "requirements.datasphere.api.txt").read_text(encoding="utf-8").splitlines()
    assert all(not line.startswith("#") for line in lines)
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in lines
    assert "torch==2.6.0" in lines
    assert "kg-gen==0.4.0" in lines
    assert "dspy==3.2.1" in lines
    assert "litellm==1.91.1" in lines


def test_official_datasphere_parser_accepts_manual_environment(tmp_path: Path, monkeypatch) -> None:
    config_module = pytest.importorskip("datasphere.config")
    pyenv_module = pytest.importorskip("datasphere.pyenv")
    output = tmp_path / "probe.yaml"
    render_job(kind="api-probe-c1", commit=SHA, run_id="probe", output=output)
    monkeypatch.chdir(ROOT)
    parsed = config_module.parse_config(output)
    environment = pyenv_module.define_py_env(parsed.python_root_modules, parsed.env.python)
    assert environment.version == "3.11"
    assert "torch==2.6.0" in environment.requirements
    assert environment.local_modules_paths == ["scripts"]


def test_submit_script_is_bash_32_portable_and_has_no_model_argument() -> None:
    script = (ROOT / "scripts/submit_datasphere_job.sh").read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(ROOT / "scripts/submit_datasphere_job.sh")], check=True)
    assert "mapfile" not in script
    assert "readarray" not in script
    assert "--model-id" not in script
    assert 'archive_name="api-probe-${RUN_ID}.tar.gz"' in script
    assert 'archive_name="api-pilot-${RUN_ID}.tar.gz"' in script
    assert 'startswith("JOB_STATUS_")' in script


def _make_submit_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "datasphere/jobs").mkdir(parents=True)
    for source in (
        "scripts/submit_datasphere_job.sh",
        "scripts/render_datasphere_job.py",
        "scripts/validate_datasphere_job.py",
        "scripts/validate_api_probe_artifact.py",
        "datasphere/jobs/api-probe-c1.template.yaml",
        "datasphere/jobs/api-pilot-c1.template.yaml",
        "requirements.datasphere.api.txt",
        "config.yaml",
    ):
        destination = repo / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, destination)
    subprocess.run(["git", "init", "-b", "clean"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "remote", "add", "origin", str(remote)], cwd=repo, check=True)
    subprocess.run(["git", "push", "-u", "origin", "clean"], cwd=repo, check=True, capture_output=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    return repo, commit


def _write_fake_datasphere(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
import pathlib
import sys

args = sys.argv[1:]
state = pathlib.Path(os.environ["FAKE_DS_STATE"])
list_state = pathlib.Path(os.environ.get("FAKE_LIST_STATE", str(state) + ".lists"))
name = os.environ["FAKE_JOB_NAME"]
mode = os.environ["FAKE_DS_MODE"]

def output_path():
    flag = "-o" if "-o" in args else "--output"
    return pathlib.Path(args[args.index(flag) + 1])

if args[-3:] == ["project", "get", "--id"]:
    raise SystemExit(2)
if "project" in args and "get" in args and "job" not in args:
    if mode == "project-fail":
        raise SystemExit(17)
    print("ok")
elif "job" in args and "list" in args:
    list_count = int(list_state.read_text() or "0") if list_state.exists() else 0
    list_count += 1
    list_state.write_text(str(list_count), encoding="utf-8")
    accepted = state.exists() or mode == "existing"
    if mode == "eventual" and state.exists() and list_count <= 3:
        accepted = False
    jobs = ([{"id": "job-1", "name": name, "status": "JOB_STATUS_SUCCESS"}]
            if accepted else [])
    output_path().write_text(json.dumps(jobs), encoding="utf-8")
elif "job" in args and "execute" in args:
    count = int(state.read_text() or "0") if state.exists() else 0
    state.write_text(str(count + 1), encoding="utf-8")
    if mode in {"ambiguous", "eventual"}:
        print("grpc unavailable after server accepted request", file=sys.stderr)
        raise SystemExit(1)
    if mode == "malformed-success":
        print("success response lost its metadata", file=sys.stderr)
        raise SystemExit(0)
    output_path().write_text(json.dumps({"job_id": "job-1"}), encoding="utf-8")
else:
    raise SystemExit(f"unexpected fake datasphere call: {args}")
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


@pytest.mark.parametrize(
    ("mode", "execute_count"),
    [("existing", 0), ("ambiguous", 1), ("eventual", 1), ("malformed-success", 1)],
)
def test_submit_is_exact_name_idempotent_after_ambiguous_execute(
    tmp_path: Path, mode: str, execute_count: int
) -> None:
    repo, commit = _make_submit_repo(tmp_path)
    fake = tmp_path / "datasphere"
    _write_fake_datasphere(fake)
    state = tmp_path / "execute-count"
    run_id = "idempotent-test"
    env = {
        **os.environ,
        "FAKE_DS_STATE": str(state),
        "FAKE_LIST_STATE": str(tmp_path / "list-count"),
        "FAKE_DS_MODE": mode,
        "FAKE_JOB_NAME": f"hallu-api-probe-c1-{run_id}",
        "DATASPHERE_EXECUTE_RECONCILE_ATTEMPTS": "4",
        "DATASPHERE_RETRY_DELAY_SECONDS": "0",
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/submit_datasphere_job.sh",
            "--kind",
            "api-probe-c1",
            "--project-id",
            "project-1",
            "--branch",
            "clean",
            "--commit",
            commit,
            "--run-id",
            run_id,
            "--python",
            sys.executable,
            "--datasphere",
            str(fake),
            "--no-wait",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    actual = int(state.read_text()) if state.exists() else 0
    assert actual == execute_count
    assert "no duplicate execute request" in result.stdout if mode == "existing" else "Recovered" in result.stdout


def test_read_only_retry_does_not_mask_final_failure(tmp_path: Path) -> None:
    repo, commit = _make_submit_repo(tmp_path)
    fake = tmp_path / "datasphere"
    _write_fake_datasphere(fake)
    env = {
        **os.environ,
        "FAKE_DS_STATE": str(tmp_path / "state"),
        "FAKE_DS_MODE": "project-fail",
        "FAKE_JOB_NAME": "hallu-api-probe-c1-retry-test",
        "DATASPHERE_RETRY_DELAY_SECONDS": "0",
    }
    result = subprocess.run(
        [
            "bash", "scripts/submit_datasphere_job.sh",
            "--kind", "api-probe-c1",
            "--project-id", "project-1",
            "--branch", "clean",
            "--commit", commit,
            "--run-id", "retry-test",
            "--python", sys.executable,
            "--datasphere", str(fake),
            "--no-wait",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "project get failed after 5 attempts" in result.stderr


def test_submit_rejects_tracked_runtime_drift_from_selected_commit(tmp_path: Path) -> None:
    repo, commit = _make_submit_repo(tmp_path)
    fake = tmp_path / "datasphere"
    _write_fake_datasphere(fake)
    (repo / "requirements.datasphere.api.txt").write_text(
        (repo / "requirements.datasphere.api.txt").read_text(encoding="utf-8")
        + "unexpected-local-runtime==1.0\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "FAKE_DS_STATE": str(tmp_path / "state"),
        "FAKE_DS_MODE": "existing",
        "FAKE_JOB_NAME": "hallu-api-probe-c1-drift-test",
        "DATASPHERE_RETRY_DELAY_SECONDS": "0",
    }
    result = subprocess.run(
        [
            "bash", "scripts/submit_datasphere_job.sh",
            "--kind", "api-probe-c1",
            "--project-id", "project-1",
            "--branch", "clean",
            "--commit", commit,
            "--run-id", "drift-test",
            "--python", sys.executable,
            "--datasphere", str(fake),
            "--no-wait",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "Tracked local files differ from --commit" in result.stderr
