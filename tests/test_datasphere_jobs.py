"""Offline contracts for shared-asset DataSphere plumbing."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
REVISION = "a" * 40


def _shared_assets(tmp_path: Path) -> tuple[Path, Path, Path]:
    shared = tmp_path / "shared"
    model = shared / "models" / "meta-llama-meta-llama-3-1-8b-instruct" / REVISION
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    files = [
        {"path": "config.json", "bytes": (model / "config.json").stat().st_size},
        {
            "path": "model-00001-of-00001.safetensors",
            "bytes": (model / "model-00001-of-00001.safetensors").stat().st_size,
        },
    ]
    (model / "model-manifest.json").write_text(json.dumps({
        "model_id": MODEL_ID, "revision": REVISION, "files": files,
    }), encoding="utf-8")
    (model / ".hallu_smiles_model_ready").write_text("ready\n", encoding="utf-8")
    active = model.parent / "active-model.json"
    active.write_text(json.dumps({
        "model_id": MODEL_ID, "revision": REVISION, "model_dir": REVISION,
    }), encoding="utf-8")

    data = shared / "ragtruth"
    data.mkdir()
    for name in ("source_info.jsonl", "response.jsonl"):
        (data / name).write_text('{"id": 1}\n', encoding="utf-8")
    (data / "ragtruth-manifest.json").write_text(json.dumps({"files": [
        {"path": name, "bytes": (data / name).stat().st_size}
        for name in ("source_info.jsonl", "response.jsonl")
    ]}), encoding="utf-8")
    return shared, model, data


def test_shared_asset_check_and_active_model_resolution(tmp_path):
    shared, model, data = _shared_assets(tmp_path)
    report = tmp_path / "report.json"
    checked = subprocess.run([
        sys.executable, str(SCRIPTS / "check_datasphere_shared_assets.py"),
        "--model-path", str(model), "--data-dir", str(data),
        "--model-id", MODEL_ID, "--report", str(report),
    ], check=True, text=True, capture_output=True)
    assert json.loads(checked.stdout)["model_revision"] == REVISION
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "ready"

    resolved = subprocess.run([
        sys.executable, str(SCRIPTS / "resolve_datasphere_shared_model.py"),
        "--shared-root", str(shared), "--model-id", MODEL_ID,
    ], check=True, text=True, capture_output=True)
    assert Path(resolved.stdout.strip()) == model


def test_runtime_config_keeps_every_mutable_path_in_job_work_dir(tmp_path):
    output = tmp_path / "runtime.yaml"
    work_dir = tmp_path / "job-output"
    subprocess.run([
        sys.executable, str(SCRIPTS / "make_datasphere_runtime_config.py"),
        "--base-config", str(ROOT / "config.yaml"), "--output", str(output),
        "--model-id", MODEL_ID, "--api-base", "http://127.0.0.1:8000/v1",
        "--data-dir", "/read-only/ragtruth", "--work-dir", str(work_dir),
    ], check=True)
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert config["data"]["dir"] == "/read-only/ragtruth"
    assert config["cache_dir"] == str(work_dir / "cache" / "kg")
    assert config["relation_verifier"]["cache_dir"] == str(work_dir / "cache" / "verdicts")


def test_gpu_job_template_is_pinned_and_has_no_gpu_time_download_or_pip(tmp_path):
    rendered = tmp_path / "qa-pilot.yaml"
    subprocess.run([
        sys.executable, str(SCRIPTS / "render_datasphere_job.py"),
        "--kind", "qa-pilot-g1", "--commit", "f" * 40,
        "--run-id", "new-metrics-20260716", "--output", str(rendered),
    ], check=True)
    text = rendered.read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    assert "__" not in text
    assert config["cloud-instance-types"] == ["g1.1"]
    assert config["working-storage"]["size"] == "100Gb"
    assert "timeout --signal=TERM --kill-after=60s 10800" in config["cmd"]
    assert config["outputs"] == [{"qa-pilot-new-metrics-20260716.tar.gz": "ARTIFACT_ARCHIVE"}]

    runner = (SCRIPTS / "run_datasphere_qa_pilot.sh").read_text(encoding="utf-8")
    assert "huggingface-cli download" not in runner
    assert "pip install" not in runner
    assert "--relation-mode strict" in runner
    assert "--relation-mode support" in runner
    subprocess.run(["bash", "-n", str(SCRIPTS / "run_datasphere_qa_pilot.sh")], check=True)


def test_stager_defaults_to_gated_llama_and_uses_model_specific_storage():
    text = (SCRIPTS / "stage_datasphere_shared_assets.py").read_text(encoding="utf-8")
    assert 'MODEL_ID_DEFAULT = "meta-llama/Meta-Llama-3.1-8B-Instruct"' in text
    assert 'shared_root / "models" / _model_family(model_id)' in text
    assert "optional for public models" in text
