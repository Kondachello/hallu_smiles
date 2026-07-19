"""Offline contracts for the parallel CPU DataSphere Vertex profile."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
IMAGE = "ghcr.io/kondachello/hallu-smiles-datasphere-vertex-cpu@sha256:" + "a" * 64


def _manifest() -> dict:
    return {
        "protocol": "hallu-vertex-openai-gateway-v1",
        "api_path": "/v1",
        "logical_model": "openai/gemini-2.5-flash",
        "vertex_model": "gemini-2.5-flash",
        "vertex_location": "europe-west4",
        "gateway_release": "git:deadbeef",
        "cloud_run_revision": "hallu-00001-abc",
    }


def test_vertex_runtime_config_derives_identity_from_authenticated_manifest(tmp_path):
    manifest = tmp_path / "gateway.json"
    runtime = tmp_path / "runtime.json"
    output = tmp_path / "runtime.yaml"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    runtime.write_text(json.dumps({"runtime_fingerprint": "cpu-image-fingerprint"}), encoding="utf-8")
    subprocess.run([
        sys.executable, str(SCRIPTS / "make_datasphere_vertex_config.py"),
        "--base-config", str(ROOT / "config.yaml"), "--gateway-manifest", str(manifest),
        "--gateway-url", "https://gateway.example.run.app", "--datasphere-runtime-manifest", str(runtime),
        "--output", str(output), "--data-dir", "/read-only/ragtruth", "--work-dir", str(tmp_path / "work"),
    ], check=True)
    cfg = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert cfg["llm"]["model"] == "openai/gemini-2.5-flash"
    assert cfg["llm"]["api_base"] == "https://gateway.example.run.app/v1"
    assert cfg["llm"]["api_key_env"] == "HALLU_GATEWAY_API_KEY"
    assert cfg["llm"]["structured_output_backend"] == "vertex"
    assert cfg["llm"]["structured_output_request_backend"] is None
    assert cfg["llm"]["concurrency"] == 2
    assert cfg["extraction"]["serial_chunking"] is False
    assert cfg["cache_dir"] == str(tmp_path / "work" / "cache" / "kg")
    assert cfg["vertex_gateway"]["gateway_manifest"] == _manifest()


def test_vertex_config_rejects_model_or_region_drift(tmp_path):
    manifest = _manifest()
    manifest["vertex_location"] = "us-central1"
    path = tmp_path / "bad.json"
    runtime = tmp_path / "runtime.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    runtime.write_text(json.dumps({"runtime_fingerprint": "cpu-image-fingerprint"}), encoding="utf-8")
    failed = subprocess.run([
        sys.executable, str(SCRIPTS / "make_datasphere_vertex_config.py"),
        "--base-config", str(ROOT / "config.yaml"), "--gateway-manifest", str(path),
        "--gateway-url", "https://gateway.example.run.app", "--datasphere-runtime-manifest", str(runtime),
        "--output", str(tmp_path / "out.yaml"), "--data-dir", "/data", "--work-dir", str(tmp_path / "work"),
    ], text=True, capture_output=True)
    assert failed.returncode != 0
    assert "vertex_location" in failed.stderr


def test_cpu_vertex_job_is_pinned_cpu_only_and_keeps_the_secret_out_of_yaml(tmp_path):
    rendered = tmp_path / "vertex-probe.yaml"
    subprocess.run([
        sys.executable, str(SCRIPTS / "render_datasphere_vertex_probe_job.py"),
        "--commit", "f" * 40, "--run-id", "vertex-3qa-20260718",
        "--gateway-url", "https://gateway.example.run.app", "--docker-image", IMAGE,
        "--output", str(rendered),
    ], check=True)
    text = rendered.read_text(encoding="utf-8")
    job = yaml.safe_load(text)
    assert job["cloud-instance-types"] == ["c1.4"]
    assert job["env"] == {"docker": {"image": IMAGE}}
    assert "HALLU_GATEWAY_URL" in text
    assert "HALLU_GATEWAY_API_KEY" not in text
    assert "vllm" not in text.lower()
    subprocess.run([sys.executable, str(SCRIPTS / "validate_datasphere_job.py"), "--job", str(rendered), "--repo-root", str(ROOT)], check=True)
    runner = (SCRIPTS / "run_datasphere_vertex_cpu_probe.sh").read_text(encoding="utf-8")
    assert "check_datasphere_gpu_runtime.py" not in runner
    assert "vllm" not in runner.lower()
    assert "--qa-pilot-limit 3" in runner
    assert "--cache-only" in runner
    assert "check_vertex_verifier_probe.py" in runner
    assert "usage-counts.json" in runner
    # Cloud Run's public frontend intercepts the reserved /healthz path;
    # the authenticated manifest is the runner's readiness probe instead.
    assert '"$HALLU_GATEWAY_URL/healthz"' not in runner
    assert '"$HALLU_GATEWAY_URL/v1/hallu/manifest"' in runner
    submitter = (SCRIPTS / "submit_datasphere_vertex_probe.sh").read_text(encoding="utf-8")
    assert 'GRPC_DNS_RESOLVER="${GRPC_DNS_RESOLVER:-ares}"' in submitter
    assert "--docker-image does not match the immutable runtime" in submitter
    assert "RESOLVED_IMAGE=" in submitter
    verifier_probe = (SCRIPTS / "check_vertex_verifier_probe.py").read_text(encoding="utf-8")
    assert 'live.verdict != "entailed"' not in verifier_probe
    assert '"failure_class"' in verifier_probe
    deploy = (SCRIPTS / "deploy_vertex_gateway.sh").read_text(encoding="utf-8")
    assert "artifacts repositories describe" in deploy


def test_cpu_dockerfile_is_pinned_and_has_no_llama_or_vllm(tmp_path):
    rendered = tmp_path / "Dockerfile"
    subprocess.run([
        sys.executable, str(SCRIPTS / "render_datasphere_cpu_vertex_dockerfile.py"),
        "--commit", "f" * 40, "--skip-pushed-check", "--output", str(rendered),
    ], check=True)
    text = rendered.read_text(encoding="utf-8").lower()
    assert "python:3.11-slim" in text
    assert "vllm" not in text
    assert "meta-llama" not in text
    assert "all-minilm-l6-v2" in text
