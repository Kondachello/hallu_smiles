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
    assert cfg["llm"]["max_tokens"] == 4096
    assert cfg["llm"]["concurrency"] == 1
    assert cfg["llm"]["max_retries"] == 7
    assert cfg["llm"]["retry_backoff_base_s"] == 5.0
    assert cfg["llm"]["retry_backoff_max_s"] == 60.0
    assert cfg["eval"]["alpha_cv_folds"] == 5
    assert cfg["extraction"]["serial_chunking"] is False
    assert cfg["cache_dir"] == str(tmp_path / "work" / "cache" / "kg")
    assert cfg["support_critical"]["claim_extractor"]["cache_dir"] == str(
        tmp_path / "work" / "cache" / "critical_claims"
    )
    assert cfg["vertex_gateway"]["gateway_manifest"] == _manifest()


def test_vertex_runtime_config_keeps_historical_kg_reads_separate_from_writes(tmp_path):
    manifest = tmp_path / "gateway.json"
    runtime = tmp_path / "runtime.json"
    output = tmp_path / "runtime.yaml"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    runtime.write_text(json.dumps({"runtime_fingerprint": "cpu-image-fingerprint"}), encoding="utf-8")
    historical = tmp_path / "historical" / "kg"
    subprocess.run([
        sys.executable, str(SCRIPTS / "make_datasphere_vertex_config.py"),
        "--base-config", str(ROOT / "config.yaml"), "--gateway-manifest", str(manifest),
        "--gateway-url", "https://gateway.example.run.app", "--datasphere-runtime-manifest", str(runtime),
        "--output", str(output), "--data-dir", "/read-only/ragtruth", "--work-dir", str(tmp_path / "work"),
        "--cache-root", str(tmp_path / "new-cache"), "--kg-cache-read-dir", str(historical),
    ], check=True)
    cfg = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert cfg["cache_dir"] == str(tmp_path / "new-cache" / "kg")
    assert cfg["cache_read_dirs"] == [str(historical)]


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
    assert job["outputs"] == [{
        "vertex-cpu-probe-vertex-3qa-20260718.tar.gz": "ARTIFACT_ARCHIVE"
    }]
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
    assert "--max-tokens 4096 --concurrency 1 --max-retries 7 --retry-backoff-base-s 5" in runner
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
    # DataSphere replaces ${ARTIFACT_ARCHIVE} with its collector-visible
    # output path before the Docker command starts. A container-local /job
    # path is not an output upload path.
    command = str(job["cmd"])
    assert 'export ARCHIVE_PATH="${ARTIFACT_ARCHIVE}"' in command
    assert 'tar -C "$(dirname "$RUN_ROOT")" -czf "$ARCHIVE_PATH"' in command
    assert "archive_artifacts()" in command
    assert "tar -tzf \"$ARCHIVE_PATH\" >/dev/null" in command
    assert command.index("if ! archive_artifacts; then status=1; fi;") < command.rindex(
        'exit "$status"'
    )


def test_cpu_vertex_qa_job_binds_the_gateway_and_parameterizes_the_sample(tmp_path):
    rendered = tmp_path / "vertex-100qa.yaml"
    subprocess.run([
        sys.executable, str(SCRIPTS / "render_datasphere_vertex_qa_pilot_job.py"),
        "--commit", "f" * 40, "--run-id", "vertex-100qa-20260719",
        "--gateway-url", "https://gateway.example.run.app",
        "--gateway-manifest-sha256", "a" * 64, "--docker-image", IMAGE,
        "--qa-sample-size", "100", "--qa-test-fraction", "0.2", "--cv-folds", "5",
        "--concurrency", "1", "--timeout-seconds", "43200",
        "--output", str(rendered),
    ], check=True)
    job = yaml.safe_load(rendered.read_text(encoding="utf-8"))
    assert job["cloud-instance-types"] == ["c1.4"]
    assert job["outputs"] == [{"vertex-cpu-qa-vertex-100qa-20260719.tar.gz": "ARTIFACT_ARCHIVE"}]
    command = str(job["cmd"])
    assert 'EXPECTED_GATEWAY_MANIFEST_SHA256="' + "a" * 64 + '"' in command
    assert 'QA_SAMPLE_SIZE="100"' in command
    assert 'QA_TEST_FRACTION="0.2"' in command
    assert 'QA_CV_FOLDS="5"' in command
    assert "timeout --signal=TERM --kill-after=60s 43200" in command
    subprocess.run([sys.executable, str(SCRIPTS / "validate_datasphere_job.py"), "--job", str(rendered), "--repo-root", str(ROOT)], check=True)
    runner = (SCRIPTS / "run_datasphere_vertex_cpu_qa_pilot.sh").read_text(encoding="utf-8")
    assert "--qa-sample --qa-sample-size \"$QA_SAMPLE_SIZE\"" in runner
    assert "--qa-manifest-out \"$QA_MANIFEST\"" in runner
    assert "require_complete_extraction \"$STRICT_OUT\" \"$QA_SAMPLE_SIZE\"" in runner
    assert "--relation-mode strict" in runner
    assert "--relation-mode support" in runner
    assert "--relation-mode support-critical" in runner
    assert "--kg-cache-only" in runner
    assert "--cache-only" in runner
    assert "--max-tokens 16384 --concurrency \"$LLM_CONCURRENCY\" --max-retries 1000" in runner
    assert "--cv-folds \"$QA_CV_FOLDS\"" in runner
    assert "--cache-root \"$BASELINE_CACHE_ROOT\"" in runner
    assert "--cache-root \"$CRITICAL_CACHE_ROOT\"" in runner
    assert "--kg-cache-read-dir \"$BASELINE_CACHE_ROOT/kg\"" in runner
    assert 'hallu-vertex-qa-support-critical-checkpoint-v1' in runner
    assert "support-critical-live-usage.jsonl" in runner
    assert "cmp \"$STRICT_OUT/metrics.csv\" \"$REPLAY_STRICT/metrics.csv\"" in runner
    assert "cmp \"$CRITICAL_OUT/metrics.csv\" \"$REPLAY_CRITICAL/metrics.csv\"" in runner
    assert "vllm" not in runner.lower()
    subprocess.run(["bash", "-n", str(SCRIPTS / "run_datasphere_vertex_cpu_qa_pilot.sh")], check=True)


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
    assert (ROOT / "datasphere/docker/client.requirements.txt").is_file()
    assert (ROOT / "datasphere/docker/write_cpu_vertex_runtime_manifest.py").is_file()
    assert "datasphere/docker/client.requirements.txt" in text
    assert "datasphere/docker/write_cpu_vertex_runtime_manifest.py" in text
