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
    assert cfg["llm"]["rate_limit_cooldown_max_s"] == 900.0
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
    critical_historical = tmp_path / "historical-critical"
    subprocess.run([
        sys.executable, str(SCRIPTS / "make_datasphere_vertex_config.py"),
        "--base-config", str(ROOT / "config.yaml"), "--gateway-manifest", str(manifest),
        "--gateway-url", "https://gateway.example.run.app", "--datasphere-runtime-manifest", str(runtime),
        "--output", str(output), "--data-dir", "/read-only/ragtruth", "--work-dir", str(tmp_path / "work"),
        "--cache-root", str(tmp_path / "new-cache"), "--kg-cache-read-dir", str(historical),
        "--relation-cache-read-dir", str(tmp_path / "historical-verdicts"),
        "--critical-cache-read-root", str(critical_historical),
    ], check=True)
    cfg = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert cfg["cache_dir"] == str(tmp_path / "new-cache" / "kg")
    assert cfg["cache_read_dirs"] == [str(historical)]
    assert cfg["relation_verifier"]["cache_read_dirs"] == [
        str(tmp_path / "historical-verdicts")
    ]
    for namespace in ("critical_claims", "critical_coverage", "critical_verdicts"):
        section = {
            "critical_claims": "claim_extractor",
            "critical_coverage": "coverage_reviewer",
            "critical_verdicts": "claim_verifier",
        }[namespace]
        assert cfg["support_critical"][section]["cache_read_dirs"] == [
            str(critical_historical / namespace)
        ]


def test_vertex_runtime_config_can_replay_a_recorded_historical_llm_identity(tmp_path):
    manifest = tmp_path / "gateway.json"
    runtime = tmp_path / "runtime.json"
    output = tmp_path / "runtime.yaml"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    runtime.write_text(json.dumps({"runtime_fingerprint": "new-cpu-image"}), encoding="utf-8")
    legacy = "vertex-gateway:historical-cache-identity"
    completed = subprocess.run([
        sys.executable, str(SCRIPTS / "make_datasphere_vertex_config.py"),
        "--base-config", str(ROOT / "config.yaml"), "--gateway-manifest", str(manifest),
        "--gateway-url", "https://gateway.example.run.app", "--datasphere-runtime-manifest", str(runtime),
        "--output", str(output), "--data-dir", "/data", "--work-dir", str(tmp_path / "work"),
        "--llm-runtime-fingerprint-override", legacy,
    ], check=True, text=True, capture_output=True)
    cfg = yaml.safe_load(output.read_text(encoding="utf-8"))
    identity = json.loads(completed.stdout)
    assert cfg["llm"]["runtime_fingerprint"] == legacy
    assert identity["historical_cache_identity"] is True
    assert identity["computed_runtime_fingerprint"] != legacy


def test_vertex_runtime_config_allows_unbounded_transient_retry_until_job_timeout(tmp_path):
    manifest = tmp_path / "gateway.json"
    runtime = tmp_path / "runtime.json"
    output = tmp_path / "runtime.yaml"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    runtime.write_text(json.dumps({"runtime_fingerprint": "cpu-image-fingerprint"}), encoding="utf-8")
    subprocess.run([
        sys.executable, str(SCRIPTS / "make_datasphere_vertex_config.py"),
        "--base-config", str(ROOT / "config.yaml"), "--gateway-manifest", str(manifest),
        "--gateway-url", "https://gateway.example.run.app", "--datasphere-runtime-manifest", str(runtime),
        "--output", str(output), "--data-dir", "/data", "--work-dir", str(tmp_path / "work"),
        "--max-retries", "0",
    ], check=True)
    assert yaml.safe_load(output.read_text(encoding="utf-8"))["llm"]["max_retries"] == 0


def test_vertex_runtime_config_can_bound_docred_adaptive_extraction_output(tmp_path):
    manifest = tmp_path / "gateway.json"
    runtime = tmp_path / "runtime.json"
    output = tmp_path / "runtime.yaml"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    runtime.write_text(json.dumps({"runtime_fingerprint": "cpu-image-fingerprint"}), encoding="utf-8")
    subprocess.run([
        sys.executable, str(SCRIPTS / "make_datasphere_vertex_config.py"),
        "--base-config", str(ROOT / "config.yaml"), "--gateway-manifest", str(manifest),
        "--gateway-url", "https://gateway.example.run.app", "--datasphere-runtime-manifest", str(runtime),
        "--output", str(output), "--data-dir", "/docred", "--work-dir", str(tmp_path / "work"),
        "--max-tokens", "4096", "--extraction-max-tokens-ceiling", "8192",
    ], check=True)
    cfg = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert cfg["llm"]["max_tokens"] == 4096
    assert cfg["extraction"]["max_tokens_ceiling"] == 8192


def test_historical_cache_lineage_requires_exact_checkpoint_gateway_and_client_runtime(tmp_path):
    registry = ROOT / "datasphere" / "historical_kg_cache_lineages.json"
    lineage = json.loads(registry.read_text(encoding="utf-8"))["lineages"][0]
    checkpoint = tmp_path / "checkpoint-identity.json"
    runtime = tmp_path / "runtime.json"
    output = tmp_path / "resolved.json"
    checkpoint.write_text(json.dumps({
        "protocol": lineage["checkpoint_protocol"],
        "source_commit": lineage["source_commit"],
        "gateway_manifest_sha256": lineage["gateway_manifest_sha256"],
        "qa_sample": {"total": 100, "train": 80, "test": 20, "alpha_cv_folds": 5},
    }), encoding="utf-8")
    runtime.write_text(json.dumps({"client_runtime": lineage["client_runtime"]}), encoding="utf-8")
    subprocess.run([
        sys.executable, str(SCRIPTS / "resolve_datasphere_historical_cache_lineage.py"),
        "--lineages", str(registry), "--checkpoint-identity", str(checkpoint),
        "--runtime-manifest", str(runtime),
        "--gateway-manifest-sha256", lineage["gateway_manifest_sha256"],
        "--qa-total", "100", "--qa-train", "80", "--qa-test", "20", "--cv-folds", "5",
        "--output", str(output),
    ], check=True)
    assert json.loads(output.read_text(encoding="utf-8"))["llm_runtime_fingerprint"] == (
        lineage["llm_runtime_fingerprint"]
    )


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
        "--concurrency", "1", "--exclude-source-id", "12448", "--timeout-seconds", "43200",
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
    assert 'QA_FORCE_CACHE_ONLY="0"' in command
    assert 'QA_EXCLUDE_SOURCE_IDS="12448"' in command
    assert 'tee "$RUN_ROOT/qa.stdout.log"' in command
    assert "timeout --signal=TERM --kill-after=60s 43200" in command
    subprocess.run([sys.executable, str(SCRIPTS / "validate_datasphere_job.py"), "--job", str(rendered), "--repo-root", str(ROOT)], check=True)
    runner = (SCRIPTS / "run_datasphere_vertex_cpu_qa_pilot.sh").read_text(encoding="utf-8")
    assert "preflight_datasphere_kg_cache.py" in runner
    assert "excluded_extractions.jsonl" in (ROOT / "run.py").read_text(encoding="utf-8")
    assert "--allow-missing" in runner
    assert "historical_kg_cache_lineages.json" in runner
    assert "--llm-runtime-fingerprint-override \"$HISTORICAL_LLM_RUNTIME_FINGERPRINT\"" in runner
    assert "--qa-manifest \"$QA_MANIFEST\"" in runner
    assert "require_complete_extraction \"$STRICT_OUT\" \"$QA_SAMPLE_SIZE\"" in runner
    assert "--relation-mode strict" in runner
    assert "--relation-mode support" in runner
    assert "--relation-mode support-critical" in runner
    assert "--kg-cache-only" in runner
    assert "--cache-only" in runner
    assert "LLM_MAX_RETRIES=\"${LLM_MAX_RETRIES:-12}\"" in runner
    assert "QA_FORCE_CACHE_ONLY=\"${QA_FORCE_CACHE_ONLY:-0}\"" in runner
    assert "force mode enabled: any missing inference entry is a hard failure" in runner
    assert "--max-tokens 16384 --concurrency \"$LLM_CONCURRENCY\" --max-retries \"$LLM_MAX_RETRIES\"" in runner
    assert "--cv-folds \"$QA_CV_FOLDS\"" in runner
    assert "--cache-root \"$BASELINE_CACHE_ROOT\"" in runner
    assert "--cache-root \"$CRITICAL_CACHE_ROOT\"" in runner
    assert "--kg-cache-read-dir \"$BASELINE_CACHE_ROOT/kg\"" in runner
    assert "--kg-cache-read-dir \"$HISTORICAL_BASELINE_CACHE_ROOT/kg\"" in runner
    assert "--relation-cache-read-dir \"$HISTORICAL_BASELINE_CACHE_ROOT/verdicts\"" in runner
    assert "--critical-cache-read-root" in runner
    assert "support-critical-v1-${GATEWAY_MANIFEST_SHA256}" in runner
    assert "check_support_critical_gateway_probe.py" in runner
    assert "preflight_support_critical_resilience.py" in runner
    assert 'hallu-vertex-qa-support-critical-checkpoint-v1' in runner
    assert "support-critical-live-usage.jsonl" in runner
    assert "strict-cache-fill-usage.jsonl" in runner
    assert "support-cache-fill-usage.jsonl" in runner
    assert "cache-before-replay.sha256" in runner
    assert "summary_metrics.csv" in runner
    assert "scored.jsonl" in runner
    assert "verify_cache_replay.py" in runner
    assert "vllm" not in runner.lower()
    subprocess.run(["bash", "-n", str(SCRIPTS / "run_datasphere_vertex_cpu_qa_pilot.sh")], check=True)


def test_cpu_vertex_qa_job_defaults_to_platform_deadline_for_resumable_long_runs(tmp_path):
    rendered = tmp_path / "vertex-1000qa.yaml"
    subprocess.run([
        sys.executable, str(SCRIPTS / "render_datasphere_vertex_qa_pilot_job.py"),
        "--commit", "f" * 40, "--run-id", "vertex-1000qa-20260721",
        "--gateway-url", "https://gateway.example.run.app",
        "--gateway-manifest-sha256", "a" * 64, "--docker-image", IMAGE,
        "--qa-sample-size", "1000", "--qa-test-fraction", "0.2", "--cv-folds", "5",
        "--concurrency", "1", "--output", str(rendered),
    ], check=True)
    command = str(yaml.safe_load(rendered.read_text(encoding="utf-8"))["cmd"])
    assert 'QA_SAMPLE_SIZE="1000"' in command
    assert "timeout --signal=TERM" not in command
    assert "bash source/scripts/run_datasphere_vertex_cpu_qa_pilot.sh" in command
    assert "#" not in command


def test_cpu_vertex_qa_job_can_forbid_inference_cache_misses(tmp_path):
    rendered = tmp_path / "vertex-750qa-cache-only.yaml"
    subprocess.run([
        sys.executable, str(SCRIPTS / "render_datasphere_vertex_qa_pilot_job.py"),
        "--commit", "f" * 40, "--run-id", "vertex-750qa-cache-only",
        "--gateway-url", "https://gateway.example.run.app",
        "--gateway-manifest-sha256", "a" * 64, "--docker-image", IMAGE,
        "--qa-sample-size", "750", "--qa-test-fraction", "0.2", "--cv-folds", "5",
        "--concurrency", "1", "--exclude-source-id", "12448", "--force-cache-only",
        "--output", str(rendered),
    ], check=True)
    job = yaml.safe_load(rendered.read_text(encoding="utf-8"))
    assert 'QA_FORCE_CACHE_ONLY="1"' in str(job["cmd"])
    subprocess.run([
        sys.executable, str(SCRIPTS / "validate_datasphere_job.py"),
        "--job", str(rendered), "--repo-root", str(ROOT),
    ], check=True)


def test_cpu_vertex_docred_job_is_fixed_budgeted_and_redacted(tmp_path):
    rendered = tmp_path / "docred.yaml"
    subprocess.run([
        sys.executable, str(SCRIPTS / "render_datasphere_vertex_docred_kg_eval_job.py"),
        "--commit", "f" * 40, "--run-id", "docred-250qa-20260731",
        "--gateway-url", "https://gateway.example.run.app",
        "--gateway-manifest-sha256", "a" * 64, "--budget-eur", "10.5",
        "--docker-image", IMAGE, "--output", str(rendered),
    ], check=True)
    job = yaml.safe_load(rendered.read_text(encoding="utf-8"))
    assert job["cloud-instance-types"] == ["c1.4"]
    assert job["outputs"] == [{
        "vertex-cpu-docred-kg-docred-250qa-20260731.tar.gz": "ARTIFACT_ARCHIVE"
    }]
    command = str(job["cmd"])
    assert 'DOCRED_BUDGET_EUR="10.50"' in command
    assert 'EXPECTED_GATEWAY_MANIFEST_SHA256="' + "a" * 64 + '"' in command
    assert "--exclude=\"*/usage.jsonl\"" in command
    assert "docred-live/graphs" in command
    assert "timeout --signal=TERM" not in command
    assert "HALLU_GATEWAY_API_KEY" not in rendered.read_text(encoding="utf-8")
    subprocess.run([
        sys.executable, str(SCRIPTS / "validate_datasphere_job.py"),
        "--job", str(rendered), "--repo-root", str(ROOT),
    ], check=True)
    runner = (SCRIPTS / "run_datasphere_vertex_cpu_docred_kg_eval.sh").read_text(encoding="utf-8")
    assert "fetch_docred_data.py" in runner
    assert "DOCRED_BUDGET_EUR" in runner
    assert "--max-tokens 4096 --extraction-max-tokens-ceiling 8192" in runner
    assert "--concurrency 1 --max-retries 0" in runner
    assert "serial_chunking'] = True" in runner
    assert "--stage replay --cache-only" in runner
    assert "verify_docred_cache_replay.py" in runner
    assert "cache-before-replay.json" in runner
    assert "cache keys" in runner.lower()
    assert "vllm" not in runner.lower()
    subprocess.run(["bash", "-n", str(SCRIPTS / "run_datasphere_vertex_cpu_docred_kg_eval.sh")], check=True)
    submitter = (SCRIPTS / "submit_datasphere_vertex_docred_kg_eval.sh").read_text(encoding="utf-8")
    assert 'GRPC_DNS_RESOLVER="${GRPC_DNS_RESOLVER:-ares}"' in submitter
    assert "--profile \"$PROFILE\" --no-browser --no-user-output iam create-token" in submitter
    assert "datasphere --profile \"$PROFILE\"" in submitter
    assert "validate_datasphere_vertex_probe_artifact.py" in submitter
    assert "yc init" not in submitter


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
