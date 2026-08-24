"""Offline checks for the Compute Engine controlled-evaluation wrapper."""
from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_gcp_config_has_no_cache_readthrough_or_secret(tmp_path):
    manifest = {
        "protocol": "hallu-vertex-openai-gateway-v1",
        "api_path": "/v1",
        "logical_model": "openai/gemini-2.5-flash",
        "vertex_model": "gemini-2.5-flash",
        "vertex_location": "europe-west4",
        "gateway_release": "release",
        "cloud_run_revision": "revision",
    }
    gateway = tmp_path / "gateway.json"
    gateway.write_text(json.dumps(manifest), encoding="utf-8")
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({
        "protocol": "hallu-gcp-compute-cpu-runtime-v1",
        "runtime_fingerprint": "gcp-compute-cpu:" + "a" * 64,
    }), encoding="utf-8")
    embedding = tmp_path / "embedding"
    embedding.mkdir()
    (embedding / "config.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "config.yaml"
    identity = tmp_path / "identity.json"
    completed = subprocess.run([
        str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/make_gcp_vertex_config.py"),
        "--base-config", str(ROOT / "config.yaml"), "--gateway-manifest", str(gateway),
        "--gateway-url", "https://gateway.example", "--gcp-runtime-manifest", str(runtime),
        "--embedding-model-path", str(embedding), "--output", str(output),
        "--identity-output", str(identity), "--data-dir", str(tmp_path / "data"),
        "--work-dir", str(tmp_path / "work"), "--cache-root", str(tmp_path / "cache"),
    ], capture_output=True, text=True, check=True)
    assert "gateway_manifest_sha256" in completed.stdout
    config = yaml.safe_load(output.read_text())
    assert config["cache_read_dirs"] == []
    assert config["llm"]["api_key_env"] == "HALLU_GATEWAY_API_KEY"
    assert "hallu-docred-gateway-bearer" not in output.read_text()
    assert json.loads(identity.read_text())["reference_graph_mode"] == "frozen_historical_artifact"


def test_gcp_archive_excludes_raw_graphs_cache_keys_and_scored_rows(tmp_path):
    root = tmp_path / "gcp-ragtruth-llama31-fixture"
    root.mkdir()
    (root / "input_provenance.json").write_text("{}", encoding="utf-8")
    (root / "run.log").write_text("[progress] safe\n", encoding="utf-8")
    for method in ("strict", "support-critical"):
        directory = root / method
        directory.mkdir()
        (directory / "extraction_summary.json").write_text(json.dumps({
            "status": "ready_with_explicit_exclusions",
            "expected_sources": 750,
            "analysis_expected_sources": 749,
            "references_completed": 749,
            "responses_completed": 749,
            "analysis_expected_responses": 749,
            "pairs_completed": 749,
            "excluded_source_ids": ["12448"],
            "failures": [],
            "cache_records": [{"cache_key": "must-not-archive"}],
        }), encoding="utf-8")
        (directory / "scored.jsonl").write_text('{"answer":"must-not-archive"}\n', encoding="utf-8")
    archive = tmp_path / "archive.tar.gz"
    subprocess.run([
        str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/archive_gcp_ragtruth_llama31_eval.py"),
        "--run-root", str(root), "--archive", str(archive),
    ], check=True)
    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    assert all("scored.jsonl" not in name and "extraction_summary.json" not in name for name in names)
    assert any(name.endswith("extraction_summary_redacted.json") for name in names)


def test_gcp_scripts_are_cpu_serial_and_do_not_grant_vertex_access():
    provision = (ROOT / "scripts/provision_gcp_ragtruth_llama31_eval.sh").read_text()
    launch = (ROOT / "scripts/launch_gcp_ragtruth_llama31_eval.sh").read_text()
    startup = (ROOT / "gcp/start_ragtruth_llama31_vm.sh").read_text()
    runner = (ROOT / "scripts/run_gcp_ragtruth_llama31_eval.sh").read_text()
    assert "--machine-type=e2-medium" in launch
    assert "--boot-disk-size=30GB" in launch and "--boot-disk-type=pd-ssd" in launch
    assert "--no-boot-disk-auto-delete" in launch
    assert "--image-family=cos-stable" in launch
    assert "gcloud compute instances create-with-container" not in launch
    assert "docker-credential-gcr configure-docker" in startup
    assert "aiplatform" not in provision.lower()
    assert "--relation-mode support --" not in runner
    assert "--relation-mode strict" in runner and "--relation-mode support-critical" in runner
    assert "--kg-cache-only" in runner and "--cache-only" in runner
