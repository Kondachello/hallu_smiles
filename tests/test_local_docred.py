"""Offline contracts for the local, cache-backed DocRED runner."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


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


def test_local_vertex_config_is_cpu_serial_and_separates_its_cache(tmp_path):
    manifest = tmp_path / "gateway.json"
    runtime = tmp_path / "runtime.json"
    model = tmp_path / "minilm"
    output = tmp_path / "runtime.yaml"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    runtime.write_text(json.dumps({"runtime_fingerprint": "local-cpu:" + "a" * 64}), encoding="utf-8")
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    completed = subprocess.run([
        sys.executable, str(SCRIPTS / "make_local_vertex_config.py"),
        "--base-config", str(ROOT / "config.yaml"), "--gateway-manifest", str(manifest),
        "--gateway-url", "https://gateway.example.run.app", "--local-runtime-manifest", str(runtime),
        "--embedding-model-path", str(model), "--output", str(output), "--data-dir", "/docred",
        "--work-dir", str(tmp_path / "work"), "--cache-root", str(tmp_path / "checkpoint"),
    ], check=True, text=True, capture_output=True)
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    identity = json.loads(completed.stdout)
    assert config["llm"]["api_base"] == "https://gateway.example.run.app/v1"
    assert config["llm"]["concurrency"] == 1
    assert config["llm"]["max_retries"] == 0
    assert config["llm"]["request_min_interval_s"] == 4
    assert config["extraction"]["serial_chunking"] is True
    assert config["extraction"]["explicit_clustering"] is True
    assert config["extraction"]["max_tokens_ceiling"] == 8192
    assert config["matching"]["embedding_model_path"] == str(model.resolve())
    assert config["cache_dir"] == str(tmp_path / "checkpoint" / "kg")
    assert identity["runtime_fingerprint"].startswith("vertex-gateway:")


def test_local_monitor_redacts_unknown_progress_fields(tmp_path):
    run_root = tmp_path / "vertex-cpu-docred-kg-artifacts-local-test"
    progress = run_root / "docred-live" / "progress.json"
    progress.parent.mkdir(parents=True)
    progress.write_text(json.dumps({
        "protocol": "docred-progress-v1", "event": "inner_progress", "phase": "smoke",
        "outer_completed": 1, "outer_total": 10, "prompt": "must never escape",
    }), encoding="utf-8")
    subprocess.run([
        sys.executable, str(SCRIPTS / "monitor_local_docred_kg_eval.py"),
        "--run-root", str(run_root), "--pid", str(os.getpid()), "--once",
    ], check=True)
    payload = json.loads((run_root / "live-snapshots" / "latest.json").read_text(encoding="utf-8"))
    assert payload["progress"]["outer_completed"] == 1
    assert "prompt" not in payload["progress"]


def test_local_archive_excludes_graphs_usage_and_logs(tmp_path):
    run_root = tmp_path / "vertex-cpu-docred-kg-artifacts-local-test"
    (run_root / "docred-live" / "graphs").mkdir(parents=True)
    (run_root / "docred-live" / "graphs" / "raw.json").write_text("secret graph", encoding="utf-8")
    (run_root / "docred-live" / "usage.jsonl").write_text("secret usage", encoding="utf-8")
    (run_root / "docred-live" / "metrics.json").write_text("{}", encoding="utf-8")
    (run_root / "docred.stdout.log").write_text("unsafe provider output", encoding="utf-8")
    (run_root / "run_metadata.json").write_text("{}", encoding="utf-8")
    archive = tmp_path / "artifact.tar.gz"
    subprocess.run([
        sys.executable, str(SCRIPTS / "archive_local_docred_kg_eval.py"),
        "--run-root", str(run_root), "--archive", str(archive),
    ], check=True)
    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    assert any(name.endswith("metrics.json") for name in names)
    assert not any("graphs" in name or name.endswith("usage.jsonl") or name.endswith(".log") for name in names)


def test_local_launcher_uses_keychain_caffeinate_replay_and_redacted_monitoring():
    launcher = (SCRIPTS / "run_local_docred_kg_eval.sh").read_text(encoding="utf-8")
    assert "security find-generic-password" in launcher
    assert '"gemini"' in launcher
    assert "caffeinate -dimsu" in launcher
    assert "TMUX" in launcher
    assert "STY" in launcher
    assert "monitor_local_docred_kg_eval.py" in launcher
    assert "--interval-seconds 900" in launcher
    assert "--stage replay --cache-only" in launcher
    assert "verify_docred_cache_replay.py" in launcher
    assert "archive_local_docred_kg_eval.py" in launcher
    subprocess.run(["bash", "-n", str(SCRIPTS / "run_local_docred_kg_eval.sh")], check=True)
    fallback = (SCRIPTS / "start_local_docred_kg_eval.sh").read_text(encoding="utf-8")
    assert "screen -dmS" in fallback
    assert "nohup" in fallback
    assert "HALLU_DOCRED_DETACHED=1" in fallback
    subprocess.run(["bash", "-n", str(SCRIPTS / "start_local_docred_kg_eval.sh")], check=True)
