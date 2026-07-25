from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "ghcr.io/kondachello/hallu-smiles-datasphere-vertex-cpu@sha256:" + "a" * 64


def test_rendered_historical_cache_replay_is_cpu_only_and_never_injects_a_secret(tmp_path) -> None:
    output = tmp_path / "job.yaml"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/render_datasphere_historical_qa_cache_replay_job.py"),
            "--commit", "a" * 40,
            "--run-id", "historical-cache-test",
            "--gateway-url", "https://gateway.example.test",
            "--docker-image", IMAGE,
            "--replay-count", "10",
            "--replay-selection-seed", "917",
            "--output", str(output),
        ],
        check=True,
    )
    text = output.read_text(encoding="utf-8")
    document = yaml.safe_load(text)
    assert document["cloud-instance-types"] == ["c1.4"]
    assert "HALLU_GATEWAY_API_KEY" not in text
    assert "HALLU_GATEWAY_URL" not in text
    assert "g1.1" not in text and "huggingface-cli" not in text.lower()
    assert "CHECKPOINT_PARENT" in text
    assert 'QA_SAMPLE_SIZE="100"' in text  # default when --qa-sample-size is not passed
    assert "timeout --signal=TERM" in text
    assert 'REPLAY_COUNT="10"' in text
    assert 'REPLAY_SELECTION_SEED="917"' in text
    assert "replay.console.log" in text
    assert yaml.safe_load(text)["name"] == "historical-qa-cache-replay-historical-cache-test"


def test_qa_sample_size_is_configurable_and_bounds_replay_count(tmp_path) -> None:
    output = tmp_path / "job-750.yaml"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/render_datasphere_historical_qa_cache_replay_job.py"),
            "--commit", "a" * 40,
            "--run-id", "historical-cache-750qa-test",
            "--gateway-url", "https://gateway.example.test",
            "--docker-image", IMAGE,
            "--qa-sample-size", "750",
            "--replay-count", "750",
            "--output", str(output),
        ],
        check=True,
    )
    text = output.read_text(encoding="utf-8")
    assert 'QA_SAMPLE_SIZE="750"' in text
    assert 'REPLAY_COUNT="750"' in text
    assert "qa-100-test-20-cv-5" not in text  # no leftover hardcoded checkpoint path

    # --replay-count above --qa-sample-size must fail fast, before any Job is rendered.
    rejected = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/render_datasphere_historical_qa_cache_replay_job.py"),
            "--commit", "a" * 40,
            "--run-id", "historical-cache-over-count-test",
            "--gateway-url", "https://gateway.example.test",
            "--docker-image", IMAGE,
            "--qa-sample-size", "100",
            "--replay-count", "101",
            "--output", str(tmp_path / "job-rejected.yaml"),
        ],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert not (tmp_path / "job-rejected.yaml").exists()
