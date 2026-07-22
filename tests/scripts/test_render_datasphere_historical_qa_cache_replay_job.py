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
    assert "HISTORICAL_CHECKPOINT_BASE" in text
    assert "timeout --signal=TERM" in text
    assert 'REPLAY_COUNT="10"' in text
    assert 'REPLAY_SELECTION_SEED="917"' in text
    assert "replay.console.log" in text
    assert yaml.safe_load(text)["name"] == "historical-qa-cache-replay-historical-cache-test"
