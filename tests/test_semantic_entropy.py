"""Offline semantic-entropy invariants and cache replay tests."""
from __future__ import annotations

import math
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.config import Config
from src.semantic_entropy import CompletionSample, SemanticEntropyDetector, SemanticEntropyError


ROOT = Path(__file__).resolve().parents[1]


def _config(cache_dir: Path) -> Config:
    return Config({
        "llm": {
            "model": "openai/gemini-2.5-flash",
            "api_base": "https://gateway.invalid/v1",
            "api_key_env": "TEST_SEMANTIC_ENTROPY_KEY",
            "temperature": 0.0,
            "request_min_interval_s": 0,
            "max_retries": 1,
            "retry_backoff_base_s": 0.01,
            "retry_backoff_max_s": 0.01,
            "rate_limit_cooldown_max_s": 0.01,
            "rate_limit_retry_deadline_s": 1,
            "retry_deadline_s": 1,
            "concurrency": 1,
            "runtime_fingerprint": "unit-test",
        },
        "semantic_entropy": {
            "protocol": "toha-semantic-entropy-v1-gemini-selected-logprobs",
            "cache_dir": str(cache_dir),
            "cache_read_dirs": [],
            "n_samples": 3,
            "temperature": 1.0,
            "max_tokens": 512,
            "request_timeout_s": 10,
            "prompt_version": "ragtruth-original-prompt-v1",
            "likelihood_normalization": "mean_selected_token_logprob",
            "strict_entailment": False,
            "nli_model": "test-nli",
            "nli_revision": "test-revision",
            "nli_model_path": None,
            "nli_device": "cpu",
            "nli_batch_size": 2,
            "nli_max_length": 512,
        },
    })


class _FakeNLI:
    def __init__(self):
        self.calls = 0

    def classify_pairs(self, pairs):
        self.calls += 1
        labels = {
            ("same-one", "same-two"): 2,
            ("same-two", "same-one"): 2,
            ("same-one", "different"): 0,
            ("different", "same-one"): 0,
        }
        return [labels[pair] for pair in pairs]


def test_toha_greedy_semantic_entropy_and_cache_only_replay(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_SEMANTIC_ENTROPY_KEY", "not-a-real-key")
    config = _config(tmp_path / "cache")
    backend = _FakeNLI()
    detector = SemanticEntropyDetector(config, entailment_backend=backend)
    samples = iter([
        CompletionSample("same-one", -1.0, 1, 1),
        CompletionSample("same-two", -1.0, 1, 1),
        CompletionSample("different", -2.0, 1, 1),
    ])
    detector._live_sample = lambda _prompt: next(samples)  # type: ignore[method-assign]

    score = detector.score_prompt("A synthetic RAGTruth prompt")
    assert score.semantic_ids == (0, 0, 1)
    assert score.n_classes == 2
    first_mass = (2 * math.exp(-1.0)) / (2 * math.exp(-1.0) + math.exp(-2.0))
    expected = -(first_mass * math.log(first_mass) + (1 - first_mass) * math.log(1 - first_mass))
    assert score.entropy == pytest.approx(expected)
    assert backend.calls == 1

    # A true cache-only replay must load both the sampled texts and the NLI
    # clustering result without requiring an API key or loading transformers.
    monkeypatch.delenv("TEST_SEMANTIC_ENTROPY_KEY")
    replay = SemanticEntropyDetector(config, cache_only=True)
    replay_score = replay.score_prompt("A synthetic RAGTruth prompt")
    assert replay_score == score
    assert replay.usage.summary()["api_calls"] == 0
    assert replay._loaded_nli is None  # noqa: SLF001 - proves no model was loaded


@pytest.mark.parametrize(
    ("ids", "values"),
    [([], []), ([0, 1], [-1.0]), ([0, 2], [-1.0, -2.0])],
)
def test_semantic_entropy_rejects_misaligned_or_sparse_classes(ids, values):
    with pytest.raises((ValueError, SemanticEntropyError)):
        SemanticEntropyDetector._entropy(ids, values)


def test_entropy_runtime_config_requires_logprob_capability(tmp_path):
    from gateway.core import GatewaySettings, gateway_manifest

    settings = GatewaySettings(
        api_key="not-a-real-key",
        logical_model="openai/gemini-2.5-flash",
        vertex_model="gemini-2.5-flash",
        project="test-project",
        location="europe-west4",
        release="test-release",
        cloud_run_revision="test-revision",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(gateway_manifest(settings)), encoding="utf-8")
    nli = tmp_path / "nli"
    nli.mkdir()
    (nli / "config.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "runtime.yaml"
    command = [
        sys.executable, str(ROOT / "scripts" / "make_local_semantic_entropy_config.py"),
        "--base-config", str(ROOT / "config.yaml"),
        "--gateway-manifest", str(manifest),
        "--gateway-url", "https://gateway.example.run.app",
        "--nli-model-path", str(nli),
        "--data-dir", str(tmp_path / "data"),
        "--cache-root", str(tmp_path / "cache"),
        "--output", str(output),
    ]
    subprocess.run(command, check=True)
    assert output.is_file()

    incompatible = gateway_manifest(settings)
    incompatible.pop("selected_token_logprobs")
    manifest.write_text(json.dumps(incompatible), encoding="utf-8")
    failed = subprocess.run(command, text=True, capture_output=True)
    assert failed.returncode != 0
    assert "selected-token logprobs" in failed.stderr
