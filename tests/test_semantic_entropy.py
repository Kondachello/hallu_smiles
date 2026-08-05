"""Offline semantic-entropy invariants and cache replay tests."""
from __future__ import annotations

import math
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.config import Config
from src.cache import llm_runtime_fingerprint
from src.semantic_entropy import CompletionSample, GatewayRequestError, SemanticEntropyDetector, SemanticEntropyError


ROOT = Path(__file__).resolve().parents[1]


def _config(
    cache_dir: Path,
    *,
    max_tokens: int = 512,
    max_tokens_read_through: list[int] | None = None,
    cache_read_dirs: list[Path] | None = None,
    runtime_fingerprint: str = "unit-test",
    legacy_sample_cache_contracts: list[dict] | None = None,
) -> Config:
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
            "runtime_fingerprint": runtime_fingerprint,
        },
        "semantic_entropy": {
            "protocol": "toha-semantic-entropy-v1-gemini-selected-logprobs",
            "cache_dir": str(cache_dir),
            "cache_read_dirs": [str(path) for path in cache_read_dirs or []],
            "n_samples": 3,
            "temperature": 1.0,
            "max_tokens": max_tokens,
            "max_tokens_read_through": max_tokens_read_through or [],
            "legacy_sample_cache_contracts": legacy_sample_cache_contracts or [],
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


def test_completed_lower_cap_samples_are_cache_compatible_with_a_higher_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_SEMANTIC_ENTROPY_KEY", "not-a-real-key")
    cache_dir = tmp_path / "cache"
    legacy = SemanticEntropyDetector(_config(cache_dir, max_tokens=8192), entailment_backend=_FakeNLI())
    samples = iter([
        CompletionSample("same-one", -1.0, 1, 1),
        CompletionSample("same-two", -1.0, 1, 1),
        CompletionSample("different", -2.0, 1, 1),
    ])
    legacy._live_sample = lambda _prompt: next(samples)  # type: ignore[method-assign]
    legacy_score = legacy.score_prompt("A synthetic RAGTruth prompt")

    # Only completed samples enter the legacy cache. Their probability is
    # unchanged under the larger cap, so a cache-only replay may reuse them.
    monkeypatch.delenv("TEST_SEMANTIC_ENTROPY_KEY")
    target = SemanticEntropyDetector(
        _config(cache_dir, max_tokens=65535, max_tokens_read_through=[8192]),
        cache_only=True,
    )
    assert target.score_prompt("A synthetic RAGTruth prompt") == legacy_score
    assert target.usage.summary()["api_calls"] == 0
    assert target.usage.summary()["cache_hits"] >= 3


def test_completed_lower_cap_samples_can_read_through_an_attested_legacy_runtime(
    tmp_path, monkeypatch
):
    """Explicitly attested, same-endpoint legacy samples need no paid replay."""
    monkeypatch.setenv("TEST_SEMANTIC_ENTROPY_KEY", "not-a-real-key")
    legacy_cache = tmp_path / "legacy-cache"
    legacy = SemanticEntropyDetector(
        _config(legacy_cache, max_tokens=4096, runtime_fingerprint="legacy-runtime"),
        entailment_backend=_FakeNLI(),
    )
    samples = iter([
        CompletionSample("same-one", -1.0, 1, 1),
        CompletionSample("same-two", -1.0, 1, 1),
        CompletionSample("different", -2.0, 1, 1),
    ])
    legacy._live_sample = lambda _prompt: next(samples)  # type: ignore[method-assign]
    legacy_score = legacy.score_prompt("A synthetic RAGTruth prompt")
    legacy_identity = llm_runtime_fingerprint(legacy.cfg)

    monkeypatch.delenv("TEST_SEMANTIC_ENTROPY_KEY")
    target = SemanticEntropyDetector(
        _config(
            tmp_path / "target-cache",
            max_tokens=65535,
            max_tokens_read_through=[4096],
            runtime_fingerprint="current-runtime",
            legacy_sample_cache_contracts=[{
                "cache_dir": str(legacy_cache),
                "llm_identity": legacy_identity,
                "api_base": "https://gateway.invalid/v1",
                "max_tokens": [4096],
            }],
        ),
        cache_only=True,
    )
    assert target.score_prompt("A synthetic RAGTruth prompt") == legacy_score
    assert target.usage.summary()["api_calls"] == 0
    assert target.usage.summary()["cache_hits"] >= 3


@pytest.mark.parametrize(
    ("ids", "values"),
    [([], []), ([0, 1], [-1.0]), ([0, 2], [-1.0, -2.0])],
)
def test_semantic_entropy_rejects_misaligned_or_sparse_classes(ids, values):
    with pytest.raises((ValueError, SemanticEntropyError)):
        SemanticEntropyDetector._entropy(ids, values)


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [(None, True), (429, True), (500, True), (503, True), (400, False), (401, False), (404, False)],
)
def test_runner_retries_only_transient_gateway_failures(status_code, expected):
    import importlib.util

    runner_path = ROOT / "scripts" / "run_ragtruth_semantic_entropy.py"
    spec = importlib.util.spec_from_file_location("semantic_entropy_runner", runner_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    error = GatewayRequestError(status_code, "redacted")
    assert module._is_retryable_gateway_failure(error) is expected


def test_runner_reuses_terminal_output_length_marker_on_resume(tmp_path):
    import importlib.util

    runner_path = ROOT / "scripts" / "run_ragtruth_semantic_entropy.py"
    spec = importlib.util.spec_from_file_location("semantic_entropy_runner", runner_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = SimpleNamespace(
        response_id="response-1", source_id="source-1", split="test", y=1, gen_model="test-model"
    )
    path = tmp_path / "score.json"
    payload = module._unscorable_output_length_checkpoint(instance, "manifest-hash", 1.0)
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert module._load_checkpoint(path, instance, "manifest-hash") == payload
    assert module._load_cache_only_unscorable_marker(path, instance, "manifest-hash") == payload
    ordinary = dict(payload, state="scored")
    path.write_text(json.dumps(ordinary), encoding="utf-8")
    assert module._load_cache_only_unscorable_marker(path, instance, "manifest-hash") is None


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


def test_entropy_runtime_config_records_an_exact_legacy_cache_contract(tmp_path):
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
    legacy_cache = tmp_path / "legacy-cache"
    legacy_cache.mkdir()
    legacy_metadata = tmp_path / "legacy-run-metadata.json"
    legacy_identity = {"runtime_fingerprint": "recorded-legacy", "packages": {"numpy": "test"}}
    legacy_metadata.write_text(
        json.dumps({"runtime": {"llm": legacy_identity}}), encoding="utf-8"
    )
    legacy_runtime = tmp_path / "legacy-runtime.yaml"
    legacy_runtime.write_text(yaml.safe_dump({
        "llm": {"api_base": "https://gateway.example.run.app/v1"},
        "semantic_entropy": {"max_tokens": 4096},
    }), encoding="utf-8")
    output = tmp_path / "runtime.yaml"
    command = [
        sys.executable, str(ROOT / "scripts" / "make_local_semantic_entropy_config.py"),
        "--base-config", str(ROOT / "config.yaml"),
        "--gateway-manifest", str(manifest),
        "--gateway-url", "https://gateway.example.run.app",
        "--nli-model-path", str(nli),
        "--data-dir", str(tmp_path / "data"),
        "--cache-root", str(tmp_path / "cache"),
        "--legacy-sample-cache-dir", str(legacy_cache),
        "--legacy-run-metadata", str(legacy_metadata),
        "--legacy-runtime-config", str(legacy_runtime),
        "--output", str(output),
    ]
    subprocess.run(command, check=True)
    generated = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert generated["semantic_entropy"]["legacy_sample_cache_contracts"] == [{
        "cache_dir": str(legacy_cache.resolve()),
        "llm_identity": legacy_identity,
        "api_base": "https://gateway.example.run.app/v1",
        "max_tokens": [4096],
    }]
