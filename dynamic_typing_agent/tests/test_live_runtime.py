from __future__ import annotations

from pathlib import Path

import pytest

from hallugraph_dynamic_typing.agent import DynamicTypingAgent
from hallugraph_dynamic_typing.errors import InputContractError
from hallugraph_dynamic_typing.transports import HhemNli


ROOT = Path(__file__).resolve().parents[1]
LIVE_CONFIG = ROOT / "config" / "live-gateway-hhem.yaml"


class _FakeHhemModel:
    def __init__(self, score: float):
        self.score = score

    def predict(self, pairs):
        assert len(pairs) == 1
        return [self.score]


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.95, "entailed"), (0.50, "neutral"), (0.05, "contradicted")],
)
def test_hhem_maps_consistency_score_to_conservative_three_way_verdict(score: float, expected: str) -> None:
    nli = HhemNli(
        model_path="not-loaded-in-test",
        revision="0e7edb3689e710c52ba120086e8f91ea3ee87f23",
        _model=_FakeHhemModel(score),
    )
    result = nli.verify(
        hypothesis_kind="answer_type_assertion",
        premise="Acme is a bank.",
        hypothesis="Acme is a commercial bank.",
        evidence_span_ids=("context:span:0",),
        idempotency_key="test",
    )
    assert result.verdict == expected
    assert "consistency score" in result.rationale


def test_live_yaml_reads_only_environment_values(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("HALLU_TYPING_MODEL", "openai/example-model")
    monkeypatch.setenv("HALLU_GATEWAY_URL", "https://gateway.example.test")
    monkeypatch.setenv("HALLU_GATEWAY_API_KEY", "test-secret-not-written-to-disk")
    monkeypatch.setenv("HALLU_HHEM_MODEL_PATH", str(tmp_path / "hhem"))
    agent = DynamicTypingAgent.from_yaml(LIVE_CONFIG, cache_root=tmp_path / "cache", artifacts_root=tmp_path / "runs")
    assert agent.backend == "live"
    assert agent.invoke_model_nodes is True
    assert agent.nli_backend == "hhem"
    assert agent.model.api_key == "test-secret-not-written-to-disk"
    assert agent.model.api_base == "https://gateway.example.test/v1"


def test_live_yaml_keeps_an_explicit_v1_suffix_once(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("HALLU_TYPING_MODEL", "openai/example-model")
    monkeypatch.setenv("HALLU_GATEWAY_URL", "https://gateway.example.test/v1/")
    monkeypatch.setenv("HALLU_GATEWAY_API_KEY", "test-secret-not-written-to-disk")
    monkeypatch.setenv("HALLU_HHEM_MODEL_PATH", str(tmp_path / "hhem"))
    agent = DynamicTypingAgent.from_yaml(LIVE_CONFIG, cache_root=tmp_path / "cache", artifacts_root=tmp_path / "runs")
    assert agent.model.api_base == "https://gateway.example.test/v1"


def test_live_yaml_fails_closed_when_gateway_secret_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HALLU_TYPING_MODEL", "HALLU_GATEWAY_URL", "HALLU_GATEWAY_API_KEY", "HALLU_HHEM_MODEL_PATH"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(InputContractError, match="HALLU_GATEWAY_URL"):
        DynamicTypingAgent.from_yaml(LIVE_CONFIG)
