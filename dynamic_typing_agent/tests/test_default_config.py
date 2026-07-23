from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_default_config_is_offline_safe_and_has_no_secret_value() -> None:
    config = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    assert config["model"]["backend"] == "fake"
    assert config["nli"]["backend"] == "fake"
    assert config["model"]["api_base"] is None
    assert config["model"]["model"] is None
    assert config["model"]["api_key_env"] == "HALLU_GATEWAY_API_KEY"
    assert "api_key" not in config["model"]


def test_default_policy_preserves_scientific_invariants() -> None:
    config = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    policy = config["policy"]
    assert policy["answer_can_extend_registry"] is False
    assert policy["definition_only_can_confirm_alias"] is False
    assert policy["neutral_is_contradiction"] is False
    assert policy["type_can_prove_identity"] is False

