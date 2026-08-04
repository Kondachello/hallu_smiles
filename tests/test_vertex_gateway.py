"""Offline wire-contract tests for the Cloud Run Vertex gateway."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.core import (
    GATEWAY_PROTOCOL,
    GatewayError,
    GatewaySettings,
    authenticate,
    canonical_manifest_sha256,
    gateway_manifest,
    openai_response,
    parse_chat_request,
    translate_vertex_error,
)


def _settings() -> GatewaySettings:
    return GatewaySettings(
        api_key="not-a-real-key",
        logical_model="openai/gemini-2.5-flash",
        vertex_model="gemini-2.5-flash",
        project="test-project",
        location="europe-west4",
        release="git:abc123",
        cloud_run_revision="hallu-00001-abc",
    )


SCHEMA = {
    "$defs": {"Relation": {"type": "object", "properties": {"subject": {"type": "string"}}}},
    "type": "object",
    "properties": {"relations": {"type": "array", "items": {"$ref": "#/$defs/Relation"}}},
    "required": ["relations"],
    "additionalProperties": False,
}


def test_manifest_is_stable_and_does_not_contain_the_secret():
    manifest = gateway_manifest(_settings())
    assert manifest["protocol"] == GATEWAY_PROTOCOL
    assert manifest["logical_model"] == "openai/gemini-2.5-flash"
    assert "not-a-real-key" not in str(manifest)
    assert canonical_manifest_sha256(manifest) == canonical_manifest_sha256(dict(manifest))


def test_settings_does_not_fall_back_to_process_environment_for_an_explicit_empty_mapping():
    with pytest.raises(RuntimeError, match="HALLU_LOGICAL_MODEL"):
        GatewaySettings.from_env({})


def test_bearer_auth_uses_a_single_public_failure_message():
    authenticate("Bearer not-a-real-key", "not-a-real-key")
    for value in (None, "Basic x", "Bearer wrong"):
        with pytest.raises(GatewayError) as raised:
            authenticate(value, "not-a-real-key")
        assert raised.value.status_code == 401
        assert raised.value.envelope()["error"]["message"] == "missing or invalid bearer credential"


def test_openai_request_becomes_vertex_contents_without_mutating_the_schema():
    payload = {
        "model": "gemini-2.5-flash",
        "messages": [
            {"role": "system", "content": "Return only JSON."},
            {"role": "user", "content": "Extract facts."},
            {"role": "assistant", "content": "{}"},
            {"role": "user", "content": "Continue."},
        ],
        "temperature": 0,
        "max_tokens": 1024,
        "response_format": {"type": "json_schema", "json_schema": {"name": "kg", "strict": True, "schema": SCHEMA}},
    }
    parsed = parse_chat_request(payload, _settings())
    assert parsed["system_instruction"] == "Return only JSON."
    assert parsed["contents"] == [
        {"role": "user", "text": "Extract facts."},
        {"role": "model", "text": "{}"},
        {"role": "user", "text": "Continue."},
    ]
    assert parsed["response_json_schema"] == SCHEMA
    assert parsed["response_json_schema"] is not SCHEMA


def test_logprobs_request_is_narrow_and_opt_in():
    parsed = parse_chat_request({
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Sample an answer."}],
        "logprobs": True,
        "top_logprobs": 0,
    }, _settings())
    assert parsed["response_logprobs"] is True
    assert parsed["top_logprobs"] == 0


@pytest.mark.parametrize("payload, message", [
    ({"logprobs": "yes"}, "logprobs must be a boolean"),
    ({"top_logprobs": 1}, "top_logprobs requires"),
    ({"logprobs": True, "top_logprobs": 21}, "top_logprobs must be"),
])
def test_invalid_logprob_requests_fail_before_vertex(payload, message):
    payload = {
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "x"}],
        **payload,
    }
    with pytest.raises(GatewayError, match=message):
        parse_chat_request(payload, _settings())


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"model": "other", "messages": [{"role": "user", "content": "x"}]}, "configured gateway model"),
        ({"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "x"}], "stream": True}, "streaming"),
        ({"model": "gemini-2.5-flash", "messages": [{"role": "tool", "content": "x"}]}, "unsupported"),
        ({"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "x"}], "response_format": {"type": "json_object"}}, "json_schema"),
    ],
)
def test_invalid_openai_features_fail_before_vertex(payload, message):
    with pytest.raises(GatewayError, match=message):
        parse_chat_request(payload, _settings())


def test_vertex_response_becomes_one_openai_choice_with_usage_and_finish_reason():
    response = SimpleNamespace(
        model_version="gemini-2.5-flash-001",
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=7, total_token_count=19),
        candidates=[
            SimpleNamespace(
                finish_reason="STOP",
                content=SimpleNamespace(parts=[SimpleNamespace(text='{"relations":[]}')]),
            )
        ],
    )
    result = openai_response(response, _settings())
    assert result["model"] == "gemini-2.5-flash"
    assert result["choices"] == [{"index": 0, "message": {"role": "assistant", "content": '{"relations":[]}'}, "finish_reason": "stop"}]
    assert result["usage"] == {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
    assert "not-a-real-key" not in str(result)


def test_protobuf_style_vertex_finish_reason_is_normalised():
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(),
        candidates=[
            SimpleNamespace(
                finish_reason="FinishReason.MAX_TOKENS",
                content=SimpleNamespace(parts=[SimpleNamespace(text="{}")]),
            )
        ],
    )
    assert openai_response(response, _settings())["choices"][0]["finish_reason"] == "length"


def test_vertex_selected_token_logprobs_are_returned_only_when_present():
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(),
        candidates=[SimpleNamespace(
            finish_reason="STOP",
            content=SimpleNamespace(parts=[SimpleNamespace(text="answer")]),
            logprobs_result=SimpleNamespace(chosen_candidates=[
                SimpleNamespace(token="answer", token_id=7, log_probability=-0.25),
                SimpleNamespace(token="<eos>", token_id=1, log_probability=-0.5),
            ]),
        )],
    )
    logprobs = openai_response(response, _settings())["choices"][0]["logprobs"]
    assert [row["logprob"] for row in logprobs["content"]] == [-0.25, -0.5]
    assert all(row["top_logprobs"] == [] for row in logprobs["content"])


def test_vertex_error_mapping_preserves_transient_retry_boundary():
    assert translate_vertex_error(SimpleNamespace(code=429)).status_code == 429
    assert translate_vertex_error(SimpleNamespace(code=503)).status_code == 503
    assert translate_vertex_error(SimpleNamespace(code=403)).status_code == 403
    assert translate_vertex_error(TimeoutError()).status_code == 503
