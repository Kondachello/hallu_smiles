from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from hallugraph_dynamic_typing.errors import TransportError
from hallugraph_dynamic_typing.transports import LiteLLMStructuredModel, parse_json_completion, vertex_compatible_schema


SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
    "additionalProperties": False,
}


class _GatewayError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def _model(completion_callable, **kwargs) -> LiteLLMStructuredModel:
    return LiteLLMStructuredModel(
        model="openai/test-model",
        api_base="https://gateway.example.test",
        api_key="secret-value",
        timeout_seconds=1,
        retry_backoff_base_seconds=0,
        retry_backoff_max_seconds=0,
        retry_jitter_seconds=0,
        completion_callable=completion_callable,
        sleep_callable=lambda _: None,
        **kwargs,
    )


def _invoke(model: LiteLLMStructuredModel) -> dict:
    return dict(
        model.invoke(
            operation="test_operation",
            messages=(HumanMessage(content="Return JSON."),),
            output_schema=SCHEMA,
            idempotency_key="fixed-key",
        )
    )


def test_rate_limit_is_retried_with_one_stable_idempotency_key() -> None:
    calls: list[dict] = []

    def completion(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise _GatewayError(429, "rate limit")
        return {"choices": [{"finish_reason": "stop", "message": {"content": '{"value":"ok"}'}}]}

    assert _invoke(_model(completion, max_attempts=3)) == {"value": "ok"}
    assert len(calls) == 3
    assert {call["metadata"]["idempotency_key"] for call in calls} == {"fixed-key"}
    assert {call["num_retries"] for call in calls} == {0}


def test_non_retryable_gateway_error_is_redacted_and_stops_immediately() -> None:
    calls = 0

    def completion(**_kwargs):
        nonlocal calls
        calls += 1
        raise _GatewayError(401, "Authorization=Bearer secret-value is rejected")

    with pytest.raises(TransportError) as raised:
        _invoke(_model(completion, max_attempts=5))
    message = str(raised.value)
    assert calls == 1
    assert "after 1/5 attempt(s)" in message
    assert "status=401" in message
    assert "secret-value" not in message
    assert "Authorization=***" in message


def test_temporary_server_error_is_bounded() -> None:
    calls = 0

    def completion(**_kwargs):
        nonlocal calls
        calls += 1
        raise _GatewayError(503, "temporary upstream failure")

    with pytest.raises(TransportError, match="after 2/2 attempt"):
        _invoke(_model(completion, max_attempts=2))
    assert calls == 2


def test_vertex_wire_schema_uses_supported_subset_but_preserves_local_contract() -> None:
    original = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:test",
        "type": "object",
        "properties": {
            "version": {"const": "v1"},
            "label": {"type": "string", "minLength": 1, "maxLength": 10, "pattern": "^[a-z]+$"},
            "items": {"type": "array", "uniqueItems": True, "minItems": 1, "maxItems": 3},
        },
        "required": ["version", "label"],
        "additionalProperties": False,
    }
    wire = vertex_compatible_schema(original)
    assert "$schema" not in wire
    assert wire["$id"] == "urn:test"
    assert wire["properties"]["version"] == {"enum": ["v1"]}
    assert wire["properties"]["label"] == {"type": "string"}
    assert wire["properties"]["items"] == {"type": "array", "minItems": 1, "maxItems": 3}
    assert original["properties"]["label"]["pattern"] == "^[a-z]+$"


def test_prompt_schema_profile_omits_wire_schema_but_keeps_it_in_system_instruction() -> None:
    calls: list[dict] = []

    def completion(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"finish_reason": "stop", "message": {"content": '{"value":"ok"}'}}]}

    model = _model(completion, max_attempts=1, structured_schema_profile="prompt")
    assert _invoke(model) == {"value": "ok"}
    assert "response_format" not in calls[0]
    assert "required JSON Schema follows" in calls[0]["messages"][0]["content"]
    assert '"value"' in calls[0]["messages"][0]["content"]


def test_json_parser_accepts_a_single_markdown_fence_but_still_validates_schema() -> None:
    assert parse_json_completion("```json\n{\"value\": \"ok\"}\n```", SCHEMA, "test") == {"value": "ok"}
    with pytest.raises(Exception, match="required schema"):
        parse_json_completion("```json\n{\"wrong\": \"ok\"}\n```", SCHEMA, "test")
