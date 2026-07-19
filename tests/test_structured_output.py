"""Offline contracts for the strict native JSON-schema transport."""
from __future__ import annotations

import json
import asyncio
from types import SimpleNamespace

import pytest

from src.dspy_adapter import (
    StructuredOutputParseError,
    StructuredOutputSchemaError,
    XGRAMMAR_STRICT_REQUEST_BACKEND,
    install_dspy_completion_guard,
    is_retryable_llm_exception,
    json_schema_response_format,
    dspy_output_schema,
    specialize_dspy_signature,
    strict_json_loads,
    strict_json_schema_adapter,
    structured_output_settings,
    validate_json_document,
)


RELATION_SCHEMA = {
    "$defs": {
        "Relation": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "object": {"type": "string"},
            },
            "required": ["subject", "predicate", "object"],
            "additionalProperties": False,
        }
    },
    "type": "object",
    "properties": {
        "relations": {
            "type": "array",
            "items": {"$ref": "#/$defs/Relation"},
        }
    },
    "required": ["relations"],
    "additionalProperties": False,
}


class _Field:
    annotation = list[dict[str, str]]


class _Signature:
    __name__ = "ExtractRelations"
    output_fields = {"relations": _Field()}


def test_response_format_wire_shape_preserves_exact_nested_schema():
    response_format = json_schema_response_format(
        RELATION_SCHEMA, name="Extract Relations!"
    )

    assert response_format == {
        "type": "json_schema",
        "json_schema": {
            "name": "Extract_Relations",
            "schema": RELATION_SCHEMA,
            "strict": True,
        },
    }
    assert response_format["json_schema"]["schema"] is not RELATION_SCHEMA


def test_strict_adapter_sends_one_native_schema_request_without_legacy_controls(monkeypatch):
    from dspy.adapters.base import Adapter

    class OutputModel:
        @staticmethod
        def model_json_schema():
            return RELATION_SCHEMA

    monkeypatch.setattr(
        "dspy.adapters.json_adapter._get_structured_outputs_response_format",
        lambda signature: OutputModel,
    )
    calls = []

    def fake_call(self, lm, lm_kwargs, signature, demos, inputs):
        calls.append(dict(lm_kwargs))
        return [{"relations": []}]

    monkeypatch.setattr(Adapter, "__call__", fake_call)
    adapter = strict_json_schema_adapter(
        request_backend=XGRAMMAR_STRICT_REQUEST_BACKEND
    )
    result = adapter(
        object(),
        {"extra_body": {"guided_json": {"old": True}, "trace": True}},
        _Signature,
        [],
        {"source_text": "x"},
    )

    assert result == [{"relations": []}]
    assert len(calls) == 1
    assert calls[0]["response_format"]["json_schema"]["schema"] == RELATION_SCHEMA
    assert calls[0]["extra_body"] == {
        "trace": True,
        "guided_decoding_backend": XGRAMMAR_STRICT_REQUEST_BACKEND,
    }


def test_strict_adapter_does_not_fallback_after_call_failure(monkeypatch):
    from dspy.adapters.base import Adapter

    class OutputModel:
        @staticmethod
        def model_json_schema():
            return RELATION_SCHEMA

    monkeypatch.setattr(
        "dspy.adapters.json_adapter._get_structured_outputs_response_format",
        lambda signature: OutputModel,
    )
    calls = 0

    def fail_once(self, lm, lm_kwargs, signature, demos, inputs):
        nonlocal calls
        calls += 1
        raise RuntimeError("server rejected schema")

    monkeypatch.setattr(Adapter, "__call__", fail_once)
    with pytest.raises(RuntimeError, match="server rejected schema"):
        strict_json_schema_adapter()(object(), {}, _Signature, [], {})
    assert calls == 1


def test_strict_async_adapter_uses_the_same_one_call_contract(monkeypatch):
    from dspy.adapters.base import Adapter

    class OutputModel:
        @staticmethod
        def model_json_schema():
            return RELATION_SCHEMA

    monkeypatch.setattr(
        "dspy.adapters.json_adapter._get_structured_outputs_response_format",
        lambda signature: OutputModel,
    )
    calls = []

    async def fake_acall(self, lm, lm_kwargs, signature, demos, inputs):
        calls.append(dict(lm_kwargs))
        return [{"relations": []}]

    monkeypatch.setattr(Adapter, "acall", fake_acall)
    result = asyncio.run(
        strict_json_schema_adapter(
            request_backend=XGRAMMAR_STRICT_REQUEST_BACKEND
        ).acall(
            object(),
            {"extra_body": {"guided_json": {"old": True}}},
            _Signature,
            [],
            {"source_text": "x"},
        )
    )

    assert result == [{"relations": []}]
    assert len(calls) == 1
    assert calls[0]["response_format"]["json_schema"]["schema"] == RELATION_SCHEMA
    assert calls[0]["extra_body"] == {
        "guided_decoding_backend": XGRAMMAR_STRICT_REQUEST_BACKEND
    }


def test_runtime_relation_contract_binds_endpoints_to_current_entities():
    from kg_gen.steps._2_get_relations import fallback_extraction_sig

    entities = ["Swiss chard", "spinach"]
    _, extract_relations = fallback_extraction_sig(
        entities, is_conversation=False
    )

    signature = specialize_dspy_signature(
        extract_relations,
        {"entities": entities},
    )
    schema = dspy_output_schema(signature)
    relation_schema = next(iter(schema["$defs"].values()))

    assert relation_schema["properties"]["subject"]["enum"] == [
        "Swiss chard",
        "spinach",
    ]
    assert relation_schema["properties"]["object"]["enum"] == [
        "Swiss chard",
        "spinach",
    ]
    predicate_schema = relation_schema["properties"]["predicate"]
    assert predicate_schema["type"] == "string"
    assert "enum" not in predicate_schema


def test_runtime_cluster_contracts_use_current_candidates_and_members():
    import dspy
    from kg_gen.steps._3_cluster_graph import (
        Cluster,
        choose_rep,
        get_check_existing_clusters_sig,
        get_extract_cluster_sig,
        get_validate_cluster_sig,
    )

    extract_cluster, _ = get_extract_cluster_sig(
        {"already-processed", "remaining-b", "remaining-a"}
    )

    cluster_schema = dspy_output_schema(
        specialize_dspy_signature(
            extract_cluster,
            {"items": {"remaining-b", "remaining-a"}},
        )
    )
    assert cluster_schema["properties"]["cluster"]["items"]["enum"] == [
        "remaining-a",
        "remaining-b",
    ]

    validate_cluster, _ = get_validate_cluster_sig(
        {"Swiss chard", "chard", "unrelated"}
    )

    validation_schema = dspy_output_schema(
        specialize_dspy_signature(
            validate_cluster,
            {"cluster": {"Swiss chard", "chard"}},
        )
    )
    assert validation_schema["properties"]["validated_items"]["items"][
        "enum"
    ] == ["Swiss chard", "chard"]

    representative_schema = dspy_output_schema(
        specialize_dspy_signature(
            choose_rep.signature,
            {"cluster": {"Swiss chard", "chard"}},
        )
    )
    assert representative_schema["properties"]["representative"]["enum"] == [
        "Swiss chard",
        "chard",
    ]

    existing_clusters = [Cluster(representative="existing", members={"old"})]
    check_existing = get_check_existing_clusters_sig(
        {"new-a", "new-b"}, existing_clusters
    )
    assignment_signature = dspy.ChainOfThought(check_existing).predict.signature

    assignment_schema = dspy_output_schema(
        specialize_dspy_signature(
            assignment_signature,
            {
                "items": ["new-a", "new-b"],
                "clusters": existing_clusters,
            },
        )
    )
    assignments = assignment_schema["properties"][
        "cluster_reps_that_items_belong_to"
    ]
    assert assignments["minItems"] == assignments["maxItems"] == 2
    assert assignments["items"]["anyOf"] == [
        {"const": "existing", "type": "string"},
        {"type": "null"},
    ]
    all_null = {"cluster_reps_that_items_belong_to": [None, None]}
    if "reasoning" in assignment_schema["properties"]:
        all_null["reasoning"] = "Neither item is equivalent to the cluster."
    validate_json_document(all_null, assignment_schema)


def test_strict_adapter_sends_and_parses_the_same_runtime_specialized_schema(
    monkeypatch,
):
    from dspy.adapters.base import Adapter
    from kg_gen.steps._2_get_relations import fallback_extraction_sig

    entities = ["Swiss chard", "spinach"]
    _, extract_relations = fallback_extraction_sig(
        entities, is_conversation=False
    )

    captured = {}

    def fake_call(self, lm, lm_kwargs, signature, demos, inputs):  # noqa: ARG001
        captured["signature"] = signature
        captured["schema"] = lm_kwargs["response_format"]["json_schema"]["schema"]
        return [{"relations": []}]

    monkeypatch.setattr(Adapter, "__call__", fake_call)
    adapter = strict_json_schema_adapter()
    result = adapter(object(), {}, extract_relations, [], {
        "entities": entities
    })

    assert result == [{"relations": []}]
    assert captured["schema"] == dspy_output_schema(captured["signature"])
    relation_schema = next(iter(captured["schema"]["$defs"].values()))
    assert relation_schema["properties"]["subject"]["enum"] == [
        "Swiss chard",
        "spinach",
    ]
    with pytest.raises(StructuredOutputSchemaError):
        adapter.parse(
            captured["signature"],
            json.dumps({
                "relations": [{
                    "subject": "chard",
                    "predicate": "is similar to",
                    "object": "spinach",
                }]
            }),
        )


def test_runtime_contract_is_per_call_immutable_and_concurrency_safe():
    from concurrent.futures import ThreadPoolExecutor
    from kg_gen.steps._2_get_relations import fallback_extraction_sig

    _, base_signature = fallback_extraction_sig(
        ["placeholder-a", "placeholder-b"], is_conversation=False
    )
    base_before = dspy_output_schema(base_signature)

    def schema_for(entities):
        signature = specialize_dspy_signature(
            base_signature, {"entities": entities}
        )
        return signature, dspy_output_schema(signature)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(schema_for, ["alpha", "beta"])
        second_future = pool.submit(schema_for, ["gamma", "delta"])
        first_signature, first_schema = first_future.result()
        second_signature, second_schema = second_future.result()

    def endpoint_enum(schema):
        relation = next(iter(schema["$defs"].values()))
        return relation["properties"]["subject"]["enum"]

    assert endpoint_enum(first_schema) == ["alpha", "beta"]
    assert endpoint_enum(second_schema) == ["delta", "gamma"]
    assert first_signature is not second_signature
    assert dspy_output_schema(base_signature) == base_before

    first_schema["$defs"].clear()
    assert endpoint_enum(dspy_output_schema(first_signature)) == ["alpha", "beta"]


def test_runtime_contract_handles_empty_relations_and_rejects_empty_cluster_calls():
    from kg_gen.steps._2_get_relations import fallback_extraction_sig
    from kg_gen.steps._3_cluster_graph import get_extract_cluster_sig

    _, relation_signature = fallback_extraction_sig([], is_conversation=False)
    empty_relation_signature = specialize_dspy_signature(
        relation_signature, {"entities": []}
    )
    relation_schema = dspy_output_schema(empty_relation_signature)
    relation_array = relation_schema["properties"]["relations"]
    assert relation_array["minItems"] == relation_array["maxItems"] == 0
    validate_json_document({"relations": []}, relation_schema)
    with pytest.raises(StructuredOutputSchemaError):
        validate_json_document(
            {
                "relations": [{
                    "subject": "invented",
                    "predicate": "mentions",
                    "object": "invented",
                }]
            },
            relation_schema,
        )

    extract_cluster, _ = get_extract_cluster_sig({"original"})
    with pytest.raises(StructuredOutputSchemaError, match="no current candidate"):
        specialize_dspy_signature(extract_cluster, {"items": set()})


def test_strict_parser_rejects_bare_relation_and_json_repair(monkeypatch):
    monkeypatch.setattr("src.dspy_adapter.dspy_output_schema", lambda signature: RELATION_SCHEMA)
    adapter = strict_json_schema_adapter()

    with pytest.raises(StructuredOutputSchemaError):
        adapter.parse(
            _Signature,
            json.dumps({"subject": "Swiss chard", "predicate": "similar", "object": "spinach"}),
        )
    with pytest.raises(StructuredOutputParseError):
        adapter.parse(_Signature, '```json\n{"relations": []}\n```')

    assert adapter.parse(_Signature, '{"relations": []}') == {"relations": []}


@pytest.mark.parametrize(
    "document,match",
    [
        ('{"relations":[],"relations":[]}', "duplicate object key"),
        ('{"value":NaN}', "non-JSON numeric constant"),
        ('{"value":Infinity}', "non-JSON numeric constant"),
    ],
)
def test_strict_json_decoder_rejects_ambiguous_non_json_inputs(document, match):
    with pytest.raises(StructuredOutputParseError, match=match):
        strict_json_loads(document)


def test_dspy_lm_guard_checks_finish_reason_before_adapter_processing():
    class LM:
        def __init__(self):
            self.response = {
                "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]
            }

        def forward(self, **kwargs):  # noqa: ARG002
            return self.response

    lm = LM()
    install_dspy_completion_guard(lm)
    assert lm.forward()["choices"][0]["finish_reason"] == "stop"
    lm.response = {
        "choices": [{"finish_reason": "length", "message": {"content": "{}"}}]
    }
    with pytest.raises(StructuredOutputParseError, match="length"):
        lm.forward()
    install_dspy_completion_guard(lm)
    with pytest.raises(StructuredOutputParseError, match="length"):
        lm.forward()


def test_structured_output_settings_validate_and_keep_legacy_mapping_explicit():
    settings = structured_output_settings(
        SimpleNamespace(
            structured_output_transport="response_format",
            structured_output_backend="xgrammar",
            model_revision="abc",
            runtime_fingerprint="image@sha256:123",
        )
    )
    assert settings.transport == "response_format"
    assert settings.backend == "xgrammar"
    assert settings.request_backend == XGRAMMAR_STRICT_REQUEST_BACKEND
    assert settings.model_revision == "abc"

    with pytest.warns(DeprecationWarning):
        legacy = structured_output_settings(SimpleNamespace(vllm_guided_json=True))
    assert legacy.transport == "guided_json"
    with pytest.raises(ValueError, match="structured_output_transport"):
        structured_output_settings(SimpleNamespace(structured_output_transport="repair"))
    with pytest.raises(ValueError, match="model_revision"):
        structured_output_settings(
            SimpleNamespace(
                structured_output_transport="response_format",
                structured_output_backend="xgrammar",
            )
        )


def test_vertex_structured_output_uses_the_same_schema_without_vllm_extra_body(monkeypatch):
    from dspy.adapters.base import Adapter

    class OutputModel:
        @staticmethod
        def model_json_schema():
            return RELATION_SCHEMA

    monkeypatch.setattr(
        "dspy.adapters.json_adapter._get_structured_outputs_response_format",
        lambda signature: OutputModel,
    )
    settings = structured_output_settings(
        SimpleNamespace(
            structured_output_transport="response_format",
            structured_output_backend="vertex",
            structured_output_request_backend=None,
            model_revision="gateway-release",
            runtime_fingerprint="gateway-manifest",
        )
    )
    assert settings.request_backend is None
    calls = []

    def fake_call(self, lm, lm_kwargs, signature, demos, inputs):
        calls.append(dict(lm_kwargs))
        return [{"relations": []}]

    monkeypatch.setattr(Adapter, "__call__", fake_call)
    strict_json_schema_adapter(request_backend=settings.request_backend)(
        object(), {"extra_body": {"guided_json": {"legacy": True}}}, _Signature, [], {}
    )
    assert calls[0]["response_format"]["json_schema"]["schema"] == RELATION_SCHEMA
    assert "extra_body" not in calls[0]


def test_retry_classifier_retries_only_transient_failures():
    class ServiceUnavailable(Exception):
        status_code = 503

    class BadRequest(Exception):
        status_code = 400

    class ArbitraryServerFailure(Exception):
        status_code = 507

    assert is_retryable_llm_exception(TimeoutError()) is True
    assert is_retryable_llm_exception(ServiceUnavailable()) is True
    assert is_retryable_llm_exception(ArbitraryServerFailure()) is True
    assert is_retryable_llm_exception(BadRequest()) is False
    assert is_retryable_llm_exception(StructuredOutputSchemaError("bad root")) is False
