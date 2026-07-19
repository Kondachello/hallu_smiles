"""Offline evidence ranking and strict relation-verifier contract tests."""
import sys
from types import SimpleNamespace

import pytest

from src.dspy_adapter import StructuredOutputParseError, StructuredOutputSchemaError
from src.verifier import (
    VERDICT_SCHEMA,
    RelationVerifier,
    RelationVerifierError,
    _parse_verdict,
    select_evidence,
)


class StubVerifier(RelationVerifier):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.backend_calls = 0

    def _call_llm(self, triple, evidence):
        self.backend_calls += 1
        return "entailed"


def _cfg(tmp_path):
    return SimpleNamespace(
        llm=SimpleNamespace(
            model="openai/test", api_base=None, temperature=0.0,
            model_revision="test-model-revision",
            runtime_fingerprint="test-runtime-fingerprint",
            max_tokens=1024,
            max_retries=1, retry_backoff_base_s=0.0,
            request_timeout_s=17,
            structured_output_transport="response_format",
            structured_output_backend="xgrammar",
        ),
        matching=SimpleNamespace(stopwords=["is", "in", "on"]),
        relation_verifier=SimpleNamespace(
            cache_dir=str(tmp_path / "verdicts"), prompt_version="test-v1", max_evidence_sentences=4,
        ),
    )


def test_evidence_ranking_prefers_both_endpoints_and_predicate():
    evidence = select_evidence(
        "Marie Curie was born in Warsaw. She moved to Paris.",
        "Where was Marie Curie born?",
        ("Marie Curie", "born in", "Warsaw"),
        stopwords=["in"],
    )
    assert evidence[0].text == "Marie Curie was born in Warsaw."
    assert evidence[0].rank == 5
    assert evidence[0].source == "context"


def test_verifier_cache_reuses_canonical_triple_and_evidence(tmp_path):
    verifier = StubVerifier(_cfg(tmp_path))
    triple = ("france", "has capital", "paris")
    first = verifier.verify(triple, "France has capital Paris.", None)
    second = verifier.verify(triple, "France has capital Paris.", None)

    assert first.verdict == second.verdict == "entailed"
    assert first.cache_hit is False and second.cache_hit is True
    assert verifier.backend_calls == 1


def test_verifier_sends_closed_native_schema_and_requires_clean_finish(monkeypatch, tmp_path):
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"verdict":"entailed"}'},
                }
            ]
        }

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    verifier = RelationVerifier(_cfg(tmp_path))
    evidence = select_evidence(
        "Paris is the capital of France.",
        None,
        ("Paris", "is capital of", "France"),
    )

    assert verifier._call_llm(("Paris", "is capital of", "France"), evidence) == "entailed"
    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "relation_verdict",
            "schema": VERDICT_SCHEMA,
            "strict": True,
        },
    }
    assert captured["extra_body"] == {
        "guided_decoding_backend": "xgrammar:disable-any-whitespace,no-fallback"
    }
    assert captured["timeout"] == 17
    assert captured["num_retries"] == 0
    assert captured["max_tokens"] == 1024


def test_verifier_cache_key_changes_with_output_budget(tmp_path):
    first_cfg = _cfg(tmp_path)
    second_cfg = _cfg(tmp_path)
    second_cfg.llm.max_tokens = 256
    triple = ("Paris", "is capital of", "France")
    evidence = select_evidence("Paris is the capital of France.", None, triple)

    assert RelationVerifier(first_cfg)._cache_key(triple, evidence, {}) != RelationVerifier(
        second_cfg
    )._cache_key(triple, evidence, {})


def test_verifier_rejects_non_stop_finish_reason_without_parsing(monkeypatch, tmp_path):
    def completion(**kwargs):  # noqa: ARG001
        return {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": '{"verdict":"entailed"}'},
                }
            ]
        }

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    verifier = RelationVerifier(_cfg(tmp_path))
    with pytest.raises(StructuredOutputParseError, match="length"):
        verifier._call_llm(
            ("Paris", "is capital of", "France"),
            select_evidence(
                "Paris is the capital of France.",
                None,
                ("Paris", "is capital of", "France"),
            ),
        )


@pytest.mark.parametrize(
    "content,error",
    [
        ('```json\n{"verdict":"entailed"}\n```', StructuredOutputParseError),
        ('{"verdict":"entailed","explanation":"repair me"}', StructuredOutputSchemaError),
        ('{"verdict":"yes"}', StructuredOutputSchemaError),
        ('["entailed"]', StructuredOutputSchemaError),
    ],
)
def test_verifier_parser_never_repairs_or_weakens_schema(content, error):
    with pytest.raises(error):
        _parse_verdict(content)


def test_verifier_fails_fast_on_parse_error_and_retries_timeout(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.llm.max_retries = 3

    class ParseFailure(RelationVerifier):
        calls = 0

        def _call_llm(self, triple, evidence):  # noqa: ARG002
            self.calls += 1
            raise StructuredOutputParseError("bare Relation-style response")

    deterministic = ParseFailure(cfg)
    with pytest.raises(RelationVerifierError, match="relation verifier failed") as caught:
        deterministic.verify(
            ("Paris", "is capital of", "France"),
            "Paris is the capital of France.",
            None,
        )
    assert isinstance(caught.value.__cause__, StructuredOutputParseError)
    assert deterministic.calls == 1

    class TimeoutThenSuccess(RelationVerifier):
        calls = 0

        def _call_llm(self, triple, evidence):  # noqa: ARG002
            self.calls += 1
            if self.calls < 3:
                raise TimeoutError("temporary verifier timeout")
            return "entailed"

    transient = TimeoutThenSuccess(cfg)
    verdict = transient.verify(
        ("Warsaw", "is birthplace of", "Marie Curie"),
        "Marie Curie was born in Warsaw.",
        None,
    )
    assert verdict.verdict == "entailed"
    assert transient.calls == 3
