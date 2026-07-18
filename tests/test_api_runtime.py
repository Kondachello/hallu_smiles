from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

import dspy
from pydantic import BaseModel

from src.api_runtime import (
    CacheOnlyMissError,
    StructuredOutputParseError,
    StructuredOutputSchemaError,
    configure_dspy_lm,
    is_retryable_exception,
    llm_runtime_fingerprint,
    retry_after_seconds,
    strict_json_object_adapter,
    strict_kggen_map_batch_items,
    validate_completion_envelope,
)
from src.extract import FakeKGGen, KGExtractor, UsageLogger
from src.verifier import RelationVerifier, _parse_verdict


def _cfg(tmp_path):
    return SimpleNamespace(
        llm=SimpleNamespace(
            model="openai/qwen3-8b",
            api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            api_key_env="DASHSCOPE_API_KEY",
            temperature=0.0,
            max_tokens=1024,
            request_timeout_s=180,
            max_retries=5,
            retry_backoff_base_s=0,
            concurrency=1,
            structured_output_transport="json_object",
            extra_body={"enable_thinking": False},
        ),
        extraction=SimpleNamespace(cluster=True, context_chunk_chars=6000),
        matching=SimpleNamespace(stopwords=[]),
        relation_verifier=SimpleNamespace(
            cache_dir=str(tmp_path / "verdicts"),
            prompt_version="relation-entailment-v1",
            max_evidence_sentences=4,
        ),
        cache_dir=str(tmp_path / "kg"),
    )


class _TestRelation(BaseModel):
    subject: str
    predicate: str
    object: str


class _TestExtractRelations(dspy.Signature):
    source_text: str = dspy.InputField()
    relations: list[_TestRelation] = dspy.OutputField()


def _relation_signature():
    return _TestExtractRelations


def test_adapter_sends_json_object_and_thinking_false():
    captured = {}

    class LM:
        def __call__(self, *, messages, **kwargs):
            captured["messages"] = messages
            captured.update(kwargs)
            return [
                '{"relations":[{"subject":"Swiss chard",'
                '"predicate":"is similar to","object":"spinach"}]}'
            ]

    result = strict_json_object_adapter(
        extra_body={"enable_thinking": False}
    )(
        LM(),
        {},
        _relation_signature(),
        [],
        {"source_text": "Swiss chard is similar to spinach."},
    )

    assert captured["response_format"] == {"type": "json_object"}
    assert captured["extra_body"] == {"enable_thinking": False}
    assert "JSON object" in json.dumps(captured["messages"])
    assert result[0]["relations"][0].subject == "Swiss chard"


@pytest.mark.parametrize(
    "document,error",
    [
        (
            '{"subject":"Swiss chard","predicate":"is similar to",'
            '"object":"spinach"}',
            StructuredOutputSchemaError,
        ),
        (
            '```json\n{"relations":[]}\n```',
            StructuredOutputParseError,
        ),
        (
            '{"relations":[],"unexpected":true}',
            StructuredOutputSchemaError,
        ),
        ('{"relations":[', StructuredOutputParseError),
    ],
)
def test_adapter_rejects_repair_wrap_fences_and_extra_fields(document, error):
    adapter = strict_json_object_adapter(extra_body={"enable_thinking": False})
    with pytest.raises(error):
        adapter.parse(_relation_signature(), document)


def test_verifier_parser_is_closed_and_does_not_strip_fences():
    assert _parse_verdict('{"verdict":"entailed"}') == "entailed"
    with pytest.raises(StructuredOutputParseError):
        _parse_verdict('```json\n{"verdict":"entailed"}\n```')
    with pytest.raises(StructuredOutputSchemaError):
        _parse_verdict('{"verdict":"entailed","reason":"looks right"}')


def test_retry_policy_is_transient_only_and_reads_retry_after():
    class HTTPError(RuntimeError):
        def __init__(self, status, headers=None):
            self.status_code = status
            self.headers = headers or {}

    assert is_retryable_exception(HTTPError(429, {"Retry-After": "7"}))
    assert retry_after_seconds(HTTPError(429, {"Retry-After": "7"})) == 7
    assert is_retryable_exception(HTTPError(503))
    assert is_retryable_exception(TimeoutError())
    assert not is_retryable_exception(HTTPError(400))
    assert not is_retryable_exception(StructuredOutputSchemaError("wrong root"))


def test_kggen_clustering_re_raises_contract_and_provider_errors():
    cluster = SimpleNamespace(representative="Paris", members={"Paris"})

    def schema_failure(**kwargs):
        raise StructuredOutputSchemaError("wrong cluster root")

    with pytest.raises(StructuredOutputSchemaError):
        strict_kggen_map_batch_items(
            {"PARIS"}, ["Paris"], {"Paris": cluster}, {}, "entities", schema_failure
        )

    def ordinary_failure(**kwargs):
        raise RuntimeError("upstream validation failure")

    assignments = strict_kggen_map_batch_items(
        {"PARIS"}, ["Paris"], {"Paris": cluster}, {}, "entities", ordinary_failure
    )
    assert assignments == {"PARIS": None}

    class AuthenticationError(RuntimeError):
        status_code = 401

    def auth_failure(**kwargs):
        raise AuthenticationError("invalid key")

    with pytest.raises(AuthenticationError):
        strict_kggen_map_batch_items(
            {"PARIS"}, ["Paris"], {"Paris": cluster}, {}, "entities", auth_failure
        )


def test_process_default_adapter_reaches_native_chunk_worker(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    class LM:
        def __init__(self):
            self.kwargs = {}
            self.num_retries = 3
            self.cache = True

        def forward(self, **kwargs):  # pragma: no cover - not called in this test
            raise AssertionError("no provider call expected")

    cfg = _cfg(tmp_path)
    lm = LM()
    installed = configure_dspy_lm(lm, cfg)

    def chunk_worker():
        # This is the same ordinary ThreadPoolExecutor + nested context pattern
        # used by KGGen 0.4's native chunker.
        with dspy.context(lm=lm):
            return dspy.settings.adapter

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker_adapter = pool.submit(chunk_worker).result()

    assert worker_adapter is installed
    assert lm.kwargs["response_format"] == {"type": "json_object"}
    assert lm.kwargs["extra_body"] == {"enable_thinking": False}
    assert lm.num_retries == 0
    assert lm.cache is False


def test_dspy_provider_gate_serializes_native_chunk_workers(tmp_path):
    class LM:
        def __init__(self):
            self.kwargs = {}
            self.num_retries = 3
            self.cache = True
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def forward(self, **kwargs):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.02)
            with self.lock:
                self.active -= 1
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content='{"relations":[]}'),
                    )
                ]
            )

    lm = LM()
    configure_dspy_lm(lm, _cfg(tmp_path))
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: lm.forward(), range(8)))

    assert lm.max_active == 1


@pytest.mark.parametrize("content", [None, "", "   "])
def test_completion_envelope_rejects_null_or_empty_content(content):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content=content)
            )
        ]
    )
    with pytest.raises(StructuredOutputParseError):
        validate_completion_envelope(response)


def test_schema_failure_is_one_contract_telemetry_event_not_success(tmp_path):
    class LM:
        def __init__(self):
            self.kwargs = {}
            self.num_retries = 3
            self.cache = True

        def forward(self, **kwargs):
            return SimpleNamespace(
                id="bad-root-request",
                usage=SimpleNamespace(
                    prompt_tokens=12, completion_tokens=6, total_tokens=18
                ),
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            content=(
                                '{"subject":"Swiss chard",'
                                '"predicate":"is similar to","object":"spinach"}'
                            )
                        ),
                    )
                ],
            )

        def __call__(self, *, messages, **kwargs):
            response = self.forward(messages=messages, **kwargs)
            return [response.choices[0].message.content]

    cfg = _cfg(tmp_path)
    usage = UsageLogger(tmp_path / "usage.jsonl")
    lm = LM()
    adapter = configure_dspy_lm(lm, cfg, usage)

    with pytest.raises(StructuredOutputSchemaError):
        adapter(
            lm,
            {},
            _relation_signature(),
            [],
            {"source_text": "Swiss chard is similar to spinach."},
        )

    records = [
        json.loads(line)
        for line in (tmp_path / "provider_calls.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["outcome"] == "contract_error"
    assert records[0]["request_id"] == "bad-root-request"
    assert records[0]["http_status"] == 200
    assert usage.summary()["provider_calls"] == 1
    assert usage.summary()["provider_successes"] == 0


def test_dspy_retries_each_provider_call_and_records_truthful_attempts(tmp_path):
    class TemporaryError(RuntimeError):
        status_code = 503
        headers = {"Retry-After": "0"}

    class LM:
        def __init__(self):
            self.kwargs = {}
            self.num_retries = 3
            self.cache = True
            self.attempts = 0

        def forward(self, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise TemporaryError("try again")
            return SimpleNamespace(
                id="retry-success",
                usage=SimpleNamespace(
                    prompt_tokens=9, completion_tokens=4, total_tokens=13
                ),
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content='{"relations":[]}'),
                    )
                ],
            )

        def __call__(self, *, messages, **kwargs):
            response = self.forward(messages=messages, **kwargs)
            return [response.choices[0].message.content]

    usage = UsageLogger(tmp_path / "usage.jsonl")
    lm = LM()
    adapter = configure_dspy_lm(lm, _cfg(tmp_path), usage)
    assert adapter(
        lm,
        {},
        _relation_signature(),
        [],
        {"source_text": "no relation"},
    )[0]["relations"] == []

    records = [
        json.loads(line)
        for line in (tmp_path / "provider_calls.jsonl").read_text().splitlines()
    ]
    assert [(row["outcome"], row["http_status"], row["retry_index"]) for row in records] == [
        ("failure", 503, 0),
        ("success", 200, 1),
    ]


def test_cache_fingerprint_contains_provider_runtime_but_not_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "never-write-this-secret")
    fingerprint = llm_runtime_fingerprint(_cfg(tmp_path))
    encoded = json.dumps(fingerprint, sort_keys=True)

    assert fingerprint["model"] == "openai/qwen3-8b"
    assert fingerprint["response_format"] == {"type": "json_object"}
    assert fingerprint["extra_body"] == {"enable_thinking": False}
    assert "runtime_versions" in fingerprint
    assert {
        "python", "torch", "kg-gen", "dspy", "litellm", "pydantic",
        "jsonschema", "sentence-transformers", "transformers", "tenacity",
    } <= set(fingerprint["runtime_versions"])
    assert "never-write-this-secret" not in encoded
    assert "DASHSCOPE_API_KEY" not in encoded


def test_provider_telemetry_is_allowlisted_and_redacted(tmp_path):
    path = tmp_path / "provider_calls.jsonl"
    usage = UsageLogger(tmp_path / "usage.jsonl", provider_calls_path=path)
    response = SimpleNamespace(
        id="request-123",
        usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=3, total_tokens=13
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content="secret prompt"))],
    )
    usage.record_provider_call(outcome="success", seconds=0.25, response=response)
    raw = path.read_text(encoding="utf-8")
    record = json.loads(raw)

    assert record == {
        "completion_tokens": 3,
        "error_type": None,
        "http_status": 200,
        "latency_s": 0.25,
        "outcome": "success",
        "prompt_tokens": 10,
        "request_id": "request-123",
        "retry_index": 0,
        "total_tokens": 13,
    }
    assert "secret prompt" not in raw
    assert usage.summary()["provider_calls"] == 1


def test_extractor_cache_only_replays_and_fails_closed_on_miss(tmp_path):
    cfg = _cfg(tmp_path)
    live = KGExtractor(cfg, backend=FakeKGGen())
    expected = live.extract("Paris is in France.")
    replay = KGExtractor(cfg, backend=FakeKGGen(), cache_only=True)

    assert replay.extract("Paris is in France.") == expected
    with pytest.raises(CacheOnlyMissError):
        replay.extract("This text was never cached.")


def test_verifier_cache_only_replays_and_fails_closed_on_miss(tmp_path):
    class StubVerifier(RelationVerifier):
        def _call_llm(self, triple, evidence):
            return "entailed"

    cfg = _cfg(tmp_path)
    triple = ("france", "has capital", "paris")
    context = "France has capital Paris."
    assert StubVerifier(cfg).verify(triple, context, None).verdict == "entailed"

    replay = RelationVerifier(cfg, cache_only=True)
    assert replay.verify(triple, context, None).cache_hit is True
    with pytest.raises(CacheOnlyMissError):
        replay.verify(("germany", "has capital", "berlin"), "Germany has capital Berlin.", None)
