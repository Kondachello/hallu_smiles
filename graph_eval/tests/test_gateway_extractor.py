import json

import pytest

from graph_eval.config import ExtractorConfig, from_dict
from graph_eval.detector import GraphEvalDetector
from graph_eval.extraction.base import ExtractionError
from graph_eval.extraction.gateway import GatewayExtractor
from graph_eval.extraction.retry import RetryPolicy
from graph_eval.factory import build_extractor, build_nli
from graph_eval.types import DetectionInput


# --- minimal OpenAI-compatible fake client -------------------------------------
class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content, finish):
        self.message = _Msg(content)
        self.finish_reason = finish


class _Usage:
    prompt_tokens = 10
    completion_tokens = 4


class _Resp:
    def __init__(self, content, finish="stop"):
        self.choices = [_Choice(content, finish)]
        self.usage = _Usage()
        self.model = "gemini"
        self.system_fingerprint = "fp1"


class _Status(Exception):
    def __init__(self, code):
        super().__init__(str(code))
        self.status_code = code


class _Completions:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _Chat:
    def __init__(self, comp):
        self.completions = comp


class _Client:
    def __init__(self, script):
        self.chat = _Chat(_Completions(script))


def _noretry():
    return RetryPolicy(max_retries=3, sleep=lambda s: None)


def _cfg(**kw):
    return ExtractorConfig(backend="gateway", **kw)


def test_success_paper_prompt_is_answer_only():
    client = _Client([_Resp(json.dumps([["a", "r", "b"]]))])
    ext = GatewayExtractor(_cfg(), client=client, retry=_noretry())
    out = ext.extract("some answer text")
    assert json.loads(out.raw_output) == [["a", "r", "b"]]
    assert out.usage["extractor_calls"] == 1 and out.usage["repair_attempts"] == 0
    assert out.usage["input_tokens"] == 10
    assert client.chat.completions.calls[0]["messages"][1]["content"] == "some answer text"


def test_structured_json_sets_response_format():
    client = _Client([_Resp(json.dumps({"triples": [["a", "r", "b"]]}))])
    ext = GatewayExtractor(_cfg(output_mode="structured_json"), client=client, retry=_noretry())
    ext.extract("ans")
    kwargs = client.chat.completions.calls[0]
    assert kwargs["response_format"]["type"] == "json_schema"


def test_retry_then_success_on_429():
    client = _Client([_Status(429), _Resp(json.dumps([["a", "r", "b"]]))])
    ext = GatewayExtractor(_cfg(), client=client, retry=_noretry())
    ext.extract("ans")
    assert len(client.chat.completions.calls) == 2


def test_fail_fast_on_400_propagates():
    client = _Client([_Status(400)])
    ext = GatewayExtractor(_cfg(), client=client, retry=_noretry())
    with pytest.raises(_Status):
        ext.extract("ans")


def test_repair_on_malformed_then_success():
    client = _Client([_Resp("garbage not json"), _Resp(json.dumps([["a", "r", "b"]]))])
    ext = GatewayExtractor(_cfg(max_repairs=2), client=client, retry=_noretry())
    out = ext.extract("ans")
    assert out.usage["repair_attempts"] == 1
    assert client.chat.completions.calls[1]["messages"][-2]["content"] == "garbage not json"


def test_repair_exhausted_raises_extraction_error():
    client = _Client([_Resp("garbage"), _Resp("still garbage")])
    ext = GatewayExtractor(_cfg(max_repairs=1), client=client, retry=_noretry())
    with pytest.raises(ExtractionError):
        ext.extract("ans")


def test_length_finish_triggers_repair_even_if_parseable():
    good = json.dumps([["a", "r", "b"]])
    client = _Client([_Resp(good, finish="length"), _Resp(good)])
    ext = GatewayExtractor(_cfg(max_repairs=1), client=client, retry=_noretry())
    out = ext.extract("ans")
    assert out.usage["repair_attempts"] == 1


def test_end_to_end_then_cache_only_replay_makes_zero_calls(tmp_path):
    triples = json.dumps([["Paris", "is", "capital"]])
    cfg = from_dict(
        {"extractor": {"backend": "gateway"}, "nli": {"backend": "fake"}, "cache_dir": str(tmp_path)}
    )
    client = _Client([_Resp(triples)])
    det = GraphEvalDetector(
        build_extractor(cfg, client=client, manifest_sha256="m1"), build_nli(cfg)
    )
    item = DetectionInput("r1", "s1", "Paris is the capital of France.", "the answer")

    first = det.predict(item)
    assert first.status == "ok"
    assert first.usage["extractor_calls"] == 1

    # cache_only replay with a client whose script is empty (any call -> IndexError)
    cfg_ro = from_dict(
        {
            "extractor": {"backend": "gateway"},
            "nli": {"backend": "fake"},
            "cache_dir": str(tmp_path),
            "cache_only": True,
        }
    )
    det_ro = GraphEvalDetector(
        build_extractor(cfg_ro, client=_Client([]), manifest_sha256="m1"), build_nli(cfg_ro)
    )
    replay = det_ro.predict(item)
    assert replay.status == "ok"
    assert replay.raw_score == first.raw_score
    assert replay.usage["extractor_calls"] == 0
    assert replay.usage["nli_calls"] == 0
    assert replay.usage["extraction_cache_hits"] == 1
