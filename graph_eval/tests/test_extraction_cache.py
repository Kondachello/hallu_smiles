import json

import pytest

from graph_eval.cache import CacheOnlyMissError, JsonCache
from graph_eval.extraction.base import ExtractionOutput
from graph_eval.extraction.cached import CachedExtractor

IDENT = {
    "prompt_version": "v1",
    "model": "gemini",
    "output_mode": "paper_prompt",
    "gateway_manifest_sha256": "abc",
    "temperature": 0.0,
}
NS = "grapheval/extraction/v1"


class RecExtractor:
    prompt_profile = "rec"

    def __init__(self, raw):
        self.raw = raw
        self.calls = 0

    def extract(self, response_text):
        self.calls += 1
        return ExtractionOutput(self.raw, {"extractor_calls": 1, "input_tokens": 7})


class BoomExtractor:
    prompt_profile = "boom"

    def extract(self, response_text):
        raise AssertionError("extractor must not be called on a warm cache")


def test_roundtrip_and_hit_flag(tmp_path):
    inner = RecExtractor(json.dumps([["a", "r", "b"]]))
    ce = CachedExtractor(inner, JsonCache(tmp_path, NS), identity=IDENT)
    first = ce.extract("ans")
    assert first.usage["extraction_cache_hit"] is False
    assert inner.calls == 1

    warm = CachedExtractor(BoomExtractor(), JsonCache(tmp_path, NS), identity=IDENT)
    second = warm.extract("ans")
    assert second.raw_output == first.raw_output
    assert second.usage == {"extractor_calls": 0, "extraction_cache_hit": True}


def test_identity_change_busts_cache(tmp_path):
    CachedExtractor(RecExtractor("[]"), JsonCache(tmp_path, NS), identity=IDENT).extract("ans")
    inner2 = RecExtractor("[]")
    ce = CachedExtractor(inner2, JsonCache(tmp_path, NS), identity={**IDENT, "model": "other"})
    ce.extract("ans")
    assert inner2.calls == 1  # different identity => recomputed


def test_cache_only_miss_raises(tmp_path):
    ce = CachedExtractor(BoomExtractor(), JsonCache(tmp_path, NS, cache_only=True), identity=IDENT)
    with pytest.raises(CacheOnlyMissError):
        ce.extract("ans")
