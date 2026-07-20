import pytest

from graph_eval.cache import CacheOnlyMissError, JsonCache
from graph_eval.nli.cached import CachedNLI

IDENTITY = {
    "nli_model_revision": "r1",
    "evidence_policy": "full_context_native",
    "verbalizer_version": "v1",
}
NS = "grapheval/nli/v1"


class RecordingNLI:
    model_label = "rec"
    revision = "r1"

    def __init__(self, table):
        self.table = table
        self.calls = []

    def score_pairs(self, pairs):
        self.calls.extend(pairs)
        return [self.table[p] for p in pairs]


class BoomNLI:
    model_label = "boom"
    revision = "r1"

    def score_pairs(self, pairs):
        raise AssertionError("inner NLI must not be called on a warm cache")


def test_roundtrip_and_stats(tmp_path):
    inner = RecordingNLI({("c", "a."): 0.9})
    nli = CachedNLI(inner, JsonCache(tmp_path, NS), identity=IDENTITY)
    assert nli.score_pairs([("c", "a.")]) == [0.9]
    assert nli.last_stats == {"hits": 0, "misses": 1}

    warm = CachedNLI(BoomNLI(), JsonCache(tmp_path, NS), identity=IDENTITY)
    assert warm.score_pairs([("c", "a.")]) == [0.9]  # served from disk cache
    assert warm.last_stats == {"hits": 1, "misses": 0}


def test_partial_hit_only_calls_inner_for_misses(tmp_path):
    cache = JsonCache(tmp_path, NS)
    inner = RecordingNLI({("c", "a."): 0.8, ("c", "b."): 0.2})
    nli = CachedNLI(inner, cache, identity=IDENTITY)
    nli.score_pairs([("c", "a.")])  # prime a.
    inner.calls.clear()
    out = nli.score_pairs([("c", "a."), ("c", "b.")])
    assert out == [0.8, 0.2]
    assert inner.calls == [("c", "b.")]  # only the miss recomputed
    assert nli.last_stats == {"hits": 1, "misses": 1}


def test_identity_change_busts_cache(tmp_path):
    CachedNLI(
        RecordingNLI({("c", "a."): 0.9}), JsonCache(tmp_path, NS), identity=IDENTITY
    ).score_pairs([("c", "a.")])
    inner2 = RecordingNLI({("c", "a."): 0.1})
    other = {**IDENTITY, "nli_model_revision": "r2"}
    nli2 = CachedNLI(inner2, JsonCache(tmp_path, NS), identity=other)
    assert nli2.score_pairs([("c", "a.")]) == [0.1]  # recomputed under new identity
    assert inner2.calls == [("c", "a.")]


def test_cache_only_warm_makes_zero_model_calls(tmp_path):
    CachedNLI(
        RecordingNLI({("c", "a."): 0.7}), JsonCache(tmp_path, NS), identity=IDENTITY
    ).score_pairs([("c", "a.")])
    warm = CachedNLI(
        BoomNLI(), JsonCache(tmp_path, NS, cache_only=True), identity=IDENTITY
    )
    assert warm.score_pairs([("c", "a.")]) == [0.7]
    assert warm.last_stats == {"hits": 1, "misses": 0}


def test_cache_only_miss_raises(tmp_path):
    nli = CachedNLI(BoomNLI(), JsonCache(tmp_path, NS, cache_only=True), identity=IDENTITY)
    with pytest.raises(CacheOnlyMissError):
        nli.score_pairs([("c", "missing.")])
