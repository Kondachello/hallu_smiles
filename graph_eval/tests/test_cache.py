import pytest

from graph_eval.cache import CacheOnlyMissError, JsonCache, make_key


def test_key_changes_with_identity_and_input():
    base = make_key({"model": "m", "revision": "r1"}, "hello")
    assert base == make_key({"model": "m", "revision": "r1"}, "hello")  # stable
    assert base != make_key({"model": "m", "revision": "r2"}, "hello")  # revision
    assert base != make_key({"model": "m", "revision": "r1"}, "HELLO")  # input


def test_roundtrip_put_get(tmp_path):
    cache = JsonCache(tmp_path, "grapheval/nli/v1")
    key = make_key({"model": "hhem", "revision": "abc"}, "premise|hypothesis")
    assert cache.get(key) is None
    cache.put(key, {"p_consistent": 0.9})
    assert cache.get(key) == {"p_consistent": 0.9}


def test_atomic_write_leaves_no_temp_files(tmp_path):
    cache = JsonCache(tmp_path, "grapheval/extraction/v1")
    key = make_key({"model": "g"}, "resp")
    cache.put(key, {"raw": "[]"})
    leftovers = [p.name for p in cache.root.rglob(".tmp-*")]
    assert leftovers == []


def test_cache_only_miss_raises_and_writes_nothing(tmp_path):
    cache = JsonCache(tmp_path, "grapheval/nli/v1", cache_only=True)
    key = make_key({"model": "hhem"}, "x")
    with pytest.raises(CacheOnlyMissError):
        cache.get(key)
    # cache_only put must be a no-op (proves a replay writes nothing)
    cache.put(key, {"p_consistent": 0.5})
    assert not cache._path(key).exists()
