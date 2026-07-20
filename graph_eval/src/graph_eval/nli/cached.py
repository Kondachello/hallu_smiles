"""Per-pair caching wrapper around any :class:`NLIModel`.

Cache key (plan section 8) = NLI identity (model revision, evidence policy,
verbalizer version) + canonical (premise, hypothesis).  A warm ``cache_only``
replay finds every pair and never calls the underlying model; a miss raises via
:class:`~graph_eval.cache.CacheOnlyMissError`.  ``last_stats`` exposes hit/miss
counts for usage accounting.
"""
from __future__ import annotations

from typing import Sequence

from ..cache import JsonCache, canonical_json, make_key


class CachedNLI:
    def __init__(self, inner, cache: JsonCache, *, identity: dict):
        self.inner = inner
        self.cache = cache
        self.identity = dict(identity)
        self.model_label = getattr(inner, "model_label", "unknown")
        self.revision = getattr(inner, "revision", None)
        self.last_stats = {"hits": 0, "misses": 0}

    def _key(self, premise: str, hypothesis: str) -> str:
        canonical_input = canonical_json(
            {
                "premise": " ".join(premise.split()),
                "hypothesis": " ".join(hypothesis.split()),
            }
        )
        return make_key(self.identity, canonical_input)

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        pairs = list(pairs)
        results: list[float | None] = [None] * len(pairs)
        miss_positions: list[int] = []
        miss_keys: list[str] = []
        miss_pairs: list[tuple[str, str]] = []

        for i, (premise, hypothesis) in enumerate(pairs):
            key = self._key(premise, hypothesis)
            cached = self.cache.get(key)  # raises in cache_only when absent
            if cached is not None:
                results[i] = float(cached["p_consistent"])
            else:
                miss_positions.append(i)
                miss_keys.append(key)
                miss_pairs.append((premise, hypothesis))

        if miss_pairs:
            fresh = self.inner.score_pairs(miss_pairs)
            if len(fresh) != len(miss_pairs):
                raise ValueError("inner NLI returned wrong number of scores")
            for pos, key, value in zip(miss_positions, miss_keys, fresh):
                value = float(value)
                results[pos] = value
                self.cache.put(key, {"p_consistent": value})

        self.last_stats = {"hits": len(pairs) - len(miss_pairs), "misses": len(miss_pairs)}
        return [float(v) for v in results]  # type: ignore[arg-type]
