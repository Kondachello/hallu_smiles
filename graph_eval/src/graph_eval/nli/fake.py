"""Deterministic, offline NLI stub with the correct score direction.

``p_consistent`` = fraction of hypothesis content words present in the premise
(lowercased token overlap), or a caller-supplied exact mapping.  A fully grounded
hypothesis scores ~1.0 (=> p_unsupported ~0); an ungrounded one scores low.
"""
from __future__ import annotations

from typing import Sequence


class FakeNLI:
    model_label = "fake-nli-v1"
    revision = "fake"

    def __init__(self, mapping: dict[tuple[str, str], float] | None = None):
        self._mapping = mapping or {}

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for premise, hypothesis in pairs:
            if (premise, hypothesis) in self._mapping:
                scores.append(float(self._mapping[(premise, hypothesis)]))
                continue
            premise_tokens = set(premise.lower().split())
            hypothesis_tokens = [
                w for w in hypothesis.lower().strip(".").split() if len(w) > 2
            ]
            if not hypothesis_tokens:
                scores.append(1.0)
            else:
                hits = sum(1 for w in hypothesis_tokens if w in premise_tokens)
                scores.append(hits / len(hypothesis_tokens))
        return scores
