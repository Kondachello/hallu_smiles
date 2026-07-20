"""Deterministic, offline extractor for tests and the fake end-to-end path.

Either return a caller-supplied mapping (response text -> list of triples) or a
deterministic fallback: one triple per sentence with >= 3 words, shaped as
``[word0, word1, rest]``.  Output is emitted as JSON so it exercises the real
parser.
"""
from __future__ import annotations

import json

from .base import ExtractionOutput


class FakeExtractor:
    prompt_profile = "fake_v1"

    def __init__(self, mapping: dict[str, list[list[str]]] | None = None):
        self._mapping = mapping or {}

    def extract(self, response_text: str) -> ExtractionOutput:
        if response_text in self._mapping:
            triples = self._mapping[response_text]
        else:
            triples = []
            normalized = response_text.replace("!", ".").replace("?", ".")
            for sentence in (s.strip() for s in normalized.split(".")):
                words = sentence.split()
                if len(words) >= 3:
                    triples.append([words[0], words[1], " ".join(words[2:])])
        return ExtractionOutput(
            raw_output=json.dumps(triples, ensure_ascii=False),
            usage={"extractor_calls": 1},
        )
