"""Extractor protocol.  Extractors see the ANSWER ONLY — never context/query."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ExtractionOutput:
    raw_output: str
    usage: dict = field(default_factory=dict)


@runtime_checkable
class Extractor(Protocol):
    prompt_profile: str

    def extract(self, response_text: str) -> ExtractionOutput:
        """Return raw triple output built from ``response_text`` alone."""
        ...
