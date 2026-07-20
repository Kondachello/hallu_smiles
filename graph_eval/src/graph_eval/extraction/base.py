"""Extractor protocol + error type.  Extractors see the ANSWER ONLY."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ExtractionOutput:
    raw_output: str
    usage: dict = field(default_factory=dict)


class ExtractionError(RuntimeError):
    """Controlled extractor failure after bounded repair; carries the raw output."""

    def __init__(self, message: str, *, raw: str = "", finish: str | None = None):
        super().__init__(message)
        self.raw = raw
        self.finish = finish


@runtime_checkable
class Extractor(Protocol):
    prompt_profile: str

    def extract(self, response_text: str) -> ExtractionOutput:
        """Return raw triple output built from ``response_text`` alone."""
        ...
