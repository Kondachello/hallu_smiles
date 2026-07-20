"""Deterministic triple -> NLI hypothesis string.

The verbalizer version is part of the NLI cache key; changing this string must
bump the version so stale verdicts are not silently reused.
"""
from __future__ import annotations

from .types import Triple

VERBALIZER_VERSION = "v1"


def verbalize(triple: Triple) -> str:
    """Render ``t`` as ``"<subject> <relation> <object>."`` with collapsed spaces."""
    subject = " ".join(triple.raw_subject.split())
    relation = " ".join(triple.raw_relation.split())
    obj = " ".join(triple.raw_object.split())
    return f"{subject} {relation} {obj}."
