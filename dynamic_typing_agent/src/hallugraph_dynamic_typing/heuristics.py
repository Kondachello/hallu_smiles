"""Small deterministic helpers that do not make semantic typing decisions.

Semantic entity typing belongs to :mod:`quality_workflow` and always passes through the
model decision and NLI stages. In particular, this module must never interpret a generic
knowledge-graph relation such as ``is`` as an entity-to-type assertion.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from .models import EvidenceSpan, normalize


SENTENCE = re.compile(r"[^.!?]+[.!?]?", re.UNICODE)


def make_spans(text: str, role: str) -> tuple[EvidenceSpan, ...]:
    spans: list[EvidenceSpan] = []
    for index, match in enumerate(SENTENCE.finditer(text)):
        raw = match.group(0)
        stripped = raw.strip()
        if not stripped:
            continue
        start = match.start() + (len(raw) - len(raw.lstrip()))
        end = start + len(stripped)
        spans.append(EvidenceSpan(span_id=f"{role}:span:{index}", source_role=role, start_char=start, end_char=end, text=stripped))
    if not spans and text.strip():
        spans.append(EvidenceSpan(span_id=f"{role}:span:0", source_role=role, start_char=0, end_char=len(text), text=text))
    return tuple(spans)


def evidence_for(surface: str, spans: Iterable[EvidenceSpan]) -> tuple[str, ...]:
    key = normalize(surface)
    ids = [item.span_id for item in spans if key and key in normalize(item.text)]
    return tuple(ids)


def registry_checksum(payload: dict) -> str:
    from .models import canonical_json

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
