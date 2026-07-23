"""Deterministic, conservative local fallback used by fake mode and tests.

These rules are not a scientific type induction method. They make the standalone graph
executable without a live model while preserving every safety boundary for later model nodes.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from .models import EvidenceLevel, EvidenceSpan, GraphInput, TypeAssignment, TypeDefinition, normalize, stable_id


SENTENCE = re.compile(r"[^.!?]+[.!?]?", re.UNICODE)
TYPE_RELATIONS = {"is a", "is", "instance of", "type"}


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


def explicit_type_pairs(graph: GraphInput) -> tuple[tuple[str, str], ...]:
    return tuple((subject, obj) for subject, relation, obj in graph.relations if normalize(relation) in TYPE_RELATIONS)


def build_registry_parts(
    *, context: GraphInput, query: GraphInput, spans: tuple[EvidenceSpan, ...]
) -> tuple[tuple[TypeDefinition, ...], tuple[TypeAssignment, ...]]:
    root_id = "T-ENTITY"
    types: dict[str, TypeDefinition] = {
        root_id: TypeDefinition(
            type_id=root_id,
            label="entity",
            definition="A local root type for every source entity.",
            evidence_span_ids=(),
            evidence_level=EvidenceLevel.SOURCE_ENTAILED,
            status="confirmed",
        )
    }
    assignments: list[TypeAssignment] = []
    for graph in (context, query):
        pairs = explicit_type_pairs(graph)
        type_by_subject: dict[str, list[str]] = {}
        for subject, label in pairs:
            normalized_label = normalize(label)
            type_id = "T-" + hashlib.sha256(normalized_label.encode("utf-8")).hexdigest()[:12].upper()
            span_ids = evidence_for(label, spans)
            if type_id not in types:
                types[type_id] = TypeDefinition(
                    type_id=type_id,
                    label=label,
                    definition=f"Source-local type explicitly named as '{label}'.",
                    parent_type_ids=(root_id,),
                    evidence_span_ids=span_ids,
                    evidence_level=EvidenceLevel.SOURCE_ENTAILED if span_ids else EvidenceLevel.UNKNOWN,
                    status="confirmed" if span_ids else "preliminary",
                )
            type_by_subject.setdefault(normalize(subject), []).append(type_id)
        for entity in graph.entities:
            type_ids = tuple(sorted(set(type_by_subject.get(normalize(entity), []))))
            assignments.append(
                TypeAssignment(
                    node_id=stable_id("node", {"graph": graph.graph_id, "entity": entity}),
                    surface_text=entity,
                    graph_role=graph.role,
                    type_ids=type_ids,
                    status="assigned" if type_ids else "unknown",
                    evidence_span_ids=evidence_for(entity, spans),
                    reason="Explicit source graph type relation." if type_ids else "No explicit source type relation.",
                )
            )
    return tuple(sorted(types.values(), key=lambda item: item.type_id)), tuple(sorted(assignments, key=lambda item: item.node_id))


def registry_checksum(payload: dict) -> str:
    from .models import canonical_json

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

