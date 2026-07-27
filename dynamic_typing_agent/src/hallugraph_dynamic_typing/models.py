"""Strict, standalone contracts for source typing and answer annotation."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .types import EvidenceLevel, NliVerdict, RunStatus


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(namespace: str, value: Any) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:20]
    return f"{namespace}:{digest}"


def normalize(value: str) -> str:
    return " ".join(value.casefold().split())


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GraphInput(StrictModel):
    graph_id: str = Field(min_length=1)
    role: str = Field(pattern="^(context|query|answer)$")
    entities: tuple[str, ...] = ()
    relations: tuple[tuple[str, str, str], ...] = ()
    input_sha256: str | None = None

    @field_validator("entities")
    @classmethod
    def nonempty_entities(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in values):
            raise ValueError("graph entities must be non-empty")
        return tuple(sorted(dict.fromkeys(values), key=normalize))

    @field_validator("relations")
    @classmethod
    def valid_relations(cls, values: tuple[tuple[str, str, str], ...]) -> tuple[tuple[str, str, str], ...]:
        if any(len(item) != 3 or any(not field.strip() for field in item) for item in values):
            raise ValueError("graph relations must be non-empty triples")
        return tuple(sorted(dict.fromkeys(values), key=lambda row: tuple(normalize(x) for x in row)))


class EvidenceSpan(StrictModel):
    span_id: str = Field(min_length=1)
    source_role: str = Field(pattern="^(context|query|answer)$")
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    text: str = Field(min_length=1)


class TypeDefinition(StrictModel):
    type_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    parent_type_ids: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    evidence_span_ids: tuple[str, ...] = ()
    evidence_level: EvidenceLevel = EvidenceLevel.UNKNOWN
    status: str = Field(pattern="^(final|confirmed|preliminary|unknown)$")


class TypeAssignment(StrictModel):
    node_id: str = Field(min_length=1)
    surface_text: str = Field(min_length=1)
    graph_role: str = Field(pattern="^(context|query|answer)$")
    type_ids: tuple[str, ...] = ()
    status: str = Field(pattern="^(assigned|unknown)$")
    evidence_span_ids: tuple[str, ...] = ()
    reason: str = Field(min_length=1)


class NliResult(StrictModel):
    request_id: str = Field(min_length=1)
    verdict: NliVerdict
    evidence_span_ids: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)
    evidence_level: EvidenceLevel = EvidenceLevel.UNKNOWN
    hypothesis_kind: str | None = None
    hypothesis: str | None = None
    subject_id: str | None = None
    target_type_id: str | None = None


class FrozenRegistry(StrictModel):
    registry_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    context_graph_id: str = Field(min_length=1)
    query_graph_id: str = Field(min_length=1)
    types: tuple[TypeDefinition, ...]
    assignments: tuple[TypeAssignment, ...]
    evidence_spans: tuple[EvidenceSpan, ...]
    nli_results: tuple[NliResult, ...] = ()
    prompt_manifest_sha256: str = Field(min_length=1)
    frozen: bool = True
    registry_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def source_only_and_unique(self) -> "FrozenRegistry":
        ids = [item.type_id for item in self.types]
        if len(ids) != len(set(ids)):
            raise ValueError("registry type IDs must be unique")
        if any(item.graph_role == "answer" for item in self.assignments):
            raise ValueError("a frozen source registry cannot contain answer assignments")
        type_ids = set(ids)
        if any(parent not in type_ids for item in self.types for parent in item.parent_type_ids):
            raise ValueError("registry parent must exist")
        if any(item.status != "final" for item in self.types):
            raise ValueError("a frozen registry may contain only final types")
        if any(item.status != "assigned" or not item.type_ids for item in self.assignments):
            raise ValueError("every frozen source assignment must contain a final type")
        if any(type_id not in type_ids for item in self.assignments for type_id in item.type_ids):
            raise ValueError("source assignment references an unknown type")
        parents = {item.type_id: set(item.parent_type_ids) for item in self.types}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(type_id: str) -> None:
            if type_id in visiting:
                raise ValueError("registry type hierarchy must be acyclic")
            if type_id in visited:
                return
            visiting.add(type_id)
            for parent_id in parents.get(type_id, ()):
                visit(parent_id)
            visiting.remove(type_id)
            visited.add(type_id)

        for type_id in parents:
            visit(type_id)
        return self


class SourceInput(StrictModel):
    source_id: str = Field(min_length=1)
    context_raw: str = Field(min_length=1)
    query_raw: str = ""
    context_graph: GraphInput
    query_graph: GraphInput

    @model_validator(mode="after")
    def graph_roles_are_source_roles(self) -> "SourceInput":
        if self.context_graph.role != "context" or self.query_graph.role != "query":
            raise ValueError("source input requires context and query graphs")
        return self


class AnswerInput(StrictModel):
    source_id: str = Field(min_length=1)
    response_id: str = Field(min_length=1)
    response_raw: str = Field(min_length=1)
    answer_graph: GraphInput
    registry: FrozenRegistry

    @model_validator(mode="after")
    def answer_is_bound_to_frozen_source(self) -> "AnswerInput":
        if self.answer_graph.role != "answer":
            raise ValueError("answer input requires an answer graph")
        if not self.registry.frozen or self.registry.source_id != self.source_id:
            raise ValueError("answer input requires a frozen registry for the same source")
        return self


class SourceRun(StrictModel):
    status: RunStatus
    registry: FrozenRegistry | None = None
    failure: str | None = None
    artifacts: tuple[dict[str, Any], ...] = ()


class AnswerAnnotation(StrictModel):
    answer_assignments: tuple[TypeAssignment, ...]
    nli_results: tuple[NliResult, ...]

    @model_validator(mode="after")
    def complete_final_assignments(self) -> "AnswerAnnotation":
        if any(item.status != "assigned" or not item.type_ids for item in self.answer_assignments):
            raise ValueError("every answer entity must receive at least one frozen-registry type")
        return self


class AnswerRun(StrictModel):
    status: RunStatus
    annotations: AnswerAnnotation | None = None
    failure: str | None = None
    artifacts: tuple[dict[str, Any], ...] = ()


class BackendKind(StrEnum):
    FAKE = "fake"
    LIVE = "live"
