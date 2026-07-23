"""Serializable LangGraph state shape for the future implementation.

Only identifiers and immutable JSON-compatible values belong in graph state. Runtime
clients, file handles and secrets are injected through graph context and never persisted.
"""

from __future__ import annotations

from typing import Any, TypedDict


class SourceInputState(TypedDict):
    source_id: str
    context_raw: str
    query_raw: str
    context_graph: dict[str, Any]
    query_graph: dict[str, Any]
    context_graph_id: str
    query_graph_id: str


class AnswerInputState(TypedDict):
    source_id: str
    response_id: str
    response_raw: str
    answer_graph: dict[str, Any]
    answer_graph_id: str
    frozen_registry: dict[str, Any]


class AgentState(TypedDict, total=False):
    run_id: str
    mode: str
    source: SourceInputState
    answer: AnswerInputState
    prompt_manifest_sha256: str
    source_spans: list[dict[str, Any]]
    schema_drafts: list[dict[str, Any]]
    entity_profiles: list[dict[str, Any]]
    candidate_sets: list[dict[str, Any]]
    typing_decisions: list[dict[str, Any]]
    registry_draft: dict[str, Any]
    registry_validation: dict[str, Any]
    frozen_registry: dict[str, Any]
    answer_annotations: list[dict[str, Any]]
    edge_verdicts: list[dict[str, Any]]
    nli_requests: list[dict[str, Any]]
    nli_results: list[dict[str, Any]]
    repair_attempt: int
    status: str
    failure: dict[str, Any]
    artifacts: list[dict[str, Any]]


FORBIDDEN_STATE_KEYS = frozenset(
    {
        "gold",
        "gold_label",
        "gold_labels",
        "hallucination_labels",
        "reference_answer",
    }
)

