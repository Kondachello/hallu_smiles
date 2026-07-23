from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from hallugraph_dynamic_typing.agent import DynamicTypingAgent, graph_from_fixture
from hallugraph_dynamic_typing.models import (
    AnswerInput,
    EvidenceLevel,
    FrozenRegistry,
    GraphInput,
    NliResult,
    SourceInput,
)
from hallugraph_dynamic_typing.quality_workflow import ROOT_TYPE_ID
from hallugraph_dynamic_typing.models import stable_id
from hallugraph_dynamic_typing.transports import FakeStructuredModel


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "examples" / "dynamic_typing_20.no_gold.jsonl"


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in CASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _source(row: dict) -> SourceInput:
    return SourceInput(
        source_id=row["source_id"],
        context_raw=row["context"],
        query_raw=row["query"],
        context_graph=graph_from_fixture(
            graph_id=f"{row['case_id']}:context",
            role="context",
            payload=row["graphs"]["context"],
        ),
        query_graph=graph_from_fixture(
            graph_id=f"{row['case_id']}:query",
            role="query",
            payload=row["graphs"]["query"],
        ),
    )


def test_all_twenty_offline_cases_have_complete_final_source_and_answer_coverage(tmp_path) -> None:
    agent = DynamicTypingAgent(cache_root=tmp_path / "cache", artifacts_root=tmp_path / "runs")
    for row in _rows():
        source = _source(row)
        source_run = agent.build_source_registry(source)
        assert source_run.status == "ok", source_run.failure
        registry = source_run.registry
        assert registry is not None
        assert all(item.status == "final" for item in registry.types)
        assert all(item.status == "assigned" and item.type_ids for item in registry.assignments)
        assert len(registry.assignments) == len(source.context_graph.entities) + len(source.query_graph.entities)
        assert len(registry.nli_results) >= len(
            {item.casefold() for item in (*source.context_graph.entities, *source.query_graph.entities)}
        )

        answer = AnswerInput(
            source_id=row["source_id"],
            response_id=row["case_id"],
            response_raw=row["response"],
            answer_graph=graph_from_fixture(
                graph_id=f"{row['case_id']}:answer",
                role="answer",
                payload=row["graphs"]["answer"],
            ),
            registry=registry,
        )
        answer_run = agent.annotate_answer(answer)
        assert answer_run.status == "ok", answer_run.failure
        assert answer_run.annotations is not None
        assert len(answer_run.annotations.answer_assignments) == len(answer.answer_graph.entities)
        assert all(
            item.status == "assigned" and item.type_ids
            for item in answer_run.annotations.answer_assignments
        )
        assert len(answer_run.annotations.nli_results) >= len(answer.answer_graph.entities)


def test_plain_is_relation_cannot_type_a_vertex_as_an_arbitrary_value(tmp_path) -> None:
    source = SourceInput(
        source_id="plain-is-is-not-a-type",
        context_raw='The measured half is 2".',
        query_raw="What is the measurement?",
        context_graph=GraphInput(
            graph_id="plain-is:context",
            role="context",
            entities=("half", '2"'),
            relations=(("half", "is", '2"'),),
        ),
        query_graph=GraphInput(graph_id="plain-is:query", role="query", entities=("half",), relations=()),
    )
    run = DynamicTypingAgent(
        cache_root=tmp_path / "cache", artifacts_root=tmp_path / "runs"
    ).build_source_registry(source)
    assert run.registry is not None
    half = [item for item in run.registry.assignments if item.surface_text == "half"]
    assert half and all(item.type_ids == (ROOT_TYPE_ID,) for item in half)
    assert {item.label for item in run.registry.types} == {"entity"}


def test_frozen_registry_rejects_preliminary_types_and_empty_assignments() -> None:
    payload = {
        "registry_id": "registry:test",
        "source_id": "source:test",
        "context_graph_id": "context:test",
        "query_graph_id": "query:test",
        "types": [
            {
                "type_id": ROOT_TYPE_ID,
                "label": "entity",
                "definition": "root",
                "parent_type_ids": [],
                "aliases": [],
                "evidence_span_ids": [],
                "evidence_level": "source_entailed",
                "status": "preliminary",
            }
        ],
        "assignments": [
            {
                "node_id": "node:test",
                "surface_text": "thing",
                "graph_role": "context",
                "type_ids": [],
                "status": "unknown",
                "evidence_span_ids": [],
                "reason": "incomplete",
            }
        ],
        "evidence_spans": [],
        "nli_results": [],
        "prompt_manifest_sha256": "a" * 64,
        "frozen": True,
        "registry_sha256": "b" * 64,
    }
    with pytest.raises(ValidationError, match="final types"):
        FrozenRegistry.model_validate(payload)


def test_cache_key_changes_when_graph_payload_changes(tmp_path) -> None:
    agent = DynamicTypingAgent(cache_root=tmp_path / "cache", artifacts_root=tmp_path / "runs")
    base = SourceInput(
        source_id="same-source",
        context_raw="A relation is present.",
        query_raw="",
        context_graph=GraphInput(
            graph_id="same-graph",
            role="context",
            entities=("A", "B"),
            relations=(("A", "related to", "B"),),
        ),
        query_graph=GraphInput(graph_id="same-query", role="query", entities=(), relations=()),
    )
    changed = base.model_copy(
        update={
            "context_graph": GraphInput(
                graph_id="same-graph",
                role="context",
                entities=("A", "B"),
                relations=(("A", "different relation", "B"),),
            )
        }
    )
    assert agent._source_cache_key(base) != agent._source_cache_key(changed)


def test_unknown_model_type_id_is_retried_then_falls_back_without_incomplete_output(tmp_path) -> None:
    source = SourceInput(
        source_id="protocol-repair",
        context_raw="Acme appears in the source.",
        query_raw="",
        context_graph=GraphInput(
            graph_id="protocol:context", role="context", entities=("Acme",), relations=()
        ),
        query_graph=GraphInput(graph_id="protocol:query", role="query", entities=(), relations=()),
    )
    entity_id = stable_id(
        "source-entity", {"source": source.source_id, "surface": "acme"}
    )
    agent = DynamicTypingAgent(
        cache_root=tmp_path / "cache",
        artifacts_root=tmp_path / "runs",
        invoke_model_nodes=True,
    )
    agent.model = FakeStructuredModel(
        {
            "schema_overview": {
                "schema_version": "schema-overview-v2",
                "source_summary": "The source mentions Acme without a reusable category.",
                "type_hints": [],
                "hierarchy_hints": [],
                "unsafe_relation_warnings": [],
            },
            "entity_type_decision": [
                {
                    "schema_version": "entity-type-decision-v2",
                    "entity_id": entity_id,
                    "selected_type_ids": ["CT-NOT-IN-REGISTRY"],
                    "proposed_types": [],
                    "role_labels": [],
                    "hypotheses": [],
                    "reason": "Invalid first attempt.",
                },
                {
                    "schema_version": "entity-type-decision-v2",
                    "entity_id": entity_id,
                    "selected_type_ids": [ROOT_TYPE_ID],
                    "proposed_types": [],
                    "role_labels": [],
                    "hypotheses": [],
                    "reason": "Use the final structural root.",
                },
            ],
        }
    )
    run = agent.build_source_registry(source)
    assert run.status == "ok", run.failure
    assert run.registry is not None
    assert run.registry.assignments[0].type_ids == (ROOT_TYPE_ID,)
    assert any(event["node"] == "validate_entity_type_decision" for event in run.artifacts)


class _FixedNli:
    def __init__(self, verdict: str):
        self.verdict = verdict

    def verify(self, *, evidence_span_ids, idempotency_key, **_) -> NliResult:
        return NliResult(
            request_id=idempotency_key,
            verdict=self.verdict,
            evidence_span_ids=tuple(evidence_span_ids),
            rationale=f"fixed {self.verdict}",
            evidence_level=EvidenceLevel.UNKNOWN,
        )


@pytest.mark.parametrize(
    ("verdict", "expected_label", "expected_level"),
    [
        ("neutral", "organization", "definition_only"),
        ("contradicted", "entity", "source_entailed"),
    ],
)
def test_source_neutral_can_finalize_but_contradiction_falls_back(
    tmp_path, verdict: str, expected_label: str, expected_level: str
) -> None:
    source = SourceInput(
        source_id=f"nli-{verdict}",
        context_raw="Acme operates in the market.",
        query_raw="",
        context_graph=GraphInput(
            graph_id=f"nli-{verdict}:context", role="context", entities=("Acme",), relations=()
        ),
        query_graph=GraphInput(graph_id=f"nli-{verdict}:query", role="query", entities=(), relations=()),
    )
    entity_id = stable_id(
        "source-entity", {"source": source.source_id, "surface": "acme"}
    )
    agent = DynamicTypingAgent(
        cache_root=tmp_path / "cache",
        artifacts_root=tmp_path / "runs",
        invoke_model_nodes=True,
        max_entity_attempts=1,
    )
    agent.nli = _FixedNli(verdict)
    agent.model = FakeStructuredModel(
        {
            "schema_overview": {
                "schema_version": "schema-overview-v2",
                "source_summary": "Acme is discussed as a market actor.",
                "type_hints": [],
                "hierarchy_hints": [],
                "unsafe_relation_warnings": [],
            },
            "entity_type_decision": {
                "schema_version": "entity-type-decision-v2",
                "entity_id": entity_id,
                "selected_type_ids": [],
                "proposed_types": [{
                    "candidate_id": "C-ORG",
                    "label": "organization",
                    "definition": "A structured collective actor.",
                    "parent_type_ids": [ROOT_TYPE_ID],
                    "aliases": [],
                    "evidence_span_ids": ["context:span:0"],
                }],
                "role_labels": [],
                "hypotheses": [{
                    "target_ref": "C-ORG",
                    "text": "Acme is an organization.",
                    "evidence_span_ids": ["context:span:0"],
                }],
                "reason": "A reusable category for Acme.",
            },
        }
    )
    run = agent.build_source_registry(source)
    assert run.status == "ok", run.failure
    assert run.registry is not None
    assigned_id = run.registry.assignments[0].type_ids[0]
    assigned_type = {item.type_id: item for item in run.registry.types}[assigned_id]
    assert assigned_type.label == expected_label
    assert assigned_type.evidence_level == expected_level
    assert assigned_type.status == "final"


def test_exact_answer_surface_keeps_frozen_source_type_even_if_nli_is_unstable(tmp_path) -> None:
    source = SourceInput(
        source_id="exact-answer",
        context_raw="Acme operates in the market.",
        query_raw="",
        context_graph=GraphInput(
            graph_id="exact:context", role="context", entities=("Acme",), relations=()
        ),
        query_graph=GraphInput(graph_id="exact:query", role="query", entities=(), relations=()),
    )
    entity_id = stable_id(
        "source-entity", {"source": source.source_id, "surface": "acme"}
    )
    agent = DynamicTypingAgent(
        cache_root=tmp_path / "cache",
        artifacts_root=tmp_path / "runs",
        invoke_model_nodes=True,
        max_entity_attempts=1,
    )
    agent.nli = _FixedNli("neutral")
    agent.model = FakeStructuredModel(
        {
            "schema_overview": {
                "schema_version": "schema-overview-v2",
                "source_summary": "Acme is a market actor.",
                "type_hints": [],
                "hierarchy_hints": [],
                "unsafe_relation_warnings": [],
            },
            "entity_type_decision": {
                "schema_version": "entity-type-decision-v2",
                "entity_id": entity_id,
                "selected_type_ids": [],
                "proposed_types": [{
                    "candidate_id": "C-ORG",
                    "label": "organization",
                    "definition": "A structured collective actor.",
                    "parent_type_ids": [ROOT_TYPE_ID],
                    "aliases": [],
                    "evidence_span_ids": ["context:span:0"],
                }],
                "role_labels": [],
                "hypotheses": [{
                    "target_ref": "C-ORG",
                    "text": "Acme is an organization.",
                    "evidence_span_ids": ["context:span:0"],
                }],
                "reason": "A reusable source category.",
            },
        }
    )
    source_run = agent.build_source_registry(source)
    assert source_run.registry is not None
    source_type_ids = source_run.registry.assignments[0].type_ids
    assert source_type_ids != (ROOT_TYPE_ID,)

    agent.nli = _FixedNli("contradicted")
    answer = AnswerInput(
        source_id=source.source_id,
        response_id="exact-answer",
        response_raw="Acme operates.",
        answer_graph=GraphInput(
            graph_id="exact:answer", role="answer", entities=("Acme",), relations=()
        ),
        registry=source_run.registry,
    )
    answer_run = agent.annotate_answer(answer)
    assert answer_run.annotations is not None
    assert answer_run.annotations.answer_assignments[0].type_ids == source_type_ids
    assert answer_run.annotations.nli_results[0].verdict == "contradicted"
