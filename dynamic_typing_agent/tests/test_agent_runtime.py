from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from hallugraph_dynamic_typing.agent import DynamicTypingAgent, graph_from_fixture
from hallugraph_dynamic_typing.models import AnswerInput, GraphInput, SourceInput, stable_id
from hallugraph_dynamic_typing.quality_workflow import ROOT_TYPE_ID, type_id_for_label
from hallugraph_dynamic_typing.transports import FakeStructuredModel


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "examples" / "dynamic_typing_20.no_gold.jsonl"


def _first_case() -> dict:
    return json.loads(CASES.read_text(encoding="utf-8").splitlines()[0])


def _source(case: dict) -> SourceInput:
    return SourceInput(
        source_id=case["source_id"],
        context_raw=case["context"],
        query_raw=case["query"],
        context_graph=graph_from_fixture(graph_id="fixture:context", role="context", payload=case["graphs"]["context"]),
        query_graph=graph_from_fixture(graph_id="fixture:query", role="query", payload=case["graphs"]["query"]),
    )


def test_source_registry_is_frozen_source_only_and_cacheable(tmp_path) -> None:
    agent = DynamicTypingAgent(cache_root=tmp_path / "cache", artifacts_root=tmp_path / "runs")
    source = _source(_first_case())
    first = agent.build_source_registry(source)
    second = agent.build_source_registry(source)
    assert first.status == "ok"
    assert second.status == "ok"
    assert first.registry is not None and second.registry is not None
    assert first.registry.registry_sha256 == second.registry.registry_sha256
    assert first.registry.frozen is True
    assert {item.graph_role for item in first.registry.assignments} <= {"context", "query"}
    assert all(item.graph_role != "answer" for item in first.registry.assignments)
    assert all(item.status == "final" for item in first.registry.types)
    assert all(item.status == "assigned" and item.type_ids for item in first.registry.assignments)
    expected_nodes = len(source.context_graph.entities) + len(source.query_graph.entities)
    assert len(first.registry.assignments) == expected_nodes


def test_run_artifacts_include_a_node_by_node_execution_trace(tmp_path) -> None:
    agent = DynamicTypingAgent(cache_root=tmp_path / "cache", artifacts_root=tmp_path / "runs")
    case = _first_case()
    source_run = agent.build_source_registry(_source(case))
    assert source_run.registry is not None
    answer = AnswerInput(
        source_id=case["source_id"], response_id="trace", response_raw="North Bank is a commercial bank.",
        answer_graph=graph_from_fixture(graph_id="trace:answer", role="answer", payload={"entities": ["North Bank"], "relations": [["North Bank", "is a", "commercial bank"]]}), registry=source_run.registry,
    )
    answer_run = agent.annotate_answer(answer)
    path = agent.write_run_artifacts(run_id="trace", source_run=source_run, answer_run=answer_run)
    trace = json.loads((path / "execution_trace.json").read_text(encoding="utf-8"))
    assert {event["node"] for event in trace["source_events"]} >= {"validate_source", "source_cache", "segment_source", "schema_overview", "derive_registry", "freeze_registry"}
    assert {event["node"] for event in trace["answer_events"]} >= {
        "validate_answer", "build_answer_profiles", "answer_typing", "nli_verify_answer",
        "annotate_answer", "nli_answer", "emit_answer"
    }


def test_answer_only_uses_frozen_types_and_nli_audits_every_vertex(tmp_path) -> None:
    case = _first_case()
    agent = DynamicTypingAgent(cache_root=tmp_path / "cache", artifacts_root=tmp_path / "runs")
    source_run = agent.build_source_registry(_source(case))
    assert source_run.registry is not None
    answer = AnswerInput(
        source_id=case["source_id"],
        response_id=case["case_id"],
        response_raw=case["response"],
        answer_graph=graph_from_fixture(graph_id="fixture:answer", role="answer", payload=case["graphs"]["answer"]),
        registry=source_run.registry,
    )
    answer_run = agent.annotate_answer(answer)
    assert answer_run.status == "ok"
    assert answer_run.annotations is not None
    registry_ids = {item.type_id for item in source_run.registry.types}
    assert all(set(item.type_ids) <= registry_ids for item in answer_run.annotations.answer_assignments)
    assert all(item.status == "assigned" and item.type_ids for item in answer_run.annotations.answer_assignments)
    assert len(answer_run.annotations.answer_assignments) == len(answer.answer_graph.entities)
    assert len(answer_run.annotations.nli_results) >= len(answer.answer_graph.entities)


def test_russian_type_relation_is_assigned_and_routed_to_nli(tmp_path) -> None:
    """Русское «является» имеет тот же смысл, что и английское ``is a``."""
    case = _first_case()
    agent = DynamicTypingAgent(cache_root=tmp_path / "cache", artifacts_root=tmp_path / "runs")
    source_run = agent.build_source_registry(_source(case))
    assert source_run.registry is not None
    answer = AnswerInput(
        source_id=case["source_id"],
        response_id="ru-answer",
        response_raw="North Bank является системно значимым банком.",
        answer_graph=graph_from_fixture(
            graph_id="fixture:answer-ru",
            role="answer",
            payload={
                "entities": ["North Bank", "commercial bank", "systemically important bank"],
                "relations": [
                    ["North Bank", "является", "commercial bank"],
                    ["North Bank", "является", "systemically important bank"],
                ],
            },
        ),
        registry=source_run.registry,
    )
    answer_run = agent.annotate_answer(answer)
    assert answer_run.annotations is not None
    assignments = {item.surface_text: item for item in answer_run.annotations.answer_assignments}
    assert assignments["North Bank"].status == "assigned"
    assert len(answer_run.annotations.nli_results) >= len(answer.answer_graph.entities)


def test_model_processes_one_entity_per_call_and_preserves_nli_validated_hierarchy(tmp_path) -> None:
    source = SourceInput(
        source_id="quality-source",
        context_raw="North Bank is a commercial bank. A commercial bank is a financial institution.",
        query_raw="What kind of institution is North Bank?",
        context_graph=GraphInput(
            graph_id="quality:context",
            role="context",
            entities=("North Bank", "commercial bank", "financial institution"),
            relations=(
                ("North Bank", "is a", "commercial bank"),
                ("commercial bank", "is a", "financial institution"),
            ),
        ),
        query_graph=GraphInput(
            graph_id="quality:query",
            role="query",
            entities=("North Bank",),
            relations=(),
        ),
    )
    entity_ids = {
        name: stable_id("source-entity", {"source": source.source_id, "surface": name.casefold()})
        for name in ("North Bank", "commercial bank", "financial institution")
    }
    financial_id = type_id_for_label("financial institution")
    commercial_id = type_id_for_label("commercial bank")
    agent = DynamicTypingAgent(
        cache_root=tmp_path / "cache",
        artifacts_root=tmp_path / "runs",
        invoke_model_nodes=True,
    )
    agent.model = FakeStructuredModel(
        {
            "schema_overview": {
                "schema_version": "schema-overview-v2",
                "source_summary": "North Bank is described through a two-level institution taxonomy.",
                "type_hints": [],
                "hierarchy_hints": [],
                "unsafe_relation_warnings": []
            },
            "entity_type_decision": [
                {
                    "schema_version": "entity-type-decision-v2",
                    "entity_id": entity_ids["commercial bank"],
                    "selected_type_ids": [],
                    "proposed_types": [{
                        "candidate_id": "C-FINANCIAL",
                        "label": "financial institution",
                        "definition": "An institution operating in finance.",
                        "parent_type_ids": [ROOT_TYPE_ID],
                        "aliases": [],
                        "evidence_span_ids": ["context:span:1"]
                    }],
                    "role_labels": [],
                    "hypotheses": [{
                        "target_ref": "C-FINANCIAL",
                        "text": "A commercial bank is a financial institution.",
                        "evidence_span_ids": ["context:span:1"]
                    }],
                    "reason": "The source explicitly provides the reusable parent category."
                },
                {
                    "schema_version": "entity-type-decision-v2",
                    "entity_id": entity_ids["financial institution"],
                    "selected_type_ids": [ROOT_TYPE_ID],
                    "proposed_types": [],
                    "role_labels": [],
                    "hypotheses": [],
                    "reason": "The graph label itself is not reused as a new identity-like type."
                },
                {
                    "schema_version": "entity-type-decision-v2",
                    "entity_id": entity_ids["North Bank"],
                    "selected_type_ids": [],
                    "proposed_types": [{
                        "candidate_id": "C-COMMERCIAL",
                        "label": "commercial bank",
                        "definition": "A bank serving commercial financial functions.",
                        "parent_type_ids": [financial_id],
                        "aliases": [],
                        "evidence_span_ids": ["context:span:0"]
                    }],
                    "role_labels": [],
                    "hypotheses": [{
                        "target_ref": "C-COMMERCIAL",
                        "text": "North Bank is a commercial bank.",
                        "evidence_span_ids": ["context:span:0"]
                    }],
                    "reason": "The source explicitly types North Bank."
                }
            ],
            "registry_consistency_review": {
                "schema_version": "registry-consistency-review-v2",
                "proposals": [{
                    "proposal_id": "P-COMMERCIAL-FINANCIAL",
                    "action": "child_of",
                    "first_type_id": commercial_id,
                    "second_type_id": financial_id,
                    "hypotheses": [{
                        "text": "A commercial bank is a financial institution.",
                        "evidence_span_ids": ["context:span:1"]
                    }],
                    "evidence_span_ids": ["context:span:1"],
                    "reason": "The source explicitly states the subtype relation."
                }]
            },
        }
    )
    run = agent.build_source_registry(source)
    assert run.status == "ok"
    assert run.registry is not None
    assert all(item.status == "final" for item in run.registry.types)
    assert all(item.status == "assigned" and item.type_ids for item in run.registry.assignments)
    assert {item.type_id: item for item in run.registry.types}[commercial_id].parent_type_ids == (financial_id,)
    assert [call["operation"] for call in agent.model.calls].count("entity_type_decision") == 3
    assert len(run.registry.nli_results) >= 4


def test_arbitrary_relation_object_never_becomes_a_type_in_offline_mode(tmp_path) -> None:
    source = SourceInput(
        source_id="unsafe-edge",
        context_raw="A bag has a zipper.",
        query_raw="What does the bag have?",
        context_graph=GraphInput(
            graph_id="unsafe:context",
            role="context",
            entities=("bag", "zipper"),
            relations=(("bag", "has", "zipper"),),
        ),
        query_graph=GraphInput(graph_id="unsafe:query", role="query", entities=("bag",), relations=()),
    )
    agent = DynamicTypingAgent(cache_root=tmp_path / "cache", artifacts_root=tmp_path / "runs")
    run = agent.build_source_registry(source)
    assert run.registry is not None
    labels = {item.label for item in run.registry.types}
    bag_assignments = [item for item in run.registry.assignments if item.surface_text == "bag"]
    assert "zipper" not in labels
    assert all(item.type_ids == (ROOT_TYPE_ID,) for item in bag_assignments)


def test_tampered_registry_checksum_fails_closed(tmp_path) -> None:
    case = _first_case()
    agent = DynamicTypingAgent(cache_root=tmp_path / "cache", artifacts_root=tmp_path / "runs")
    source_run = agent.build_source_registry(_source(case))
    assert source_run.registry is not None
    tampered = source_run.registry.model_copy(update={"registry_sha256": "0" * 64})
    answer = AnswerInput(
        source_id=case["source_id"],
        response_id=case["case_id"],
        response_raw=case["response"],
        answer_graph=graph_from_fixture(graph_id="fixture:answer", role="answer", payload=case["graphs"]["answer"]),
        registry=tampered,
    )
    result = agent.annotate_answer(answer)
    assert result.status == "failed"
    assert "checksum" in (result.failure or "")


def test_cli_runs_no_gold_fixture_with_output_after_subcommand(tmp_path) -> None:
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hallugraph_dynamic_typing",
            "run-fixture",
            "--input",
            str(CASES),
            "--output",
            str(tmp_path / "artifacts"),
            "--limit",
            "1",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "artifacts" / "summary.json").read_text(encoding="utf-8"))
    assert summary[0]["status"] == "ok"
    assert (tmp_path / "artifacts" / "dt-001-bank-generalization" / "source_registry.json").is_file()
    snapshot = json.loads((tmp_path / "artifacts" / "dt-001-bank-generalization" / "input_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["source"]["context_raw"] == _first_case()["context"]
    assert snapshot["answer"]["response_raw"] == _first_case()["response"]
    assert "registry" not in snapshot["answer"]
