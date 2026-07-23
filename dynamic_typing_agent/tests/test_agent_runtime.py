from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from hallugraph_dynamic_typing.agent import DynamicTypingAgent, graph_from_fixture
from hallugraph_dynamic_typing.models import AnswerInput, SourceInput
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
    assert any(item.label == "commercial bank" for item in first.registry.types)


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
    assert {event["node"] for event in trace["answer_events"]} == {"validate_answer", "annotate_answer", "nli_answer", "emit_answer"}


def test_answer_only_uses_frozen_types_or_unknown_and_routes_nli(tmp_path) -> None:
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
    assert any(item.status == "unknown" for item in answer_run.annotations.answer_assignments)
    assert [item.verdict for item in answer_run.annotations.nli_results] == ["neutral"]


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
    assert [item.verdict for item in answer_run.annotations.nli_results] == ["neutral"]


def test_schema_overview_model_output_is_strictly_validated_and_added_as_preliminary_type(tmp_path) -> None:
    agent = DynamicTypingAgent(
        cache_root=tmp_path / "cache",
        artifacts_root=tmp_path / "runs",
        invoke_model_nodes=True,
    )
    agent.model = FakeStructuredModel(
        {
            "schema_overview": {
                "schema_version": "schema-overview-v1",
                "source_summary": "A financial institution issues a loan.",
                "draft_types": [
                    {
                        "candidate_type_id": "CT-FINANCIAL-ORG",
                        "label": "financial organization",
                        "definition": "An organization providing financial services in this source.",
                        "parent_candidate_ids": [],
                        "aliases": [],
                        "distinctions": ["more general than commercial bank"],
                        "role_signatures": ["subject:issued"],
                        "evidence_span_ids": ["context:span:0"],
                        "evidence_level": "example_supported"
                    }
                ],
                "contextual_roles": [],
                "open_questions": []
            }
        }
    )
    run = agent.build_source_registry(_source(_first_case()))
    assert run.status == "ok"
    assert run.registry is not None
    assert any(item.label == "financial organization" and item.status == "preliminary" for item in run.registry.types)


def test_model_nli_uses_versioned_prompt_and_three_way_schema(tmp_path) -> None:
    case = _first_case()
    agent = DynamicTypingAgent(cache_root=tmp_path / "cache", artifacts_root=tmp_path / "runs", invoke_model_nodes=True)
    agent.model = FakeStructuredModel(
        {
            "schema_overview": {
                "schema_version": "schema-overview-v1",
                "source_summary": "A bank finances a project.",
                "draft_types": [],
                "contextual_roles": [],
                "open_questions": []
            },
            "nli_verification": {
                "schema_version": "nli-verification-v1",
                "verdict": "neutral",
                "evidence_level": "unknown",
                "supporting_span_ids": [],
                "conflicting_span_ids": [],
                "confidence": 0.5,
                "rationale": "The supplied evidence does not establish the specialization."
            }
        }
    )
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
    assert answer_run.annotations.nli_results[0].verdict == "neutral"
    assert [call["operation"] for call in agent.model.calls] == ["schema_overview", "nli_verification"]


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
