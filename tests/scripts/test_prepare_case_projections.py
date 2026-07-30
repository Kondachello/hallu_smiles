from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "audit_agents" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve annotations through sys.modules[cls.__module__]; register first
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _package() -> dict:
    """Shape mirrors export_historical_replay_audit_case.build_case_packages output."""
    return {
        "schema_version": "historical-replay-agent-audit-case-v1",
        "analysis_only": True,
        "case_id": "r1",
        "provenance": {"archive_dir": "/tmp/runs/sealed", "gold_join_timing": "post_seal_evaluation_only"},
        "ragtruth": {
            "response_id": "r1", "split": "train", "gold_response_label": 1,
            "labels": [{"start": 21, "end": 25, "text": "2019", "label_type": "Evident Conflict"}],
        },
        "input": {
            "response_id": "r1", "query_raw": "What happened?",
            "context_raw": "Ada founded Acme in 2020.", "response_raw": "Ada founded Acme in 2019.",
        },
        "graphs": {
            "context": {"role": "context", "entities": ["Ada", "Acme", "2020"], "relations": [["Ada", "founded", "Acme"]]},
            "query": {"role": "query", "entities": ["What"], "relations": []},
            "response": {"role": "response", "entities": ["Ada", "Acme", "2019"], "relations": [["Ada", "founded", "Acme"]]},
        },
        "methods": {
            "hallugraph": {
                "status": "ok", "raw_score": 0.1, "threshold": 0.5, "threshold_comparator": ">", "decision": False,
                "prediction": {"method": "hallugraph", "components": {"CFI": 0.9, "EG": 1.0, "RP": 0.8}},
            },
            "grapheval": {
                "status": "ok", "raw_score": 0.9, "threshold": 0.5, "threshold_comparator": ">", "decision": True,
                "prediction": {"method": "grapheval", "flagged_unit_ids": ["t1"]},
            },
        },
        "classification": {
            "gold_response_label": 1, "hallugraph_outcome": "FN",
            "grapheval_outcome": "CORRECT", "paired_score_available": True,
        },
    }


def test_hallugraph_projection_hides_grapheval_entirely() -> None:
    module = _load_module("prepare_case_projections")
    projection = module.project(_package(), "hallugraph")

    assert set(projection["methods"]) == {"hallugraph"}
    assert projection["classification"]["hallugraph_outcome"] == "FN"
    assert "grapheval_outcome" not in projection["classification"]
    assert "paired_score_available" not in projection["classification"]
    assert projection["audit_scope"]["method_under_audit"] == "hallugraph"
    assert projection["audit_scope"]["withheld_comparison_methods"] == 1
    # the evidence the auditor still needs must survive untouched
    assert projection["graphs"]["context"]["entities"] == ["Ada", "Acme", "2020"]
    assert projection["ragtruth"]["labels"][0]["label_type"] == "Evident Conflict"
    assert projection["methods"]["hallugraph"]["prediction"]["components"]["CFI"] == 0.9
    module.assert_no_leak(projection, "hallugraph")


def test_grapheval_projection_hides_hallugraph_entirely() -> None:
    module = _load_module("prepare_case_projections")
    projection = module.project(_package(), "grapheval")

    assert set(projection["methods"]) == {"grapheval"}
    assert projection["classification"]["grapheval_outcome"] == "CORRECT"
    assert "hallugraph_outcome" not in projection["classification"]
    module.assert_no_leak(projection, "grapheval")


def test_projection_does_not_mutate_the_source_package() -> None:
    module = _load_module("prepare_case_projections")
    package = _package()
    before = json.dumps(package, sort_keys=True)
    module.project(package, "hallugraph")
    assert json.dumps(package, sort_keys=True) == before


def test_leak_guard_catches_a_stray_mention_of_the_hidden_method() -> None:
    module = _load_module("prepare_case_projections")
    projection = module.project(_package(), "hallugraph")
    # simulate a future schema change that smuggles the other verdict back in
    projection["methods"]["hallugraph"]["prediction"]["note"] = "agrees with GraphEval"

    with pytest.raises(module.LeakError) as excinfo:
        module.assert_no_leak(projection, "hallugraph")
    assert "r1" in str(excinfo.value)


def test_registry_ids_are_stable_and_collision_free() -> None:
    registry = _load_module("aspect_registry")
    assert registry.make_aspect_id("Числовая нормализация", []) == "UNNAMED_ASPECT"
    assert registry.make_aspect_id("Numeric unit mismatch", []) == "NUMERIC_UNIT_MISMATCH"
    assert registry.make_aspect_id("Numeric unit mismatch", ["NUMERIC_UNIT_MISMATCH"]) == "NUMERIC_UNIT_MISMATCH_2"


def test_registry_round_trips_and_renders(tmp_path: Path) -> None:
    registry = _load_module("aspect_registry")
    jsonl, md = tmp_path / "aspects.jsonl", tmp_path / "aspects.md"

    assert registry.load_registry(jsonl) == []
    assert "Реестр пока пуст" in registry.render_markdown([], method="HalluGraph")

    aspect = registry.Aspect(
        aspect_id="NUMERIC_UNIT_MISMATCH", title="Numeric unit mismatch",
        definition="Число совпадает, единица измерения — нет.",
        how_to_check="Сравни единицы у числовых узлов Context и Response.",
        why_it_matters="ER не отличает 5 км от 5 миль.",
        proposed_by_case="r1", proposed_by_agent="agent-3", wave=1,
    )
    registry.append_accepted(jsonl, aspect)
    assert registry.load_registry(jsonl) == [aspect]

    count = registry.sync_markdown(jsonl, md, method="HalluGraph")
    assert count == 1
    rendered = md.read_text(encoding="utf-8")
    assert "NUMERIC_UNIT_MISMATCH" in rendered
    assert "кейсе `r1`" in rendered


def test_log_event_rejects_an_unknown_status(tmp_path: Path) -> None:
    registry = _load_module("aspect_registry")
    with pytest.raises(ValueError):
        registry.log_event(tmp_path / "log.jsonl", {"status": "maybe"})
