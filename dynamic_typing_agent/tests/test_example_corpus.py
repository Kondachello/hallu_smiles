from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NO_GOLD = ROOT / "examples" / "dynamic_typing_20.no_gold.jsonl"
EXPECTATIONS = ROOT / "examples" / "dynamic_typing_20.expectations.jsonl"
FORBIDDEN_INPUT_KEYS = {
    "expected_type_concepts",
    "expected_roles",
    "expected_answer_behavior",
    "expected_nli_routes",
    "gold",
    "gold_label",
    "gold_labels",
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_all_keys(child) for child in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_keys(child) for child in value), set())
    return set()


def test_corpus_has_twenty_unique_linked_cases() -> None:
    no_gold = _read_jsonl(NO_GOLD)
    expectations = _read_jsonl(EXPECTATIONS)
    assert len(no_gold) == 20
    assert len(expectations) == 20
    no_gold_ids = [row["case_id"] for row in no_gold]
    expected_ids = [row["case_id"] for row in expectations]
    assert len(no_gold_ids) == len(set(no_gold_ids))
    assert set(no_gold_ids) == set(expected_ids)


def test_agent_inputs_contain_no_expectation_or_gold_keys() -> None:
    for row in _read_jsonl(NO_GOLD):
        assert _all_keys(row).isdisjoint(FORBIDDEN_INPUT_KEYS), row["case_id"]


def test_every_graph_fixture_has_three_roles_and_valid_triples() -> None:
    for row in _read_jsonl(NO_GOLD):
        assert set(row["graphs"]) == {"context", "query", "answer"}
        for graph in row["graphs"].values():
            assert isinstance(graph["entities"], list)
            assert all(len(triple) == 3 for triple in graph["relations"])


def test_expectations_remain_semantic_not_exact_registry_snapshots() -> None:
    allowed = {
        "case_id",
        "expected_type_concepts",
        "expected_roles",
        "expected_answer_behavior",
        "expected_nli_routes",
    }
    for row in _read_jsonl(EXPECTATIONS):
        assert set(row) == allowed
        assert "registry" not in row

