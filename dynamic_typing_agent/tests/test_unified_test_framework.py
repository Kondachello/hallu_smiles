from __future__ import annotations

import json
from codecs import BOM_UTF8
from pathlib import Path
from types import SimpleNamespace

from hallugraph_dynamic_typing.agent import DynamicTypingAgent
from hallugraph_dynamic_typing.kggen_pipeline import _set_config_value
from hallugraph_dynamic_typing.kggen_pipeline import _unicode_fake_graph
from hallugraph_dynamic_typing.test_framework import (
    detect_input_mode,
    load_no_gold_cases,
    run_test_suite,
)


class _AttributeConfig:
    def __init__(self, **values):
        self._data = dict(values)
        for name, value in values.items():
            setattr(self, name, value)

    def get(self, name, default=None):
        return self._data.get(name, default)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_runtime_config_override_updates_attribute_and_mapping_views() -> None:
    config = _AttributeConfig(structured_output_transport="none")
    _set_config_value(config, "structured_output_transport", "response_format")
    assert config.structured_output_transport == "response_format"
    assert config.get("structured_output_transport") == "response_format"


def test_unicode_fake_graph_preserves_non_latin_offline_smoke_coverage() -> None:
    class Graph:
        def __init__(self, *, entities, relations):
            self.entities = entities
            self.relations = relations

    graph = _unicode_fake_graph("Конструкторское бюро разработало модель аппарата", Graph)
    assert graph.entities
    assert graph.relations


def test_auto_detects_supplied_graphs_and_text() -> None:
    assert detect_input_mode({"case_id": "a", "graphs": {}}, "auto") == "graphs"
    assert detect_input_mode({"case_id": "b"}, "auto") == "text"
    assert detect_input_mode({"case_id": "a", "graphs": {}}, "text") == "text"


def test_loader_rejects_gold_and_duplicate_case_ids(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    _write_jsonl(path, [{"case_id": "a", "context": "x", "gold_label": 1}])
    try:
        load_no_gold_cases(path)
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("gold field was accepted")

    _write_jsonl(
        path,
        [
            {"case_id": "a", "context": "x"},
            {"case_id": "a", "context": "y"},
        ],
    )
    try:
        load_no_gold_cases(path)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate case_id was accepted")


def test_loader_selects_named_case_without_reading_review_metadata(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    _write_jsonl(
        path,
        [
            {"case_id": "ru-long", "context": "long Russian context"},
            {"case_id": "en-short", "context": "short English context"},
        ],
    )
    selected = load_no_gold_cases(path, case_ids=("ru-long",))
    assert [row["case_id"] for row in selected] == ["ru-long"]
    try:
        load_no_gold_cases(path, case_ids=("absent",))
    except ValueError as exc:
        assert "absent" in str(exc)
    else:
        raise AssertionError("missing selected case was accepted")


def test_loader_accepts_standard_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.jsonl"
    path.write_bytes(BOM_UTF8 + b'{"case_id":"bom-case","context":"text"}\n')
    assert load_no_gold_cases(path)[0]["case_id"] == "bom-case"


def test_run_output_is_immutable_and_must_start_empty(tmp_path: Path) -> None:
    input_path = tmp_path / "cases.jsonl"
    _write_jsonl(input_path, [{"case_id": "a", "context": "x"}])
    output = tmp_path / "existing-run"
    output.mkdir()
    (output / "run_manifest.json").write_text("{}", encoding="utf-8")
    agent = DynamicTypingAgent(
        cache_root=tmp_path / "cache",
        artifacts_root=output,
        backend="fake",
    )
    try:
        run_test_suite(agent=agent, input_path=input_path, output=output)
    except FileExistsError as exc:
        assert "new or empty" in str(exc)
    else:
        raise AssertionError("existing run output was overwritten")


def test_mixed_text_and_graph_cases_share_one_output_contract(
    tmp_path: Path, monkeypatch
) -> None:
    rows = [
        {
            "case_id": "graph-case",
            "source_id": "source-graph",
            "context": "North Bank issued a loan.",
            "query": "Who issued the loan?",
            "response": "North Bank issued it.",
            "graphs": {
                "context": {
                    "entities": ["North Bank", "loan"],
                    "relations": [["North Bank", "issued", "loan"]],
                },
                "query": {
                    "entities": ["loan"],
                    "relations": [],
                },
                "answer": {
                    "entities": ["North Bank"],
                    "relations": [],
                },
            },
        },
        {
            "case_id": "text-case",
            "source_id": "source-text",
            "context": "Aurora Observatory uses a telescope.",
            "query": "What does the observatory use?",
        },
    ]
    input_path = tmp_path / "cases.jsonl"
    _write_jsonl(input_path, rows)

    class Extractor:
        def extract(self, text: str, kind: str):
            if kind == "context":
                return SimpleNamespace(
                    entities={"Aurora Observatory", "telescope"},
                    relations={("Aurora Observatory", "uses", "telescope")},
                )
            return SimpleNamespace(entities={"telescope"}, relations=set())

    monkeypatch.setattr(
        "hallugraph_dynamic_typing.kggen_pipeline.make_extractor",
        lambda **_: (Extractor(), "test-kggen-v1"),
    )
    output = tmp_path / "run"
    agent = DynamicTypingAgent(
        cache_root=tmp_path / "cache",
        artifacts_root=output,
        backend="fake",
    )
    manifest = run_test_suite(
        agent=agent,
        input_path=input_path,
        output=output,
        input_mode="auto",
        kggen_mode="fake",
    )

    assert manifest["case_count"] == 2
    assert manifest["status_counts"] == {"ok": 2}
    assert manifest["viewer"] == "viewer/index.html"
    assert (output / "run_manifest.json").is_file()
    assert (output / "viewer" / "index.html").is_file()
    for case_id, expected_mode in (("graph-case", "graphs"), ("text-case", "text")):
        case = output / case_id
        assert (case / "source_registry.json").is_file()
        assert (case / "execution_trace.json").is_file()
        assert (case / "input_snapshot.json").is_file()
        snapshot = json.loads((case / "input_snapshot.json").read_text(encoding="utf-8"))
        trace = json.loads((case / "execution_trace.json").read_text(encoding="utf-8"))
        assert snapshot["input_mode"] == expected_mode
        assert snapshot["source"]["context_graph"]["entities"]
        assert trace["schema_version"] == "execution-trace-v2"
        assert trace["input_events"][0]["node"] == "detect_input_mode"
        assert trace["input_events"][1]["node"] == (
            "load_supplied_graph" if expected_mode == "graphs" else "kggen_extract_graph"
        )
        assert (
            output / "viewer" / "cases" / case_id / "index.html"
        ).is_file()
