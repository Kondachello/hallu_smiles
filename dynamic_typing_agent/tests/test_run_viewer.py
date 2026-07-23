from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from hallugraph_dynamic_typing.viewer import (
    load_viewer_payload,
    write_viewer_html,
    write_viewer_site,
)


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "examples" / "dynamic_typing_20.no_gold.jsonl"


def _fixture_snapshot() -> dict:
    row = json.loads(CASES.read_text(encoding="utf-8").splitlines()[0])
    return {
        "schema_version": "typing-test-input-v2",
        "case_id": row["case_id"],
        "input_mode": "graphs",
        "graph_provenance": {"kind": "supplied", "protocol": "supplied-graphs-v1"},
        "source": {
            "source_id": row["source_id"],
            "context_raw": row["context"],
            "query_raw": row["query"],
            "context_graph": {
                "graph_id": f"{row['case_id']}:context",
                "role": "context",
                **row["graphs"]["context"],
            },
            "query_graph": {
                "graph_id": f"{row['case_id']}:query",
                "role": "query",
                **row["graphs"]["query"],
            },
        },
        "answer": {
            "source_id": row["source_id"],
            "response_id": row["case_id"],
            "response_raw": row["response"],
            "answer_graph": {
                "graph_id": f"{row['case_id']}:answer",
                "role": "answer",
                **row["graphs"]["answer"],
            },
        },
    }


def _artifact_case(tmp_path: Path) -> Path:
    case = tmp_path / "run" / "dt-001-bank-generalization"
    case.mkdir(parents=True)
    registry = {
        "registry_id": "registry:test",
        "source_id": "src-bank-1",
        "context_graph_id": "ctx",
        "query_graph_id": "qry",
        "types": [
            {
                "type_id": "T-ENTITY",
                "label": "entity",
                "definition": "root",
                "parent_type_ids": [],
                "aliases": [],
                "evidence_span_ids": [],
                "evidence_level": "source_entailed",
                "status": "final",
            },
            {
                "type_id": "T-BANK",
                "label": "commercial bank",
                "definition": "a bank",
                "parent_type_ids": ["T-ENTITY"],
                "aliases": [],
                "evidence_span_ids": ["context:span:0"],
                "evidence_level": "source_entailed",
                "status": "final",
            },
        ],
        "assignments": [
            {
                "node_id": "n1",
                "surface_text": "North Bank",
                "graph_role": "context",
                "type_ids": ["T-BANK"],
                "status": "assigned",
                "evidence_span_ids": ["context:span:0"],
                "reason": "explicit",
            }
        ],
        "evidence_spans": [
            {
                "span_id": "context:span:0",
                "source_role": "context",
                "start_char": 0,
                "end_char": 32,
                "text": "North Bank is a commercial bank.",
            }
        ],
        "nli_results": [],
        "prompt_manifest_sha256": "a" * 64,
        "frozen": True,
        "registry_sha256": "b" * 64,
    }
    (case / "source_registry.json").write_text(
        json.dumps({"status": "ok", "registry": registry}, ensure_ascii=False),
        encoding="utf-8",
    )
    (case / "answer_annotations.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "annotations": {"answer_assignments": [], "nli_results": []},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (case / "manifest.json").write_text(
        json.dumps({"backend": "fake", "prompt_manifest_sha256": "a" * 64}),
        encoding="utf-8",
    )
    (case / "execution_trace.json").write_text(
        json.dumps(
            {
                "schema_version": "execution-trace-v1",
                "source_events": [
                    {
                        "event": "node_completed",
                        "node": "entity_type_decision",
                        "inputs": {"surface_text": "North Bank"},
                        "outputs": {
                            "response": {
                                "selected_type_ids": ["T-BANK"],
                                "reason": "model choice",
                            }
                        },
                    }
                ],
                "answer_events": [],
            }
        ),
        encoding="utf-8",
    )
    (case / "input_snapshot.json").write_text(
        json.dumps(_fixture_snapshot(), ensure_ascii=False), encoding="utf-8"
    )
    return case


def test_single_case_compatibility_document_is_self_contained(tmp_path: Path) -> None:
    case = _artifact_case(tmp_path)
    payload = load_viewer_payload(case)
    html = write_viewer_html(tmp_path / "viewer.html", payload).read_text(
        encoding="utf-8"
    )
    assert "Подробный разбор" in html
    assert "Словарь типов" in html
    assert "Как агент пришёл к результату" in html
    assert "North Bank is a commercial bank." in html
    assert "Модель ответила / предложила" in html
    assert "fetch(" not in html
    assert "<canvas" in html


def test_viewer_site_has_dashboard_assets_and_case_page(tmp_path: Path) -> None:
    case = _artifact_case(tmp_path)
    run = case.parent
    (run / "summary.json").write_text(
        json.dumps([{"case_id": case.name, "status": "ok", "input_mode": "graphs"}]),
        encoding="utf-8",
    )
    index = write_viewer_site(run)
    assert index == run / "viewer" / "index.html"
    assert index.is_file()
    assert (run / "viewer" / "cases" / case.name / "index.html").is_file()
    assert (run / "viewer" / "cases" / case.name / "data.js").is_file()
    javascript = (run / "viewer" / "assets" / "case.js").read_text(encoding="utf-8")
    assert "class CanvasGraph" in javascript
    assert "selectEntity" in javascript
    assert "eventNarrative" in javascript
    assert "fetch(" not in javascript


def test_renderer_builds_dashboard_for_old_summary(tmp_path: Path) -> None:
    case = _artifact_case(tmp_path)
    run = case.parent
    (run / "summary.json").write_text(
        json.dumps([{"case_id": case.name, "status": "ok"}]), encoding="utf-8"
    )
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "scripts/render_run_viewer.py",
            "--run",
            str(run),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (run / "viewer" / "index.html").is_file()
