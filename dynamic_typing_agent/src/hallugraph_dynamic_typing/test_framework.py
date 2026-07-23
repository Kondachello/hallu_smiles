"""Unified no-gold test runner for supplied graphs and raw-text KGGen inputs."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from .agent import DynamicTypingAgent, graph_from_fixture
from .models import AnswerInput, SourceInput
from .persistence import ArtifactWriter


InputMode = Literal["auto", "graphs", "text"]
KggenMode = Literal["fake", "live"]
SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
FORBIDDEN_FIELDS = frozenset(
    {"gold", "gold_label", "gold_labels", "labels", "hallucination_labels"}
)


def load_no_gold_cases(
    path: str | Path,
    *,
    limit: int | None = None,
    case_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Read a JSONL test suite and reject labels, unsafe IDs and duplicates."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    requested_ids = set(case_ids or ())
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number}: each test case must be a JSON object")
        leaked = FORBIDDEN_FIELDS & set(row)
        if leaked:
            raise ValueError(
                f"line {line_number}: no-gold input contains forbidden fields: "
                f"{', '.join(sorted(leaked))}"
            )
        case_id = str(row.get("case_id") or f"case-{line_number:04d}")
        if not SAFE_CASE_ID.fullmatch(case_id):
            raise ValueError(
                f"line {line_number}: case_id must match {SAFE_CASE_ID.pattern}"
            )
        if case_id in seen:
            raise ValueError(f"line {line_number}: duplicate case_id {case_id!r}")
        if not str(row.get("context", "")).strip():
            raise ValueError(f"line {line_number}: context is required")
        row = dict(row)
        row["case_id"] = case_id
        row.setdefault("source_id", case_id)
        seen.add(case_id)
        if requested_ids and case_id not in requested_ids:
            continue
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    missing_ids = requested_ids - seen
    if missing_ids:
        raise ValueError(
            "requested case_id is absent from test suite: "
            + ", ".join(sorted(missing_ids))
        )
    if not rows:
        raise ValueError("test suite contains no cases")
    return rows


def detect_input_mode(row: Mapping[str, Any], requested: InputMode = "auto") -> Literal["graphs", "text"]:
    """Choose the per-case input adapter without changing the output contract."""
    has_graphs = isinstance(row.get("graphs"), Mapping)
    if requested == "graphs" and not has_graphs:
        raise ValueError(f"case {row.get('case_id')}: graphs mode requires a graphs object")
    if requested == "text" and has_graphs:
        return "text"
    if requested == "graphs":
        return "graphs"
    return "graphs" if has_graphs else "text"


def _graph_payload(graph: Any) -> dict[str, Any]:
    return {
        "entities": sorted(str(item) for item in graph.entities),
        "relations": sorted(
            [list(map(str, relation)) for relation in graph.relations]
        ),
    }


def _supplied_graphs(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    graphs = row.get("graphs")
    if not isinstance(graphs, Mapping):
        raise ValueError(f"case {row['case_id']}: graphs object is missing")
    required = {"context", "query"}
    missing = required - set(graphs)
    if missing:
        raise ValueError(
            f"case {row['case_id']}: missing supplied graph roles: {', '.join(sorted(missing))}"
        )
    result = {role: dict(graphs[role]) for role in required}
    if str(row.get("response", "")).strip():
        if "answer" not in graphs:
            raise ValueError(
                f"case {row['case_id']}: response requires a supplied answer graph"
            )
        result["answer"] = dict(graphs["answer"])
    return result


def _extract_graphs(
    row: Mapping[str, Any], extractor: Any
) -> dict[str, dict[str, Any]]:
    result = {
        "context": _graph_payload(
            extractor.extract(str(row["context"]), kind="context")
        ),
        "query": _graph_payload(
            extractor.extract(str(row.get("query", "")), kind="query")
        ),
    }
    response = str(row.get("response", "")).strip()
    if response:
        result["answer"] = _graph_payload(extractor.extract(response, kind="answer"))
    return result


def _case_metrics(
    *,
    graphs: Mapping[str, Mapping[str, Any]],
    source_run: Any,
    answer_run: Any | None,
) -> dict[str, int]:
    registry = source_run.registry
    annotations = answer_run.annotations if answer_run is not None else None
    return {
        "graph_entities": sum(len(graph.get("entities", [])) for graph in graphs.values()),
        "graph_relations": sum(
            len(graph.get("relations", [])) for graph in graphs.values()
        ),
        "types": len(registry.types) if registry is not None else 0,
        "source_assignments": len(registry.assignments) if registry is not None else 0,
        "answer_assignments": (
            len(annotations.answer_assignments) if annotations is not None else 0
        ),
        "nli_results": (
            (len(registry.nli_results) if registry is not None else 0)
            + (len(annotations.nli_results) if annotations is not None else 0)
        ),
    }


def run_test_suite(
    *,
    agent: DynamicTypingAgent,
    input_path: str | Path,
    output: str | Path,
    limit: int | None = None,
    input_mode: InputMode = "auto",
    kggen_mode: KggenMode = "fake",
    kggen_config: str | None = None,
    kggen_cache_root: str | None = None,
    case_ids: Iterable[str] | None = None,
    render_viewer: bool = True,
) -> dict[str, Any]:
    """Run heterogeneous cases into one normalized, locally browsable suite."""
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"test output must be a new or empty directory: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    rows = load_no_gold_cases(input_path, limit=limit, case_ids=case_ids)
    modes = [detect_input_mode(row, input_mode) for row in rows]

    extractor = None
    kggen_protocol = None
    if "text" in modes:
        from .kggen_pipeline import make_extractor

        extractor, kggen_protocol = make_extractor(
            kggen_config=kggen_config,
            fake=kggen_mode == "fake",
            cache_root=kggen_cache_root,
        )

    # One output root is an invariant of the test framework, even when an agent was
    # constructed by a caller with a different default artifact directory.
    agent.artifacts_root = root
    case_results: list[dict[str, Any]] = []
    for row, mode in zip(rows, modes, strict=True):
        case_id = str(row["case_id"])
        case_dir = root / case_id
        input_events: list[dict[str, Any]] = [
            {
                "event": "node_completed",
                "node": "detect_input_mode",
                "inputs": {
                    "requested_mode": input_mode,
                    "has_supplied_graphs": isinstance(row.get("graphs"), Mapping),
                },
                "outputs": {"selected_mode": mode},
            }
        ]
        try:
            graphs = (
                _supplied_graphs(row)
                if mode == "graphs"
                else _extract_graphs(row, extractor)
            )
            for role, graph in graphs.items():
                input_events.append(
                    {
                        "event": "node_completed",
                        "node": (
                            "load_supplied_graph"
                            if mode == "graphs"
                            else "kggen_extract_graph"
                        ),
                        "inputs": {
                            "role": role,
                            "protocol": (
                                "supplied-graphs-v1"
                                if mode == "graphs"
                                else kggen_protocol
                            ),
                            "text_chars": len(
                                str(
                                    row.get(
                                        "response"
                                        if role == "answer"
                                        else role,
                                        "",
                                    )
                                )
                            ),
                        },
                        "outputs": {
                            "role": role,
                            "entity_count": len(graph.get("entities", [])),
                            "relation_count": len(graph.get("relations", [])),
                            "graph": graph,
                        },
                    }
                )
            source = SourceInput(
                source_id=str(row["source_id"]),
                context_raw=str(row["context"]),
                query_raw=str(row.get("query", "")),
                context_graph=graph_from_fixture(
                    graph_id=f"{case_id}:context",
                    role="context",
                    payload=graphs["context"],
                ),
                query_graph=graph_from_fixture(
                    graph_id=f"{case_id}:query",
                    role="query",
                    payload=graphs["query"],
                ),
            )
            source_run = agent.build_source_registry(source)
            answer = None
            answer_run = None
            response = str(row.get("response", "")).strip()
            if source_run.registry is not None and response:
                answer = AnswerInput(
                    source_id=source.source_id,
                    response_id=str(row.get("response_id", case_id)),
                    response_raw=response,
                    answer_graph=graph_from_fixture(
                        graph_id=f"{case_id}:answer",
                        role="answer",
                        payload=graphs["answer"],
                    ),
                    registry=source_run.registry,
                )
                answer_run = agent.annotate_answer(answer)
            artifact_dir = agent.write_run_artifacts(
                run_id=case_id, source_run=source_run, answer_run=answer_run
            )
            trace_path = artifact_dir / "execution_trace.json"
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            ArtifactWriter(artifact_dir).write_json(
                "execution_trace.json",
                {
                    **trace,
                    "schema_version": "execution-trace-v2",
                    "input_events": input_events,
                },
            )
            snapshot = {
                "schema_version": "typing-test-input-v2",
                "case_id": case_id,
                "input_mode": mode,
                "source": source.model_dump(mode="json"),
                "answer": (
                    answer.model_dump(mode="json", exclude={"registry"})
                    if answer is not None
                    else None
                ),
                "graph_provenance": {
                    "kind": "supplied" if mode == "graphs" else "kggen",
                    "protocol": (
                        "supplied-graphs-v1" if mode == "graphs" else kggen_protocol
                    ),
                },
                "metadata": {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "context",
                        "query",
                        "response",
                        "graphs",
                    }
                },
            }
            ArtifactWriter(artifact_dir).write_json("input_snapshot.json", snapshot)
            status = (
                answer_run.status.value
                if answer_run is not None
                else source_run.status.value
            )
            case_results.append(
                {
                    "case_id": case_id,
                    "status": status,
                    "input_mode": mode,
                    "has_answer": bool(response),
                    "artifact_dir": case_id,
                    "metrics": _case_metrics(
                        graphs=graphs,
                        source_run=source_run,
                        answer_run=answer_run,
                    ),
                    "failure": (
                        answer_run.failure
                        if answer_run is not None and answer_run.failure
                        else source_run.failure
                    ),
                }
            )
        except Exception as exc:
            case_writer = ArtifactWriter(case_dir)
            case_writer.write_json(
                "failure.json",
                {
                    "schema_version": "typing-test-failure-v1",
                    "case_id": case_id,
                    "input_mode": mode,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            case_writer.write_json(
                "execution_trace.json",
                {
                    "schema_version": "execution-trace-v2",
                    "input_events": [
                        *input_events,
                        {
                            "event": "node_failed",
                            "node": "prepare_case_input",
                            "inputs": {"case_id": case_id, "input_mode": mode},
                            "outputs": {
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                            },
                        },
                    ],
                    "source_events": [],
                    "answer_events": [],
                },
            )
            case_results.append(
                {
                    "case_id": case_id,
                    "status": "failed",
                    "input_mode": mode,
                    "has_answer": bool(str(row.get("response", "")).strip()),
                    "artifact_dir": case_id,
                    "metrics": {},
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            )

    status_counts = Counter(item["status"] for item in case_results)
    run_manifest = {
        "schema_version": "typing-test-run-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": root.name,
        "input": {
            "path": str(Path(input_path)),
            "requested_mode": input_mode,
            "kggen_mode": kggen_mode if "text" in modes else None,
            "kggen_protocol": kggen_protocol,
            "no_gold": True,
        },
        "case_count": len(case_results),
        "status_counts": dict(sorted(status_counts.items())),
        "cases": case_results,
    }
    writer = ArtifactWriter(root)
    writer.write_json("run_manifest.json", run_manifest)
    # Keep the historical filename so older discovery scripts can still find the run.
    writer.write_json("summary.json", {"cases": case_results})
    if render_viewer:
        from .viewer import write_viewer_site

        viewer = write_viewer_site(root)
        run_manifest = {
            **run_manifest,
            "viewer": viewer.relative_to(root).as_posix(),
        }
        writer.write_json("run_manifest.json", run_manifest)
    return run_manifest
