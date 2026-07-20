"""A visually readable, deterministic mock experiment; it never calls network/model backends."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import RunArchive, atomic_write_jsonl
from .mocks import demo_detectors
from .runner import run_paired, seal_run


def demo_instances() -> list[dict[str, Any]]:
    return [
        {
            "dataset_record_id": "mock-001",
            "source_id": "mock-source-001",
            "response_id": "mock-response-001",
            "context_raw": "Paris is the capital of France. The Eiffel Tower is in Paris.",
            "query_raw": "What is the capital of France?",
            "response_raw": "Paris is the capital of France.",
            "metadata": {"dataset_record_id": "mock-001", "task": "QA", "source_dataset": "mock", "generator_model": "mock-llm", "generator_temperature": 0.0, "context_document_ids": ["source:mock-source-001"], "context_document_order": ["source:mock-source-001"]},
            "gold_access_state": "hidden",
        },
        {
            "dataset_record_id": "mock-002",
            "source_id": "mock-source-002",
            "response_id": "mock-response-002",
            "context_raw": "Paris is the capital of France. The Eiffel Tower is in Paris.",
            "query_raw": "What is the capital of France?",
            "response_raw": "Berlin is the capital of France. The Eiffel Tower is in Berlin.",
            "metadata": {"dataset_record_id": "mock-002", "task": "QA", "source_dataset": "mock", "generator_model": "mock-llm", "generator_temperature": 0.0, "context_document_ids": ["source:mock-source-002"], "context_document_order": ["source:mock-source-002"]},
            "gold_access_state": "hidden",
        },
    ]


def _bar(score: float | None, width: int = 16) -> str:
    if score is None:
        return "·" * width
    filled = round(max(0.0, min(1.0, score)) * width)
    return "█" * filled + "░" * (width - filled)


def pretty_summary(archive: RunArchive) -> str:
    paired = archive.read_jsonl("predictions/paired_predictions.jsonl")
    lines = [
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║  MOCK EXPERIMENT — GraphEval × HalluGraph (offline, no secrets)     ║",
        "╠══════════════════════════════════════════════════════════════════════╣",
        "║ response             HalluGraph risk          GraphEval risk         ║",
        "╠══════════════════════════════════════════════════════════════════════╣",
    ]
    for row in paired:
        left = f"{_bar(row['hallugraph_score'])} {row['hallugraph_score']!s:>5}"
        right = f"{_bar(row['grapheval_score'])} {row['grapheval_score']!s:>5}"
        lines.append(f"║ {row['response_id']:<20} {left:<25} {right:<24} ║")
    lines.extend(
        [
            "╠══════════════════════════════════════════════════════════════════════╣",
            "║ Higher risk means less support from the supplied context.             ║",
            f"║ Archive: {str(archive.path):<59} ║",
            "╚══════════════════════════════════════════════════════════════════════╝",
        ]
    )
    return "\n".join(lines)


def run_demo(output_root: str | Path, *, run_id: str = "mock-demo") -> dict[str, Any]:
    root = Path(output_root)
    archive = RunArchive.create(root, run_id=run_id, manifest={"run_purpose": "mock_demo", "comparison_track": "exploratory", "network_access": False})
    instances_path = archive.path / "instances.no_gold.jsonl"
    atomic_write_jsonl(instances_path, demo_instances())
    summary = run_paired(archive, instances_path=instances_path, detectors=demo_detectors())
    seal = seal_run(archive, instances_path)
    text = pretty_summary(archive)
    (archive.path / "reports" / "mock-summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return {**summary, "run_id": run_id, "archive": str(archive.path), "sealed_methods": seal["methods"]}
