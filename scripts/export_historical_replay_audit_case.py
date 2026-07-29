#!/usr/bin/env python3
"""Export post-seal HalluGraph/GraphEval cases for a human or agent audit.

This command is intentionally analysis-only.  It validates an existing sealed archive,
joins the official RAGTruth labels afterwards, and writes case packages outside the
archive.  It never invokes KGGen, HalluGraph, GraphEval, an LLM, or a gateway.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.artifacts import RunArchive, sha256_file


METHODS = ("hallugraph", "grapheval")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite or choose a new output directory")
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _method_thresholds(metrics_path: Path, archive_dir: Path, responses_path: Path) -> tuple[dict[str, float], dict[str, Any]]:
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not report.get("analysis_only"):
        raise ValueError("metrics report must declare analysis_only=true")
    if Path(str(report.get("archive_dir", ""))).resolve() != archive_dir.resolve():
        raise ValueError("metrics report belongs to another archive")
    if report.get("responses_sha256") != sha256_file(responses_path):
        raise ValueError("metrics report was built with another response.jsonl")
    if report.get("threshold_protocol") != "choose_max_F1_on_train_then_evaluate_once_on_test":
        raise ValueError("metrics report does not declare the expected train-only threshold protocol")
    by_method = {str(row.get("method")): row for row in report.get("methods") or []}
    if set(by_method) != set(METHODS):
        raise ValueError("metrics report must contain HalluGraph and GraphEval thresholds")
    thresholds: dict[str, float] = {}
    for method in METHODS:
        row = by_method[method]
        if row.get("selection_split") != "train" or row.get("selection_objective") != "max_F1; ties: max_recall, then lower_threshold":
            raise ValueError(f"metrics threshold provenance is incomplete for {method}")
        thresholds[method] = float(row["threshold"])
    return thresholds, report


def _decision(prediction: dict[str, Any], threshold: float) -> bool | None:
    if prediction.get("status") != "ok" or prediction.get("raw_score") is None:
        return None
    return float(prediction["raw_score"]) > threshold


def _error_kind(*, gold: int, decision: bool | None) -> str:
    if decision is None:
        return "UNSCORABLE"
    if decision == bool(gold):
        return "CORRECT"
    return "FP" if decision else "FN"


def _markdown(package: dict[str, Any], *, json_name: str, prompt_path: str) -> str:
    classification = package["classification"]
    h = package["methods"]["hallugraph"]
    g = package["methods"]["grapheval"]
    return f"""# Вход для аудита случая {package['case_id']}

Это постфактум-пакет: детекторы и извлечение графов уже завершены. Передайте
`{json_name}` вместе с системным промптом `{prompt_path}` агенту-аудитору.
Не запускайте на этом пакете HalluGraph, GraphEval, KGGen или gateway.

## Краткая сводка

- Официальная RAGTruth-метка: `{classification['gold_response_label']}`
- Решение HalluGraph: `{h['decision']}`; балл `{h['raw_score']}`; порог строго `> {h['threshold']}`
- Классификация HalluGraph: `{classification['hallugraph_outcome']}`
- Решение GraphEval: `{g['decision']}`; балл `{g['raw_score']}`; порог строго `> {g['threshold']}`
- Часть данных: `{package['ragtruth']['split']}`

## Содержимое JSON

- `input` — исходные query, context, response и неизменяемые хэши;
- `graphs` — полные графы контекста, запроса и ответа;
- `methods` — исходные записи предсказаний обоих методов, включая компоненты;
- `ragtruth.labels` — исходные размеченные интервалы и комментарии RAGTruth;
- `classification` — только постфактум-сопоставление с официальной меткой;
- `provenance` — проверка sealing, хэши и происхождение порогов.

Не считайте GraphEval эталоном: сначала проведите независимый аудит HalluGraph,
затем разберите GraphEval как вторичное сравнение.
"""


def build_case_packages(
    *,
    archive_dir: Path,
    responses_path: Path,
    metrics_path: Path,
    response_ids: Iterable[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return full packages and compact manifest rows without modifying the archive."""
    archive_dir = archive_dir.resolve()
    responses_path = responses_path.resolve()
    metrics_path = metrics_path.resolve()
    archive = RunArchive(archive_dir.parent, archive_dir.name)
    validation = archive.validate()
    if not validation["valid"]:
        raise ValueError(f"prediction archive seal is invalid: {validation['errors']}")
    thresholds, metrics = _method_thresholds(metrics_path, archive_dir, responses_path)

    predictions = read_jsonl(archive_dir / "predictions" / "raw_predictions.jsonl")
    if not predictions:
        raise ValueError("archive has no raw predictions")
    if any(row.get("gold_access_state") != "hidden" for row in predictions):
        raise ValueError("prediction rows must retain gold_access_state='hidden'")
    instances = {str(row["response_id"]): row for row in read_jsonl(archive_dir / "instances.no_gold.jsonl")}
    if not instances:
        raise ValueError("archive has no instances.no_gold.jsonl")
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in predictions:
        method = str(row.get("method"))
        if method not in METHODS:
            raise ValueError(f"unexpected method in archive: {method!r}")
        response_id = str(row.get("response_id"))
        if method in by_id[response_id]:
            raise ValueError(f"duplicate prediction for {response_id}/{method}")
        by_id[response_id][method] = row
    if set(instances) != set(by_id):
        raise ValueError("prediction and instance response IDs differ")
    if any(set(rows) != set(METHODS) for rows in by_id.values()):
        raise ValueError("every response must contain exactly one HalluGraph and GraphEval prediction")

    gold_rows = {str(row["id"]): row for row in read_jsonl(responses_path)}
    missing_gold = sorted(set(instances) - set(gold_rows))
    if missing_gold:
        raise ValueError(f"official response.jsonl lacks archive IDs: {missing_gold[:10]}")
    split_mismatches = [response_id for response_id, instance in instances.items() if gold_rows[response_id].get("split") != instance.get("split")]
    if split_mismatches:
        raise ValueError(f"official response.jsonl and archive split differ for {len(split_mismatches)} records")

    graph_by_hash = {str(row["input_sha256"]): row for row in read_jsonl(archive_dir / "shared_graphs" / "graph_index.jsonl")}
    requested = [str(response_id) for response_id in response_ids]
    unknown = sorted(set(requested) - set(instances))
    if unknown:
        raise ValueError(f"response IDs are not in archive: {unknown}")

    packages: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for response_id in requested:
        instance = instances[response_id]
        gold_row = gold_rows[response_id]
        gold = int(bool(gold_row.get("labels") or []))
        methods: dict[str, Any] = {}
        for method in METHODS:
            prediction = by_id[response_id][method]
            score = prediction.get("raw_score")
            decision = _decision(prediction, thresholds[method])
            methods[method] = {
                "status": prediction.get("status"),
                "raw_score": None if score is None else float(score),
                "score_direction": prediction.get("score_direction"),
                "threshold": thresholds[method],
                "threshold_comparator": ">",
                "decision": decision,
                "prediction": prediction,
            }
        graphs: dict[str, Any] = {}
        for role, hash_field in (("context", "context_hash"), ("query", "query_hash"), ("response", "response_hash")):
            graph = graph_by_hash.get(str(instance[hash_field]))
            if graph is None:
                raise ValueError(f"missing {role} graph for response_id={response_id}")
            graphs[role] = graph
        hallugraph_outcome = _error_kind(gold=gold, decision=methods["hallugraph"]["decision"])
        grapheval_outcome = _error_kind(gold=gold, decision=methods["grapheval"]["decision"])
        package = {
            "schema_version": "historical-replay-agent-audit-case-v1",
            "analysis_only": True,
            "case_id": response_id,
            "provenance": {
                "archive_dir": str(archive_dir),
                "archive_validation": validation,
                "responses_path": str(responses_path),
                "responses_sha256": sha256_file(responses_path),
                "metrics_path": str(metrics_path),
                "threshold_protocol": metrics["threshold_protocol"],
                "gold_join_timing": "post_seal_evaluation_only",
            },
            "ragtruth": {
                "response_id": response_id,
                "source_id": gold_row.get("source_id"),
                "split": gold_row.get("split"),
                "quality": gold_row.get("quality"),
                "model": gold_row.get("model"),
                "temperature": gold_row.get("temperature"),
                "gold_response_label": gold,
                "labels": gold_row.get("labels") or [],
            },
            "input": instance,
            "graphs": graphs,
            "methods": methods,
            "classification": {
                "gold_response_label": gold,
                "hallugraph_outcome": hallugraph_outcome,
                "grapheval_outcome": grapheval_outcome,
                "paired_score_available": all(methods[method]["decision"] is not None for method in METHODS),
            },
        }
        packages.append(package)
        manifest.append({
            "case_id": response_id,
            "split": package["ragtruth"]["split"],
            "gold_response_label": gold,
            "hallugraph_outcome": hallugraph_outcome,
            "hallugraph_score": methods["hallugraph"]["raw_score"],
            "hallugraph_threshold": thresholds["hallugraph"],
            "grapheval_outcome": grapheval_outcome,
            "grapheval_score": methods["grapheval"]["raw_score"],
            "grapheval_threshold": thresholds["grapheval"],
            "paired_score_available": package["classification"]["paired_score_available"],
        })
    return packages, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True, help="extracted sealed replay archive")
    parser.add_argument("--responses", type=Path, required=True, help="official RAGTruth response.jsonl used after sealing")
    parser.add_argument("--metrics", type=Path, required=True, help="gold-audit-metrics.json produced by build_historical_replay_gold_audit.py")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for exported JSON/Markdown packages")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--response-id", action="append", help="one response ID; may be supplied repeatedly")
    selection.add_argument("--hallugraph-errors", choices=("fp", "fn", "all"), help="export all scored HalluGraph false positives, false negatives or both")
    parser.add_argument("--no-markdown", action="store_true", help="write JSON packages and manifest only")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing packages in --output-dir")
    args = parser.parse_args()

    archive_dir = args.archive_dir.resolve()
    output_dir = args.output_dir.resolve()
    if archive_dir in output_dir.parents or output_dir == archive_dir:
        parser.error("--output-dir must be outside the sealed archive")

    if args.response_id:
        selected = list(dict.fromkeys(args.response_id))
    else:
        # Build compact classifications first, then export exactly the requested error slice.
        all_ids = [str(row["response_id"]) for row in read_jsonl(archive_dir / "instances.no_gold.jsonl")]
        _, preliminary = build_case_packages(
            archive_dir=archive_dir, responses_path=args.responses, metrics_path=args.metrics, response_ids=all_ids,
        )
        allowed = {"FP", "FN"} if args.hallugraph_errors == "all" else {args.hallugraph_errors.upper()}
        selected = [row["case_id"] for row in preliminary if row["hallugraph_outcome"] in allowed]

    packages, manifest = build_case_packages(
        archive_dir=archive_dir, responses_path=args.responses, metrics_path=args.metrics, response_ids=selected,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for package in packages:
        case_id = package["case_id"]
        json_path = output_dir / f"audit-case-{case_id}.json"
        if json_path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {json_path}; pass --overwrite or choose a new output directory")
        json_path.write_text(json.dumps(package, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not args.no_markdown:
            markdown_path = output_dir / f"audit-case-{case_id}.md"
            if markdown_path.exists() and not args.overwrite:
                raise FileExistsError(f"refusing to overwrite {markdown_path}; pass --overwrite or choose a new output directory")
            markdown_path.write_text(_markdown(package, json_name=json_path.name, prompt_path="docs/hallugraph-error-audit-system-prompt.md"), encoding="utf-8")
    write_jsonl(output_dir / "audit-manifest.jsonl", manifest, overwrite=args.overwrite)
    counts = {kind: sum(row["hallugraph_outcome"] == kind for row in manifest) for kind in ("FP", "FN", "CORRECT", "UNSCORABLE")}
    print(json.dumps({"output_dir": str(output_dir), "exported_cases": len(packages), "hallugraph_outcomes": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
