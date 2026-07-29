from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from experiments.artifacts import RunArchive, sha256_file


ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "export_historical_replay_audit_case", ROOT / "scripts" / "export_historical_replay_audit_case.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    archive = RunArchive.create(tmp_path / "runs", run_id="sealed", manifest={})
    instance = {
        "response_id": "r1", "source_id": "s1", "split": "train", "gold_access_state": "hidden",
        "query_raw": "What happened?", "context_raw": "Ada founded Acme in 2020.", "response_raw": "Ada founded Acme in 2019.",
        "context_hash": "context-hash", "query_hash": "query-hash", "response_hash": "response-hash", "metadata": {"task": "QA"},
    }
    archive.write_jsonl("instances.no_gold.jsonl", [instance])
    archive.write_jsonl("shared_graphs/graph_index.jsonl", [
        {"input_sha256": "context-hash", "role": "context", "entities": ["Ada", "Acme", "2020"], "relations": [["Ada", "founded", "Acme"]]},
        {"input_sha256": "query-hash", "role": "query", "entities": ["What"], "relations": []},
        {"input_sha256": "response-hash", "role": "response", "entities": ["Ada", "Acme", "2019"], "relations": [["Ada", "founded", "Acme"]]},
    ])
    predictions = [
        {"method": "hallugraph", "response_id": "r1", "status": "ok", "raw_score": 0.1, "gold_access_state": "hidden", "components": {"CFI": 0.9, "EG": 1.0, "RP": 0.8, "unsupported_relations": [["Ada", "founded", "Acme"]]}},
        {"method": "grapheval", "response_id": "r1", "status": "ok", "raw_score": 0.9, "gold_access_state": "hidden", "components": {"triples": [{"triple_id": "t1", "p_unsupported": 0.9}]}, "flagged_unit_ids": ["t1"]},
    ]
    archive.write_jsonl("predictions/raw_predictions.jsonl", predictions)
    archive.seal_predictions(expected_response_ids=["r1"])
    responses = tmp_path / "response.jsonl"
    _jsonl(responses, [{"id": "r1", "source_id": "s1", "split": "train", "quality": "good", "labels": [{"start": 21, "end": 25, "text": "2019", "label_type": "Evident Conflict"}], "response": instance["response_raw"]}])
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({
        "analysis_only": True,
        "archive_dir": str(archive.path.resolve()),
        "responses_sha256": sha256_file(responses),
        "threshold_protocol": "choose_max_F1_on_train_then_evaluate_once_on_test",
        "methods": [
            {"method": "hallugraph", "threshold": 0.5, "selection_split": "train", "selection_objective": "max_F1; ties: max_recall, then lower_threshold"},
            {"method": "grapheval", "threshold": 0.5, "selection_split": "train", "selection_objective": "max_F1; ties: max_recall, then lower_threshold"},
        ],
    }), encoding="utf-8")
    return archive.path, responses, metrics


def test_case_package_contains_raw_artifacts_and_post_seal_gold_only(tmp_path: Path) -> None:
    module = _load_module()
    archive_dir, responses, metrics = _fixture(tmp_path)
    before = (archive_dir / "predictions" / "raw_predictions.jsonl").read_bytes()

    packages, manifest = module.build_case_packages(
        archive_dir=archive_dir, responses_path=responses, metrics_path=metrics, response_ids=["r1"],
    )

    package = packages[0]
    assert manifest[0]["hallugraph_outcome"] == "FN"
    assert package["ragtruth"]["labels"][0]["label_type"] == "Evident Conflict"
    assert package["graphs"]["context"]["entities"] == ["Ada", "Acme", "2020"]
    assert package["methods"]["hallugraph"]["prediction"]["components"]["CFI"] == 0.9
    assert package["methods"]["grapheval"]["prediction"]["flagged_unit_ids"] == ["t1"]
    assert package["classification"]["gold_response_label"] == 1
    assert package["analysis_only"] is True
    assert "gold_response_label" not in (archive_dir / "predictions" / "raw_predictions.jsonl").read_text(encoding="utf-8")
    assert (archive_dir / "predictions" / "raw_predictions.jsonl").read_bytes() == before


def test_case_package_rejects_metrics_from_another_response_file(tmp_path: Path) -> None:
    module = _load_module()
    archive_dir, responses, metrics = _fixture(tmp_path)
    other = tmp_path / "other-response.jsonl"
    _jsonl(other, [{"id": "r1", "split": "train", "labels": []}])

    try:
        module.build_case_packages(archive_dir=archive_dir, responses_path=other, metrics_path=metrics, response_ids=["r1"])
    except ValueError as error:
        assert "another response.jsonl" in str(error)
    else:
        raise AssertionError("expected response file provenance validation")
