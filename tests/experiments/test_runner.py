from __future__ import annotations

from experiments.artifacts import RunArchive, atomic_write_jsonl
from experiments.demo import demo_instances
from experiments.evaluation import evaluate_joined_predictions, join_gold
from experiments.mocks import demo_detectors
from experiments.runner import run_paired, seal_run


def test_paired_mock_run_is_sealed_and_checksum_validated(tmp_path) -> None:
    archive = RunArchive.create(tmp_path, run_id="r1", manifest={"run_purpose": "test", "comparison_track": "exploratory"})
    instances = archive.path / "instances.no_gold.jsonl"
    atomic_write_jsonl(instances, demo_instances())

    summary = run_paired(archive, instances_path=instances, detectors=demo_detectors())
    assert summary["n_predictions"] == 4
    paired = archive.read_jsonl("predictions/paired_predictions.jsonl")
    assert len(paired) == 2
    assert all(row["both_status_ok"] for row in paired)

    seal = seal_run(archive, instances)
    assert seal["methods"] == ["grapheval", "hallugraph"]
    assert archive.validate() == {"run_id": "r1", "valid": True, "errors": []}


def test_resume_does_not_duplicate_predictions(tmp_path) -> None:
    archive = RunArchive.create(tmp_path, run_id="r2", manifest={"run_purpose": "test", "comparison_track": "exploratory"})
    instances = archive.path / "instances.no_gold.jsonl"
    atomic_write_jsonl(instances, demo_instances())
    run_paired(archive, instances_path=instances, detectors=demo_detectors())
    run_paired(archive, instances_path=instances, detectors=demo_detectors(), resume=True)
    rows = archive.read_jsonl("predictions/raw_predictions.jsonl")
    assert len(rows) == 4
    assert len({(row["method"], row["response_id"]) for row in rows}) == 4


def test_gold_join_requires_seal_and_evaluation_is_post_prediction(tmp_path) -> None:
    archive = RunArchive.create(tmp_path, run_id="r3", manifest={"run_purpose": "test", "comparison_track": "exploratory"})
    instances = archive.path / "instances.no_gold.jsonl"
    atomic_write_jsonl(instances, demo_instances())
    run_paired(archive, instances_path=instances, detectors=demo_detectors())
    gold = archive.path / "external-response-gold.jsonl"
    atomic_write_jsonl(gold, [
        {"response_id": "mock-response-001", "gold_response_label": 0, "quality_raw": "good"},
        {"response_id": "mock-response-002", "gold_response_label": 1, "quality_raw": "good"},
    ])
    import pytest

    with pytest.raises(ValueError, match="prediction_seal"):
        join_gold(archive, response_gold_path=str(gold))
    seal_run(archive, instances)
    joined = join_gold(archive, response_gold_path=str(gold))
    assert len(joined) == 4
    metrics = evaluate_joined_predictions(archive, thresholds={"hallugraph": 0.5, "grapheval": 0.5})
    assert {row["method"] for row in metrics} == {"hallugraph", "grapheval"}
