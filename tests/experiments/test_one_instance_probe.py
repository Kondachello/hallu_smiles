from __future__ import annotations

import json
from pathlib import Path

from experiments.artifacts import atomic_write_jsonl, read_jsonl
from experiments.contracts import assert_no_gold
from experiments.one_instance import render_probe_summary, run_ragtruth_one_instance_probe

ROOT = Path(__file__).resolve().parents[2]


def test_one_ragtruth_response_runs_real_adapters_with_fake_backends_without_gold_leakage(tmp_path, monkeypatch) -> None:
    """This is deliberately one object: it validates wiring, not detector quality."""
    source_info = tmp_path / "source_info.jsonl"
    responses = tmp_path / "response.jsonl"
    atomic_write_jsonl(
        source_info,
        [
            {
                "source_id": "source-one",
                "task_type": "QA",
                "source": "fixture",
                "source_info": {
                    "question": "Which city hosts the museum?",
                    "passages": "The Aurora Museum is located in Northbridge. It opens daily.",
                },
                "prompt": "Answer only from the passages.",
            }
        ],
    )
    # Evaluation-only fields simulate a real raw RAGTruth response.  The probe must
    # never propagate them into its no-gold input, predictions, or human summary.
    atomic_write_jsonl(
        responses,
        [
            {
                "id": "response-one",
                "source_id": "source-one",
                "model": "fixture-generator",
                "temperature": 0.0,
                "split": "test",
                "response": "The Aurora Museum is located in Northbridge.",
                "quality": "good",
                "labels": [{"start": 0, "end": 3, "text": "The", "label_type": "unsupported"}],
            }
        ],
    )
    monkeypatch.chdir(tmp_path)  # keep FakeKGGen's cache outside the repository worktree

    archive, report = run_ragtruth_one_instance_probe(
        source_info_path=source_info,
        response_path=responses,
        response_id="response-one",
        output_root=tmp_path / "runs",
        hallugraph_config=ROOT / "config.yaml",
    )

    assert report["validation"] == {"run_id": "ragtruth-one-response-one", "valid": True, "errors": []}
    instances = read_jsonl(archive.path / "instances.no_gold.jsonl")
    assert len(instances) == 1
    assert_no_gold(instances[0])
    assert "labels" not in json.dumps(instances[0], sort_keys=True)
    assert "quality" not in json.dumps(instances[0], sort_keys=True)

    predictions = archive.read_jsonl("predictions/raw_predictions.jsonl")
    assert {row["method"] for row in predictions} == {"hallugraph", "grapheval"}
    assert all(row["status"] == "ok" for row in predictions)
    assert all(row["gold_access_state"] == "hidden" for row in predictions)
    assert len(archive.read_jsonl("predictions/paired_predictions.jsonl")) == 1
    assert archive.validate()["valid"] is True

    rendered = render_probe_summary(archive)
    assert "RAGTRUTH ONE-INSTANCE PAIRED PROBE (OFFLINE)" in rendered
    assert "gold passed to detectors: no" in rendered
    assert "unsupported" not in rendered
    assert "quality" not in rendered
