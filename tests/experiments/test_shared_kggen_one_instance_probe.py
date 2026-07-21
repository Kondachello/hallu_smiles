from __future__ import annotations

from pathlib import Path

from experiments.artifacts import atomic_write_jsonl
from experiments.one_instance import (
    render_shared_kggen_mock_probe_summary,
    run_ragtruth_one_instance_shared_kggen_mock_probe,
)

ROOT = Path(__file__).resolve().parents[2]


def test_response_id_two_pass_shared_kggen_cache_probe(tmp_path, monkeypatch) -> None:
    source_info = tmp_path / "source_info.jsonl"
    responses = tmp_path / "responses.jsonl"
    atomic_write_jsonl(
        source_info,
        [{
            "source_id": "source-42", "task_type": "QA", "source": "fixture",
            "source_info": {"question": "Where is Aurora Museum?", "passages": "Aurora Museum is in Northbridge."},
            "prompt": "Answer from passages.",
        }],
    )
    atomic_write_jsonl(
        responses,
        [{
            "id": "response-42", "source_id": "source-42", "model": "fixture", "temperature": 0.0,
            "split": "test", "response": "Aurora Museum is in Northbridge.",
            # These evaluation fields must not enter either detector pass.
            "quality": "good", "labels": [{"start": 0, "end": 6, "label_type": "unsupported"}],
        }],
    )
    monkeypatch.chdir(tmp_path)
    cold, replay, report = run_ragtruth_one_instance_shared_kggen_mock_probe(
        source_info_path=source_info, response_path=responses, response_id="response-42",
        output_root=tmp_path / "runs", cache_root=tmp_path / "project-cache",
        hallugraph_config=ROOT / "config.yaml",
    )

    assert report["response_id"] == "response-42"
    assert report["materialize_kggen_api_calls"] > 0
    assert report["cache_replay_kggen_api_calls"] == 0
    assert report["shared_graph_consistent_across_passes"] is True
    assert cold.validate()["valid"] is True
    assert replay.validate()["valid"] is True
    for archive in (cold, replay):
        predictions = archive.read_jsonl("predictions/raw_predictions.jsonl")
        assert {row["method"] for row in predictions} == {"hallugraph", "grapheval"}
        assert all(row["status"] == "ok" for row in predictions)
        assert len({row["shared_graph_sha256"] for row in predictions}) == 1
        assert "labels" not in (archive.path / "instances.no_gold.jsonl").read_text(encoding="utf-8")
        assert "quality" not in (archive.path / "instances.no_gold.jsonl").read_text(encoding="utf-8")
    rendered = render_shared_kggen_mock_probe_summary(report)
    assert "cache replay KGGen calls : 0" in rendered
