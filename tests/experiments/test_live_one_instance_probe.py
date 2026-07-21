from __future__ import annotations

from pathlib import Path

from experiments.artifacts import atomic_write_json, atomic_write_jsonl
from experiments.live_one_instance import render_live_probe_summary, run_ragtruth_one_instance_live_probe
from experiments.mocks import demo_detectors


def test_live_probe_contract_reads_environment_and_redacts_secret_without_live_calls(tmp_path) -> None:
    """Exercise the live Job contract with injected fake detectors, never a network call."""
    source_info = tmp_path / "source_info.jsonl"
    responses = tmp_path / "response.jsonl"
    hhem = tmp_path / "hhem"
    hhem.mkdir()
    (hhem / "config.json").write_text("{}\n", encoding="utf-8")
    atomic_write_jsonl(source_info, [{
        "source_id": "source-live-one",
        "task_type": "QA",
        "source": "fixture",
        "source_info": {"question": "Where is the archive?", "passages": "The archive is in Northbridge."},
        "prompt": "Use only the passage.",
    }])
    atomic_write_jsonl(responses, [{
        "id": "response-live-one",
        "source_id": "source-live-one",
        "model": "fixture-generator",
        "temperature": 0.0,
        "split": "test",
        "response": "The archive is in Northbridge.",
        "quality": "good",
        "labels": [{"start": 0, "end": 3, "text": "The", "label_type": "unsupported"}],
    }])
    hallu_config = tmp_path / "hallu.yaml"
    hallu_config.write_text("llm:\n  model: openai/gemini-2.5-flash\n", encoding="utf-8")
    graph_config = tmp_path / "graph.yaml"
    graph_config.write_text(
        "extractor:\n  backend: gateway\nnli:\n  backend: hhem\n  model: " + str(hhem).replace("\\", "/") + "\n  revision: 0e7edb3689e710c52ba120086e8f91ea3ee87f23\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "gateway-manifest.json"
    atomic_write_json(manifest, {
        "protocol": "hallu-vertex-openai-gateway-v1",
        "api_path": "/v1",
        "logical_model": "openai/gemini-2.5-flash",
        "vertex_model": "gemini-2.5-flash",
        "vertex_location": "europe-west4",
        "gateway_release": "test-release",
        "cloud_run_revision": "test-revision",
    })
    secret = "do-not-leak-live-secret"
    archive, report = run_ragtruth_one_instance_live_probe(
        source_info_path=source_info,
        response_path=responses,
        response_id="response-live-one",
        output_root=tmp_path / "runs",
        hallugraph_config=hallu_config,
        grapheval_config=graph_config,
        gateway_manifest_path=manifest,
        environ={"HALLU_GATEWAY_URL": "https://example.test", "HALLU_GATEWAY_API_KEY": secret},
        detector_factory=lambda **_kwargs: demo_detectors(),
    )

    assert report["validation"]["valid"] is True
    assert report["detector_statuses"] == {"grapheval": "ok", "hallugraph": "ok"}
    events = archive.read_jsonl("audit/live_one_instance_events.jsonl")
    assert [event["stage"] for event in events] == [
        "environment", "gateway_manifest", "hallugraph_token_policy", "hallugraph_transport_retry_policy", "input_materialization", "detector_construction", "paired_inference", "archive_seal",
    ]
    for path in archive.path.rglob("*"):
        if path.is_file():
            assert secret not in path.read_text(encoding="utf-8")
    rendered = render_live_probe_summary(archive)
    assert "LIVE DATASPHERE" in rendered
    assert "not passed to detectors" in rendered
