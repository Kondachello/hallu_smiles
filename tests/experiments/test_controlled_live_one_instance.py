from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from experiments.artifacts import atomic_write_json, atomic_write_jsonl
from experiments.controlled_live_one_instance import run_ragtruth_one_instance_controlled_live_probe
from experiments.detectors import build_controlled_shared_kggen_fake

ROOT = Path(__file__).resolve().parents[2]


def _inputs(tmp_path):
    source, response = tmp_path / "source.jsonl", tmp_path / "response.jsonl"
    atomic_write_jsonl(source, [{"source_id":"s", "task_type":"QA", "source":"fixture", "source_info":{"question":"Where?","passages":"Museum is in Northbridge."}, "prompt":"use context"}])
    atomic_write_jsonl(response, [{"id":"6845", "source_id":"s", "model":"fixture", "temperature":0.0, "split":"test", "response":"Museum is in Northbridge.", "quality":"good", "labels":[{"label_type":"unsupported"}]}])
    hallu = tmp_path / "hallu.yaml"; shutil.copy(ROOT / "config.yaml", hallu)
    hallu.write_text(hallu.read_text(encoding="utf-8") + f"\ncache_dir: {str(tmp_path / 'kg-cache').replace(chr(92), '/')}\n", encoding="utf-8")
    hhem = tmp_path / "hhem"; hhem.mkdir(); (hhem / "config.json").write_text("{}", encoding="utf-8")
    graph = tmp_path / "graph.yaml"; graph.write_text(f"extractor:\n  backend: shared_kggen\nnli:\n  backend: hhem\n  model: {str(hhem).replace(chr(92), '/')}\n  revision: 0e7edb3689e710c52ba120086e8f91ea3ee87f23\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"; atomic_write_json(manifest, {"protocol":"hallu-vertex-openai-gateway-v1", "api_path":"/v1", "logical_model":"openai/gemini-2.5-flash", "vertex_model":"gemini-2.5-flash", "vertex_location":"europe-west4", "gateway_release":"test", "cloud_run_revision":"test"})
    return source, response, hallu, graph, manifest


def test_controlled_live_entrypoint_selects_injected_factory_and_fails_closed_cache_replay(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source, response, hallu, graph, manifest = _inputs(tmp_path)
    calls = []
    def factory(**kwargs):
        calls.append(kwargs["cache_mode"])
        return build_controlled_shared_kggen_fake(kwargs["hallugraph_config"], cache_mode=kwargs["cache_mode"])
    secret = "must-not-appear"
    first, second, report = run_ragtruth_one_instance_controlled_live_probe(
        source_info_path=source, response_path=response, response_id="6845", output_root=tmp_path / "runs",
        hallugraph_config=hallu, grapheval_config=graph, gateway_manifest_path=manifest,
        cache_root=tmp_path / "kg-cache", environ={"HALLU_GATEWAY_URL":"https://example.test", "HALLU_GATEWAY_API_KEY":secret}, detector_factory=factory,
    )
    assert calls == ["read_write", "cache_only"]
    assert report["materialize_real_kggen_calls"] > 0
    assert report["cache_replay_real_kggen_calls"] == 0
    assert report["cache_replay_gateway_calls"] == 0
    assert report["shared_graph_consistent_across_methods_and_passes"] is True
    assert first.validate()["valid"] and second.validate()["valid"]
    for archive in (first, second):
        assert {r["method"]: r["status"] for r in archive.read_jsonl("predictions/raw_predictions.jsonl")} == {"hallugraph":"ok", "grapheval":"ok"}
        assert all(secret not in p.read_text(encoding="utf-8", errors="ignore") for p in archive.path.rglob("*") if p.is_file())


def test_controlled_live_entrypoint_requires_project_secret(tmp_path):
    source, response, hallu, graph, manifest = _inputs(tmp_path)
    with pytest.raises(RuntimeError, match="HALLU_GATEWAY_API_KEY"):
        run_ragtruth_one_instance_controlled_live_probe(source_info_path=source, response_path=response, response_id="6845", output_root=tmp_path / "runs", hallugraph_config=hallu, grapheval_config=graph, gateway_manifest_path=manifest, cache_root=tmp_path / "cache", environ={"HALLU_GATEWAY_URL":"https://example.test"})
