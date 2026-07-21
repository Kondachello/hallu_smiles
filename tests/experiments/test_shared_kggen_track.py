from __future__ import annotations

from pathlib import Path
from argparse import Namespace

import pytest

from experiments.artifacts import RunArchive, atomic_write_jsonl
from experiments.cli import cmd_cache_inspect
from experiments.demo import demo_instances
from experiments.detectors import build_controlled_shared_kggen_fake
from experiments.runner import run_paired, seal_run
from experiments.shared_graphs import CachePreflightError, GraphCacheSource, SharedKGGraphProvider, inspect_cache_sources
from src.config import load_config
from src.extract import FakeKGGen, KGExtractor, UsageLogger

ROOT = Path(__file__).resolve().parents[2]


def _run(tmp_path, *, run_id: str, cache_mode: str):
    detectors, provider = build_controlled_shared_kggen_fake(ROOT / "config.yaml", cache_mode=cache_mode)
    archive = RunArchive.create(tmp_path / "runs", run_id=run_id, manifest={"comparison_track": "controlled_shared_kggen_response_v1"})
    instances = archive.path / "instances.no_gold.jsonl"
    atomic_write_jsonl(instances, demo_instances()[:1])
    summary = run_paired(archive, instances_path=instances, detectors=detectors, shared_graph_provider=provider)
    seal_run(archive, instances)
    return archive, provider, summary


def test_controlled_track_records_one_identical_response_graph(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    archive, provider, summary = _run(tmp_path, run_id="shared-first", cache_mode="read_write")

    assert summary["shared_graphs"] == 3  # response, context, and query
    rows = archive.read_jsonl("predictions/raw_predictions.jsonl")
    assert {row["method"] for row in rows} == {"hallugraph", "grapheval"}
    assert len({row["shared_graph_id"] for row in rows}) == 1
    assert len({row["shared_graph_sha256"] for row in rows}) == 1
    assert all(row["artifact_refs"]["shared_graph_id"] == row["shared_graph_id"] for row in rows)
    paired = archive.read_jsonl("predictions/paired_predictions.jsonl")
    assert paired[0]["shared_response_graph_consistent"] is True
    assert paired[0]["both_status_ok"] is True
    assert provider.extractor.usage.summary()["api_calls"] == 3
    assert archive.validate()["valid"] is True


def test_controlled_cache_only_replay_has_zero_kggen_calls(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _run(tmp_path, run_id="shared-warm", cache_mode="read_write")
    archive, provider, _summary = _run(tmp_path, run_id="shared-replay", cache_mode="cache_only")

    assert provider.extractor.usage.summary()["api_calls"] == 0
    assert all(row["status"] == "ok" for row in archive.read_jsonl("predictions/raw_predictions.jsonl"))
    assert archive.validate()["valid"] is True


def test_cache_inspection_rejects_conflicting_same_key(tmp_path) -> None:
    source_a = tmp_path / "a"
    source_b = tmp_path / "b"
    source_a.mkdir()
    source_b.mkdir()
    # A deliberately hand-made pair checks the source inspector before any model call.
    (source_a / ("a" * 64 + ".json")).write_text('{"protocol":"hallu-kg-cache-v2","cache_key":"' + "a" * 64 + '","graph":{"entities":["A"],"relations":[]},"graph_sha256":"bad"}', encoding="utf-8")
    report = inspect_cache_sources([GraphCacheSource("a", source_a), GraphCacheSource("b", source_b)])
    assert report["valid"] is False
    assert len(report["invalid_entries"]) == 1


def test_cache_only_preflight_fails_before_backend(tmp_path) -> None:
    cfg = load_config(ROOT / "config.yaml")
    cfg._data["cache_dir"] = str(tmp_path / "empty")
    cfg.cache_dir = str(tmp_path / "empty")
    provider = SharedKGGraphProvider(KGExtractor(cfg, backend=FakeKGGen(), usage=UsageLogger(None), cache_only=True), cache_mode="cache_only")
    with pytest.raises(CachePreflightError):
        provider.preflight([{"response_id": "r", "response_raw": "Not cached"}])


def test_cache_only_reads_a_declared_external_source(tmp_path) -> None:
    source_cfg = load_config(ROOT / "config.yaml")
    source_cfg._data["cache_dir"] = str(tmp_path / "historical")
    source_cfg.cache_dir = source_cfg._data["cache_dir"]
    source_extractor = KGExtractor(source_cfg, backend=FakeKGGen(), usage=UsageLogger(None))
    expected = source_extractor.extract("External cache graph", kind="response")

    replay_cfg = load_config(ROOT / "config.yaml")
    replay_cfg._data["cache_dir"] = str(tmp_path / "current")
    replay_cfg.cache_dir = replay_cfg._data["cache_dir"]
    replay = KGExtractor(replay_cfg, backend=FakeKGGen(), usage=UsageLogger(None), cache_only=True)
    provider = SharedKGGraphProvider(
        replay, sources=[GraphCacheSource("historical", tmp_path / "historical")], cache_mode="cache_only"
    )
    artifact = provider.materialize("External cache graph", role="response")
    assert artifact.graph.to_dict() == expected.to_dict()
    assert artifact.source_id == "historical"
    assert replay.usage.summary()["api_calls"] == 0


def test_cache_inspect_cli_is_read_only(tmp_path) -> None:
    output = tmp_path / "report.json"
    result = cmd_cache_inspect(
        Namespace(
            hallugraph_config=str(ROOT / "config.yaml"), source=[],
            writable_cache=str(tmp_path / "current"), instances=None,
            role=["response"], output=str(output),
        )
    )
    assert result == 0
    assert output.exists()
