from __future__ import annotations

import json
from pathlib import Path

import yaml
import pytest

from experiments.artifacts import RunArchive
from experiments.detectors import build_controlled_shared_kggen_fake
from experiments.historical_qa_cache_replay import (
    _select_replay_records,
    run_historical_qa_cache_controlled_replay,
)
from src.config import load_config
from src.extract import CACHE_KEY_SCHEMA_V11_PRE_LENGTH_RETRY, FakeKGGen, KGExtractor, UsageLogger

ROOT = Path(__file__).resolve().parents[2]


def test_historical_replay_selects_a_fully_warm_record_without_llm_calls(tmp_path, monkeypatch) -> None:
    """A partial historical population must not prevent replaying its first full row."""
    record = {
        "response_id": "historical-response-1",
        "source_id": "historical-source-1",
        "dataset_record_id": "ragtruth:historical-response-1",
        "task": "QA",
        "context_raw": "Northbridge is the home of Aurora Museum.",
        "query_raw": "Where is Aurora Museum?",
        "response_raw": "Aurora Museum is in Northbridge.",
        "metadata": {"source_dataset": "RAGTruth"},
    }
    partial = {**record, "response_id": "historical-response-0", "response_raw": "This is not in the cache."}
    historical_root = tmp_path / "historical-kg"
    historical_root.mkdir()

    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    seed_config = {**config, "cache_dir": str(historical_root)}
    seed_config_path = tmp_path / "historical-seed.yaml"
    seed_config_path.write_text(yaml.safe_dump(seed_config, sort_keys=False), encoding="utf-8")
    config["cache_dir"] = str(tmp_path / "current-run-cache")
    config_path = tmp_path / "historical-replay.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    # Populate exactly the three HalluGraph/GraphEval shared graphs for the usable row.
    cfg = load_config(seed_config_path)
    seed = KGExtractor(cfg, backend=FakeKGGen(), usage=UsageLogger(None))
    for role in ("response_raw", "context_raw", "query_raw"):
        text = record[role]
        seed.extract(text, kind=role.removesuffix("_raw"))
        current_key = seed._cache_key(text)
        historical_key = seed.cache_key_for_schema(
            text, schema=CACHE_KEY_SCHEMA_V11_PRE_LENGTH_RETRY
        )
        current_path = historical_root / f"{current_key}.json"
        envelope = json.loads(current_path.read_text(encoding="utf-8"))
        envelope["cache_key"] = historical_key
        (historical_root / f"{historical_key}.json").write_text(
            json.dumps(envelope), encoding="utf-8"
        )
        current_path.unlink()

    hhem_root = tmp_path / "hhem"
    hhem_root.mkdir()
    (hhem_root / "config.json").write_text("{}", encoding="utf-8")
    graph_config = {
        "extractor": {"backend": "shared_kggen"},
        "nli": {"backend": "hhem", "model": str(hhem_root)},
    }
    graph_config_path = tmp_path / "graph-eval.yaml"
    graph_config_path.write_text(yaml.safe_dump(graph_config), encoding="utf-8")
    lineage_path = tmp_path / "lineage.json"
    lineage_path.write_text(json.dumps({
        "lineage_id": "fixture-historical",
        "llm_runtime_fingerprint": "vertex-gateway:" + "a" * 64,
        "gateway_manifest_sha256": "b" * 64,
    }), encoding="utf-8")

    monkeypatch.setattr(
        "experiments.historical_qa_cache_replay.materialize_historical_qa_no_gold",
        lambda *_args, **_kwargs: [partial, record],
    )

    def fake_factory(**kwargs):
        return build_controlled_shared_kggen_fake(
            kwargs["hallugraph_config"],
            cache_mode=kwargs["cache_mode"],
            cache_sources=kwargs["cache_sources"],
        )

    archive, report = run_historical_qa_cache_controlled_replay(
        data_dir=tmp_path / "unused-ragtruth",
        output_root=tmp_path / "runs",
        hallugraph_config=config_path,
        grapheval_config=graph_config_path,
        historical_cache_root=historical_root,
        lineage_path=lineage_path,
        run_id="historical-cache-test",
        detector_factory=fake_factory,
    )

    assert report["selected_response_id"] == record["response_id"]
    assert report["kggen_api_calls"] == 0
    assert report["grapheval_extractor_calls"] == 0
    assert report["graph_sources"] == ["historical_100qa"]
    assert report["detector_statuses"] == {"hallugraph": "ok", "grapheval": "ok"}
    assert all(row["status"] == "compatible_hit" for row in report["cache_preflight"]["rows"] if row["response_id"] == record["response_id"])
    assert all(
        row["cache_key_schema"] == CACHE_KEY_SCHEMA_V11_PRE_LENGTH_RETRY
        for row in report["cache_preflight"]["rows"]
        if row["response_id"] == record["response_id"]
    )
    assert archive.validate()["valid"] is True
    assert "label" not in (archive.path / "instances.no_gold.jsonl").read_text(encoding="utf-8")


def test_historical_replay_preserves_coverage_when_no_triplet_is_available(tmp_path, monkeypatch) -> None:
    record = {
        "response_id": "historical-response-miss",
        "source_id": "historical-source-miss",
        "context_raw": "Uncached context.",
        "query_raw": "Uncached query?",
        "response_raw": "Uncached response.",
        "metadata": {"source_dataset": "RAGTruth"},
    }
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    config["cache_dir"] = str(tmp_path / "current-run-cache")
    config_path = tmp_path / "historical-replay.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    hhem_root = tmp_path / "hhem"
    hhem_root.mkdir()
    (hhem_root / "config.json").write_text("{}", encoding="utf-8")
    graph_config_path = tmp_path / "graph-eval.yaml"
    graph_config_path.write_text(yaml.safe_dump({
        "extractor": {"backend": "shared_kggen"},
        "nli": {"backend": "hhem", "model": str(hhem_root)},
    }), encoding="utf-8")
    lineage_path = tmp_path / "lineage.json"
    lineage_path.write_text(json.dumps({
        "lineage_id": "fixture-historical", "llm_runtime_fingerprint": "vertex-gateway:" + "a" * 64,
    }), encoding="utf-8")
    historical_root = tmp_path / "historical-empty"
    historical_root.mkdir()
    monkeypatch.setattr(
        "experiments.historical_qa_cache_replay.materialize_historical_qa_no_gold",
        lambda *_args, **_kwargs: [record],
    )

    def fake_factory(**kwargs):
        return build_controlled_shared_kggen_fake(
            kwargs["hallugraph_config"], cache_mode=kwargs["cache_mode"], cache_sources=kwargs["cache_sources"],
        )

    with pytest.raises(RuntimeError, match="no selected QA record"):
        run_historical_qa_cache_controlled_replay(
            data_dir=tmp_path / "unused-ragtruth", output_root=tmp_path / "runs",
            hallugraph_config=config_path, grapheval_config=graph_config_path,
            historical_cache_root=historical_root, lineage_path=lineage_path,
            run_id="historical-cache-miss", detector_factory=fake_factory,
        )

    archive = RunArchive(tmp_path / "runs", "historical-cache-miss")
    coverage = json.loads((archive.path / "reports/historical_cache_coverage.json").read_text(encoding="utf-8"))
    manifest = json.loads((archive.path / "run_manifest.json").read_text(encoding="utf-8"))
    assert coverage["misses"] == 3
    assert manifest["run_status"] == "cache_preflight_failed"


def test_historical_replay_selects_reproducible_random_complete_subset() -> None:
    records = [{"response_id": str(index), "source_id": f"source-{index}"} for index in range(12)]
    coverage = {
        "rows": [
            {"response_id": str(index), "role": role, "status": "compatible_hit"}
            for index in range(12)
            for role in ("response", "context", "query")
        ]
    }
    first = _select_replay_records(records, coverage, replay_count=10, replay_selection_seed=917)
    second = _select_replay_records(records, coverage, replay_count=10, replay_selection_seed=917)
    assert [row["response_id"] for row in first] == [row["response_id"] for row in second]
    assert len({row["response_id"] for row in first}) == 10
