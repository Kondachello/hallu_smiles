"""Offline proof of the shared-graph three-way experiment infrastructure."""
from __future__ import annotations

from pathlib import Path

from experiments.artifacts import RunArchive, atomic_write_jsonl
from experiments.demo import demo_instances
from experiments.detectors import build_three_way_shared_kggen_fake
from experiments.runner import run_paired, seal_run


ROOT = Path(__file__).resolve().parents[2]


def _execute(tmp_path, *, run_id: str, cache_mode: str):
    archive = RunArchive.create(
        tmp_path / "runs",
        run_id=run_id,
        manifest={
            "comparison_track": "controlled_shared_all_graphs_three_way_stub_v1",
            "run_purpose": "offline_infrastructure_test",
        },
    )
    instances = archive.path / "instances.no_gold.jsonl"
    atomic_write_jsonl(instances, demo_instances()[:1])
    detectors, provider = build_three_way_shared_kggen_fake(
        ROOT / "config.yaml",
        cache_mode=cache_mode,
        cache_root=tmp_path / "kg-cache",
    )
    summary = run_paired(
        archive,
        instances_path=instances,
        detectors=detectors,
        shared_graph_provider=provider,
    )
    seal_run(archive, instances)
    return archive, provider, summary


def test_three_variants_share_one_sealed_graph_bundle_and_unknown_types_preserve_b0(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    archive, provider, summary = _execute(tmp_path, run_id="three-way-cold", cache_mode="read_write")

    assert summary["n_predictions"] == 3
    assert summary["shared_graphs"] == 3
    assert summary["shared_graph_bundles"] == 1
    assert set(summary["variants"]) == {
        "grapheval_kggen_shared_answer_fake_v1",
        "hallugraph_untyped_shared_kggen_fake_v1",
        "hallugraph_dynamic_types_unknown_stub_v1",
    }
    assert provider.extractor.usage.summary()["api_calls"] == 3

    predictions = archive.read_jsonl("predictions/raw_predictions.jsonl")
    assert len({row["shared_graph_bundle_id"] for row in predictions}) == 1
    assert len({row["shared_context_graph_id"] for row in predictions}) == 1
    assert len({row["shared_query_graph_id"] for row in predictions}) == 1
    assert len({row["shared_answer_graph_id"] for row in predictions}) == 1

    by_variant = {row["variant"]: row for row in predictions}
    untyped = by_variant["hallugraph_untyped_shared_kggen_fake_v1"]
    typed = by_variant["hallugraph_dynamic_types_unknown_stub_v1"]
    assert typed["raw_score"] == untyped["raw_score"]
    assert typed["flagged_unit_ids"] == untyped["flagged_unit_ids"]
    assert typed["components"]["dynamic_typing"]["typing_status"] == "ok"
    assert typed["components"]["dynamic_typing"]["answer_available_type_count"] == 0

    comparison = archive.read_jsonl("predictions/paired_predictions.jsonl")[0]
    assert comparison["all_variants_status_ok"] is True
    assert comparison["all_variants_same_graph_bundle"] is True
    assert set(comparison["variant_prediction_ids"]) == set(summary["variants"])

    assert len(archive.read_jsonl("shared_graphs/bundles.jsonl")) == 1
    assert len(archive.read_jsonl("typing/type_registries.jsonl")) == 1
    assert len(archive.read_jsonl("typing/type_annotation_bundles.jsonl")) == 1
    assert archive.validate()["valid"] is True

    replay, replay_provider, replay_summary = _execute(
        tmp_path, run_id="three-way-cache-only", cache_mode="cache_only"
    )
    assert replay_summary["shared_graph_bundles"] == 1
    assert replay_provider.extractor.usage.summary()["api_calls"] == 0
    assert replay.validate()["valid"] is True
