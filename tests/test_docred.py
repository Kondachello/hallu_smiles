"""Offline contracts for deterministic, held-out DocRED KG evaluation."""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from src.docred import (
    BudgetExceeded,
    BudgetGuard,
    DOCRED_HF_REPO,
    DOCRED_HF_REVISION,
    DocREDDocument,
    EntityResolver,
    RelationAligner,
    Triple,
    align_graph,
    documents_from_manifest,
    evaluate_documents,
    load_docred_documents,
    load_relation_info,
    make_manifest,
    select_relation_threshold,
)
from src.extract import Graph
from src.matching import DictEmbedder


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _document(
    *,
    split: str = "dev",
    source_index: int = 0,
    entities: tuple[tuple[str, ...], ...] = (("Alice",), ("Paris",), ("Acme",)),
    gold: frozenset[Triple] = frozenset({Triple(0, "P1", 1)}),
) -> DocREDDocument:
    return DocREDDocument(
        split=split,
        source_index=source_index,
        document_id=f"{split}-{source_index}-stable",
        text=f"Synthetic document {split} {source_index}.",
        entities=entities,
        gold=gold,
    )


def _aligner() -> RelationAligner:
    return RelationAligner(
        {"P1": "birthplace", "P2": "employed by"},
        DictEmbedder(
            {
                "birthplace": (1.0, 0.0),
                "born in": (1.0, 0.0),
                "employed by": (0.0, 1.0),
                "works for": (0.0, 1.0),
            },
            dim=2,
        ),
    )


def test_alias_alignment_is_conservative_and_directional_with_multilabel_pairs():
    document = _document(gold=frozenset({Triple(0, "P1", 1), Triple(0, "P2", 2)}))
    aligner = _aligner()
    graph = Graph(
        {"Alice", "Paris", "Acme"},
        {
            ("Alice", "born in", "Paris"),
            ("Alice", "works for", "Acme"),
            ("Paris", "born in", "Alice"),
        },
    )
    aligned = align_graph(document, graph, aligner, 0.75)
    assert aligned.triples == {Triple(0, "P1", 1), Triple(0, "P2", 2), Triple(1, "P1", 0)}
    assert aligned.entity_pairs == {(0, 1), (0, 2), (1, 0)}
    # Direction is retained: reverse gold triples never count as recovered.
    summary, _ = evaluate_documents([document], {document.document_id: graph}, {}, aligner, 0.75, n_bootstrap=10)
    assert summary["matched_triples"] == 2
    assert summary["gold_triples"] == 2
    assert summary["predicted_triples"] == 3


def test_ambiguous_entity_alias_is_not_guessed():
    document = _document(entities=(("Alex",), ("Alex",), ("Paris",)))
    resolver = EntityResolver(document)
    assert resolver.resolve("Alex").status == "ambiguous"
    graph = Graph({"Alex", "Paris"}, {("Alex", "born in", "Paris")})
    aligned = align_graph(document, graph, _aligner(), 0.75)
    assert aligned.triples == set()
    assert aligned.diagnostics["entity_ambiguous"] == 1


def test_threshold_is_selected_on_train_only_and_failure_scores_zero_predictions():
    train = [_document(split="train_annotated", source_index=0)]
    dev = [_document(split="dev", source_index=1)]
    aligner = _aligner()
    # The predicate has cosine 0.8 with the train relation description: it is
    # included at .65/.75 but excluded at .85, so train chooses .75 on tie.
    aligner = RelationAligner(
        {"P1": "birthplace"},
        DictEmbedder({"birthplace": (1.0, 0.0), "rough birthplace": (0.8, 0.6)}, dim=2),
    )
    graph = Graph({"Alice", "Paris"}, {("Alice", "rough birthplace", "Paris")})
    threshold, tuning = select_relation_threshold(train, {train[0].document_id: graph}, {}, aligner)
    assert threshold == 0.75
    assert tuning["selected"]["threshold"] == 0.75

    # A held-out extraction failure is retained in its denominator rather than
    # quietly dropped; it contributes no predicted triples and lowers coverage.
    summary, scores = evaluate_documents(
        dev, {dev[0].document_id: None}, {dev[0].document_id: "ExtractionError"},
        aligner, threshold, n_bootstrap=20,
    )
    assert scores[0]["extraction_failed"] is True
    assert summary["documents"] == 1
    assert summary["predicted_triples"] == 0
    assert summary["extraction_coverage"] == 0.0
    assert summary["extraction_failures"] == 1


def test_loader_manifest_and_relation_info_are_deterministic(tmp_path):
    records = []
    for index in range(3):
        records.append({
            "sents": [["Alice", "visited", "Paris", str(index)]],
            "vertexSet": [[{"name": "Alice"}], [{"name": "Paris"}]],
            "labels": [{"h": 0, "t": 1, "r": "P1"}],
        })
    for name, payload in (("train_annotated.json.gz", records), ("dev.json.gz", records)):
        with gzip.open(tmp_path / name, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
    with gzip.open(tmp_path / "rel_info.json.gz", "wt", encoding="utf-8") as handle:
        json.dump({"P1": "birthplace"}, handle)
    train = load_docred_documents(tmp_path, "train_annotated")
    dev = load_docred_documents(tmp_path, "dev")
    manifest_a = make_manifest(
        train_documents=train, dev_documents=dev, train_count=2, dev_count=2, seed=42, data_dir=tmp_path,
    )
    manifest_b = make_manifest(
        train_documents=train, dev_documents=dev, train_count=2, dev_count=2, seed=42, data_dir=tmp_path,
    )
    assert manifest_a == manifest_b
    assert manifest_a["dataset"]["repository"] == DOCRED_HF_REPO
    assert manifest_a["dataset"]["revision"] == DOCRED_HF_REVISION
    assert [item["split"] for item in manifest_a["documents"]].count("train_annotated") == 2
    assert [item["split"] for item in manifest_a["documents"]].count("dev") == 2
    assert load_relation_info(tmp_path) == {"P1": "birthplace"}


def test_manifest_refuses_non_preregistered_split_counts():
    manifest = {
        "protocol": "docred-kg-eval-manifest-v1",
        "seed": 42,
        "calibration": {"split": "train_annotated", "count": 50},
        "heldout": {"split": "dev", "count": 199},
    }
    with pytest.raises(ValueError, match="fixed 50/200"):
        documents_from_manifest(manifest, [], [])


def test_budget_guard_reserves_before_next_live_document():
    guard = BudgetGuard(10.5)
    assert guard.estimate_eur({"api_calls": 524}) == 10.48
    with pytest.raises(BudgetExceeded):
        guard.assert_can_start_document({"api_calls": 524})
    guard.assert_can_reserve_documents({"api_calls": 10}, 240)
    with pytest.raises(BudgetExceeded):
        guard.assert_can_reserve_documents({"api_calls": 50}, 240)
    assert guard.estimate_eur({"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}) > 2.5


def test_budget_guard_reserves_each_raw_live_operation_before_it_is_sent():
    guard = BudgetGuard(0.06)
    usage = {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    guard.reserve_live_request(usage)
    guard.reserve_live_request(usage)
    guard.reserve_live_request(usage)
    assert guard.reserved_live_requests == 3
    assert guard.estimate_eur(usage) == 0.06
    with pytest.raises(BudgetExceeded, match="before live request"):
        guard.reserve_live_request(usage)


def test_cache_replay_verifier_rejects_scientific_drift_but_allows_usage_drift(tmp_path):
    live = tmp_path / "live"
    replay = tmp_path / "replay"
    live.mkdir()
    replay.mkdir()
    for root, calls in ((live, 1), (replay, 0)):
        (root / "run_metadata.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")
        (root / "relation_alignment_tuning.json").write_text("{}\n", encoding="utf-8")
        (root / "document_scores.jsonl").write_text('{"document_id":"safe"}\n', encoding="utf-8")
        (root / "metrics.json").write_text(json.dumps({"score": 1, "usage": {"api_calls": calls}, "budget": {"x": calls}}), encoding="utf-8")
    subprocess.run([
        sys.executable, str(SCRIPTS / "verify_docred_cache_replay.py"),
        "--live-dir", str(live), "--replay-dir", str(replay),
    ], check=True)
    (replay / "document_scores.jsonl").write_text('{"document_id":"changed"}\n', encoding="utf-8")
    failed = subprocess.run([
        sys.executable, str(SCRIPTS / "verify_docred_cache_replay.py"),
        "--live-dir", str(live), "--replay-dir", str(replay),
    ])
    assert failed.returncode != 0


def test_tex_report_is_generated_only_from_completed_archive(tmp_path):
    root = tmp_path / "vertex-cpu-docred-kg-artifacts-run"
    (root / "docred-live").mkdir(parents=True)
    manifest = {
        "dataset": {"repository": DOCRED_HF_REPO, "revision": DOCRED_HF_REVISION},
        "documents": ([{"split": "train_annotated"}] * 50) + ([{"split": "dev"}] * 200),
    }
    (root / "docred_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "run_metadata.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")
    metrics = {
        "documents": 200, "extraction_coverage": 1.0, "extraction_failures": 0,
        "evaluation_split": "held-out-development", "triple_recall": 0.2,
        "gold_supported_precision": 0.1, "triple_f1": 0.133333,
        "matched_triples": 2, "gold_triples": 10, "predicted_triples": 20,
        "entity_pair_recall": 0.3, "entity_pair_gold_supported_precision": 0.25,
        "entity_pair_f1": 0.272727, "matched_entity_pairs": 3,
        "gold_entity_pairs": 10, "predicted_entity_pairs": 12,
        "bootstrap": {
            "replicates": 1000, "triple_recall_ci95": [0.1, 0.3],
            "gold_supported_precision_ci95": [0.05, 0.2], "triple_f1_ci95": [0.08, 0.24],
        },
        "alignment_diagnostics": {
            "raw_predicted_triples": 20, "entity_aligned_predictions": 15,
            "relation_aligned_predictions": 12, "entity_unmatched": 3,
            "entity_ambiguous": 1, "relation_unmatched": 2, "relation_ambiguous": 1,
        },
        "budget": {"max_eur": 10.5, "estimated_spend_eur": 1.2},
    }
    (root / "docred-live" / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (root / "docred-live" / "relation_alignment_tuning.json").write_text(json.dumps({"selected_threshold": 0.75}), encoding="utf-8")
    usage = {"live": {"api_calls": 10, "cache_hits": 4, "retries": 1}, "replay": {"api_calls": 0}}
    (root / "usage-counts.json").write_text(json.dumps(usage), encoding="utf-8")
    inventory = {"files": 5, "aggregate_sha256": "a" * 64}
    (root / "cache-before-replay.json").write_text(json.dumps(inventory), encoding="utf-8")
    (root / "cache-after-replay.json").write_text(json.dumps(inventory), encoding="utf-8")
    artifact = tmp_path / "docred.tar.gz"
    with tarfile.open(artifact, "w:gz") as archive:
        archive.add(root, arcname=root.name)
    output = tmp_path / "docred.tex"
    subprocess.run([
        sys.executable, str(SCRIPTS / "write_docred_kg_results_tex.py"),
        "--artifact", str(artifact), "--output", str(output),
    ], check=True)
    text = output.read_text(encoding="utf-8")
    assert "gold-supported precision" in text
    assert "held-out development" in text
    assert "0.200" in text
