"""Paired R12 comparison invariants independent of prompts and completions."""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.candidate_agreement import (
    GRAPH_REFERENCE_PROTOCOL,
    GraphReferenceError,
    evaluate_paired_candidate_agreement,
    load_graph_reference,
)


ROOT = Path(__file__).resolve().parents[1]


MANIFEST = "19cb9472e1662ac029dab7e144e07267c9e43f7ca50556aa92123a5e268e4f86"


def _reference_payload():
    records = []
    # Source 12448 is quarantined; 17 factual train and three factual test
    # answer graphs are explicitly unscorable in the historical graph protocol.
    for index in range(582):
        y = int(index < 300)
        records.append({
            "source_id": f"train-source-{index}", "response_id": f"train-response-{index}",
            "split": "train", "y": y,
            "scores": {"strict": 0.1 + .8 * y, "support": 0.2 + .6 * y, "support_critical": 0.15 + .7 * y},
        })
    for index in range(147):
        y = int(index < 75)
        records.append({
            "source_id": f"test-source-{index}", "response_id": f"test-response-{index}",
            "split": "test", "y": y,
            "scores": {"strict": 0.1 + .8 * y, "support": 0.2 + .6 * y, "support_critical": 0.15 + .7 * y},
        })
    return {
        "protocol": GRAPH_REFERENCE_PROTOCOL,
        "manifest_sha256": MANIFEST,
        "archive_sha256": "a" * 64,
        "frozen_thresholds": {"strict": 0.5, "support": 0.5, "support_critical": 0.5},
        "records": records,
    }


def _candidate_rows(payload, *, flip_test_scores=False):
    rows = []
    for record in payload["records"]:
        score = .2 + .6 * record["y"]
        if flip_test_scores and record["split"] == "test":
            score = 1.0 - score
        rows.append({
            "source_id": record["source_id"], "response_id": record["response_id"],
            "split": record["split"], "y": record["y"], "candidate_disagreement": score,
        })
    return rows


def test_graph_reference_requires_exact_r12_pairing_and_train_only_threshold(tmp_path):
    payload = _reference_payload()
    source = tmp_path / "r12-reference.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    reference = load_graph_reference(source, manifest_sha256=MANIFEST)

    baseline = evaluate_paired_candidate_agreement(
        _candidate_rows(payload), reference, n_bootstrap=5
    )
    perturbed = evaluate_paired_candidate_agreement(
        _candidate_rows(payload, flip_test_scores=True), reference, n_bootstrap=5
    )
    assert baseline["heldout_test"]["n"] == 147
    assert baseline["heldout_test"]["n_hallucinated"] == 75
    assert baseline["heldout_test"]["n_factual"] == 72
    assert baseline["threshold_selection"]["candidate_agreement"]["n"] == 582
    # Test labels/scores can change held-out metrics but never threshold choice.
    assert (
        baseline["threshold_selection"]["candidate_agreement"]["theta"]
        == perturbed["threshold_selection"]["candidate_agreement"]["theta"]
    )
    assert "candidate_agreement" in baseline["paired_vs_support_critical"]


def test_graph_reference_rejects_wrong_r12_heldout_balance(tmp_path):
    payload = _reference_payload()
    payload["records"][-1]["y"] = 1
    source = tmp_path / "bad-r12-reference.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GraphReferenceError, match="label balance"):
        load_graph_reference(source, manifest_sha256=MANIFEST)


def test_candidate_runner_records_explicit_unscorable_states_without_text(tmp_path):
    runner_path = ROOT / "scripts" / "run_ragtruth_candidate_agreement.py"
    spec = importlib.util.spec_from_file_location("candidate_agreement_runner", runner_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = SimpleNamespace(
        response_id="response-1", source_id="source-1", split="test", y=1, gen_model="test-model"
    )
    for state in ("unscorable_output_length", "unscorable_empty_candidate"):
        record = module._unscorable_record(instance, MANIFEST, state, 1.0)
        assert record["state"] == state
        assert "response" not in record
        assert "prompt" not in record
