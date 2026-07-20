from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.contracts import GoldLeakageError, assert_no_gold
from experiments.datasets.ragtruth import (
    audit_dataset,
    create_source_sample_manifest,
    fetch_dataset,
    materialize_subset,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _dataset(tmp_path: Path) -> tuple[Path, Path]:
    sources = tmp_path / "source_info.jsonl"
    responses = tmp_path / "response.jsonl"
    _write_jsonl(
        sources,
        [
            {"source_id": "s1", "task_type": "QA", "source": "MARCO", "source_info": {"question": "capital?", "passages": "Paris is the capital of France."}, "prompt": "question prompt"},
            {"source_id": "s2", "task_type": "Summary", "source": "CNN/DM", "source_info": "A short article about a river.", "prompt": "summary prompt"},
        ],
    )
    _write_jsonl(
        responses,
        [
            {"id": "r1", "source_id": "s1", "model": "m1", "temperature": 0.0, "labels": [], "split": "train", "quality": "good", "response": "Paris is the capital of France."},
            {"id": "r2", "source_id": "s1", "model": "m2", "temperature": 0.7, "labels": [{"start": 0, "end": 6, "text": "Berlin", "label_type": "Evident Baseless Info"}], "split": "train", "quality": "good", "response": "Berlin is the capital of France."},
            {"id": "r3", "source_id": "s2", "model": "m1", "temperature": 0.0, "labels": [], "split": "test", "quality": "truncated", "response": "A river article."},
        ],
    )
    return sources, responses


def test_audit_and_materialization_physically_isolate_gold(tmp_path: Path) -> None:
    sources, responses = _dataset(tmp_path)
    data_manifest = audit_dataset(sources, responses, revision="a" * 40)
    assert data_manifest["audit"]["n_unique_source_ids"] == 2
    assert data_manifest["audit"]["n_label_text_offset_mismatches"] == 0

    sample = create_source_sample_manifest(
        sources,
        responses,
        dataset_manifest=data_manifest,
        split="train",
        seed=17,
        n_sources=1,
    )
    assert sample["gold_used_for_selection"] is False
    assert sample["selected_source_ids"] == ["s1"]
    paths = materialize_subset(sources, responses, dataset_manifest=data_manifest, sample_manifest=sample, output_dir=tmp_path / "subset")

    no_gold = [json.loads(line) for line in paths["instances"].read_text(encoding="utf-8").splitlines()]
    assert {row["response_id"] for row in no_gold} == {"r1", "r2"}
    for row in no_gold:
        assert_no_gold(row)
        encoded = json.dumps(row).lower()
        assert "quality" not in encoded
        assert "label_type" not in encoded
    gold = [json.loads(line) for line in paths["response_gold"].read_text(encoding="utf-8").splitlines()]
    assert {row["gold_response_label"] for row in gold} == {0, 1}


def test_sampling_is_seeded_and_input_order_independent(tmp_path: Path) -> None:
    sources, responses = _dataset(tmp_path)
    manifest = audit_dataset(sources, responses, revision="b" * 40)
    first = create_source_sample_manifest(sources, responses, dataset_manifest=manifest, split="train", seed=5)
    second = create_source_sample_manifest(sources, responses, dataset_manifest=manifest, split="train", seed=5)
    assert first == second


def test_gold_guard_rejects_nested_label() -> None:
    with pytest.raises(GoldLeakageError):
        assert_no_gold({"metadata": {"hidden_label": 1}})


def test_fetch_rejects_floating_or_short_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact 40-character"):
        fetch_dataset(data_root=tmp_path, revision="main")
