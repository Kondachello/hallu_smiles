"""Source-level QA pilot manifest tests."""
from collections import Counter

from src.data import Instance
import pytest

from src.sampling import (
    load_manifest_instances,
    qa_sample_quotas,
    select_qa_pilot,
    write_manifest,
)


def _instances():
    rows = []
    models = ["gpt", "llama", "mistral", "qwen"]
    for split, count in (("train", 16), ("test", 4)):
        for index in range(count):
            y = index % 2
            rows.append(Instance(
                response_id=f"{split}-r{index}", source_id=f"{split}-s{index}", task="QA",
                gen_model=models[index % len(models)], split=split, context=f"context {index}",
                query=f"query {index}", response=f"response {index}", y=y,
            ))
    return rows


def test_select_qa_pilot_is_balanced_one_response_per_source_and_reproducible(tmp_path):
    source = _instances()
    first = select_qa_pilot(source, seed=42)
    second = select_qa_pilot(list(reversed(source)), seed=42)

    assert [i.response_id for i in first] == [i.response_id for i in second]
    assert len(first) == 20 == len({i.source_id for i in first})
    assert Counter((i.split, i.y) for i in first) == Counter({
        ("train", 0): 8, ("train", 1): 8, ("test", 0): 2, ("test", 1): 2,
    })

    path = write_manifest(tmp_path / "pilot.json", first, seed=42, train_sources=16, test_sources=4)
    restored = load_manifest_instances(path, source)
    assert [i.response_id for i in restored] == [i.response_id for i in first]


def test_qa_sample_quotas_make_a_100_record_80_20_split_without_rounding():
    assert qa_sample_quotas(100, "0.2") == (80, 20)
    assert qa_sample_quotas(20, "1/5") == (16, 4)
    with pytest.raises(ValueError, match="integer test size"):
        qa_sample_quotas(101, "0.2")
    with pytest.raises(ValueError, match="positive and even"):
        qa_sample_quotas(10, "0.1")
