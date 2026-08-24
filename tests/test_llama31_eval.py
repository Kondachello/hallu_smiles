"""Offline contracts for the controlled Llama-3.1 RAGTruth evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from run import extract_all, write_extraction_summary
from src.data import Instance
from src.dspy_adapter import GatewayIdentityDriftError, validate_gateway_identity
from src.extract import Graph, UsageLogger
from src.llama31_eval import (
    QUARANTINED_SOURCE_ID,
    build_llama31_instances,
    frozen_reference_artifact_dict,
    load_frozen_reference_graphs,
    load_llama31_manifest_instances_with_historical_manifest,
    write_llama31_manifest,
)


def _source_ids() -> list[str]:
    return [QUARANTINED_SOURCE_ID, *[str(index) for index in range(1, 750)]]


def _controlled_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    data = tmp_path / "data"
    data.mkdir()
    source_ids = _source_ids()
    with (data / "source_info.jsonl").open("w", encoding="utf-8") as handle:
        for source_id in source_ids:
            handle.write(json.dumps({
                "source_id": source_id,
                "task_type": "QA",
                "source_info": {"passages": f"context {source_id}", "question": f"query {source_id}"},
            }) + "\n")
    (data / "response.jsonl").write_text("", encoding="utf-8")
    historical = tmp_path / "historical.json"
    historical.write_text(json.dumps({
        "version": 1,
        "task": "QA",
        "seed": 42,
        "quotas": {"train_sources": 600, "test_sources": 150},
        "records": [
            {
                "source_id": source_id,
                "response_id": f"historical_{source_id}",
                "split": "train" if index < 600 else "test",
                "y": index % 2,
                "gen_model": "historical",
            }
            for index, source_id in enumerate(source_ids)
        ],
    }), encoding="utf-8")
    annotations = tmp_path / "annotations.csv"
    with annotations.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "id", "generated_response", "prompt", "hallucination", "annotation_reason",
            "annotation_raw", "annotation_model",
        ])
        writer.writeheader()
        for index, source_id in enumerate(source_ids):
            writer.writerow({
                "id": f"llama31_8b_{source_id}",
                "generated_response": f"answer {source_id}",
                "prompt": f"context {source_id}\nquery {source_id}",
                "hallucination": 1 if index < 256 else 0,
                "annotation_reason": "fixture",
                "annotation_raw": "fixture",
                "annotation_model": "openai/gpt-4o",
            })
    return annotations, data, historical


def test_controlled_manifest_preserves_historical_split_and_csv_labels(tmp_path):
    annotations, data, historical = _controlled_inputs(tmp_path)
    instances, provenance = build_llama31_instances(annotations, data, historical)
    assert len(instances) == 750
    assert sum(instance.split == "train" for instance in instances) == 600
    assert sum(instance.split == "test" for instance in instances) == 150
    assert sum(instance.y for instance in instances) == 256
    assert next(instance for instance in instances if instance.source_id == QUARANTINED_SOURCE_ID).split == "train"
    manifest = write_llama31_manifest(tmp_path / "llama31_manifest.json", instances, provenance)
    restored = load_llama31_manifest_instances_with_historical_manifest(
        manifest, annotations, data, historical
    )
    assert [instance.response_id for instance in restored] == [instance.response_id for instance in instances]


def test_frozen_reference_loader_rejects_missing_and_corrupt_graphs(tmp_path):
    annotations, data, historical = _controlled_inputs(tmp_path)
    instances, _ = build_llama31_instances(annotations, data, historical)
    references = {
        instance.source_id: (Graph({"context"}, set()), Graph({"query"}, set()))
        for instance in instances
        if instance.source_id != QUARANTINED_SOURCE_ID
    }
    provenance = {
        "gateway_manifest_sha256": "9407591410b215ba41478290526acd3a4ea32f3dd70a63076c6394c95e37c845",
        "llm_runtime_fingerprint": "vertex-gateway:9ba169c4f2de8a246c756948b24a3860a54cc419957004d9ed351c2ad538b3bd",
    }
    artifact = frozen_reference_artifact_dict(instances, references, historical_provenance=provenance)
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    loaded, details = load_frozen_reference_graphs(
        path, instances, excluded_source_ids={QUARANTINED_SOURCE_ID}
    )
    assert len(loaded) == 749
    assert details["reference_origin"] == "frozen_historical_artifact"
    artifact["records"].pop()
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage"):
        load_frozen_reference_graphs(path, instances, excluded_source_ids={QUARANTINED_SOURCE_ID})


def test_extract_all_never_calls_reference_extraction_when_frozen(tmp_path):
    class Extractor:
        usage = UsageLogger(None)

        def __init__(self):
            self.reference_calls = 0
            self.answer_calls = 0

        def extract_reference(self, context, query):
            self.reference_calls += 1
            raise AssertionError("frozen references must not invoke extractor.extract_reference")

        def extract(self, response, *, kind):
            assert kind == "response"
            self.answer_calls += 1
            return Graph({response}, set())

    instances = [
        Instance("r1", "1", "QA", "llama", "train", "c1", "q1", "a1", 0),
        Instance("r2", "2", "QA", "llama", "test", "c2", "q2", "a2", 1),
    ]
    frozen = {source: (Graph({"c"}, set()), Graph({"q"}, set())) for source in ("1", "2")}
    extractor = Extractor()
    refs, answers, failures = extract_all(
        SimpleNamespace(llm=SimpleNamespace(concurrency=1)),
        instances,
        extractor,
        tmp_path,
        frozen_reference_graphs=frozen,
    )
    assert refs == frozen
    assert len(answers) == 2
    assert not failures
    assert extractor.reference_calls == 0
    assert extractor.answer_calls == 2


def test_frozen_summary_has_no_reference_cache_records(tmp_path):
    class Extractor:
        usage = UsageLogger(None)

        def _cache_key(self, text):
            return f"key-{text}"

        def _cache_path(self, key):
            return tmp_path / f"{key}.json"

        def cache_location(self, key):
            return "primary", self._cache_path(key)

    instance = Instance("r1", "1", "QA", "llama", "train", "c", "q", "a", 0)
    path = write_extraction_summary(
        [instance],
        {"1": (Graph({"c"}, set()), Graph({"q"}, set()))},
        {"r1": Graph({"a"}, set())},
        [],
        Extractor(),
        tmp_path,
        frozen_reference_provenance={"reference_origin": "frozen_historical_artifact"},
    )
    summary = json.loads(path.read_text())
    assert summary["reference_graph_provenance"]["reference_origin"] == "frozen_historical_artifact"
    assert summary["graph_records"][0]["cache"]["context"] is None
    assert summary["graph_records"][0]["cache"]["query"] is None


def test_gateway_fingerprint_drift_is_rejected(monkeypatch):
    expected = "a" * 64
    monkeypatch.setenv("EXPECTED_GATEWAY_MANIFEST_SHA256", expected)
    validate_gateway_identity({"system_fingerprint": "aaaaaaaaaaaaaaaa:model"})
    with pytest.raises(GatewayIdentityDriftError):
        validate_gateway_identity({"system_fingerprint": "bbbbbbbbbbbbbbbb:model"})
