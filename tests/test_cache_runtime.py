"""Offline tests for reproducible runtime fingerprints and warm-cache replay."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.cache import CacheOnlyMissError, evaluation_runtime_metadata
from src.data import Instance
from src.dspy_adapter import StructuredOutputSchemaError
from src.extract import (
    CLUSTER_EQUIVALENCE_POLICY,
    ClusteringCollapseError,
    FakeKGGen,
    KGExtractor,
)
from src.matching import SBERTEmbedder
from src.verifier import RelationVerifier


def _cfg(tmp_path):
    return SimpleNamespace(
        llm=SimpleNamespace(
            model="openai/local-model",
            model_revision="model-revision-a",
            runtime_fingerprint="image-digest-a",
            structured_output_transport="none",
            structured_output_backend="xgrammar",
            api_base="http://127.0.0.1:8000/v1",
            temperature=0.0,
            max_tokens=256,
            max_retries=1,
            retry_backoff_base_s=0.0,
            request_timeout_s=10,
            concurrency=1,
        ),
        extraction=SimpleNamespace(
            cluster=True,
            cluster_context_mode="source_text",
            cluster_max_items=None,
            cluster_min_retention_ratio=None,
            cluster_retention_min_items=5,
            context_chunk_chars=6000,
            serial_chunking=True,
            explicit_clustering=True,
        ),
        matching=SimpleNamespace(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_model_revision="embedding-revision-a",
            stopwords=[],
        ),
        relation_verifier=SimpleNamespace(
            cache_dir=str(tmp_path / "verdicts"),
            prompt_version="test-v1",
            max_evidence_sentences=4,
        ),
        cache_dir=str(tmp_path / "kg"),
    )


def test_graph_cache_key_fingerprints_runtime_revision_and_clustering(tmp_path):
    cfg = _cfg(tmp_path)
    original = KGExtractor(cfg, backend=FakeKGGen())._cache_key("same text")

    changed_revision = copy.deepcopy(cfg)
    changed_revision.llm.model_revision = "model-revision-b"
    assert KGExtractor(changed_revision, backend=FakeKGGen())._cache_key("same text") != original

    changed_runtime = copy.deepcopy(cfg)
    changed_runtime.llm.runtime_fingerprint = "image-digest-b"
    assert KGExtractor(changed_runtime, backend=FakeKGGen())._cache_key("same text") != original

    changed_clustering = copy.deepcopy(cfg)
    changed_clustering.extraction.cluster_max_items = 25
    assert KGExtractor(changed_clustering, backend=FakeKGGen())._cache_key("same text") != original

    changed_context = copy.deepcopy(cfg)
    changed_context.extraction.cluster_context_mode = "empty"
    assert KGExtractor(changed_context, backend=FakeKGGen())._cache_key("same text") != original

    changed_concurrency = copy.deepcopy(cfg)
    changed_concurrency.llm.concurrency = 2
    assert KGExtractor(changed_concurrency, backend=FakeKGGen())._cache_key("same text") != original


def test_cache_only_extractor_reads_warm_cache_and_fails_before_backend(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.extraction.explicit_clustering = False

    class FixtureBackend(FakeKGGen):
        def generate(self, **kwargs):  # pragma: no cover - must never be entered
            if getattr(self, "forbid", False):
                raise AssertionError(f"live backend called with {kwargs}")
            return super().generate(**kwargs)

    writer_backend = FixtureBackend()
    expected = KGExtractor(cfg, backend=writer_backend).extract("Alice sees Bob")
    replay_backend = FixtureBackend()
    replay_backend.forbid = True
    replay = KGExtractor(cfg, backend=replay_backend, cache_only=True)
    assert replay.extract("Alice sees Bob") == expected
    with pytest.raises(CacheOnlyMissError, match="cache-only miss"):
        replay.extract("this graph was never cached")


def test_cache_only_extractor_reads_historical_read_through_root(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.extraction.explicit_clustering = False
    historical = tmp_path / "historical-kg"
    writer_cfg = copy.deepcopy(cfg)
    writer_cfg.cache_dir = str(historical)
    writer = KGExtractor(writer_cfg, backend=FakeKGGen())
    expected = writer.extract("Alice sees Bob")

    replay_cfg = copy.deepcopy(cfg)
    replay_cfg.cache_read_dirs = [str(historical)]
    replay = KGExtractor(replay_cfg, backend=FakeKGGen(), cache_only=True)
    assert replay.extract("Alice sees Bob") == expected
    assert not list((tmp_path / "kg").glob("*.json"))


def test_kg_cache_preflight_reports_every_missing_graph_before_extraction(tmp_path):
    from src.cache_preflight import verify_kg_cache
    from src.extract import Graph

    cfg = _cfg(tmp_path)
    cfg.extraction.explicit_clustering = False
    Path(cfg.cache_dir).mkdir(parents=True)
    writer = KGExtractor(cfg, cache_only=True)
    rows = [
        Instance(
            response_id="r1", source_id="s1", task="QA", gen_model="fixture", split="train",
            context="Context one", query="Question one", response="Answer one", y=0,
        ),
        Instance(
            response_id="r2", source_id="s2", task="QA", gen_model="fixture", split="test",
            context="Context two", query="Question two", response="Answer two", y=1,
        ),
    ]
    graph = Graph({"entity"}, {("entity", "rel", "entity")})
    for row in rows:
        for text in (row.context, row.query, row.response):
            writer._save_cache(writer._cache_key(text), graph)

    assert verify_kg_cache(cfg, rows)["status"] == "ready"
    missing_key = writer._cache_key(rows[1].response)
    (Path(cfg.cache_dir) / f"{missing_key}.json").unlink()
    report = verify_kg_cache(cfg, rows)
    assert report["status"] == "missing"
    assert report["missing_count"] == 1
    assert report["missing"][0]["response_id"] == "r2"


def test_fake_and_live_extractors_cannot_share_cache_entries(tmp_path):
    cfg = _cfg(tmp_path)
    fake_key = KGExtractor(cfg, backend=FakeKGGen())._cache_key("same text")
    live_key = KGExtractor(cfg, cache_only=True)._cache_key("same text")
    assert fake_key != live_key


def test_graph_cache_envelope_rejects_corruption_as_cache_only_miss(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.extraction.explicit_clustering = False
    writer = KGExtractor(cfg, backend=FakeKGGen())
    writer.extract("Alice sees Bob")
    key = writer._cache_key("Alice sees Bob")
    path = writer._cache_path(key)
    payload = path.read_text(encoding="utf-8")
    assert '"protocol": "hallu-kg-cache-v2"' in payload

    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(CacheOnlyMissError, match="cache-only miss"):
        KGExtractor(cfg, backend=FakeKGGen(), cache_only=True).extract("Alice sees Bob")


def test_pipeline_does_not_downgrade_cache_only_miss_to_failed_extraction(tmp_path):
    from run import extract_all

    instance = Instance(
        response_id="r1",
        source_id="s1",
        task="QA",
        gen_model="fixture",
        split="train",
        context="context",
        query="query",
        response="answer",
        y=0,
    )

    class MissingExtractor:
        @staticmethod
        def extract_reference(context, query):
            raise CacheOnlyMissError("context", "missing-key", tmp_path / "missing.json")

    with pytest.raises(CacheOnlyMissError, match="missing-key"):
        extract_all(
            SimpleNamespace(llm=SimpleNamespace(concurrency=1)),
            [instance],
            MissingExtractor(),
            tmp_path,
        )


def test_extraction_summary_proves_exact_reference_answer_pairs_and_cache(tmp_path):
    from run import write_extraction_summary
    from src.extract import Graph

    cfg = _cfg(tmp_path)
    extractor = KGExtractor(cfg, backend=FakeKGGen())
    instances = [
        Instance(
            response_id="r1",
            source_id="s1",
            task="QA",
            gen_model="fixture",
            split="test",
            context="Context one",
            query="Question one",
            response="Answer one",
            y=0,
        ),
        Instance(
            response_id="r2",
            source_id="s2",
            task="QA",
            gen_model="fixture",
            split="test",
            context="Context two",
            query="Question two",
            response="Answer two",
            y=1,
        ),
    ]
    graph = Graph({"entity"}, {("entity", "rel", "entity")})
    for inst in instances:
        for text in (inst.context, inst.query, inst.response):
            key = extractor._cache_key(text)
            extractor._save_cache(key, graph)
    refs = {inst.source_id: (graph, graph) for inst in instances}
    answers = {inst.response_id: graph for inst in instances}

    path = write_extraction_summary(
        instances, refs, answers, [], extractor, tmp_path / "result"
    )
    payload = __import__("json").loads(path.read_text(encoding="utf-8"))

    assert payload["status"] == "ready"
    assert payload["pairs_completed"] == 2
    assert payload["completed_records"] == payload["expected_records"]
    assert len(payload["expected_cache_keys"]) == 6
    assert all(record["cache_file_exists"] for record in payload["cache_records"])
    assert {record["cache_origin"] for record in payload["cache_records"]} == {"primary"}


def test_extraction_summary_accepts_valid_read_through_graph_cache(tmp_path):
    from run import write_extraction_summary
    from src.extract import Graph

    historical = tmp_path / "historical-kg"
    writer_cfg = _cfg(tmp_path)
    writer_cfg.cache_dir = str(historical)
    writer = KGExtractor(writer_cfg, backend=FakeKGGen())
    replay_cfg = _cfg(tmp_path)
    replay_cfg.cache_read_dirs = [str(historical)]
    replay = KGExtractor(replay_cfg, backend=FakeKGGen(), cache_only=True)
    instance = Instance(
        response_id="r1", source_id="s1", task="QA", gen_model="fixture", split="test",
        context="Context one", query="Question one", response="Answer one", y=0,
    )
    graph = Graph({"entity"}, {("entity", "rel", "entity")})
    for text in (instance.context, instance.query, instance.response):
        writer._save_cache(writer._cache_key(text), graph)

    path = write_extraction_summary(
        [instance], {"s1": (graph, graph)}, {"r1": graph}, [], replay, tmp_path / "result"
    )
    payload = __import__("json").loads(path.read_text(encoding="utf-8"))

    assert payload["status"] == "ready"
    assert not list((tmp_path / "kg").glob("*.json"))
    assert {record["cache_origin"] for record in payload["cache_records"]} == {"read-through-1"}


def test_corrupt_primary_cache_does_not_hide_valid_read_through_graph(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.extraction.explicit_clustering = False
    historical = tmp_path / "historical-kg"
    writer_cfg = copy.deepcopy(cfg)
    writer_cfg.cache_dir = str(historical)
    writer = KGExtractor(writer_cfg, backend=FakeKGGen())
    expected = writer.extract("Alice sees Bob")
    key = writer._cache_key("Alice sees Bob")

    replay_cfg = copy.deepcopy(cfg)
    replay_cfg.cache_read_dirs = [str(historical)]
    primary = Path(replay_cfg.cache_dir)
    primary.mkdir(parents=True)
    (primary / f"{key}.json").write_text("{}\n", encoding="utf-8")
    replay = KGExtractor(replay_cfg, backend=FakeKGGen(), cache_only=True)

    assert replay.extract("Alice sees Bob") == expected
    assert replay.cache_location(key) == ("read-through-1", historical / f"{key}.json")


def test_cache_only_zero_live_call_invariant_rejects_any_recorded_call():
    from run import _assert_cache_only_no_live_calls
    from src.extract import UsageLogger

    usage = UsageLogger(None)
    _assert_cache_only_no_live_calls(True, usage)
    usage.record_call("kg_context", "key", 0.1, cached=False)
    with pytest.raises(RuntimeError, match="recorded 1 live inference"):
        _assert_cache_only_no_live_calls(True, usage)


def test_verifier_cache_only_replays_and_runtime_change_is_a_miss(tmp_path):
    class StubVerifier(RelationVerifier):
        calls = 0

        def _call_llm(self, triple, evidence):
            self.calls += 1
            return "entailed"

    cfg = _cfg(tmp_path)
    triple = ("france", "has capital", "paris")
    context = "France has capital Paris."
    first = StubVerifier(cfg).verify(triple, context, None, matching_params={"tau_e": 0.9})
    assert first.verdict == "entailed"

    replay = RelationVerifier(cfg, cache_only=True)
    cached = replay.verify(triple, context, None, matching_params={"tau_e": 0.9})
    assert cached.verdict == "entailed" and cached.cache_hit is True

    changed_runtime = copy.deepcopy(cfg)
    changed_runtime.llm.runtime_fingerprint = "image-digest-b"
    with pytest.raises(CacheOnlyMissError, match="relation_verifier"):
        RelationVerifier(changed_runtime, cache_only=True).verify(
            triple, context, None, matching_params={"tau_e": 0.9}
        )


def test_verifier_reads_historical_verdict_after_corrupt_primary(tmp_path):
    cfg = _cfg(tmp_path)

    class StubVerifier(RelationVerifier):
        def _call_llm(self, triple, evidence):  # noqa: ARG002
            return "entailed"

    triple = ("France", "has capital", "Paris")
    context = "France has capital Paris."
    historical = tmp_path / "historical-verdicts"
    writer_cfg = copy.deepcopy(cfg)
    writer_cfg.relation_verifier.cache_dir = str(historical)
    StubVerifier(writer_cfg).verify(triple, context, None, matching_params={"tau_e": 0.9})

    reader_cfg = copy.deepcopy(cfg)
    reader_cfg.relation_verifier.cache_read_dirs = [str(historical)]
    # Use the real cached key rather than trusting a primary file to exist.
    cached_path = next(historical.glob("*.json"))
    (tmp_path / "verdicts").mkdir(exist_ok=True)
    (tmp_path / "verdicts" / cached_path.name).write_text("{not-json", encoding="utf-8")
    replay = RelationVerifier(reader_cfg, cache_only=True)
    result = replay.verify(triple, context, None, matching_params={"tau_e": 0.9})
    assert result.verdict == "entailed" and result.cache_hit is True


def test_sbert_uses_staged_path_on_cpu_without_hub_access(monkeypatch):
    captured = {}

    class FakeSentenceTransformer:
        def __init__(self, target, **kwargs):
            captured["target"] = target
            captured["kwargs"] = kwargs

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    embedder = SBERTEmbedder(
        "sentence-transformers/all-MiniLM-L6-v2",
        model_revision="revision-ignored-for-local-snapshot",
        model_path="/opt/hallu/models/all-MiniLM-L6-v2",
        device="cpu",
        local_files_only=True,
    )

    embedder._ensure()

    assert captured == {
        "target": "/opt/hallu/models/all-MiniLM-L6-v2",
        "kwargs": {"device": "cpu", "local_files_only": True},
    }


def test_cache_only_forces_sbert_cpu_and_offline_even_for_stale_config(tmp_path):
    from run import get_embedder

    cfg = _cfg(tmp_path)
    cfg.matching.embedding_device = "cuda"
    cfg.matching.local_files_only = False
    cfg.matching.embedding_model_path = "/opt/hallu/models/all-MiniLM-L6-v2"

    embedder = get_embedder(cfg, False, cache_only=True)

    assert embedder.device == "cpu"
    assert embedder.local_files_only is True


def test_evaluation_runtime_metadata_records_exact_model_and_embedding_identity(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    cfg.matching.embedding_model_path = "/opt/hallu/models/all-MiniLM-L6-v2"
    cfg.matching.embedding_device = "cpu"
    cfg.matching.local_files_only = True
    monkeypatch.setenv("PYTHONHASHSEED", "42")

    metadata = evaluation_runtime_metadata(cfg)

    assert metadata["llm_model_revision"] == "model-revision-a"
    assert metadata["runtime_fingerprint"] == "image-digest-a"
    assert metadata["structured_output_backend"] == "xgrammar"
    assert metadata["embedding_model_revision"] == "embedding-revision-a"
    assert metadata["embedding_model_path"] == "/opt/hallu/models/all-MiniLM-L6-v2"
    assert metadata["embedding_device"] == "cpu"
    assert metadata["embedding_local_files_only"] is True
    assert metadata["python_hash_seed"] == "42"


def test_summary_csv_flattens_runtime_package_versions():
    from src.evaluate import _flatten_summary

    flattened = _flatten_summary({
        "runtime": {"client_packages": {"kg-gen": "0.4.0", "dspy": "2.6.27"}},
        "overall_AUC_ci95": [0.25, 0.75],
    })

    assert flattened["runtime.client_packages.kg-gen"] == "0.4.0"
    assert flattened["runtime.client_packages.dspy"] == "2.6.27"
    assert flattened["overall_AUC_ci95_lo"] == 0.25
    assert flattened["overall_AUC_ci95_hi"] == 0.75


def test_clustering_retention_gate_reports_and_rejects_collapse(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    cfg.extraction.cluster_min_retention_ratio = 0.20

    raw = SimpleNamespace(
        entities={f"entity-{index}" for index in range(6)},
        edges={f"predicate-{index}" for index in range(6)},
        relations={(f"entity-{index}", f"predicate-{index}", "entity-0") for index in range(6)},
    )
    collapsed = SimpleNamespace(
        entities={"entity-0"},
        edges={"predicate-0"},
        relations={("entity-0", "predicate-0", "entity-0")},
        entity_clusters={"entity-0": set(raw.entities)},
        edge_clusters={"predicate-0": set(raw.edges)},
    )

    class Backend:
        seen_context = None

        @staticmethod
        def cluster(graph, context=""):
            assert graph is raw
            Backend.seen_context = context
            return collapsed

    extractor = KGExtractor(cfg, backend=Backend())
    with pytest.raises(ClusteringCollapseError, match="retention gate failed"):
        extractor._cluster_backend_graph(
            Backend(), raw, source_text="A source about six entities."
        )

    assert Backend.seen_context == (
        CLUSTER_EQUIVALENCE_POLICY
        + "\nSource evidence:\nA source about six entities."
    )

    output = capsys.readouterr().out
    assert "cluster:start entities=6 predicates=6 relations=6" in output
    assert "cluster:retention entities=1/6 (0.166667)" in output


def test_extractor_fails_fast_on_schema_error_but_retries_timeout(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.extraction.cluster = False
    cfg.llm.max_retries = 3

    class DeterministicFailure:
        calls = 0

        def generate(self, **kwargs):  # noqa: ARG002
            self.calls += 1
            raise StructuredOutputSchemaError("bare Relation root")

    deterministic = DeterministicFailure()
    with pytest.raises(StructuredOutputSchemaError, match="bare Relation"):
        KGExtractor(cfg, backend=deterministic).extract("Swiss chard")
    assert deterministic.calls == 1

    class TransientFailure:
        calls = 0

        def generate(self, **kwargs):  # noqa: ARG002
            self.calls += 1
            if self.calls < 3:
                raise TimeoutError("temporary localhost timeout")
            return SimpleNamespace(
                entities={"Swiss chard", "spinach"},
                relations={("Swiss chard", "similar to", "spinach")},
            )

    transient = TransientFailure()
    graph = KGExtractor(cfg, backend=transient).extract("Swiss chard after timeout")
    assert transient.calls == 3
    assert graph.relations == {("Swiss chard", "similar to", "spinach")}
