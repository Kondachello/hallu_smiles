from __future__ import annotations

import json
from types import SimpleNamespace

from src.extract import Graph, KGExtractor
from src.matching import DictEmbedder
from src.micro_qa_demo import (
    audit_micro_graphs,
    graph_stats,
    list_qa_candidates,
    mermaid_graph,
    write_obsidian_artifacts,
)


def test_graph_stats_excludes_self_loops_from_density():
    graph = Graph(
        {"A", "B"},
        {("A", "supports", "B"), ("A", "mentions", "A")},
    )

    assert graph_stats(graph) == {
        "nodes": 2,
        "edges": 2,
        "self_loops": 1,
        "average_out_degree": 1.0,
        "directed_density": 0.5,
    }


def test_mermaid_escapes_labels_and_emits_all_relation_nodes():
    graph = Graph({"A [one]"}, {("A [one]", "rel|x", 'B "two"')})

    rendered = mermaid_graph("G_A", graph)

    assert "A &#91;one&#93;" in rendered
    assert "rel&#124;x" in rendered
    assert "B &quot;two&quot;" in rendered
    assert rendered.count("[") >= 2


def test_obsidian_artifacts_have_links_and_match_stats(tmp_path):
    gc = Graph({"Paris", "France"}, {("Paris", "is capital of", "France")})
    gq = Graph({"Paris"}, set())
    ga = Graph({"Paris", "France"}, {("Paris", "governs", "France")})
    graphs = {"G_C": gc, "G_Q": gq, "G_A": ga, "G_ref": gc.union(gq)}
    stats = {name: graph_stats(graph) for name, graph in graphs.items()}
    metadata = {
        "model": "example/model",
        "selected_instance": {"source_id": "s1", "response_id": "r1", "split": "test", "label": 0},
        "input_lengths_chars": {"context": 10, "query": 5, "response": 8},
    }

    write_obsidian_artifacts(tmp_path, graphs, stats, metadata)

    overview = (tmp_path / "overview.md").read_text(encoding="utf-8")
    assert "```mermaid" in overview
    assert "| G_ref | 2 | 1 |" in overview
    index = (tmp_path / "entity_index.md").read_text(encoding="utf-8")
    links = [line.split("[[entities/", 1)[1].split("|", 1)[0] for line in index.splitlines() if "[[entities/" in line]
    assert links
    for stem in links:
        assert (tmp_path / "entities" / f"{stem}.md").exists()
    entity_pages = [
        (tmp_path / "entities" / f"{stem}.md").read_text(encoding="utf-8") for stem in links
    ]
    assert any("[[entities/" in page for page in entity_pages)

    # The JSON hand-off format retains all graph facts without any Mermaid escaping.
    sample = {"graphs": {name: graph.to_dict() for name, graph in graphs.items()}, "statistics": stats}
    round_trip = json.loads(json.dumps(sample))
    assert round_trip["graphs"]["G_A"]["relations"] == [["Paris", "governs", "France"]]


def test_kg_extractor_keeps_kggen_clustering_and_native_chunking(tmp_path):
    class Backend:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                entities={"source", "target"},
                relations={("source", "supports", "target")},
            )

    cfg = SimpleNamespace(
        llm=SimpleNamespace(
            model="openrouter/nvidia/nemotron-nano-9b-v2:free",
            temperature=0.0,
            max_retries=1,
            retry_backoff_base_s=0.0,
        ),
        extraction=SimpleNamespace(cluster=True, context_chunk_chars=8),
        cache_dir=str(tmp_path / "cache"),
    )
    backend = Backend()
    extractor = KGExtractor(cfg, backend=backend)

    graph = extractor._call_backend("a context that exceeds eight characters")

    assert graph == Graph({"source", "target"}, {("source", "supports", "target")})
    assert backend.calls == [
        {
            "input_data": "a context that exceeds eight characters",
            "cluster": True,
            "chunk_size": 8,
        }
    ]


def test_kg_extractor_can_schedule_kggen_chunks_serially_for_local_vllm(tmp_path, monkeypatch):
    class RawGraph:
        def __init__(self, value):
            self.entities = {value}
            self.relations = {(value, "rel", value)}

    class Backend:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(("generate", kwargs))
            return RawGraph(kwargs["input_data"])

        def aggregate(self, graphs):
            self.calls.append(("aggregate", [next(iter(graph.entities)) for graph in graphs]))
            return RawGraph("aggregate")

        def cluster(self, graph):
            self.calls.append(("cluster", next(iter(graph.entities))))
            return RawGraph("cluster")

    cfg = SimpleNamespace(
        llm=SimpleNamespace(model="test", temperature=0.0, max_retries=1, retry_backoff_base_s=0.0),
        extraction=SimpleNamespace(cluster=True, context_chunk_chars=8, serial_chunking=True),
        cache_dir=str(tmp_path / "cache"),
    )
    backend = Backend()
    extractor = KGExtractor(cfg, backend=backend)
    monkeypatch.setattr(extractor, "_split_text", lambda text, size: ["first", "second"])
    graph = extractor._call_backend("a context that exceeds eight characters")

    assert graph.entities == {"cluster"}
    assert backend.calls == [
        ("generate", {"input_data": "first", "cluster": False}),
        ("generate", {"input_data": "second", "cluster": False}),
        ("aggregate", ["first", "second"]),
        ("cluster", "aggregate"),
    ]


def test_micro_audit_uses_the_normal_eg_rp_audit_contract():
    cfg = SimpleNamespace(
        matching=SimpleNamespace(
            entity_sim_threshold=0.99,
            relation_sim_threshold=0.99,
            allow_substring_match=True,
            direction_sensitive_edges=True,
            inverse_edge_match=False,
            min_substring_chars=2,
            stopwords=[],
            embedding_model="unused-in-test",
        )
    )
    instance = SimpleNamespace(
        response_id="r1", source_id="s1", task="QA", gen_model="model", split="train",
        y=1, gt_span_types=["Type"],
    )
    gc = Graph({"Paris", "France"}, {("Paris", "is capital of", "France")})
    gq = Graph(set(), set())
    ga = Graph({"Paris", "Berlin"}, {("Berlin", "is capital of", "France")})

    audit = audit_micro_graphs(
        cfg,
        instance,
        {"G_C": gc, "G_Q": gq, "G_A": ga, "G_ref": gc.union(gq)},
        alpha=0.7,
        embedder=DictEmbedder(),
    )

    assert audit["EG"] == 0.5
    assert audit["RP"] == 0.0
    assert audit["H"] == 0.65
    assert audit["ungrounded_entities"] == ["berlin"]
    assert audit["unsupported_relations"] == [["berlin", "is capital of", "france"]]


def test_candidate_listing_filters_and_balances_all_three_inputs():
    def instance(rid, c, q, a):
        return SimpleNamespace(
            response_id=rid, source_id=f"s{rid}", task="QA", context="c" * c,
            query="q" * q, response="a" * a,
        )

    selected = list_qa_candidates(
        [
            instance("context-only", 3000, 10, 1000),
            instance("balanced", 900, 100, 1000),
            instance("short-answer", 900, 100, 200),
        ],
        min_context_chars=700,
        min_query_chars=50,
        min_response_chars=500,
    )

    assert [item.response_id for item in selected] == ["balanced"]
