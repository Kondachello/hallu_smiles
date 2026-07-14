from __future__ import annotations

import json
from types import SimpleNamespace

from src.extract import Graph, KGExtractor
from src.micro_qa_demo import graph_stats, mermaid_graph, write_obsidian_artifacts


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
