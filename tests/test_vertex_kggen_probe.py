"""Offline contracts for the Vertex KGGen synthetic probe."""
from scripts.check_vertex_kggen_probe import _has_ada_lovelace_anchor, _structural_failure
from src.extract import Graph


def test_vertex_probe_accepts_direct_and_babbage_mediated_ada_fact():
    assert _has_ada_lovelace_anchor(
        {("Ada Lovelace", "wrote notes about", "Analytical Engine")}
    )
    assert _has_ada_lovelace_anchor(
        {("Ada Lovelace", "wrote notes about", "Charles Babbage")}
    )
    assert not _has_ada_lovelace_anchor(
        {("Charles Babbage", "designed", "Analytical Engine")}
    )


def test_vertex_probe_requires_a_well_formed_nontrivial_graph_not_one_interpretation():
    graph = Graph(
        entities={"Ada Lovelace", "Charles Babbage", "Marie Curie", "Warsaw"},
        relations={
            ("Ada Lovelace", "wrote notes about", "Charles Babbage"),
            ("Marie Curie", "was born in", "Warsaw"),
        },
    )
    assert _structural_failure(graph) is None

    dangling = Graph(
        entities={"Ada Lovelace", "Marie Curie", "Warsaw", "X"},
        relations={
            ("Ada Lovelace", "wrote notes about", "Charles Babbage"),
            ("Marie Curie", "was born in", "Warsaw"),
        },
    )
    assert "missing endpoint" in (_structural_failure(dangling) or "")
