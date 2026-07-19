"""Offline semantic contracts for the Vertex KGGen synthetic probe."""
from scripts.check_vertex_kggen_probe import _has_ada_lovelace_anchor


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
