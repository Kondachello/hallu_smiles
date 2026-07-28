"""Local unit tests for the type-aware vertex metric (no torch, cache, or gateway)."""
from src.typed_matching import (
    TypedRefGraph,
    typed_cfi,
    typed_entity_grounding,
)


def test_exact_shared_type_grounds_vertex():
    ref = TypedRefGraph({"North Bank": ["commercial bank"], "Delta Ltd": ["company"]})
    # answer vertex whose assigned type equals a reference type (case-insensitive)
    m = ref.match_entity_types(["Commercial Bank"])
    assert m.matched and m.method == "type_exact" and m.ref == "North Bank"


def test_no_shared_type_is_ungrounded():
    ref = TypedRefGraph({"North Bank": ["commercial bank"]})
    assert ref.match_entity_types(["planet"]).matched is False
    # a vertex with no assigned type carries no vertex-level signal
    assert ref.match_entity_types([]).matched is False


def test_substring_relaxation_is_opt_in():
    ref_strict = TypedRefGraph({"North Bank": ["commercial bank"]})
    ref_relaxed = TypedRefGraph({"North Bank": ["commercial bank"]}, allow_substring=True)
    assert ref_strict.match_entity_types(["bank"]).matched is False
    hit = ref_relaxed.match_entity_types(["bank"])
    assert hit.matched and hit.method == "type_substring"


def test_eg_type_fraction_and_audit():
    ref = TypedRefGraph(
        {"North Bank": ["financial institution"], "Delta Ltd": ["company"]}
    )
    answer = {
        "North Bank": ["financial institution"],  # grounded
        "the loan": ["debt instrument"],            # ungrounded
        "Delta Ltd": ["company"],                   # grounded
    }
    out = typed_entity_grounding(answer, ref)
    assert out["total_vertices"] == 3
    assert out["grounded_vertices"] == 2
    assert abs(out["eg"] - 2 / 3) < 1e-9
    assert out["ungrounded"] == ["the loan"]


def test_typed_cfi_blends_eg_and_rp():
    # CFI = alpha*EG + (1-alpha)*RP
    assert abs(typed_cfi(1.0, 0.0, 0.5) - 0.5) < 1e-9
    assert abs(typed_cfi(0.6, 0.4, 0.75) - (0.75 * 0.6 + 0.25 * 0.4)) < 1e-9
