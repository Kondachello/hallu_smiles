"""Unit tests for EG, RP, CFI, H including edge-aware (|E_a|=0) and unscorable (|V_a|=0)."""
import math
from types import SimpleNamespace

from src.extract import Graph
from src.matching import RefGraph
from src.metrics import ScoreResult, cfi, score_response


def mcfg(**over):
    base = dict(
        entity_sim_threshold=0.90,
        relation_sim_threshold=0.75,
        allow_substring_match=True,
        direction_sensitive_edges=True,
        inverse_edge_match=False,
        min_substring_chars=2,
        stopwords=[],
    )
    base.update(over)
    return SimpleNamespace(**base)


def ref():
    return RefGraph(
        ["france", "paris", "europe"],
        [("france", "has capital", "paris")],
        mcfg(),
        embedder=None,
    )


# --------------------------------------------------------------------------------------
# EG + RP + CFI with distinct EG and RP so alpha matters
# --------------------------------------------------------------------------------------
def test_eg_rp_cfi_and_audit_lists():
    g_c = Graph({"france", "paris", "europe"}, {("france", "has capital", "paris")})
    g_q = Graph.empty()
    g_a = Graph({"paris", "france", "berlin"}, {("france", "has capital", "paris")})
    res = score_response(g_a, ref(), g_c, g_q)

    assert math.isclose(res.EG, 2 / 3, rel_tol=1e-9)     # paris, france grounded; berlin not
    assert res.ungrounded_entities == ["berlin"]
    assert res.RP == 1.0 and res.RP_defined is True
    assert res.unsupported_relations == []

    alpha = 0.7
    expected_cfi = 0.7 * (2 / 3) + 0.3 * 1.0
    assert math.isclose(res.cfi(alpha), expected_cfi, rel_tol=1e-9)
    assert math.isclose(res.h(alpha), 1 - expected_cfi, rel_tol=1e-9)


def test_unsupported_relation_recorded():
    g_a = Graph({"berlin", "france"}, {("berlin", "is in", "france")})
    res = score_response(g_a, ref(), Graph.empty(), Graph.empty())
    # subject 'berlin' ungrounded -> relation unsupported
    assert res.RP == 0.0
    assert ["berlin", "is in", "france"] in res.unsupported_relations


# --------------------------------------------------------------------------------------
# Edge-aware convention: |E_a| = 0  -> RP undefined, CFI reduces to EG
# --------------------------------------------------------------------------------------
def test_empty_edges_cfi_reduces_to_eg():
    g_a = Graph({"paris"}, set())  # no relations
    res = score_response(g_a, ref(), Graph.empty(), Graph.empty())
    assert res.EG == 1.0
    assert res.RP is None and res.RP_defined is False
    # CFI == EG regardless of alpha
    assert res.cfi(0.1) == 1.0
    assert res.cfi(0.9) == 1.0
    assert res.h(0.5) == 0.0


def test_strict_only_edges_do_not_fabricate_support_score():
    g_a = Graph({"france", "paris"}, {("france", "has capital", "paris")})
    res = score_response(g_a, ref(), Graph.empty(), Graph.empty(), verifier=None)

    assert res.Ea == 1 and res.support_verified is False
    assert res.RP_support is None and res.RP_support_defined is False
    assert res.cfi_for_mode(0.7, "support") is None
    assert res.h_for_mode(0.7, "support") is None


def test_strict_report_suppresses_support_detector_even_without_edges():
    from run import build_rows

    result = score_response(
        Graph({"paris"}, set()), ref(), Graph.empty(), Graph.empty(), verifier=None
    )
    rows = build_rows([
        {
            "response_id": "r1", "source_id": "s1", "task": "QA",
            "gen_model": "fixture", "split": "test", "y": 0,
            "context_len": 1, "score": result.to_dict(),
        }
    ], 0.7, 0.7, "strict")

    assert rows[0]["H_strict"] == 0.0
    assert rows[0]["CFI_support"] is None
    assert rows[0]["H_support"] is None


# --------------------------------------------------------------------------------------
# Degenerate: |V_a| = 0 -> unscorable
# --------------------------------------------------------------------------------------
def test_empty_response_graph_unscorable():
    g_a = Graph(set(), set())
    res = score_response(g_a, ref(), Graph.empty(), Graph.empty())
    assert res.unscorable is True
    assert res.EG is None
    assert res.cfi(0.7) is None
    assert res.h(0.7) is None
    # imputation policy
    assert res.h(0.7, impute=0.5) == 0.5


def test_ref_empty_flag():
    empty_ref = RefGraph([], [], mcfg(), embedder=None)
    g_a = Graph({"paris"}, set())
    res = score_response(g_a, empty_ref, Graph.empty(), Graph.empty())
    assert res.ref_empty is True
    assert res.EG == 0.0  # nothing to ground against


# --------------------------------------------------------------------------------------
# pure cfi helper
# --------------------------------------------------------------------------------------
def test_cfi_helper_formula():
    assert math.isclose(cfi(0.5, 1.0, 0.7), 0.7 * 0.5 + 0.3 * 1.0)
    assert cfi(0.8, None, 0.7) == 0.8  # RP undefined -> EG


def test_scoreresult_roundtrip():
    r = ScoreResult(Va=3, Ea=1, EG=0.66, RP=1.0, RP_defined=True,
                    ungrounded_entities=["berlin"])
    r2 = ScoreResult.from_dict(r.to_dict())
    assert r2.EG == r.EG and r2.ungrounded_entities == ["berlin"] and r2.RP_defined
