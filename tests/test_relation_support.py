"""Offline unit tests for RP_grounded / RP_entailed_cond / RP_support."""
import math
from types import SimpleNamespace

from src.extract import Graph
from src.matching import RefGraph
from src.metrics import ScoreResult, score_response
from src.verifier import EvidenceSpan, FakeRelationVerifier, RelationVerdict


def mcfg():
    return SimpleNamespace(
        entity_sim_threshold=0.9,
        relation_sim_threshold=0.75,
        allow_substring_match=True,
        direction_sensitive_edges=True,
        inverse_edge_match=False,
        min_substring_chars=2,
        stopwords=["is", "in", "on"],
    )


def test_entailed_text_edge_counts_without_strict_reference_edge():
    ref = RefGraph({"france", "paris"}, set(), mcfg(), embedder=None)
    answer = Graph({"France", "Paris"}, {("France", "has capital", "Paris")})
    verifier = FakeRelationVerifier({("france", "has capital", "paris"): "entailed"})

    result = score_response(
        answer, ref, Graph.empty(), Graph.empty(),
        context="France has capital Paris.", verifier=verifier,
    )

    assert result.RP == result.RP_strict == 0.0  # old baseline is unchanged
    assert result.RP_grounded == 1.0
    assert result.RP_entailed_cond == 1.0
    assert result.RP_support == 1.0
    assert result.relation_audits[0]["status"] == "entailed_from_text"
    assert result.relation_audits[0]["evidence"][0]["text"] == "France has capital Paris."


def test_grounded_unknown_is_not_support():
    ref = RefGraph({"france", "berlin"}, set(), mcfg(), embedder=None)
    answer = Graph({"France", "Berlin"}, {("France", "is in", "Berlin")})
    result = score_response(
        answer, ref, Graph.empty(), Graph.empty(),
        context="France and Berlin are mentioned.", verifier=FakeRelationVerifier(default="unknown"),
    )

    assert result.RP_grounded == 1.0
    assert result.RP_entailed_cond == 0.0
    assert result.RP_support == 0.0
    assert result.relation_audits[0]["status"] == "grounded_unknown"


def test_relation_fallback_is_explicitly_audited_as_unknown():
    class FallbackVerifier:
        def verify(self, canonical_triple, context, query, *, matching_params=None):  # noqa: ARG002
            return RelationVerdict(
                "unknown",
                (EvidenceSpan("context", 0, 0, 1, "France", 1),),
                protocol_fallback=True,
                fallback_reason="transient_exhausted",
            )

    ref = RefGraph({"france", "berlin"}, set(), mcfg(), embedder=None)
    answer = Graph({"France", "Berlin"}, {("France", "is in", "Berlin")})
    result = score_response(
        answer, ref, Graph.empty(), Graph.empty(), context="France and Berlin.", verifier=FallbackVerifier(),
    )
    audit = result.relation_audits[0]
    assert audit["status"] == "grounded_unknown"
    assert audit["verifier_protocol_fallback"] is True
    assert audit["verifier_fallback_reason"] == "transient_exhausted"


def test_contradicted_relation_is_not_support_even_with_grounded_endpoints():
    ref = RefGraph({"marie curie", "warsaw", "paris"}, set(), mcfg(), embedder=None)
    answer = Graph({"Marie Curie", "Paris"}, {("Marie Curie", "born in", "Paris")})
    verifier = FakeRelationVerifier({("marie curie", "born in", "paris"): "contradicted"})
    result = score_response(
        answer, ref, Graph.empty(), Graph.empty(),
        context="Marie Curie was born in Warsaw and later moved to Paris.", verifier=verifier,
    )

    assert result.RP_grounded == 1.0
    assert result.RP_support == 0.0
    assert result.relation_audits[0]["status"] == "contradicted"


def test_ungrounded_edge_does_not_receive_verifier_call():
    ref = RefGraph({"france"}, set(), mcfg(), embedder=None)
    answer = Graph({"France", "Berlin"}, {("France", "is in", "Berlin")})
    verifier = FakeRelationVerifier(default="entailed")
    result = score_response(answer, ref, Graph.empty(), Graph.empty(), context="France.", verifier=verifier)

    assert verifier.calls == 0
    assert result.RP_grounded == 0.0
    assert result.RP_entailed_cond is None
    assert result.RP_support == 0.0
    assert result.relation_audits[0]["status"] == "ungrounded_object"


def test_edge_less_graph_keeps_all_relation_metrics_undefined_and_cfi_reduces_to_eg():
    ref = RefGraph({"france"}, set(), mcfg(), embedder=None)
    result = score_response(Graph({"France"}, set()), ref, Graph.empty(), Graph.empty(), verifier=FakeRelationVerifier())

    assert result.RP is None and result.RP_support is None
    assert result.RP_grounded is None and result.RP_entailed_cond is None
    assert result.cfi_for_mode(0.1, "support") == 1.0
    assert result.h_for_mode(0.9, "support") == 0.0


def test_scoreresult_roundtrip_keeps_support_fields():
    score = ScoreResult(EG=0.8, RP=0.5, RP_defined=True, RP_strict=0.5, RP_strict_defined=True,
                        RP_grounded=1.0, RP_grounded_defined=True,
                        RP_entailed_cond=0.5, RP_entailed_cond_defined=True,
                        RP_support=0.5, RP_support_defined=True, support_verified=True)
    restored = ScoreResult.from_dict(score.to_dict())
    assert math.isclose(restored.RP_support or 0, 0.5)
    assert restored.support_verified is True
