"""Offline evidence ranking and disk-cache tests for the relation verifier."""
from types import SimpleNamespace

from src.verifier import RelationVerifier, select_evidence


class StubVerifier(RelationVerifier):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.backend_calls = 0

    def _call_llm(self, triple, evidence):
        self.backend_calls += 1
        return "entailed"


def _cfg(tmp_path):
    return SimpleNamespace(
        llm=SimpleNamespace(
            model="openai/test", api_base=None, temperature=0.0,
            max_retries=1, retry_backoff_base_s=0.0,
        ),
        matching=SimpleNamespace(stopwords=["is", "in", "on"]),
        relation_verifier=SimpleNamespace(
            cache_dir=str(tmp_path / "verdicts"), prompt_version="test-v1", max_evidence_sentences=4,
        ),
    )


def test_evidence_ranking_prefers_both_endpoints_and_predicate():
    evidence = select_evidence(
        "Marie Curie was born in Warsaw. She moved to Paris.",
        "Where was Marie Curie born?",
        ("Marie Curie", "born in", "Warsaw"),
        stopwords=["in"],
    )
    assert evidence[0].text == "Marie Curie was born in Warsaw."
    assert evidence[0].rank == 5
    assert evidence[0].source == "context"


def test_verifier_cache_reuses_canonical_triple_and_evidence(tmp_path):
    verifier = StubVerifier(_cfg(tmp_path))
    triple = ("france", "has capital", "paris")
    first = verifier.verify(triple, "France has capital Paris.", None)
    second = verifier.verify(triple, "France has capital Paris.", None)

    assert first.verdict == second.verdict == "entailed"
    assert first.cache_hit is False and second.cache_hit is True
    assert verifier.backend_calls == 1
