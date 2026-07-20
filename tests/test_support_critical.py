"""Offline contracts for the claim-level support-critical detector."""
from types import SimpleNamespace

import pytest

from src.critical import (
    AtomicClaim,
    AtomicClaimExtractor,
    CriticalCompletionTruncatedError,
    CriticalClaimVerifier,
    StructuredOutputParseError,
    FakeCriticalClaimPipeline,
    _validate_claims,
    merge_claims,
    select_claim_evidence,
)
from src.cache import CacheOnlyMissError
from src.extract import Graph
from src.matching import DictEmbedder, RefGraph
from src.metrics import ScoreResult, score_response
from src.tune import critical_cv, critical_h_array


def _matching():
    return SimpleNamespace(
        entity_sim_threshold=0.9,
        relation_sim_threshold=0.75,
        allow_substring_match=True,
        direction_sensitive_edges=True,
        inverse_edge_match=False,
        min_substring_chars=2,
        stopwords=["is", "in", "on"],
    )


def _cfg(tmp_path):
    return SimpleNamespace(
        llm=SimpleNamespace(
            model="openai/test", api_base=None, temperature=0.0,
            model_revision="model-a", runtime_fingerprint="runtime-a", max_retries=1,
            retry_backoff_base_s=0.0, retry_backoff_max_s=0.0, request_timeout_s=10,
            structured_output_transport="none", structured_output_backend="vertex",
        ),
        matching=SimpleNamespace(
            stopwords=["is", "in", "on"], embedding_model="fixture",
            embedding_model_revision="fixture-a",
        ),
        support_critical=SimpleNamespace(
            claim_extractor=SimpleNamespace(
                cache_dir=str(tmp_path / "claims"), prompt_version="claims-v1", max_tokens=64,
            ),
            coverage_reviewer=SimpleNamespace(
                cache_dir=str(tmp_path / "coverage"), prompt_version="coverage-v1", max_tokens=64,
            ),
            claim_verifier=SimpleNamespace(
                cache_dir=str(tmp_path / "verdicts"), prompt_version="verdict-v1", max_tokens=64,
                max_evidence_sentences=8,
            ),
        ),
    )


def _critical_score() -> ScoreResult:
    return ScoreResult(
        EG=0.9,
        Ea=2,
        relation_audits=[{"verdict": "entailed"}, {"verdict": "unsupported"}],
        critical={
            "protocol": "support-critical-v1",
            "claim_audits": [
                {"verdict": "entailed"},
                {"verdict": "entailed"},
                {"verdict": "unsupported"},
            ],
        },
    )


def test_one_unsupported_claim_is_hard_top_k_signal():
    score = _critical_score()
    assert score.critical_relation_rp(0.0) == 0.5
    assert score.critical_claim_topk(1, 0.0) == 1.0
    # H_graph = 1 - (.7*.9 + .3*.5) = .22; beta=.5 adds the hidden claim.
    assert score.critical_h(0.7, 0.5, 1, 0.0) == pytest.approx(0.61)


def test_unknown_penalty_is_train_tunable_but_unsupported_is_always_hard():
    score = ScoreResult(
        EG=1.0, Ea=1, relation_audits=[{"verdict": "unknown"}],
        critical={"claim_audits": [{"verdict": "unsupported"}]},
    )
    assert score.critical_relation_rp(0.0) == 1.0
    assert score.critical_relation_rp(0.25) == 0.75
    assert score.critical_claim_topk(1, 0.0) == 1.0
    assert score.critical_claim_topk(1, 0.25) == 1.0


def test_graph_empty_claim_layer_is_diagnostic_but_remains_unscorable():
    score = ScoreResult(
        unscorable=True,
        critical={"claim_audits": [{"verdict": "contradicted"}]},
    )
    assert score.critical_h(0.7, 0.5, 1, 0.0) == 1.0
    H, mask = critical_h_array([score], 0.7, 0.5, 1, 0.0)
    assert not mask[0]
    H, mask = critical_h_array([score], 0.7, 0.5, 1, 0.0, include_unscorable=True)
    assert mask[0] and H[0] == 1.0


def test_strict_serialization_does_not_grow_a_critical_field():
    assert "critical" not in ScoreResult(EG=1.0).to_dict()
    assert ScoreResult.from_dict(ScoreResult(EG=1.0).to_dict()).critical is None


def test_claim_offsets_must_be_exact_and_sources_merge():
    response = "Paris has 2 museums."
    claims = _validate_claims(
        {"claims": [{"text": "Paris has 2 museums", "start": 0, "end": 19}]},
        response,
        "atomic",
    )
    assert claims[0].sources == ("atomic",)
    merged = merge_claims(claims, [AtomicClaim(claims[0].text, 0, 19, ("global_review",))])
    assert merged[0].sources == ("atomic", "global_review")
    with pytest.raises(Exception):
        _validate_claims(
            {"claims": [{"text": "Paris has 3 museums", "start": 0, "end": 19}]}, response, "atomic"
        )


def test_evidence_retrieval_combines_lexical_and_local_embeddings():
    evidence = select_claim_evidence(
        "Paris has two museums. Berlin has one museum.", None, "Paris has two museums.",
        max_sentences=2, embedder=DictEmbedder(),
    )
    assert evidence[0].text == "Paris has two museums."


def test_critical_pipeline_records_claim_audit_even_when_answer_graph_is_empty():
    ref = RefGraph(set(), set(), _matching(), embedder=None)
    pipeline = FakeCriticalClaimPipeline(
        [AtomicClaim("Paris has 2 museums", 0, 19, ("atomic",))],
        {"Paris has 2 museums": "unsupported"},
    )
    result = score_response(
        Graph.empty(), ref, Graph.empty(), Graph.empty(), answer_text="Paris has 2 museums.",
        critical_pipeline=pipeline,
    )
    assert result.unscorable is True
    assert result.critical["claim_audits"][0]["verdict"] == "unsupported"


def test_critical_cv_uses_only_supplied_train_scores():
    scores = [_critical_score(), _critical_score(), _critical_score(), _critical_score()]
    scores[0].critical["claim_audits"] = [{"verdict": "entailed"}]
    scores[1].critical["claim_audits"] = [{"verdict": "unsupported"}]
    scores[2].critical["claim_audits"] = [{"verdict": "entailed"}]
    scores[3].critical["claim_audits"] = [{"verdict": "unsupported"}]
    rows = critical_cv(
        scores, [0, 1, 0, 1], alpha_grid=[0.7], beta_grid=[0.5],
        top_k_grid=[1], unknown_risk_grid=[0.0], folds=2, seed=42,
    )
    assert len(rows) == 1
    assert rows[0]["top_k"] == 1


def test_claim_and_verdict_caches_replay_without_live_calls(tmp_path):
    cfg = _cfg(tmp_path)

    class StubClaims(AtomicClaimExtractor):
        calls = 0

        def _retry_json(self, messages, schema, name):  # noqa: ARG002
            self.calls += 1
            return {"claims": [{"text": "Paris has two museums", "start": 0, "end": 21}]}

    response = "Paris has two museums."
    writer = StubClaims(cfg)
    assert writer.extract(response)[0].text == "Paris has two museums"
    assert writer.calls == 1
    replay = AtomicClaimExtractor(cfg, cache_only=True)
    assert replay.extract(response)[0].start == 0
    with pytest.raises(CacheOnlyMissError):
        replay.extract("A different response.")

    class StubVerifier(CriticalClaimVerifier):
        calls = 0

        def _retry_json(self, messages, schema, name):  # noqa: ARG002
            self.calls += 1
            return {"verdict": "entailed"}

    verifier = StubVerifier(cfg, embedder=DictEmbedder())
    first = verifier.verify_claim("Paris has two museums.", "Paris has two museums.", None)
    assert first.verdict == "entailed" and verifier.calls == 1
    replay_verifier = CriticalClaimVerifier(cfg, cache_only=True, embedder=DictEmbedder())
    second = replay_verifier.verify_claim("Paris has two museums.", "Paris has two museums.", None)
    assert second.verdict == "entailed" and second.cache_hit is True


def test_critical_truncation_escalates_only_the_transport_budget(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.llm.max_retries = 3
    cfg.support_critical.claim_verifier.max_tokens = 64
    cfg.support_critical.claim_verifier.max_tokens_ceiling = 256

    class TruncatesOnce(CriticalClaimVerifier):
        calls: list[int] = []

        def _call_json(self, messages, schema, name, *, max_tokens):  # noqa: ARG002
            self.calls.append(max_tokens)
            if len(self.calls) == 1:
                raise CriticalCompletionTruncatedError("simulated length")
            return {"verdict": "entailed"}

    verifier = TruncatesOnce(cfg, embedder=DictEmbedder())
    assert verifier._retry_json([], {}, "test") == {"verdict": "entailed"}
    assert verifier.calls == [64, 128]


def test_critical_truncation_fails_at_the_configured_ceiling(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.llm.max_retries = 4
    cfg.support_critical.claim_verifier.max_tokens = 64
    cfg.support_critical.claim_verifier.max_tokens_ceiling = 128

    class AlwaysTruncated(CriticalClaimVerifier):
        calls: list[int] = []

        def _call_json(self, messages, schema, name, *, max_tokens):  # noqa: ARG002
            self.calls.append(max_tokens)
            raise CriticalCompletionTruncatedError("simulated length")

    verifier = AlwaysTruncated(cfg, embedder=DictEmbedder())
    with pytest.raises(StructuredOutputParseError, match="remained truncated"):
        verifier._retry_json([], {}, "test")
    assert verifier.calls == [64, 128]


def test_critical_verdict_cache_reads_previous_namespace_without_writing_it(tmp_path):
    old_cache = tmp_path / "prior" / "critical_verdicts"
    cfg = _cfg(tmp_path)
    cfg.support_critical.claim_verifier.cache_dir = str(old_cache)

    class StubVerifier(CriticalClaimVerifier):
        def _retry_json(self, messages, schema, name):  # noqa: ARG002
            return {"verdict": "entailed"}

    claim = "Paris has two museums."
    writer = StubVerifier(cfg, embedder=DictEmbedder())
    assert writer.verify_claim(claim, claim, None).cache_hit is False
    written = list(old_cache.glob("*.json"))
    assert len(written) == 1

    fresh_cache = tmp_path / "active" / "critical_verdicts"
    cfg.support_critical.claim_verifier.cache_dir = str(fresh_cache)
    cfg.support_critical.claim_verifier.cache_read_dirs = [str(old_cache)]
    replay = CriticalClaimVerifier(cfg, cache_only=True, embedder=DictEmbedder())
    result = replay.verify_claim(claim, claim, None)
    assert result.verdict == "entailed" and result.cache_hit is True
    assert not fresh_cache.exists()
