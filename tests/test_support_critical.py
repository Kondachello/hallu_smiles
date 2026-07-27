"""Offline contracts for the claim-level support-critical detector."""
from types import SimpleNamespace

import pytest

from src.critical import (
    AtomicClaim,
    AtomicClaimExtractor,
    CriticalCompletionTruncatedError,
    CriticalClaimVerifier,
    CriticalOutputLimitError,
    FullContextReviewer,
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


def test_claim_offsets_are_repaired_only_for_a_unique_verbatim_fragment_and_sources_merge():
    response = "Paris has 2 museums."
    claims = _validate_claims(
        # The model's end index is deliberately wrong. The claim text is a
        # unique answer substring, so local code can recover the exact span
        # without altering the factual fragment.
        {"claims": [{"text": "Paris has 2 museums", "start": 0, "end": 18}]},
        response,
        "atomic",
    )
    assert claims[0].sources == ("atomic",) and claims[0].end == 19
    merged = merge_claims(claims, [AtomicClaim(claims[0].text, 0, 19, ("global_review",))])
    assert merged[0].sources == ("atomic", "global_review")
    with pytest.raises(Exception):
        _validate_claims(
            {"claims": [{"text": "Paris has 3 museums", "start": 0, "end": 19}]}, response, "atomic"
        )


def test_repaired_offset_never_guesses_between_repeated_answer_substrings():
    response = "Paris is old. Paris is old."
    with pytest.raises(StructuredOutputParseError, match="one exact response substring"):
        _validate_claims(
            {"claims": [{"text": "Paris is old", "start": 1, "end": 13}]}, response, "atomic"
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


def test_malformed_claim_payload_gets_bounded_schema_retry_after_offset_recovery(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.support_critical.claim_extractor.max_protocol_retries = 2

    class MalformedThenValid(AtomicClaimExtractor):
        calls = 0

        def _retry_json(self, messages, schema, name):  # noqa: ARG002
            self.calls += 1
            if self.calls == 1:
                return {"claims": [{"text": "a claim absent from the answer", "start": 0, "end": 1}]}
            return {"claims": [{"text": "Paris has two museums", "start": 2, "end": 3}]}

    extractor = MalformedThenValid(cfg)
    extracted = extractor.extract("Paris has two museums.")
    assert extracted[0].start == 0 and extracted[0].end == 21
    assert extracted[0].text == "Paris has two museums"
    assert extractor.calls == 2


def test_long_atomic_claims_are_chunked_with_absolute_offsets_and_cache_only_replay(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.support_critical.claim_extractor.chunk_chars = 20
    cfg.support_critical.claim_extractor.min_chunk_chars = 5
    response = "Alpha has one. Beta has two. Gamma has three."

    class SegmentClaims(AtomicClaimExtractor):
        calls = 0

        def _retry_json(self, messages, schema, name):  # noqa: ARG002
            self.calls += 1
            segment = messages[-1]["content"].split("Answer segment:\n", 1)[1]
            return {"claims": [{"text": segment, "start": 0, "end": len(segment)}]}

    writer = SegmentClaims(cfg)
    claims = writer.extract(response)
    assert writer.calls > 1
    assert [(claim.start, claim.end) for claim in claims] == [
        (0, claims[0].end),
        (claims[0].end, claims[1].end),
        (claims[1].end, len(response)),
    ]
    assert "".join(response[claim.start:claim.end] for claim in claims) == response

    replay = AtomicClaimExtractor(cfg, cache_only=True)
    assert replay.extract(response) == claims


def test_output_ceiling_bisects_claim_segment_and_leaves_replay_marker(tmp_path):
    cfg = _cfg(tmp_path)
    # The initial request is deliberately one segment; simulated provider
    # truncation must recursively split it rather than fail the Job.
    cfg.support_critical.claim_extractor.chunk_chars = 10_000
    cfg.support_critical.claim_extractor.min_chunk_chars = 5
    response = "Alpha one. Beta two. Gamma three. Delta four."

    class TruncateDenseSegment(AtomicClaimExtractor):
        calls = 0

        def _retry_json(self, messages, schema, name):  # noqa: ARG002
            self.calls += 1
            marker = "Answer segment:\n" if "Answer segment:\n" in messages[-1]["content"] else "Answer:\n"
            segment = messages[-1]["content"].split(marker, 1)[1]
            if len(segment) > 10:
                raise CriticalOutputLimitError("simulated provider output ceiling")
            return {"claims": [{"text": segment, "start": 0, "end": len(segment)}]}

    writer = TruncateDenseSegment(cfg)
    claims = writer.extract(response)
    assert writer.calls > 2
    assert "".join(response[claim.start:claim.end] for claim in claims) == response
    # The full-answer replay marker prevents the exact same dynamic fallback
    # from being rediscovered during a zero-HTTP cache-only run.
    assert AtomicClaimExtractor(cfg, cache_only=True).extract(response) == claims


def test_long_coverage_review_is_chunked_and_replays_without_calls(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.support_critical.coverage_reviewer.chunk_chars = 18
    cfg.support_critical.coverage_reviewer.min_chunk_chars = 5
    response = "Alpha is red. Beta is blue. Gamma is green."

    class SegmentReview(FullContextReviewer):
        calls = 0

        def _retry_json(self, messages, schema, name):  # noqa: ARG002
            self.calls += 1
            segment = messages[-1]["content"].split("Answer segment:\n", 1)[1].split(
                "\n\nAlready extracted", 1
            )[0]
            return {"claims": [{"text": segment, "start": 0, "end": len(segment)}]}

    writer = SegmentReview(cfg)
    claims = writer.review(response, "Alpha is red.", None, [])
    assert writer.calls > 1
    assert "".join(response[claim.start:claim.end] for claim in claims) == response
    assert FullContextReviewer(cfg, cache_only=True).review(response, "Alpha is red.", None, []) == claims


def test_invalid_claim_offsets_fall_back_to_exact_sentences_and_replay(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.support_critical.claim_extractor.max_protocol_retries = 2
    response = "Alpha has one. Beta has two."

    class AlwaysParaphrases(AtomicClaimExtractor):
        calls = 0

        def _retry_json(self, messages, schema, name):  # noqa: ARG002
            self.calls += 1
            return {"claims": [{"text": "Alpha owns one", "start": 0, "end": 14}]}

    writer = AlwaysParaphrases(cfg)
    claims = writer.extract(response)
    assert writer.calls == 2
    assert [claim.text for claim in claims] == ["Alpha has one.", "Beta has two."]
    assert all(claim.sources == ("atomic_fallback_sentence",) for claim in claims)
    assert AtomicClaimExtractor(cfg, cache_only=True).extract(response) == claims


def test_invalid_coverage_offsets_fall_back_to_exact_sentences_and_replay(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.support_critical.coverage_reviewer.max_protocol_retries = 2
    response = "Alpha has one. Beta has two."

    class AlwaysParaphrases(FullContextReviewer):
        calls = 0

        def _retry_json(self, messages, schema, name):  # noqa: ARG002
            self.calls += 1
            return {"claims": [{"text": "Beta owns two", "start": 0, "end": 13}]}

    writer = AlwaysParaphrases(cfg)
    claims = writer.review(response, "", None, [])
    assert writer.calls == 2
    assert [claim.text for claim in claims] == ["Alpha has one.", "Beta has two."]
    assert all(claim.sources == ("global_review_fallback_sentence",) for claim in claims)
    assert FullContextReviewer(cfg, cache_only=True).review(response, "", None, []) == claims


def test_scalar_verifier_schema_failure_is_cached_as_conservative_unknown(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.support_critical.claim_verifier.max_protocol_retries = 2

    class InvalidVerdict(CriticalClaimVerifier):
        calls = 0

        def _retry_json(self, messages, schema, name):  # noqa: ARG002
            self.calls += 1
            return {"verdict": "maybe"}

    writer = InvalidVerdict(cfg, embedder=DictEmbedder())
    first = writer.verify_claim("Alpha has one.", "Alpha has one.", None)
    assert first.verdict == "unknown" and first.protocol_fallback is True and writer.calls == 2
    replay = CriticalClaimVerifier(cfg, cache_only=True, embedder=DictEmbedder()).verify_claim(
        "Alpha has one.", "Alpha has one.", None
    )
    assert replay.verdict == "unknown" and replay.cache_hit and replay.protocol_fallback


def test_scalar_verifier_transient_exhaustion_is_cached_as_conservative_unknown(tmp_path):
    cfg = _cfg(tmp_path)

    class ExhaustedTimeout(CriticalClaimVerifier):
        calls = 0

        def _retry_validated_json(self, messages, schema, name, validator):  # noqa: ARG002
            self.calls += 1
            raise TimeoutError("gateway read timeout")

    writer = ExhaustedTimeout(cfg, embedder=DictEmbedder())
    first = writer.verify_claim("Alpha has one.", "Alpha has one.", None)
    assert first.verdict == "unknown"
    assert first.protocol_fallback and first.fallback_reason == "transient_exhausted"
    assert writer.calls == 1
    replay = CriticalClaimVerifier(cfg, cache_only=True, embedder=DictEmbedder()).verify_claim(
        "Alpha has one.", "Alpha has one.", None
    )
    assert replay.verdict == "unknown" and replay.cache_hit and replay.protocol_fallback
    assert replay.fallback_reason == "transient_exhausted"


def test_transient_claim_extraction_and_coverage_preserve_sentence_candidates(tmp_path):
    cfg = _cfg(tmp_path)
    response = "Alpha has one. Beta has two."

    class TimeoutAtomic(AtomicClaimExtractor):
        def _retry_validated_json(self, messages, schema, name, validator):  # noqa: ARG002
            raise TimeoutError("gateway read timeout")

    extracted = TimeoutAtomic(cfg).extract(response)
    assert [claim.text for claim in extracted] == ["Alpha has one.", "Beta has two."]
    assert all(claim.sources == ("atomic_fallback_sentence",) for claim in extracted)
    assert AtomicClaimExtractor(cfg, cache_only=True).extract(response) == extracted

    class TimeoutCoverage(FullContextReviewer):
        def _retry_validated_json(self, messages, schema, name, validator):  # noqa: ARG002
            raise TimeoutError("gateway read timeout")

    reviewed = TimeoutCoverage(cfg).review(response, "", None, [])
    assert [claim.text for claim in reviewed] == ["Alpha has one.", "Beta has two."]
    assert all(claim.sources == ("global_review_fallback_sentence",) for claim in reviewed)
    assert FullContextReviewer(cfg, cache_only=True).review(response, "", None, []) == reviewed


def test_recursive_claim_and_coverage_rate_limit_context_preserves_replayable_candidates(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.support_critical.claim_extractor.chunk_chars = 5
    cfg.support_critical.coverage_reviewer.chunk_chars = 5
    response = "Alpha has one. Beta has two."

    class RateLimitError(Exception):
        pass

    class InterruptedAtomic(AtomicClaimExtractor):
        def _extract_chunk(self, chunk):  # noqa: ARG002
            try:
                raise CriticalOutputLimitError("previous segment hit output ceiling")
            except CriticalOutputLimitError:
                raise RateLimitError("Vertex capacity is temporarily exhausted")

    extracted = InterruptedAtomic(cfg).extract(response)
    assert extracted
    assert all(claim.sources == ("atomic_fallback_sentence",) for claim in extracted)
    assert AtomicClaimExtractor(cfg, cache_only=True).extract(response) == extracted

    class InterruptedCoverage(FullContextReviewer):
        def _review_chunk(self, chunk, context, query, known_claims):  # noqa: ARG002
            try:
                raise CriticalOutputLimitError("previous segment hit output ceiling")
            except CriticalOutputLimitError:
                raise RateLimitError("Vertex capacity is temporarily exhausted")

    reviewed = InterruptedCoverage(cfg).review(response, "", None, [])
    assert reviewed
    assert all(claim.sources == ("global_review_fallback_sentence",) for claim in reviewed)
    assert FullContextReviewer(cfg, cache_only=True).review(response, "", None, []) == reviewed


def test_corrupt_live_claim_cache_is_recomputed_but_cache_only_stays_strict(tmp_path):
    cfg = _cfg(tmp_path)
    response = "Alpha has one."

    class ValidClaims(AtomicClaimExtractor):
        calls = 0

        def _retry_json(self, messages, schema, name):  # noqa: ARG002
            self.calls += 1
            return {"claims": [{"text": response, "start": 0, "end": len(response)}]}

    writer = ValidClaims(cfg)
    key = writer._cache_key({"response": response})
    writer._save(key, {"claims": [{"text": "not present", "start": 0, "end": 11}]})
    assert writer.extract(response)[0].text == response and writer.calls == 1

    bad = "Beta has two."
    bad_key = writer._cache_key({"response": bad})
    writer._save(bad_key, {"claims": [{"text": "not present", "start": 0, "end": 11}]})
    with pytest.raises(CacheOnlyMissError):
        AtomicClaimExtractor(cfg, cache_only=True).extract(bad)
