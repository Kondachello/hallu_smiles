"""GraphEvalDetector: answer-only extraction -> per-triple NLI -> max aggregation.

Contract guarantees exercised by the tests:
  * the extractor is handed ``item.response`` and nothing else;
  * the NLI premise is ``item.context`` (the query is never used as evidence);
  * an empty/all-invalid graph yields ``empty_graph`` with ``raw_score=None``;
  * any extractor/NLI/parse exception yields ``failed`` with ``raw_score=None``,
    i.e. a transport error is never turned into a hallucination score.
"""
from __future__ import annotations

import time

from .scoring import (
    AGGREGATION_MAX_UNSUPPORTED,
    DEFAULT_PAPER_THRESHOLD,
    aggregate,
    decision_at_threshold,
    p_unsupported,
)
from .cache import CacheOnlyMissError
from .parser import STATUS_MALFORMED, parse_triples
from .types import (
    STATUS_EMPTY_GRAPH,
    STATUS_FAILED,
    STATUS_OK,
    DetectionInput,
    DetectionResult,
)
from .usage import Usage
from .verbalize import VERBALIZER_VERSION, verbalize

METHOD = "grapheval"


class GraphEvalDetector:
    def __init__(
        self,
        extractor,
        nli,
        *,
        paper_threshold: float = DEFAULT_PAPER_THRESHOLD,
        aggregation: str = AGGREGATION_MAX_UNSUPPORTED,
        method: str = METHOD,
    ):
        self.extractor = extractor
        self.nli = nli
        self.paper_threshold = paper_threshold
        self.aggregation = aggregation
        self.method = method

    def predict(self, item: DetectionInput) -> DetectionResult:
        usage = Usage()

        # 1. Extraction — answer only.
        try:
            start = time.perf_counter()
            extraction = self.extractor.extract(item.response)
            usage.wall_time_ms_extract = (time.perf_counter() - start) * 1000.0
            usage.extractor_calls += int(extraction.usage.get("extractor_calls", 1))
        except CacheOnlyMissError:
            raise  # replay-integrity failure, not a per-item state
        except Exception as exc:  # noqa: BLE001 - transport/model failure is a state
            return self._failed(item, {"stage": "extraction", "error": repr(exc)}, usage)

        parsed = parse_triples(extraction.raw_output)
        if parsed.status == STATUS_MALFORMED:
            return self._failed(
                item,
                {"stage": "parse", "error": parsed.error, "raw_output": parsed.raw_output},
                usage,
            )

        valid = list(parsed.valid_triples)
        if not valid:
            return self._empty(item, parsed, usage)

        # 2. NLI verification — premise is the context.
        hypotheses = [verbalize(t) for t in valid]
        pairs = [(item.context, h) for h in hypotheses]
        try:
            start = time.perf_counter()
            p_consistent = self.nli.score_pairs(pairs)
            usage.wall_time_ms_verify = (time.perf_counter() - start) * 1000.0
        except CacheOnlyMissError:
            raise  # replay-integrity failure, not a per-item state
        except Exception as exc:  # noqa: BLE001
            return self._failed(item, {"stage": "nli", "error": repr(exc)}, usage)

        if len(p_consistent) != len(valid):
            return self._failed(
                item, {"stage": "nli", "error": "score count != triple count"}, usage
            )

        # Account model calls vs cache hits when the NLI layer is cache-backed;
        # this is what lets a warm replay prove 0 model calls.
        stats = getattr(self.nli, "last_stats", None)
        if isinstance(stats, dict):
            usage.nli_calls += int(stats.get("misses", len(pairs)))
            usage.nli_cache_hits += int(stats.get("hits", 0))
        else:
            usage.nli_calls += len(pairs)

        per_triple = [p_unsupported(p) for p in p_consistent]
        raw_score = aggregate(per_triple, self.aggregation)

        # 3. Audit + flags.
        triples_detail = []
        flagged: list[str] = []
        for triple, hypothesis, pc, pu in zip(valid, hypotheses, p_consistent, per_triple):
            is_flagged = pu > self.paper_threshold
            if is_flagged:
                flagged.append(triple.triple_id)
            triples_detail.append(
                {
                    "triple_id": triple.triple_id,
                    "raw_subject": triple.raw_subject,
                    "raw_relation": triple.raw_relation,
                    "raw_object": triple.raw_object,
                    "verbalized_hypothesis": hypothesis,
                    "p_consistent": pc,
                    "p_unsupported": pu,
                    "flagged_at_paper_threshold": is_flagged,
                }
            )

        components = {
            "triples": triples_detail,
            "n_triples_total": len(parsed.triples),
            "n_triples_valid": len(valid),
            "n_triples_invalid": parsed.invalid_count,
            "aggregation": self.aggregation,
            "verbalizer_version": VERBALIZER_VERSION,
            "paper_threshold": self.paper_threshold,
            "paper_threshold_decision": decision_at_threshold(raw_score, self.paper_threshold),
        }
        return DetectionResult(
            response_id=item.response_id,
            source_id=item.source_id,
            method=self.method,
            raw_score=raw_score,
            components=components,
            flagged_unit_ids=tuple(flagged),
            status=STATUS_OK,
            failure=None,
            usage=usage.to_dict(),
            artifact_refs={},
        )

    def _empty(self, item: DetectionInput, parsed, usage: Usage) -> DetectionResult:
        return DetectionResult(
            item.response_id, item.source_id, self.method, None,
            {
                "reason": "empty_answer_graph",
                "n_triples_total": len(parsed.triples),
                "n_triples_invalid": parsed.invalid_count,
            },
            (), STATUS_EMPTY_GRAPH, None, usage.to_dict(), {},
        )

    def _failed(self, item: DetectionInput, failure: dict, usage: Usage) -> DetectionResult:
        return DetectionResult(
            item.response_id, item.source_id, self.method, None,
            {}, (), STATUS_FAILED, failure, usage.to_dict(), {},
        )
