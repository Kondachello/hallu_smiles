"""Deterministic mock detectors used only for framework tests and the visual demo."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .contracts import DetectionInput, DetectionResult, STATUS_EMPTY_GRAPH, STATUS_OK

_WORD = re.compile(r"[\w'-]+", flags=re.UNICODE)
_STOP = frozenset({"the", "a", "an", "is", "was", "were", "of", "to", "and", "in", "on", "for"})


def _content_words(text: str) -> set[str]:
    return {word.lower() for word in _WORD.findall(text) if len(word) > 2 and word.lower() not in _STOP}


def _claims(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"[.!?]+", text) if len(_content_words(sentence)) >= 2]


@dataclass
class LexicalGraphEvalMock:
    """Claim-wise lexical mock preserving GraphEval's max-risk shape."""

    method_name: str = "grapheval"
    variant_name: str = "mock_lexical_v1"

    def predict(self, item: DetectionInput) -> DetectionResult:
        claims = _claims(item.response)
        if not claims:
            return DetectionResult(item.response_id, item.source_id, self.method_name, None, {"reason": "empty_answer_graph"}, (), STATUS_EMPTY_GRAPH, None, {}, {})
        context_words = _content_words(item.context)
        details = []
        flagged = []
        scores: list[float] = []
        for index, claim in enumerate(claims):
            words = _content_words(claim)
            support = len(words & context_words) / len(words) if words else 1.0
            risk = round(1.0 - support, 6)
            claim_id = f"claim:{index}"
            scores.append(risk)
            if risk > 0.5:
                flagged.append(claim_id)
            details.append({"claim_id": claim_id, "text": claim, "p_consistent": round(support, 6), "p_unsupported": risk})
        score = max(scores)
        return DetectionResult(item.response_id, item.source_id, self.method_name, score, {"triples": details, "aggregation": "max_unsupported", "paper_threshold": 0.5}, tuple(flagged), STATUS_OK, None, {"extractor_calls": 1, "nli_calls": len(claims)}, {})


@dataclass
class LexicalHalluGraphMock:
    """Structural-looking mock with EG/RP/CFI components for archive plumbing tests."""

    method_name: str = "hallugraph"
    variant_name: str = "mock_structural_v1"
    alpha: float = 0.7

    def predict(self, item: DetectionInput) -> DetectionResult:
        answer_words = _content_words(item.response)
        if not answer_words:
            return DetectionResult(item.response_id, item.source_id, self.method_name, None, {"reason": "empty_answer_graph"}, (), STATUS_EMPTY_GRAPH, None, {}, {})
        context_words = _content_words(item.context)
        grounded = sorted(answer_words & context_words)
        ungrounded = sorted(answer_words - context_words)
        eg = len(grounded) / len(answer_words)
        # A relation proxy deliberately differs from claim-wise GraphEval aggregation.
        sentences = _claims(item.response)
        supported_sentences = sum(_content_words(sentence) <= context_words for sentence in sentences)
        rp = supported_sentences / len(sentences) if sentences else None
        cfi = eg if rp is None else self.alpha * eg + (1.0 - self.alpha) * rp
        risk = round(1.0 - cfi, 6)
        flagged = tuple(f"entity:{word}" for word in ungrounded)
        return DetectionResult(item.response_id, item.source_id, self.method_name, risk, {"EG": round(eg, 6), "RP": round(rp, 6) if rp is not None else None, "CFI": round(cfi, 6), "alpha": self.alpha, "ungrounded_entities": ungrounded, "unsupported_relations": []}, flagged, STATUS_OK, None, {"extractor_calls": 2, "embedding_calls": 0}, {})


def demo_detectors() -> dict[str, object]:
    return {"hallugraph": LexicalHalluGraphMock(), "grapheval": LexicalGraphEvalMock()}
