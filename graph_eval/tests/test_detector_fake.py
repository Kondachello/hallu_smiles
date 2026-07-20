import json

import pytest

from graph_eval.detector import GraphEvalDetector
from graph_eval.extraction.base import ExtractionOutput
from graph_eval.extraction.fake import FakeExtractor
from graph_eval.nli.fake import FakeNLI
from graph_eval.types import (
    STATUS_EMPTY_GRAPH,
    STATUS_FAILED,
    STATUS_OK,
    DetectionInput,
)

CTX = "Paris is the capital of France."


def _item(response, response_id="r1"):
    return DetectionInput(
        response_id=response_id, source_id="s1", context=CTX,
        response=response, query="What is the capital of France?",
    )


def test_grounded_answer_scores_low():
    extractor = FakeExtractor({"grounded": [["Paris", "is", "the capital of France"]]})
    nli = FakeNLI({(CTX, "Paris is the capital of France."): 0.98})
    det = GraphEvalDetector(extractor, nli)
    res = det.predict(_item("grounded"))
    assert res.status == STATUS_OK
    assert res.raw_score == pytest.approx(0.02)
    assert res.components["paper_threshold_decision"] is False
    assert res.flagged_unit_ids == ()


def test_ungrounded_answer_scores_high_and_flags():
    extractor = FakeExtractor({"bad": [["Berlin", "is", "the capital of France"]]})
    nli = FakeNLI({(CTX, "Berlin is the capital of France."): 0.05})
    det = GraphEvalDetector(extractor, nli)
    res = det.predict(_item("bad"))
    assert res.raw_score == pytest.approx(0.95)
    assert res.components["paper_threshold_decision"] is True
    assert res.flagged_unit_ids == ("t_1",)


def test_response_score_is_max_over_triples():
    extractor = FakeExtractor(
        {"mix": [["Paris", "is", "the capital of France"], ["Cats", "orbit", "moons"]]}
    )
    nli = FakeNLI({
        (CTX, "Paris is the capital of France."): 0.95,
        (CTX, "Cats orbit moons."): 0.10,
    })
    det = GraphEvalDetector(extractor, nli)
    res = det.predict(_item("mix"))
    assert res.raw_score == pytest.approx(0.90)  # max(0.05, 0.90)


def test_empty_answer_graph_is_special_state_not_a_score():
    det = GraphEvalDetector(FakeExtractor(), FakeNLI())
    res = det.predict(_item("hi"))  # too short -> zero triples
    assert res.status == STATUS_EMPTY_GRAPH
    assert res.raw_score is None


def test_extractor_failure_becomes_failed_not_a_score():
    class Boom:
        prompt_profile = "boom"

        def extract(self, response_text):
            raise RuntimeError("gateway 503")

    det = GraphEvalDetector(Boom(), FakeNLI())
    res = det.predict(_item("anything here please"))
    assert res.status == STATUS_FAILED
    assert res.raw_score is None
    assert res.failure["stage"] == "extraction"


def test_extractor_sees_only_the_answer():
    seen = []

    class RecordingExtractor:
        prompt_profile = "rec"

        def extract(self, response_text):
            seen.append(response_text)
            return ExtractionOutput(json.dumps([["Paris", "is", "here"]]))

    det = GraphEvalDetector(RecordingExtractor(), FakeNLI())
    det.predict(_item("Paris is here"))
    assert seen == ["Paris is here"]
    assert all(CTX not in s for s in seen)  # context never handed to extractor


def test_nli_premise_is_context_and_hypothesis_is_verbalized_triple():
    captured = []

    class RecordingNLI:
        model_label = "rec"
        revision = None

        def score_pairs(self, pairs):
            captured.extend(pairs)
            return [0.5] * len(pairs)

    extractor = FakeExtractor({"x": [["Paris", "is", "capital"]]})
    det = GraphEvalDetector(extractor, RecordingNLI())
    det.predict(_item("x"))
    assert captured == [(CTX, "Paris is capital.")]


def test_detection_input_has_no_gold_fields():
    fields = set(DetectionInput.__dataclass_fields__)
    assert not fields & {"gold_response_label", "gold_spans", "label", "labels"}
