from graph_eval.config import from_dict
from graph_eval.detector import GraphEvalDetector
from graph_eval.extraction.fake import FakeExtractor
from graph_eval.factory import build_nli
from graph_eval.types import DetectionInput


def test_fake_nli_via_factory_and_detector_cache_accounting(tmp_path):
    cfg = from_dict({"nli": {"backend": "fake"}, "cache_dir": str(tmp_path)})
    nli = build_nli(cfg)
    extractor = FakeExtractor({"x": [["Paris", "is", "capital"]]})
    detector = GraphEvalDetector(extractor, nli)
    item = DetectionInput(
        "r1", "s1", "Paris is the capital of France.", "x", query="q"
    )

    first = detector.predict(item)
    assert first.status == "ok"
    assert first.usage["nli_calls"] == 1
    assert first.usage["nli_cache_hits"] == 0

    second = detector.predict(item)  # warm cache => zero model calls
    assert second.usage["nli_calls"] == 0
    assert second.usage["nli_cache_hits"] == 1
    assert second.raw_score == first.raw_score


def test_build_hhem_backend_does_not_import_torch(tmp_path):
    cfg = from_dict(
        {"nli": {"backend": "hhem", "revision": "abc123"}, "cache_dir": str(tmp_path)}
    )
    nli = build_nli(cfg)  # constructing HHEM adapter must not import torch/transformers
    assert nli.model_label == "hhem-2.1-open"
