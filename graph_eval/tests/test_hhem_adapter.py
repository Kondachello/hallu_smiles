import pytest
import sys
from types import SimpleNamespace

from graph_eval.config import NLIConfig, from_dict
from graph_eval.nli.hhem import HHEMNLIModel, _as_floats
from graph_eval.scoring import p_unsupported


class FakeHHEM:
    """Stand-in for the real HHEM model: predict(pairs) -> consistency scores."""

    def __init__(self, table):
        self.table = table
        self.batches = []

    def predict(self, pairs):
        self.batches.append(list(pairs))
        return [self.table[p] for p in pairs]


def test_passthrough_is_p_consistent_with_correct_direction():
    ctx = "Paris is the capital of France."
    supported = (ctx, "Paris is the capital of France.")
    contradiction = (ctx, "Berlin is the capital of France.")
    model = FakeHHEM({supported: 0.95, contradiction: 0.03})
    nli = HHEMNLIModel(NLIConfig(backend="hhem", revision="deadbeef"), model=model)

    p_consistent = nli.score_pairs([supported, contradiction])
    assert p_consistent == [0.95, 0.03]
    # contradiction must carry the higher hallucination score
    assert p_unsupported(p_consistent[1]) > p_unsupported(p_consistent[0])


def test_batching_respects_batch_size():
    pairs = [("p", f"h{i}") for i in range(5)]
    model = FakeHHEM({pr: 0.5 for pr in pairs})
    nli = HHEMNLIModel(NLIConfig(backend="hhem", revision="sha", batch_size=2), model=model)
    out = nli.score_pairs(pairs)
    assert len(out) == 5
    assert [len(b) for b in model.batches] == [2, 2, 1]


def test_empty_pairs_never_touch_the_model():
    model = FakeHHEM({})
    nli = HHEMNLIModel(NLIConfig(backend="hhem", revision="sha"), model=model)
    assert nli.score_pairs([]) == []
    assert model.batches == []


def test_as_floats_coerces_tensor_like_and_lists():
    class TensorLike:
        def tolist(self):
            return [0.1, 0.2]

    assert _as_floats(TensorLike()) == [0.1, 0.2]
    assert _as_floats([0.3, 0.4]) == [0.3, 0.4]


def test_unpinned_revision_refuses_to_load_without_touching_transformers():
    nli = HHEMNLIModel(NLIConfig(backend="hhem", revision=None))  # no injected model
    with pytest.raises(ValueError):
        nli.score_pairs([("p", "h")])  # _ensure_model -> _default_load -> revision guard


def test_config_requires_pinned_hhem_revision():
    with pytest.raises(ValueError):
        from_dict({"nli": {"backend": "hhem"}})  # missing revision
    with pytest.raises(ValueError):
        from_dict({"nli": {"backend": "hhem", "revision": "main"}})  # floating tag
    cfg = from_dict({"nli": {"backend": "hhem", "revision": "abc123"}})
    assert cfg.nli.revision == "abc123"


def test_real_loader_forces_local_files_only(monkeypatch):
    seen = {}

    class AutoModelForSequenceClassification:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs
            return object()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForSequenceClassification=AutoModelForSequenceClassification
    ))
    nli = HHEMNLIModel(NLIConfig(backend="hhem", model="/mounted/hhem", revision="pinned-sha"))
    assert nli._default_load() is not None
    assert seen["args"] == ("/mounted/hhem",)
    assert seen["kwargs"] == {
        "trust_remote_code": True,
        "revision": "pinned-sha",
        "local_files_only": True,
    }
