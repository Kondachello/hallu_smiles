"""Local test for TypedVertexDetector scoring (fake provider + fake typer; no cache/LLM)."""
from types import SimpleNamespace

# Importing the detector first runs ensure_graph_eval_importable() so the nested
# graph_eval package is on sys.path before we import its types below.
from experiments.typed_vertex_detector import TypedVertexDetector
from graph_eval.types import STATUS_EMPTY_GRAPH, STATUS_OK, DetectionInput


def _graph(entities, relations):
    return SimpleNamespace(entities=list(entities), relations=[tuple(r) for r in relations])


class _FakeProvider:
    """Serves fixed response/reference graphs, mimicking GraphCacheSource."""

    def __init__(self, response, context, query):
        self._response = response
        self._context = context
        self._query = query

    def prepare_response(self, item):
        return SimpleNamespace(graph=self._response)

    def extract_reference(self, context, query):
        return self._context, self._query


def _item():
    return DetectionInput(
        response_id="r1", source_id="s1",
        context="North Bank issued a loan to Delta Ltd.",
        response="The financial organization North Bank issued the loan.",
        query="Which organization issued the loan?",
    )


def test_typed_detector_scores_grounded_and_ungrounded_vertices():
    provider = _FakeProvider(
        response=_graph(["North Bank", "loan"], [("North Bank", "issued", "loan")]),
        context=_graph(["North Bank", "Delta Ltd", "loan"], [("North Bank", "issued", "loan")]),
        query=_graph(["organization", "loan"], [("organization", "issued", "loan")]),
    )

    def typer(**kw):
        ref_types = {"North Bank": ["financial institution"], "Delta Ltd": ["company"],
                     "loan": ["debt instrument"], "organization": ["organization"]}
        answer_types = {"North Bank": ["financial institution"], "loan": ["debt instrument"]}
        return ref_types, answer_types

    det = TypedVertexDetector(shared_graph_provider=provider, typer=typer, alpha=0.5)
    res = det.predict(_item())
    assert res.status == STATUS_OK
    # both answer vertices are type-grounded -> EG_type = 1.0
    assert res.components["eg_type"] == 1.0
    # the single answer edge is grounded (both endpoints in reference) -> RP_grounded = 1.0
    assert res.components["rp_grounded"] == 1.0
    assert abs(res.components["cfi_type"] - 1.0) < 1e-9
    assert abs(res.raw_score - 0.0) < 1e-9  # fully grounded => low hallucination


def test_ungrounded_type_raises_hallucination_score():
    provider = _FakeProvider(
        response=_graph(["Mars", "loan"], [("Mars", "issued", "loan")]),
        context=_graph(["North Bank", "loan"], [("North Bank", "issued", "loan")]),
        query=_graph(["organization"], []),
    )

    def typer(**kw):
        ref_types = {"North Bank": ["financial institution"], "loan": ["debt instrument"],
                     "organization": ["organization"]}
        answer_types = {"Mars": ["planet"], "loan": ["debt instrument"]}  # planet ungrounded
        return ref_types, answer_types

    det = TypedVertexDetector(shared_graph_provider=provider, typer=typer, alpha=1.0)
    res = det.predict(_item())
    assert res.status == STATUS_OK
    assert res.components["eg_type"] == 0.5  # one of two vertices grounded
    assert "Mars" in res.flagged_unit_ids
    assert abs(res.raw_score - 0.5) < 1e-9  # alpha=1 => score = 1 - EG_type


def test_empty_answer_graph_is_empty_status():
    provider = _FakeProvider(
        response=_graph([], []),
        context=_graph(["North Bank"], []),
        query=_graph(["organization"], []),
    )
    det = TypedVertexDetector(shared_graph_provider=provider, typer=lambda **kw: ({}, {}))
    res = det.predict(_item())
    assert res.status == STATUS_EMPTY_GRAPH and res.raw_score is None
