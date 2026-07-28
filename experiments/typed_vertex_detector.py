"""Typed-vertex HalluGraph detector (fresh, self-contained metric for one replay).

This detector recomputes the HalluGraph confabulation index for one record, but
grounds answer *vertices* by their assigned **type** (from the dynamic typing
agent) instead of by surface-string similarity. The *edge* component (RP) is the
original HalluGraph relation grounding, computed fresh here over the same cached
graphs::

    EG_type  = |{v in V_A : type_match(v, V_ref)}| / |V_A|        # NEW: by type
    RP       = |{e=(s,r,o) in E_A : match(s,V_ref) & match(o,V_ref)}| / |E_A|
    CFI_type = alpha * EG_type + (1 - alpha) * RP
    raw_score (higher = more hallucination) = 1 - CFI_type

Everything is computed within this run; no other experiment's results are read.
The already-computed strict/support results stay untouched for a post-hoc compare.

Graphs come from the shared cache provider (the same one the hallugraph/grapheval
detectors read), so extraction identity is shared. Typing is injected as a
``typer`` callable so the pure scoring logic is unit-testable without a gateway,
HHEM, or the agent import; production wiring supplies the real agent-backed typer.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Mapping, Protocol, Sequence

from ._compat import ensure_graph_eval_importable

ensure_graph_eval_importable()

from graph_eval.types import (  # noqa: E402
    STATUS_EMPTY_GRAPH,
    STATUS_FAILED,
    STATUS_OK,
    DetectionInput,
    DetectionResult,
)

from src.matching import RefGraph, normalize
from src.typed_matching import TypedRefGraph, typed_cfi, typed_entity_grounding


# Result of typing one record: surface -> assigned type labels, for reference
# (context+query) vertices and answer vertices respectively.
TyperResult = tuple[Mapping[str, Sequence[str]], Mapping[str, Sequence[str]]]


class Typer(Protocol):
    """Types one record's graphs. Implementations may call an LLM/HHEM."""

    def __call__(
        self,
        *,
        context_raw: str,
        query_raw: str,
        response_raw: str,
        context_entities: Sequence[str],
        context_relations: Sequence[tuple[str, str, str]],
        query_entities: Sequence[str],
        query_relations: Sequence[tuple[str, str, str]],
        response_entities: Sequence[str],
        response_relations: Sequence[tuple[str, str, str]],
        record_id: str,
    ) -> TyperResult: ...


_DEFAULT_MATCHING = {
    "entity_sim_threshold": 0.72,
    "relation_sim_threshold": 0.72,
    "allow_substring_match": True,
    "direction_sensitive_edges": True,
    "inverse_edge_match": False,
    "min_substring_chars": 3,
    "stopwords": [],
}


def _matching_cfg(overrides: Mapping[str, Any] | None) -> SimpleNamespace:
    merged = {**_DEFAULT_MATCHING, **(dict(overrides) if overrides else {})}
    return SimpleNamespace(**merged)


def _relation_grounding(
    response_relations: Sequence[tuple[str, str, str]],
    ref: RefGraph,
) -> dict[str, Any]:
    """RP_grounded / RP_strict over answer edges (original HalluGraph edge metric)."""
    total = 0
    grounded = 0
    aligned = 0
    unsupported: list[tuple[str, str, str]] = []
    for edge in response_relations:
        if len(edge) != 3 or not (normalize(edge[0]) and normalize(edge[2])):
            continue
        total += 1
        s_ok = ref.match_entity(edge[0]).matched
        o_ok = ref.match_entity(edge[2]).matched
        if s_ok and o_ok:
            grounded += 1
        if ref.align_relation(tuple(edge)).matched:
            aligned += 1
        else:
            unsupported.append(tuple(edge))
    rp_grounded = (grounded / total) if total else 0.0
    rp_strict = (aligned / total) if total else 0.0
    return {
        "rp_grounded": rp_grounded,
        "rp_strict": rp_strict,
        "total_edges": total,
        "grounded_edges": grounded,
        "aligned_edges": aligned,
        "unsupported_edges": unsupported,
    }


class TypedVertexDetector:
    """DetectorProtocol adapter for the typed-vertex confabulation index."""

    method_name = "typed_vertex"

    def __init__(
        self,
        *,
        shared_graph_provider: Any,
        typer: Typer,
        embedder: Any | None = None,
        matching_config: Mapping[str, Any] | None = None,
        alpha: float = 0.5,
        variant_name: str = "typed_vertex_cfi",
    ) -> None:
        self.provider = shared_graph_provider
        self.typer = typer
        self.embedder = embedder
        self.cfg_matching = _matching_cfg(matching_config)
        self.alpha = float(alpha)
        self.variant_name = variant_name

    def predict(self, item: DetectionInput) -> DetectionResult:
        base = dict(response_id=item.response_id, source_id=item.source_id, method=self.method_name)
        try:
            response = self.provider.prepare_response(item).graph
            context_graph, query_graph = self.provider.extract_reference(item.context, item.query)
        except Exception as exc:  # a graph/cache failure is not a positive prediction
            return DetectionResult(
                **base, raw_score=None, components={}, flagged_unit_ids=(),
                status=STATUS_FAILED, failure={"error": f"shared_graph: {exc!r}"},
                usage={}, artifact_refs={},
            )

        response_entities = list(response.entities)
        response_relations = [tuple(r) for r in response.relations]
        # An empty/invalid answer graph is a legitimate LLM-free outcome, not a failure.
        if not response_entities:
            return DetectionResult(
                **base, raw_score=None, components={"reason": "empty_answer_graph"},
                flagged_unit_ids=(), status=STATUS_EMPTY_GRAPH, failure=None, usage={}, artifact_refs={},
            )

        try:
            ref_vertex_types, answer_vertex_types = self.typer(
                context_raw=item.context,
                query_raw=item.query or "",
                response_raw=item.response,
                context_entities=list(context_graph.entities),
                context_relations=[tuple(r) for r in context_graph.relations],
                query_entities=list(query_graph.entities),
                query_relations=[tuple(r) for r in query_graph.relations],
                response_entities=response_entities,
                response_relations=response_relations,
                record_id=item.response_id,
            )
        except Exception as exc:
            return DetectionResult(
                **base, raw_score=None, components={}, flagged_unit_ids=(),
                status=STATUS_FAILED, failure={"error": f"typing: {exc!r}"}, usage={}, artifact_refs={},
            )

        # EG_type: answer vertices grounded by assigned type against reference types.
        typed_ref = TypedRefGraph(
            ref_vertex_types,
            allow_substring=bool(self.cfg_matching.allow_substring_match),
            min_substring_chars=int(self.cfg_matching.min_substring_chars),
            stopwords=frozenset(self.cfg_matching.stopwords or []),
        )
        eg = typed_entity_grounding(answer_vertex_types, typed_ref)

        # RP: original HalluGraph edge grounding over the same reference graph.
        ref = RefGraph(
            entities=list(context_graph.entities) + list(query_graph.entities),
            relations=[tuple(r) for r in context_graph.relations]
            + [tuple(r) for r in query_graph.relations],
            cfg_matching=self.cfg_matching,
            embedder=self.embedder,
        )
        rp = _relation_grounding(response_relations, ref)

        cfi = typed_cfi(eg["eg"], rp["rp_grounded"], self.alpha)
        raw_score = 1.0 - cfi  # higher = more hallucination
        components = {
            "eg_type": eg["eg"],
            "rp_grounded": rp["rp_grounded"],
            "rp_strict": rp["rp_strict"],
            "cfi_type": cfi,
            "alpha": self.alpha,
            "total_vertices": eg["total_vertices"],
            "grounded_vertices": eg["grounded_vertices"],
            "total_edges": rp["total_edges"],
            "grounded_edges": rp["grounded_edges"],
            "aligned_edges": rp["aligned_edges"],
            "ungrounded_vertices": list(eg["ungrounded"]),
        }
        return DetectionResult(
            **base, raw_score=raw_score, components=components,
            flagged_unit_ids=tuple(eg["ungrounded"]), status=STATUS_OK,
            failure=None, usage={}, artifact_refs={},
        )
