"""Type-aware vertex matching for the typed HalluGraph metric.

Background
----------
The strict/support HalluGraph metric grounds an answer vertex ``v`` against the
reference graph with :class:`src.matching.RefGraph`, whose ``match_entity`` uses
normalized string equality, guarded substring, and S-BERT cosine over the vertex
*surface strings*. The relation component (RP) is edge-based and unchanged.

Previous QA runs ranked answers by the edge component only. This module adds the
missing vertex signal as a **type-aware** grounding: an answer vertex is grounded
iff some reference vertex carries a compatible *assigned type* (from the dynamic
typing agent), not merely a similar surface string. It mirrors ``RefGraph`` exactly
so it can be dropped into the same EG / CFI formulas::

    EG_type = |{v in V_A : type_match(v, V_ref)}| / |V_A|
    CFI_type = alpha * EG_type + (1 - alpha) * RP        # RP: existing edge metric

The class deliberately does not touch ``src.matching`` or the existing detectors,
so the already-computed strict/support results stay valid for the final comparison.

Type compatibility
------------------
``type_match(v, w)`` is TRUE if the normalized assigned-type sets of ``v`` and ``w``
intersect. Each vertex may carry several types (the agent can assign more than one);
a single shared type grounds the vertex. Normalization reuses ``src.matching.normalize``
so ``"Commercial Bank"`` and ``"commercial bank"`` agree. Guarded token-boundary
substring on type labels is optional (``allow_substring``) so ``"bank"`` grounds
``"commercial bank"`` when enabled, matching the surface matcher's relaxation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .matching import EntityMatch, _token_boundary_substring, normalize


def _normalized_type_set(labels: Iterable[str]) -> frozenset[str]:
    """Normalize a vertex's assigned type labels, dropping empties."""
    out: set[str] = set()
    for label in labels:
        n = normalize(label)
        if n:
            out.add(n)
    return frozenset(out)


@dataclass
class TypedRefGraph:
    """Reference graph that matches answer vertices by assigned type.

    Parameters
    ----------
    ref_vertex_types:
        maps each reference (context+query) vertex surface -> its assigned type
        labels. Only the *types* are used for matching; surfaces are kept for audit.
    allow_substring:
        also ground on guarded token-boundary substring between type labels
        (e.g. answer type ``"bank"`` vs reference type ``"commercial bank"``).
    min_substring_chars / stopwords:
        same guards as :class:`src.matching.RefGraph` to avoid trivial substrings.
    """

    ref_vertex_types: Mapping[str, Sequence[str]]
    allow_substring: bool = False
    min_substring_chars: int = 2
    stopwords: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # Union of all normalized reference types, plus a per-type audit of which
        # reference surface(s) carry it (first surface kept, mirroring RefGraph).
        self._ref_types: set[str] = set()
        self._type_to_ref_surface: dict[str, str] = {}
        for surface, labels in self.ref_vertex_types.items():
            for t in _normalized_type_set(labels):
                self._ref_types.add(t)
                self._type_to_ref_surface.setdefault(t, str(surface))
        self._match_cache: dict[frozenset[str], EntityMatch] = {}

    # -- entity (vertex) matching, by type -----------------------------------
    def match_entity_types(self, vertex_types: Iterable[str]) -> EntityMatch:
        """Return an :class:`EntityMatch` grounding a vertex by its assigned types."""
        vtypes = _normalized_type_set(vertex_types)
        if not vtypes:
            return EntityMatch(False)
        if vtypes in self._match_cache:
            return self._match_cache[vtypes]
        res = self._match_types(vtypes)
        self._match_cache[vtypes] = res
        return res

    def _match_types(self, vtypes: frozenset[str]) -> EntityMatch:
        # 1) exact shared type
        exact = vtypes & self._ref_types
        if exact:
            t = sorted(exact)[0]
            return EntityMatch(True, self._type_to_ref_surface[t], "type_exact")
        # 2) guarded token-boundary substring between type labels
        if self.allow_substring:
            for vt in vtypes:
                for rt in self._ref_types:
                    shorter, longer = (vt, rt) if len(vt) <= len(rt) else (rt, vt)
                    if len(shorter) < self.min_substring_chars or shorter in self.stopwords:
                        continue
                    if _token_boundary_substring(shorter, longer):
                        return EntityMatch(True, self._type_to_ref_surface[rt], "type_substring")
        return EntityMatch(False)


def typed_entity_grounding(
    answer_vertex_types: Mapping[str, Sequence[str]],
    ref: TypedRefGraph,
) -> dict[str, object]:
    """Compute EG_type and its audit over an answer graph's vertices.

    Returns a dict with ``eg`` (grounded fraction), the grounded/ungrounded vertex
    lists, and matched (answer_surface -> ref_surface, how) pairs. Vertices with no
    assigned type are counted as ungrounded (they carry no vertex-level signal),
    mirroring how RefGraph treats an unmatched surface.
    """
    total = 0
    grounded = 0
    matched: list[tuple[str, str, str]] = []
    ungrounded: list[str] = []
    for surface, labels in answer_vertex_types.items():
        total += 1
        m = ref.match_entity_types(labels)
        if m.matched:
            grounded += 1
            matched.append((str(surface), m.ref, m.method))
        else:
            ungrounded.append(str(surface))
    eg = (grounded / total) if total else 0.0
    return {
        "eg": eg,
        "total_vertices": total,
        "grounded_vertices": grounded,
        "matched": matched,
        "ungrounded": ungrounded,
    }


def typed_cfi(eg_type: float, rp: float, alpha: float) -> float:
    """CFI_type = alpha * EG_type + (1 - alpha) * RP (RP is the existing edge metric)."""
    a = float(alpha)
    return a * float(eg_type) + (1.0 - a) * float(rp)
