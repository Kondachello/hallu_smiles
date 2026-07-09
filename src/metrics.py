"""HalluGraph metrics: Entity Grounding, edge-aware Relation Preservation, CFI, H.

  EG  = |{ v in V_a : exists w in V_ref, match(v,w) }| / |V_a|
  RP  = (1/|E_a|) * sum_{e in E_a} 1[ exists e' in E_ref : align(e,e') ]      (undefined if |E_a|=0)
  CFI = alpha*EG + (1-alpha)*RP        (or CFI = EG when RP is undefined -- edge-aware)
  H   = 1 - CFI

Degenerate cases:
  |V_a| = 0  -> response is unscorable by structure (EG undefined). Handled by policy at report time.
  |V_ref| = 0 -> instance flagged & excluded.

EG/RP are computed once (they do NOT depend on alpha); CFI/H are derived later so alpha
tuning and the tau sweep never require re-extraction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .matching import RefGraph


@dataclass
class ScoreResult:
    # graph sizes
    Vc: int = 0
    Ec: int = 0
    Vq: int = 0
    Eq: int = 0
    Va: int = 0
    Ea: int = 0
    # core alpha-independent metrics
    EG: float | None = None          # None => |V_a| = 0 (unscorable)
    RP: float | None = None          # None => |E_a| = 0 (edge-aware: excluded)
    RP_defined: bool = False
    # audit detail
    matched_entities: list[tuple[str, str, str]] = field(default_factory=list)   # (v, ref, method)
    ungrounded_entities: list[str] = field(default_factory=list)
    supported_relations: list[list[str]] = field(default_factory=list)
    unsupported_relations: list[list[str]] = field(default_factory=list)
    # flags
    unscorable: bool = False         # |V_a| = 0
    ref_empty: bool = False          # |V_ref| = 0

    def cfi(self, alpha: float) -> float | None:
        """Composite Fidelity Index for a given alpha (None if unscorable)."""
        if self.unscorable or self.EG is None:
            return None
        if not self.RP_defined or self.RP is None:
            return self.EG  # edge-aware: CFI reduces to EG
        return alpha * self.EG + (1.0 - alpha) * self.RP

    def h(self, alpha: float, impute: float | None = None) -> float | None:
        """Hallucination score H = 1 - CFI. If unscorable, return `impute` (or None)."""
        c = self.cfi(alpha)
        if c is None:
            return impute
        return 1.0 - c

    def eg_only_h(self) -> float | None:
        return None if self.EG is None else 1.0 - self.EG

    def rp_only_h(self) -> float | None:
        return None if self.RP is None else 1.0 - self.RP

    def to_dict(self) -> dict[str, Any]:
        return {
            "Vc": self.Vc, "Ec": self.Ec, "Vq": self.Vq, "Eq": self.Eq,
            "Va": self.Va, "Ea": self.Ea,
            "EG": self.EG, "RP": self.RP, "RP_defined": self.RP_defined,
            "matched_entities": [list(m) for m in self.matched_entities],
            "ungrounded_entities": self.ungrounded_entities,
            "supported_relations": self.supported_relations,
            "unsupported_relations": self.unsupported_relations,
            "unscorable": self.unscorable, "ref_empty": self.ref_empty,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScoreResult":
        return cls(
            Vc=d["Vc"], Ec=d["Ec"], Vq=d["Vq"], Eq=d["Eq"], Va=d["Va"], Ea=d["Ea"],
            EG=d["EG"], RP=d["RP"], RP_defined=d["RP_defined"],
            matched_entities=[tuple(m) for m in d.get("matched_entities", [])],
            ungrounded_entities=list(d.get("ungrounded_entities", [])),
            supported_relations=[list(r) for r in d.get("supported_relations", [])],
            unsupported_relations=[list(r) for r in d.get("unsupported_relations", [])],
            unscorable=d["unscorable"], ref_empty=d["ref_empty"],
        )


def score_response(g_a, refgraph: RefGraph, g_c=None, g_q=None) -> ScoreResult:
    """Compute EG, RP and the audit lists for one response graph against the ref graph.

    g_a is a Graph (entities: set[str], relations: set[tuple]); refgraph is a RefGraph
    built from V_ref = V_c ∪ V_q, E_ref = E_c ∪ E_q.
    """
    Va = len(g_a.entities)
    Ea = len(g_a.relations)
    res = ScoreResult(
        Va=Va, Ea=Ea,
        Vc=len(g_c.entities) if g_c is not None else 0,
        Ec=len(g_c.relations) if g_c is not None else 0,
        Vq=len(g_q.entities) if g_q is not None else 0,
        Eq=len(g_q.relations) if g_q is not None else 0,
    )

    ref_empty = len(refgraph.ent_norm) == 0
    res.ref_empty = ref_empty

    # ---- Entity Grounding ----
    if Va == 0:
        res.unscorable = True
        res.EG = None
    else:
        grounded = 0
        for v in sorted(g_a.entities):
            m = refgraph.match_entity(v)
            if m.matched:
                grounded += 1
                res.matched_entities.append((_norm(v, refgraph), m.ref or "", m.method or ""))
            else:
                res.ungrounded_entities.append(_norm(v, refgraph))
        res.EG = grounded / Va

    # ---- Relation Preservation (edge-aware) ----
    if Ea == 0:
        res.RP = None
        res.RP_defined = False
    else:
        supported = 0
        for e in sorted(g_a.relations):
            a = refgraph.align_relation(e)
            triple = [_norm(e[0], refgraph), _norm(e[1], refgraph), _norm(e[2], refgraph)]
            if a.matched:
                supported += 1
                res.supported_relations.append(triple)
            else:
                res.unsupported_relations.append(triple)
        res.RP = supported / Ea
        res.RP_defined = True

    return res


def _norm(s: str, refgraph: RefGraph) -> str:
    from .matching import normalize

    return normalize(s)


# --------------------------------------------------------------------------------------
# Small helpers used by tuning/eval
# --------------------------------------------------------------------------------------
def cfi(eg: float, rp: float | None, alpha: float) -> float:
    """Pure CFI given values (rp=None => edge-aware reduction to EG)."""
    if rp is None:
        return eg
    return alpha * eg + (1.0 - alpha) * rp


def hallucination(cfi_value: float) -> float:
    return 1.0 - cfi_value
