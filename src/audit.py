"""Per-response audit-trail writer (HalluGraph's core deliverable).

One JSON record per scored response at results/audit/{response_id}.json whose
ungrounded_entities / unsupported_relations EXACTLY explain the EG/RP values.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data import Instance
from .metrics import ScoreResult


def build_audit_record(
    inst: Instance,
    res: ScoreResult,
    alpha: float,
    alpha_support: float | None = None,
    relation_mode: str = "strict",
    timings: dict[str, float] | None = None,
    impute_h: float | None = None,
) -> dict[str, Any]:
    alpha_support = alpha if alpha_support is None else alpha_support
    cfi = res.cfi_for_mode(alpha, "strict")
    h = res.h_for_mode(alpha, "strict", impute=impute_h)
    cfi_support = (
        res.cfi_for_mode(alpha_support, "support") if relation_mode == "support" else None
    )
    h_support = (
        res.h_for_mode(alpha_support, "support", impute=impute_h)
        if relation_mode == "support"
        else None
    )
    return {
        "response_id": inst.response_id,
        "source_id": inst.source_id,
        "task": inst.task,
        "gen_model": inst.gen_model,
        "split": inst.split,
        "relation_mode": relation_mode,
        "alpha": alpha,  # legacy strict alpha
        "alpha_strict": alpha,
        "alpha_support": alpha_support,
        "EG": _round(res.EG),
        "RP": _round(res.RP), "RP_defined": res.RP_defined,
        "RP_strict": _round(res.RP_strict),
        "RP_grounded": _round(res.RP_grounded),
        "RP_entailed_cond": _round(res.RP_entailed_cond),
        "RP_support": _round(res.RP_support),
        "RP_support_defined": res.RP_support_defined,
        "CFI": _round(cfi), "H": _round(h),  # legacy strict aliases
        "CFI_strict": _round(cfi), "H_strict": _round(h),
        "CFI_support": _round(cfi_support), "H_support": _round(h_support),
        "unscorable": res.unscorable,
        "ref_empty": res.ref_empty,
        "graph_sizes": {
            "Vc": res.Vc, "Ec": res.Ec, "Vq": res.Vq, "Eq": res.Eq,
            "Va": res.Va, "Ea": res.Ea,
        },
        "ungrounded_entities": res.ungrounded_entities,
        "unsupported_relations": res.unsupported_relations,
        "matched_entities": [list(m) for m in res.matched_entities],
        "supported_relations": res.supported_relations,
        "relation_audits": res.relation_audits,
        "gt_label": inst.y,
        "gt_span_types": inst.gt_span_types,
        "timings_s": timings or {},
    }


def write_audit(record: dict[str, Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{record['response_id']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _round(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(float(x), nd)
