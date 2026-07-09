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
    timings: dict[str, float] | None = None,
    impute_h: float | None = None,
) -> dict[str, Any]:
    cfi = res.cfi(alpha)
    h = res.h(alpha, impute=impute_h)
    return {
        "response_id": inst.response_id,
        "source_id": inst.source_id,
        "task": inst.task,
        "gen_model": inst.gen_model,
        "split": inst.split,
        "alpha": alpha,
        "EG": _round(res.EG),
        "RP": _round(res.RP),
        "RP_defined": res.RP_defined,
        "CFI": _round(cfi),
        "H": _round(h),
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
