"""Score direction, aggregation, and the paper-threshold decision.

Global convention: every score means *more hallucination when larger*.
NLI models emit ``p_consistent`` (higher = better supported); we convert to
``p_unsupported = 1 - p_consistent`` before aggregating.
"""
from __future__ import annotations

from typing import Iterable

AGGREGATION_MAX_UNSUPPORTED = "max_unsupported"
VALID_AGGREGATIONS = frozenset({AGGREGATION_MAX_UNSUPPORTED})
DEFAULT_PAPER_THRESHOLD = 0.5


def clamp_unit(x: float) -> float:
    """Clamp to [0, 1]; reject NaN loudly rather than let it poison a max()."""
    xf = float(x)
    if xf != xf:  # NaN
        raise ValueError("NLI score is NaN")
    if xf < 0.0:
        return 0.0
    if xf > 1.0:
        return 1.0
    return xf


def p_unsupported(p_consistent: float) -> float:
    """Flip NLI consistency into the global hallucination direction."""
    return 1.0 - clamp_unit(p_consistent)


def aggregate(
    per_triple_unsupported: Iterable[float],
    method: str = AGGREGATION_MAX_UNSUPPORTED,
) -> float | None:
    """GraphEval response score.  Returns ``None`` for an empty triple set."""
    if method not in VALID_AGGREGATIONS:
        raise ValueError(f"unknown aggregation: {method!r}")
    values = [clamp_unit(v) for v in per_triple_unsupported]
    if not values:
        return None
    return max(values)


def decision_at_threshold(
    raw_score: float | None, threshold: float = DEFAULT_PAPER_THRESHOLD
) -> bool | None:
    """Paper-compatible verdict: hallucinated iff score strictly exceeds threshold."""
    if raw_score is None:
        return None
    return raw_score > threshold
