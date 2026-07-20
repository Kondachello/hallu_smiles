import math

import pytest

from graph_eval.scoring import (
    DEFAULT_PAPER_THRESHOLD,
    aggregate,
    clamp_unit,
    decision_at_threshold,
    p_unsupported,
)


def test_direction_p_unsupported():
    # higher p_consistent -> lower hallucination score
    assert p_unsupported(1.0) == 0.0
    assert p_unsupported(0.0) == 1.0
    assert p_unsupported(0.75) == pytest.approx(0.25)


def test_clamp_and_nan():
    assert clamp_unit(-0.2) == 0.0
    assert clamp_unit(1.5) == 1.0
    with pytest.raises(ValueError):
        clamp_unit(float("nan"))


def test_max_aggregation_picks_worst_triple():
    assert aggregate([0.1, 0.9, 0.3]) == pytest.approx(0.9)


def test_empty_aggregation_is_none_not_zero():
    assert aggregate([]) is None


def test_unknown_aggregation_rejected():
    with pytest.raises(ValueError):
        aggregate([0.5], method="mean")


def test_threshold_decision_strict_and_none():
    assert decision_at_threshold(0.6) is True
    assert decision_at_threshold(0.5) is False  # strictly greater
    assert decision_at_threshold(None) is None
    assert DEFAULT_PAPER_THRESHOLD == 0.5
