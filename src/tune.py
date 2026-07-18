"""Hyperparameter tuning -- TRAIN SPLIT ONLY.

  - alpha via 5-fold CV maximizing mean ROC-AUC of H (grid 0.0..1.0).
  - decision threshold theta maximizing F1.
  - tau_e x tau_r sensitivity sweep (train AUC per combination). The re-scoring loop that
    the sweep needs lives in run.py (it re-runs matching over cached graphs, never
    re-extracting); this module provides the pure aggregation helpers.

Nothing here ever touches the test split.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from .metrics import ScoreResult


def h_array(
    scores: Sequence[ScoreResult],
    alpha: float,
    impute: float | None = None,
    include_unscorable: bool = False,
    mode: str = "strict",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (H, mask) where mask marks entries that carry a usable score.

    If include_unscorable and impute is not None, unscorable responses get H=impute and are
    kept; otherwise they are masked out.
    """
    H = np.full(len(scores), np.nan, dtype=float)
    mask = np.zeros(len(scores), dtype=bool)
    for i, s in enumerate(scores):
        h = s.h_for_mode(alpha, mode=mode, impute=impute if include_unscorable else None)
        if h is None:
            continue
        H[i] = h
        mask[i] = True
    return H, mask


def safe_auc(h: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC of H vs y (higher H => positive). NaN if only one class present."""
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, h))


def alpha_cv(
    scores: Sequence[ScoreResult],
    y: Sequence[int],
    grid: Sequence[float],
    folds: int = 5,
    seed: int = 42,
    mode: str = "strict",
) -> tuple[float, dict[float, float]]:
    """Pick alpha maximizing mean cross-validated ROC-AUC of H on the train split.

    Unscorable responses (|V_a|=0) are excluded here -- their H is alpha-independent, so
    including them only adds constant noise to the ranking of alphas.
    """
    y = np.asarray(list(y), dtype=int)
    # Restrict to scorable responses (EG defined) once; the subset is the same for all alpha.
    scorable_idx = np.array([i for i, s in enumerate(scores) if not s.unscorable], dtype=int)
    if len(scorable_idx) == 0:
        return (grid[len(grid) // 2], {a: float("nan") for a in grid})
    y_s = y[scorable_idx]
    sub = [scores[i] for i in scorable_idx]

    n_splits = min(folds, _max_stratified_splits(y_s))
    per_alpha: dict[float, float] = {}
    if n_splits < 2:
        # Not enough per-class samples to CV; fall back to whole-train AUC.
        for a in grid:
            H, m = h_array(sub, a, mode=mode)
            per_alpha[a] = safe_auc(H[m], y_s[m])
        best = _argmax_alpha(per_alpha)
        return best, per_alpha

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds_idx = list(skf.split(np.zeros(len(sub)), y_s))
    for a in grid:
        H, m = h_array(sub, a, mode=mode)
        fold_aucs = []
        for _, val_idx in folds_idx:
            vi = val_idx[m[val_idx]]
            if len(vi) == 0:
                continue
            fold_aucs.append(safe_auc(H[vi], y_s[vi]))
        fold_aucs = [x for x in fold_aucs if not np.isnan(x)]
        per_alpha[a] = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
    best = _argmax_alpha(per_alpha)
    return best, per_alpha


def _argmax_alpha(per_alpha: dict[float, float]) -> float:
    valid = {a: v for a, v in per_alpha.items() if not np.isnan(v)}
    if not valid:
        # default to paper reference alpha ~= 0.7 (nearest grid point)
        return min(per_alpha, key=lambda a: abs(a - 0.7))
    return max(valid, key=lambda a: valid[a])


def _max_stratified_splits(y: np.ndarray) -> int:
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min()) if len(counts) else 0


def select_f1_threshold(h: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Choose theta maximizing F1 (H >= theta => predicted hallucinated)."""
    h = np.asarray(h, dtype=float)
    y = np.asarray(y, dtype=int)
    finite = np.isfinite(h)
    h, y = h[finite], y[finite]
    if len(h) == 0 or len(np.unique(y)) < 2:
        return 0.5, float("nan")
    cands = np.unique(h)
    # candidate thresholds: each unique value and midpoints, plus a point below the min
    mids = (cands[:-1] + cands[1:]) / 2 if len(cands) > 1 else cands
    thresholds = np.unique(np.concatenate([[cands.min() - 1e-9], cands, mids]))
    best_theta, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (h >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1, best_theta = float(f1), float(t)
    return best_theta, best_f1


def prf_at_threshold(h: np.ndarray, y: np.ndarray, theta: float) -> tuple[float, float, float]:
    h = np.asarray(h, dtype=float)
    y = np.asarray(y, dtype=int)
    finite = np.isfinite(h)
    h, y = h[finite], y[finite]
    if len(h) == 0:
        return float("nan"), float("nan"), float("nan")
    pred = (h >= theta).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return float(p), float(r), float(f1)
