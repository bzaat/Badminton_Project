"""Evaluation of temporal events (hits) against ground truth with a frame tolerance.

Matching is one-to-one: each ground-truth event can be matched by at most one prediction and
vice versa, and a pair counts only if |pred - gt| <= tol. Both lists are sorted; the greedy
scan below (each GT, in time order, takes the earliest unused prediction inside its window)
gives the maximum number of matched pairs for equal-width windows on a line, so TP/FP/FN are
optimal. Events of different matches must be evaluated separately (frame numbers restart).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MatchResult:
    pairs: list[tuple[int, int]]   # (gt index, pred index)
    fn: list[int]                  # unmatched gt indices
    fp: list[int]                  # unmatched pred indices

    @property
    def counts(self) -> tuple[int, int, int]:
        return len(self.pairs), len(self.fp), len(self.fn)


def match_events(gt: np.ndarray, pred: np.ndarray, tol: int) -> MatchResult:
    gt = np.asarray(gt, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    if np.any(np.diff(gt) < 0) or np.any(np.diff(pred) < 0):
        raise ValueError("gt and pred must be sorted")
    pairs, fn = [], []
    used = np.zeros(len(pred), dtype=bool)
    j = 0
    for i, g in enumerate(gt):
        while j < len(pred) and pred[j] < g - tol:
            j += 1
        if j < len(pred) and pred[j] <= g + tol:
            pairs.append((i, j))
            used[j] = True
            j += 1
        else:
            fn.append(i)
    return MatchResult(pairs=pairs, fn=fn, fp=[int(k) for k in np.flatnonzero(~used)])


def count_matches(gt: np.ndarray, pred: np.ndarray, tol: int) -> tuple[int, int, int]:
    """(TP, FP, FN) only; same matching as match_events."""
    return match_events(gt, pred, tol).counts


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": int(tp), "fp": int(fp), "fn": int(fn)}
