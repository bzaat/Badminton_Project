"""Tests for shuttle cleaning and event matching."""
import itertools

import numpy as np
import pytest

from badminton_analysis.events_eval import match_events, prf
from badminton_analysis.shuttle import clean_track, interpolate_gaps


def _hit_trajectory():
    """Shuttle flies right, is hit at frame 20 and flies back left/up (a sharp reversal)."""
    t = np.arange(40, dtype=float)
    x = np.where(t <= 20, 100 + 20 * t, 500 - 25 * (t - 20))
    y = np.where(t <= 20, 300 + 2 * t, 340 - 10 * (t - 20))
    return np.c_[x, y]


def test_clean_track_keeps_hit_and_removes_glitches():
    xy = _hit_trajectory()
    xy[10] = [900, 50]               # single-frame glitch
    xy[30:32] = [[50, 650], [55, 655]]  # two-frame glitch cluster
    clean, outlier, filled = clean_track(xy, tol_px=40, speed_frac=0.5, max_gap=5)
    assert outlier[10] and outlier[30] and outlier[31]
    assert outlier.sum() == 3                       # the hit point (frame 20) is kept
    assert filled[10] and filled[30] and filled[31]
    np.testing.assert_allclose(clean[10], _hit_trajectory()[10])  # linear segment restored exactly


def test_isolated_point_is_outlier():
    xy = np.full((20, 2), np.nan)
    xy[10] = [500, 300]
    _, outlier, _ = clean_track(xy)
    assert outlier[10]


def test_interpolate_respects_max_gap():
    xy = np.full((12, 2), np.nan)
    xy[0] = [0, 0]; xy[4] = [40, 0]; xy[11] = [110, 0]   # gaps of 3 and 6
    out, filled = interpolate_gaps(xy, max_gap=5)
    np.testing.assert_allclose(out[1:4, 0], [10, 20, 30])
    assert np.isnan(out[5:11]).all() and filled.sum() == 3


# --------------------------------------------------------------------------- matching
def test_match_events_basic():
    r = match_events(gt=[10, 40, 70], pred=[11, 45, 69, 100], tol=2)
    assert r.pairs == [(0, 0), (2, 2)] and r.fn == [1] and r.fp == [1, 3]
    m = prf(*r.counts)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(2 / 3)


def test_one_prediction_cannot_match_two_hits():
    r = match_events(gt=[10, 12], pred=[11], tol=2)
    assert len(r.pairs) == 1 and len(r.fn) == 1


def test_two_predictions_one_hit():
    r = match_events(gt=[10], pred=[9, 11], tol=2)
    assert len(r.pairs) == 1 and len(r.fp) == 1


def _max_matching_bruteforce(gt, pred, tol):
    best = 0
    for perm in itertools.permutations(range(len(pred)), min(len(gt), len(pred))):
        best = max(best, sum(abs(pred[p] - g) <= tol for g, p in zip(gt, perm)))
    # permutations over len(gt) slots: also allow fewer gts than preds
    return best


def test_greedy_matching_is_maximum_cardinality():
    rng = np.random.default_rng(0)
    for _ in range(300):
        gt = np.sort(rng.choice(30, size=rng.integers(1, 5), replace=False))
        pred = np.sort(rng.choice(30, size=rng.integers(1, 5), replace=False))
        tol = int(rng.integers(0, 4))
        greedy = len(match_events(gt, pred, tol).pairs)
        # exact maximum via scipy assignment
        from scipy.optimize import linear_sum_assignment
        cost = (np.abs(gt[:, None] - pred[None, :]) > tol).astype(float)
        r, c = linear_sum_assignment(cost)
        assert greedy == int((cost[r, c] == 0).sum())


def test_unsorted_input_rejected():
    with pytest.raises(ValueError):
        match_events([5, 3], [1], 2)
