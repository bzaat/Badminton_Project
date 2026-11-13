"""Movement measurement tests with known trajectories."""
import numpy as np
import pytest

from badminton_analysis.movement import (ZONE_NAMES, fill_gaps, own_frame, smooth, speed_and_distance,
                                         zone_index)
from badminton_analysis.schema import BOTTOM, TOP

FPS = 30.0


def test_own_frame_conventions():
    # bottom player at x=4 (image right of centre) -> their right; top player at x=4 -> their left
    d, lat = own_frame(np.array([4.0, 10.0]), BOTTOM)
    assert d == pytest.approx(3.3) and lat == pytest.approx(0.95)
    d, lat = own_frame(np.array([4.0, 3.0]), TOP)
    assert d == pytest.approx(3.7) and lat == pytest.approx(-0.95)


def test_straight_line_speed_and_distance():
    t = np.arange(61) / FPS                               # 2 s at 3 m/s along x
    xy = np.c_[3 * t, np.full_like(t, 10.0)]
    valid = np.ones(len(t), bool)
    s, dist = speed_and_distance(smooth(xy, valid, 9), valid, FPS)
    assert dist == pytest.approx(6.0, rel=1e-6)
    np.testing.assert_allclose(s[5:-5], 3.0, rtol=1e-6)    # moving average keeps a linear track


def test_smoothing_reduces_noise_distance():
    rng = np.random.default_rng(0)
    xy = np.full((300, 2), 5.0) + rng.normal(scale=0.05, size=(300, 2))   # standing still, 5 cm noise
    valid = np.ones(300, bool)
    _, d_raw = speed_and_distance(smooth(xy, valid, 1), valid, FPS)
    s9, d9 = speed_and_distance(smooth(xy, valid, 9), valid, FPS)
    assert d9 < d_raw / 4                                  # noise inflates raw distance a lot
    assert np.nanmedian(s9) < 0.6


def test_gaps_and_runs():
    xy = np.c_[np.arange(10, dtype=float), np.zeros(10)]
    valid = np.ones(10, bool); valid[[3, 4]] = False; valid[7] = False
    xy2, v2 = fill_gaps(xy, valid, max_gap=1)
    assert v2[7] and not v2[3] and not v2[4]                # 2-frame gap stays open
    assert xy2[7, 0] == pytest.approx(7.0)
    s, dist = speed_and_distance(xy2, v2, FPS)
    assert dist == pytest.approx(2 + 4)                     # runs 0-2 and 5-9, no step across the gap
    assert np.isnan(s[3]) and np.isnan(s[4])


def test_zones():
    depth = np.array([0.5, 3.3, 6.0, 6.9, 9.0, np.nan])
    lat = np.array([-2.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    z = zone_index(depth, lat)
    assert [ZONE_NAMES[i] if i >= 0 else None for i in z] == ["front_left", "mid_centre", "rear_right", "rear_centre", None, None]
