"""Task 5 comparison logic on the synthetic match (no video, no TrackNet needed)."""
import numpy as np
import pytest

from badminton_analysis.build import build_match
from badminton_analysis.task5 import compare, load_tracknet_csv

from . import synthetic_bfmd as syn
from .test_build_synthetic import cfg, truth  # noqa: F401  (pytest fixtures)


def test_load_tracknet_csv(tmp_path):
    p = tmp_path / "clip_ball.csv"
    p.write_text("Frame,Visibility,X,Y\n0,1,400,300\n1,0,0,0\n2,1,420.5,301\n")
    d = load_tracknet_csv(p)
    np.testing.assert_array_equal(d["frame"], [0, 1, 2])
    np.testing.assert_array_equal(d["visible"], [True, False, True])
    np.testing.assert_allclose(d["xy"][2], [420.5, 301])


def test_compare_counts_and_errors(cfg):  # noqa: F811
    md = build_match(cfg, syn.MATCH)[0]
    # clip starts at video frame 98: clip frames 0..13 -> video 98..111 (rally 0 = 100..109)
    n = 14
    frame = np.arange(n)
    vf = frame + 98
    xy = np.c_[400 + 10 * (vf - 100), np.full(n, 300.0)]   # BFMD synthetic shuttle: x = 400 + 10 (f-100), y = 300
    xy[2 + 5] += [3, 4]                                    # video 105: 5 px off
    visible = np.ones(n, bool)
    visible[2 + 6] = False                                 # video 106: TrackNetV3 misses it
    r = compare(md, {"frame": frame, "visible": visible, "xy": xy}, clip_start=98)
    assert r["frames_in_clip"] == 14 and r["frames_in_rallies"] == 10   # 100..109
    # BFMD has no shuttle at 103 (0,0 entry) and 104 (disabled)
    assert r["only_tracknetv3"] == 2 and r["only_bfmd"] == 1 and r["both_visible"] == 7
    assert r["error_px_median"] == pytest.approx(0.0, abs=0.06) and r["within_5px"] == pytest.approx(1.0)
    assert r["visibility_agreement"] == pytest.approx(7 / 10)
