"""Hit-detector tests on a synthetic rally with one known hit."""
import copy

import numpy as np

from badminton_analysis.build import build_match
from badminton_analysis.config import load_config
from badminton_analysis.hits import HitParams, candidate_features, candidates, frame_features, predict, select
from badminton_analysis.schema import BOTTOM, TOP

from . import synthetic_bfmd as syn
from .test_build_synthetic import cfg, truth  # noqa: F401  (pytest fixtures)

HIT_ROW = 4  # frame 104


def _rally_with_hit(md):
    """Rally 0: shuttle flies towards the bottom player, reverses at row 4 (frame 104); the bottom
    player's right wrist is at the shuttle at that frame and moves fast around it."""
    md = copy.deepcopy(md)
    rows = np.arange(10)
    x = np.where(rows <= HIT_ROW, 500 + 20 * rows, 580 - 25 * (rows - HIT_ROW))
    y = np.where(rows <= HIT_ROW, 300 + 30 * rows, 420 - 35 * (rows - HIT_ROW))
    md.arrays["shuttle_px"][:10] = np.c_[x, y].astype(np.float32)
    kp = md.arrays["pose_kp_px"]
    for r in range(9):  # frames with pose
        if md.arrays["pose_assign"][r, BOTTOM]:
            kp[r, BOTTOM, 10, :2] = [x[HIT_ROW] + 15 * (r - HIT_ROW), y[HIT_ROW] + 5]  # moving wrist
            kp[r, BOTTOM, 10, 2] = 0.9
    return md


def _cfg_hits():
    return load_config()["hits"]


def test_accel_peaks_at_the_hit(cfg):  # noqa: F811
    md = _rally_with_hit(build_match(cfg, syn.MATCH)[0])
    f = frame_features(md, _cfg_hits())
    sl = md.rally_slice(0)
    assert int(np.nanargmax(f["accel"][sl])) == HIT_ROW
    assert f["turn_deg"][HIT_ROW] > 90                       # direction reversal
    assert f["prox"][HIT_ROW, BOTTOM] < 0.2 < f["prox"][HIT_ROW, TOP]
    assert f["wspeed"][HIT_ROW, BOTTOM] > 0.05


def test_candidates_and_prediction(cfg):  # noqa: F811
    md = _rally_with_hit(build_match(cfg, syn.MATCH)[0])
    f = frame_features(md, _cfg_hits())
    c = candidates(md, f, peak_radius=3)
    k = list(c["row"]).index(HIT_ROW)
    assert c["hitter"][k] == BOTTOM
    p = predict(md, c, HitParams(accel=c["accel"][k] - 1e-6, proximity=0.5, wrist_speed=0.01, offset=1))
    assert 105 in p["frame"].tolist()                        # frame 104 + offset 1
    # a proximity threshold below the actual distance rejects it
    assert not select(c, HitParams(accel=0, proximity=1e-6, keep_without_pose=False))[k]
    X = candidate_features(c)
    assert X.shape == (len(c["row"]), 7) and np.isfinite(X).all()


def test_no_shuttle_no_candidates(cfg):  # noqa: F811
    md = build_match(cfg, syn.MATCH)[0]
    md.arrays["shuttle_px"][:] = np.nan
    f = frame_features(md, _cfg_hits())
    assert len(candidates(md, f, 3)["row"]) == 0
