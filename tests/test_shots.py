"""Shot-feature tests on the synthetic match (values known by construction)."""
import numpy as np
import pytest

from badminton_analysis.build import build_match
from badminton_analysis.config import load_config
from badminton_analysis.hits import frame_features
from badminton_analysis.schema import SIDE_INDEX
from badminton_analysis.shots import FEATURE_NAMES, FINE_CLASSES, event_features, merged_label

from . import synthetic_bfmd as syn
from .test_build_synthetic import cfg, truth  # noqa: F401  (pytest fixtures)


@pytest.fixture()
def X(cfg):  # noqa: F811
    md = build_match(cfg, syn.MATCH)[0]
    f = frame_features(md, load_config()["hits"])
    ev = [{"frame": h["frame"], "side": SIDE_INDEX[h["side"]], "rally_idx": h["rally_idx"]} for h in md.hits]
    return event_features(md, f, ev), {n: i for i, n in enumerate(FEATURE_NAMES)}


def test_shape_and_context(X):
    X, c = X
    assert X.shape == (3, len(FEATURE_NAMES))
    # hits: 102 top (rally 0), 106 bottom (rally 0), 203 bottom (rally 1)
    np.testing.assert_array_equal(X[:, c["is_top"]], [1, 0, 0])
    np.testing.assert_array_equal(X[:, c["hit_index"]], [0, 1, 0])
    np.testing.assert_array_equal(X[:, c["is_first"]], [1, 0, 1])
    assert X[1, c["dt_prev"]] == 4 and X[0, c["dt_next"]] == 4
    assert np.isnan(X[0, c["dt_prev"]]) and np.isnan(X[1, c["dt_next"]]) and np.isnan(X[2, c["dt_next"]])


def test_court_position_features(X):
    X, c = X
    # top player stands at (3.05, 2.0) m, bottom at (2.0, 9.0) m (synthetic truth)
    assert X[0, c["hit_depth"]] == pytest.approx(6.7 - 2.0, abs=1e-3)
    assert X[0, c["hit_x_own"]] == pytest.approx(6.1 - 3.05, abs=1e-3)     # top player mirrored
    assert X[0, c["opp_depth"]] == pytest.approx(9.0 - 6.7, abs=1e-3)
    assert X[0, c["opp_x_own"]] == pytest.approx(2.0, abs=1e-3)
    assert X[0, c["next_depth"]] == pytest.approx(9.0 - 6.7, abs=1e-3)     # next hitter = bottom player
    assert np.isnan(X[2, c["hit_depth"]])                                  # rally 1 is 'no_court'


def test_shuttle_features_use_body_frame(X):
    X, c = X
    # shuttle moves +10 px/frame in image x, 0 in y; top hitter box height 150 px; top player is mirrored
    assert X[0, c["sh_after_vx"]] == pytest.approx(-10 / 150, abs=1e-6)
    assert X[1, c["sh_before_vx"]] == pytest.approx(+10 / 150, abs=1e-6)   # bottom player not mirrored
    assert X[0, c["sh_after_vfwd"]] == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(X[2, c["sh_after_speed"]])                             # no shuttle in rally 1


def test_pose_features(X):
    X, c = X
    # synthetic keypoints sit at the box centre except the ankles -> joints at the hip
    assert X[0, c["l_wrist_x"]] == pytest.approx(0.0) and X[0, c["wrist_above_shoulder"]] == pytest.approx(0.0)


def test_label_merge():
    assert merged_label("flick_serve") == "serve" and merged_label("press") == "net_attack"
    assert merged_label("smash") == "smash"
    assert len({merged_label(x) for x in FINE_CLASSES}) == 10
