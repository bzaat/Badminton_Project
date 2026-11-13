"""Loader tests on a synthetic BFMD folder where every correct value is known."""
import copy

import numpy as np
import pytest

from badminton_analysis.build import build_match, load_match, singles_matches
from badminton_analysis.config import load_config
from badminton_analysis.schema import BOTTOM, TOP, SchemaError

from . import synthetic_bfmd as syn


@pytest.fixture()
def truth(tmp_path):
    return syn.write(tmp_path / "BFMD_data")


@pytest.fixture()
def cfg(tmp_path, truth):
    base = load_config()
    raw = copy.deepcopy(base.raw)
    raw["paths"].update(bfmd_root=str(tmp_path / "BFMD_data"), cache_dir=str(tmp_path / "cache"))
    raw["frame_shift"] = {"player_bbox": syn.BBOX_SHIFT, "shuttle": 0, "pose": 0}
    # synthetic ankles sit exactly on the ground point, so exact-position tests use no offset
    raw["pose"].update(ankle_ground_offset=0.0, box_ground_offset=0.0)
    return type(base)(raw=raw, path=base.path)


@pytest.fixture()
def md(cfg):
    return build_match(cfg, syn.MATCH)[0]


def test_singles_filter_excludes_doubles(cfg):
    assert singles_matches(cfg) == [syn.MATCH]


def test_frame_axis_and_rallies(md):
    assert md.n_frames == 20
    np.testing.assert_array_equal(md.frame[:10], np.arange(100, 110))
    np.testing.assert_array_equal(md.rally_idx, [0] * 10 + [1] * 10)
    assert [(r["start"], r["end"]) for r in md.rallies] == syn.RALLIES
    assert md.rallies[0]["winner"] == "Alpha A" and md.rallies[1]["score_after"] == {"Alpha A": 1, "Beta B": 1}
    assert md.rallies[0]["winner_from"] == "metadata"
    # null winner field -> derived from the score going 1-0 -> 1-1
    assert md.rallies[1]["winner"] == "Beta B" and md.rallies[1]["winner_from"] == "score_diff"
    assert md.rallies[0]["top_player"] == "Alpha A" and md.rallies[0]["bottom_player"] == "Beta B"
    # rally 1 only has a bottom hit; the top player comes from the neighbouring rally
    assert md.rallies[1]["top_player"] == "Alpha A" and md.rallies[1]["players_from"] == "hits"
    assert all(r["in_timeline"] for r in md.rallies)
    # rally 0 is on the main camera with both players in their own half; rally 1 is 'no_court'
    assert md.rallies[0]["wrong_half_frac"] == 0.0 and not md.rallies[0]["suspect"]
    assert md.rallies[1]["main_camera_frames"] == 0 and md.rallies[1]["wrong_half_frac"] is None


def test_hits(md):
    assert [h["frame"] for h in md.hits] == [102, 106, 203]
    assert [h["rally_idx"] for h in md.hits] == [0, 0, 1]
    assert md.rallies[0]["n_hits"] == 2
    assert md.rallies[0]["starts_with_serve"] and not md.rallies[1]["starts_with_serve"]


def test_player_bbox_shift_applied(truth, md):
    # every rally frame (100..109) must have a box, including the last one, thanks to the shift
    assert not np.isnan(md.player_box_px[:10]).any()
    np.testing.assert_allclose(md.player_box_px[0, TOP], truth["top_box"], atol=1e-3)
    np.testing.assert_allclose(md.player_box_px[0, BOTTOM], truth["bot_box"], atol=1e-3)


def test_player_bbox_without_shift_misses_last_frame(cfg):
    cfg.raw["frame_shift"]["player_bbox"] = 0
    md0 = build_match(cfg, syn.MATCH)[0]
    assert np.isnan(md0.player_box_px[9]).all() and not np.isnan(md0.player_box_px[0]).any()


def test_pose_assigned_to_correct_side(truth, md):
    # bottom person is listed first in the raw file; the loader must still put it in slot BOTTOM
    np.testing.assert_allclose(md.pose_box_px[0, TOP], truth["top_box"], atol=1e-6)
    np.testing.assert_allclose(md.pose_box_px[0, BOTTOM], truth["bot_box"], atol=1e-6)
    two = np.r_[0:5, 6:9]                              # rally-0 rows with both persons
    assert (md.pose_assign[two] == 1).all()            # matched by IoU with the manual box
    # frame 105 has only the top player
    r = 5
    assert md.pose_assign[r, TOP] == 1 and md.pose_assign[r, BOTTOM] == 0
    assert np.isnan(md.pose_kp_px[r, BOTTOM]).all()
    # frame 109 has no pose at all
    assert (md.pose_assign[9] == 0).all()


def test_keypoints_converted_to_pixels(truth, md):
    np.testing.assert_allclose(md.pose_kp_px[0, TOP, 15, :2], truth["top_foot_px"], atol=1e-3)
    assert md.pose_kp_px[0, TOP, 15, 2] == pytest.approx(0.9)


def test_feet_and_court_projection(cfg, md):
    # rally 0: confident ankles -> exact court position
    two = np.r_[0:5, 6:9]
    assert (md.foot_src[two] == 0).all()
    # frame 105: bottom player has no pose -> foot from the manual box
    assert md.foot_src[5, TOP] == 0 and md.foot_src[5, BOTTOM] == 1
    np.testing.assert_allclose(md.foot_court[5, BOTTOM], syn.BOTTOM_FOOT_M, atol=1e-3)
    np.testing.assert_allclose(md.foot_court[0, TOP], syn.TOP_FOOT_M, atol=1e-3)
    np.testing.assert_allclose(md.foot_court[0, BOTTOM], syn.BOTTOM_FOOT_M, atol=1e-3)
    # frame 109: no pose -> manual box bottom-centre (same point by construction)
    assert (md.foot_src[9] == 1).all()
    np.testing.assert_allclose(md.foot_court[9, TOP], syn.TOP_FOOT_M, atol=1e-3)
    # rally 1: ankle score below threshold -> box fallback
    assert (md.foot_src[10:19] == 1).all()
    # top player is in the far half, bottom in the near half
    assert (md.foot_court[:, TOP, 1] < 6.7).all() and (md.foot_court[:, BOTTOM, 1] > 6.7).all()


def test_ground_offsets_move_point_by_box_fraction(cfg, truth):
    cfg.raw["pose"].update(ankle_ground_offset=0.1, box_ground_offset=-0.05)
    md = build_match(cfg, syn.MATCH)[0]
    h_top = truth["top_box"][3] - truth["top_box"][1]
    # row 0: ankles -> +0.1 * box height in image y, x unchanged
    np.testing.assert_allclose(md.foot_px[0, TOP], truth["top_foot_px"] + [0, 0.1 * h_top], atol=1e-3)
    # row 9: box fallback -> bottom - 0.05 * box height
    np.testing.assert_allclose(md.foot_px[9, TOP], truth["top_foot_px"] + [0, -0.05 * h_top], atol=1e-3)
    # moving the point down in the image moves the far player towards the net (larger y)
    assert md.foot_court[0, TOP, 1] > syn.TOP_FOOT_M[1]


def test_foot_point_function_edge_cases():
    from badminton_analysis.build import _foot_point
    kp = np.zeros((17, 3)); kp[15] = [100, 200, 0.9]; kp[16] = [120, 210, 0.1]
    box = np.array([90.0, 100, 130, 220])
    pt, src = _foot_point(kp, box, 0.3, 0.1, 0.0)   # only the confident ankle is used
    np.testing.assert_allclose(pt, [100, 212]); assert src == 0
    pt, src = _foot_point(None, np.full(4, np.nan), 0.3)
    assert src == -1 and np.isnan(pt).all()


def test_suspect_rally_flag(cfg, tmp_path):
    """Swap the manual top/bottom labels in rally 0 -> both players are in the wrong half."""
    import json
    p = tmp_path / "BFMD_data" / "annotations" / "player_bbox" / f"{syn.MATCH}.json"
    j = json.loads(p.read_text())
    res = j["annotations"][0]["result"]
    res[0]["value"]["labels"], res[1]["value"]["labels"] = ["Bottom_Player"], ["Top_Player"]
    p.write_text(json.dumps(j))
    md = build_match(cfg, syn.MATCH)[0]
    assert md.rallies[0]["suspect"] and md.rallies[0]["wrong_half_frac"] == 1.0


def test_shuttle_missing_values_dropped(md):
    assert np.isnan(md.shuttle_px[3]).all() and np.isnan(md.shuttle_px[4]).all()
    np.testing.assert_allclose(md.shuttle_px[0], [400, 300], atol=0.05)
    assert np.isnan(md.shuttle_px[10:]).all()  # no shuttle in rally 1


def test_court_flag(md):
    assert (md.court_flag[:10] == 0).all()      # 90 / 120 'valid' samples are within 20 frames
    assert (md.court_flag[10:] == 2).all()      # 205 'no_court'


def test_cache_round_trip_and_settings_rebuild(cfg):
    a = load_match(cfg, syn.MATCH)
    b = load_match(cfg, syn.MATCH)              # from cache
    for k in a.arrays:
        np.testing.assert_array_equal(a.arrays[k], b.arrays[k])
    assert a.hits == b.hits and a.rallies == b.rallies
    cfg.raw["frame_shift"]["player_bbox"] = 0  # different settings -> silently rebuilt
    c = load_match(cfg, syn.MATCH)
    assert c.info["frame_shift"]["player_bbox"] == 0
    assert np.isnan(c.player_box_px[9]).all()


def test_schema_validation_catches_errors(md):
    md.validate()
    bad = copy.deepcopy(md)
    bad.arrays["shuttle_px"] = bad.arrays["shuttle_px"].astype(np.float64)
    with pytest.raises(SchemaError):
        bad.validate()
    bad = copy.deepcopy(md)
    bad.hits[0]["frame"] = 150
    with pytest.raises(SchemaError):
        bad.validate()
    bad = copy.deepcopy(md)
    bad.arrays["foot_src"][0, 0] = 7
    with pytest.raises(SchemaError):
        bad.validate()
