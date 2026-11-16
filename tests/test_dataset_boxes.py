"""badminton_dataset.py: shuttle boxes centred on the TrackNet point; stale caches rebuilt."""
import json

import numpy as np
import pytest

pytest.importorskip("cv2")
import badminton_dataset as bd  # noqa: E402

MATCH = "tiny-match"


def _write(ann, shuttle_xy_pct, player_xywh_pct):
    (ann / "player_bbox").mkdir(parents=True)
    (ann / "shuttle").mkdir(parents=True)
    x, y, w, h = player_xywh_pct
    pb = {"annotations": [{"result": [{"value": {"labels": ["Top_Player"], "sequence": [
        {"frame": 10, "x": x, "y": y, "width": w, "height": h, "enabled": True}]}}]}]}
    sx, sy = shuttle_xy_pct
    sh = {"predictions": [{"result": [{"value": {"sequence": [
        {"frame": 10, "x": sx, "y": sy, "width": 0.8, "height": 1.4222222, "rotation": 0, "enabled": True},
        {"frame": 11, "x": sx, "y": sy, "width": 0.8, "height": 1.4222222, "rotation": 0, "enabled": False},
    ]}}]}]}
    (ann / "player_bbox" / f"{MATCH}.json").write_text(json.dumps(pb))
    (ann / "shuttle" / f"{MATCH}.json").write_text(json.dumps(sh))


def test_shuttle_box_is_centred_on_xy_and_players_unchanged(tmp_path):
    _write(tmp_path, shuttle_xy_pct=(50.0, 25.0), player_xywh_pct=(10.0, 20.0, 5.0, 10.0))
    d = bd.parse_match(str(tmp_path), MATCH)
    np.testing.assert_array_equal(d["frames"], [10])          # disabled frame 11 dropped
    shuttle = d["boxes"][0, bd.SHUTTLE - 1]
    np.testing.assert_allclose((shuttle[:2] + shuttle[2:]) / 2, [640.0, 180.0], atol=1e-3)
    np.testing.assert_allclose(shuttle[2:] - shuttle[:2], [10.24, 10.24], atol=1e-3)
    # player boxes stay Label Studio top-left + size
    np.testing.assert_allclose(d["boxes"][0, 0], [128.0, 144.0, 192.0, 216.0], atol=1e-3)


def test_unversioned_cache_is_rebuilt(tmp_path):
    ann, cache = tmp_path / "ann", tmp_path / "cache"
    _write(ann, shuttle_xy_pct=(50.0, 25.0), player_xywh_pct=(10.0, 20.0, 5.0, 10.0))
    cache.mkdir()
    stale = cache / f"{MATCH}.npz"
    np.savez_compressed(stale, frames=np.array([10]), boxes=np.zeros((1, 3, 4), np.float32))  # v1: no key
    d = bd.load_match(str(ann), MATCH, str(cache))
    shuttle = d["boxes"][0, bd.SHUTTLE - 1]
    np.testing.assert_allclose((shuttle[:2] + shuttle[2:]) / 2, [640.0, 180.0], atol=1e-3)
    with np.load(stale) as z:
        assert int(z["version"]) == bd.CACHE_VERSION
    again = bd.load_match(str(ann), MATCH, str(cache))           # now served from the cache
    np.testing.assert_array_equal(again["boxes"], d["boxes"])
