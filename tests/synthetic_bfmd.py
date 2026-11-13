"""Writes a tiny BFMD-format annotation folder whose correct parse is known exactly."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from badminton_analysis.court import CourtModel, apply_homography

W, H = 1280, 720
MATCH = "TEST-Open-2025-Alpha-A-vs.-Beta-B-F"
DOUBLES = "TEST-Open-2024-Gamma-Delta-vs.-Eps-Zeta-QF"
H_C2I = np.array([[60.0, -8.0, 450.0], [0.0, 20.0, 290.0], [0.0, -0.035, 1.0]])  # court m -> px
RALLIES = [(100, 109), (200, 209)]
TOP_FOOT_M = np.array([3.05, 2.0])      # where the top player's ankles stand (court metres)
BOTTOM_FOOT_M = np.array([2.0, 9.0])
BBOX_SHIFT = 1                           # player_bbox raw frames are 1 behind the others


def court_px(pt_m) -> np.ndarray:
    return apply_homography(H_C2I, np.asarray(pt_m, float))


def box_around_foot(foot_px: np.ndarray, w: float = 60, h: float = 150) -> np.ndarray:
    return np.array([foot_px[0] - w / 2, foot_px[1] - h, foot_px[0] + w / 2, foot_px[1]])


def _pct(box: np.ndarray) -> dict:
    return {"x": box[0] / W * 100, "y": box[1] / H * 100,
            "width": (box[2] - box[0]) / W * 100, "height": (box[3] - box[1]) / H * 100}


def _kps(box: np.ndarray, ankle_px: np.ndarray, ankle_score: float = 0.9) -> list:
    kp = np.zeros((17, 3))
    kp[:, 0] = (box[0] + box[2]) / 2 / W
    kp[:, 1] = (box[1] + box[3]) / 2 / H
    kp[:, 2] = 0.8
    kp[15] = [ankle_px[0] / W, ankle_px[1] / H, ankle_score]
    kp[16] = [ankle_px[0] / W, ankle_px[1] / H, ankle_score]
    return kp.tolist()


def write(root: Path) -> dict:
    """Create <root>/annotations/... and return the ground truth used by the tests."""
    ann = root / "annotations"
    for d in ["metadata", "hit_inferred", "shot_type", "pose", "shuttle", "player_bbox",
              "court", "court_perframe", "hits_doubles"]:
        (ann / d).mkdir(parents=True, exist_ok=True)
    model = CourtModel()
    top_foot, bot_foot = court_px(TOP_FOOT_M), court_px(BOTTOM_FOOT_M)
    top_box, bot_box = box_around_foot(top_foot), box_around_foot(bot_foot)

    meta = [{
        "match_name": MATCH, "player1_name": "Alpha A", "player2_name": "Beta B",
        "game1_top_player": "player1", "match_winner": "player1",
        "games": {"game1": {
            "rallies": [{"start": s, "end": e} for s, e in RALLIES],
            "score_events": [
                {"rally_start": 100, "rally_end": 109, "winner": "player1",
                 "score_after_rally": {"player1": 1, "player2": 0}},
                {"rally_start": 200, "rally_end": 209, "winner": None,
                 "score_after_rally": {"player1": 1, "player2": 1}}]}},
    }]
    hits = [
        {"frame": 102, "shot_type": "serve", "side": "top", "player": "Alpha A", "game": 1, "rally": 0},
        {"frame": 106, "shot_type": "clear", "side": "bottom", "player": "Beta B", "game": 1, "rally": 0},
        {"frame": 203, "shot_type": "smash", "side": "bottom", "player": "Beta B", "game": 1, "rally": 1},
    ]
    timeline = {"annotations": [{"result": [
        {"value": {"ranges": [{"start": s, "end": e}], "timelinelabels": ["Rally"]}} for s, e in RALLIES
    ] + [{"value": {"ranges": [{"start": h["frame"], "end": h["frame"]}], "timelinelabels": ["hit"]}} for h in hits]}]}

    # player_bbox stored BBOX_SHIFT frames earlier than the true frame
    seq_top, seq_bot = [], []
    for s, e in RALLIES:
        for f in range(s, e + 1):
            seq_top.append({"frame": f - BBOX_SHIFT, "enabled": True, **_pct(top_box)})
            seq_bot.append({"frame": f - BBOX_SHIFT, "enabled": True, **_pct(bot_box)})
    player_bbox = {"annotations": [{"result": [
        {"value": {"labels": ["Top_Player"], "sequence": seq_top}},
        {"value": {"labels": ["Bottom_Player"], "sequence": seq_bot}}]}]}

    # pose: bottom person listed first; frame 105 has only the top player; frame 109 missing
    frames = {}
    for f in range(100, 109):
        persons = [{"bbox": bot_box.tolist(), "keypoints": _kps(bot_box, bot_foot)},
                   {"bbox": top_box.tolist(), "keypoints": _kps(top_box, top_foot)}]
        if f == 105:
            persons = persons[1:]
        frames[str(f)] = persons
    # rally 2: low ankle confidence -> foot falls back to box bottom-centre
    frames2 = {str(f): [{"bbox": top_box.tolist(), "keypoints": _kps(top_box, top_foot, 0.05)},
                        {"bbox": bot_box.tolist(), "keypoints": _kps(bot_box, bot_foot, 0.05)}]
               for f in range(200, 209)}
    pose = {"match_name": MATCH, "videos": {
        "rally_000_100_109": {"fps": 30.0, "width": W, "height": H, "frames": frames},
        "rally_001_200_209": {"fps": 30.0, "width": W, "height": H, "frames": frames2}}}

    sh_seq = []
    for f in range(100, 110):
        px = np.array([400 + 10 * (f - 100), 300.0])
        sh_seq.append({"frame": f, "enabled": True, "x": px[0] / W * 100 - 0.4, "y": px[1] / H * 100 - 0.7111,
                       "width": 0.8, "height": 1.4222, "rotation": 0})
    sh_seq[3].update({"x": 0, "y": 0})          # TrackNet "not visible" -> dropped
    sh_seq[4]["enabled"] = False               # disabled -> dropped
    shuttle = {"predictions": [{"result": [{"value": {"sequence": sh_seq}}]}]}

    corners = court_px(model.corners("doubles_outer"))
    net = court_px([[0, model.net_y], [model.width_doubles, model.net_y]])
    court = {"match_name": MATCH, "court": dict(zip(["top_left", "top_right", "bottom_right", "bottom_left"],
                                                  corners.tolist())),
             "net": {"left_base": net[0].tolist(), "right_base": net[1].tolist(),
                     "left_top": (net[0] - [0, 100]).tolist(), "right_top": (net[1] - [0, 100]).tolist()}}
    perframe = {"frames": [
        {"frame_idx": 90, "corners": corners.tolist(), "fit_score": 0.8, "flag": "valid"},
        {"frame_idx": 120, "corners": corners.tolist(), "fit_score": 0.8, "flag": "valid"},
        {"frame_idx": 205, "corners": None, "fit_score": 0.0, "flag": "no_court"}]}

    def dump(sub, name, obj, suffix=".json"):
        with open(ann / sub / f"{name}{suffix}", "w") as f:
            json.dump(obj, f)

    dump("metadata", MATCH, meta)
    dump("hit_inferred", MATCH, {"match_name": MATCH, "n_hits": len(hits), "hits": hits})
    dump("shot_type", MATCH, timeline)
    dump("pose", MATCH, pose)
    dump("shuttle", MATCH, shuttle)
    dump("player_bbox", MATCH, player_bbox)
    dump("court", MATCH, court, "_court.json")
    dump("court_perframe", MATCH, perframe)
    dump("hits_doubles", DOUBLES, timeline)
    # a doubles match that also has metadata must still be excluded
    dump("metadata", DOUBLES, meta)
    return {"top_box": top_box, "bot_box": bot_box, "top_foot_px": top_foot, "bot_foot_px": bot_foot}
