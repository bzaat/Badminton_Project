"""Per-hit features for shot-type classification (skeleton + shuttle trajectory + court position).

A hit event is (frame, hitter side). Features are hand-crafted and named so models stay
interpretable. Conventions:
    bh       hitter box height in pixels (pose box, else manual box): the unit for image distances
    fwd      image-y direction towards the opponent (+1 for the top player, -1 for the bottom one)
    mx       mirror so x is in the hitter's body frame: the top player faces the camera, so image
             right is their left (mx = -1); the bottom player faces away (mx = +1)
    court    metres; only used on main-camera frames (court_flag == 0), else NaN
Uses information after the hit too (shuttle flight, time to the next hit): this is an offline,
whole-rally classifier, not a real-time one.
"""
from __future__ import annotations

import numpy as np

from .schema import TOP, MatchData

UPPER_BODY = {"nose": 0, "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8, "l_wrist": 9, "r_wrist": 10}
L_HIP, R_HIP = 11, 12

FEATURE_GROUPS: dict[str, list[str]] = {
    "context": ["is_top", "hit_index", "is_first", "dt_prev", "dt_next"],
    "shuttle": ["sh_after_vx", "sh_after_vfwd", "sh_after_speed", "sh_before_vx", "sh_before_vfwd",
                "sh_before_speed", "sh_long_vfwd", "sh_rel_x", "sh_rel_y", "sh_rise"],
    "position": ["hit_depth", "hit_x_own", "opp_depth", "opp_x_own", "next_depth", "next_x_own"],
    "pose": [f"{n}_{a}" for n in UPPER_BODY for a in ("x", "y")] + ["wrist_above_shoulder", "wrist_speed"],
}
FEATURE_NAMES = [n for g in FEATURE_GROUPS.values() for n in g]

FINE_CLASSES = ["serve", "flick_serve", "clear", "drop", "smash", "drive", "net_shot", "net_kill",
                "lift", "push", "press", "block"]
MERGE = {"flick_serve": "serve", "net_kill": "net_attack", "press": "net_attack"}


def merged_label(label: str) -> str:
    return MERGE.get(label, label)


def _box(md: MatchData, row: int, s: int) -> np.ndarray:
    b = md.pose_box_px[row, s].astype(np.float64)
    return b if not np.isnan(b).any() else md.player_box_px[row, s].astype(np.float64)


def _mean_velocity(xy: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Average velocity (px/frame) between the first and last valid points of xy[lo:hi]."""
    idx = np.arange(max(lo, 0), min(hi, len(xy)))
    idx = idx[~np.isnan(xy[idx, 0])] if len(idx) else idx
    if len(idx) < 2:
        return np.array([np.nan, np.nan])
    return (xy[idx[-1]] - xy[idx[0]]) / (idx[-1] - idx[0])


def _court_feet(md: MatchData, row: int, s: int, net_y: float, width: float) -> tuple[float, float]:
    """(distance from net, x from the player's own left sideline) in metres, or NaN."""
    if row < 0 or md.court_flag[row] != 0 or md.foot_src[row, s] < 0:
        return np.nan, np.nan
    x, y = md.foot_court[row, s].astype(np.float64)
    return abs(y - net_y), (x if s != TOP else width - x)


def event_features(md: MatchData, feats: dict, events: list[dict], min_score: float = 0.3,
                   net_y: float = 6.7, width: float = 6.1) -> np.ndarray:
    """Feature matrix (len(events), len(FEATURE_NAMES)).

    events: dicts with frame, side (0 top / 1 bottom), rally_idx; any order.
    """
    X = np.full((len(events), len(FEATURE_NAMES)), np.nan)
    col = {n: i for i, n in enumerate(FEATURE_NAMES)}
    order = sorted(range(len(events)), key=lambda i: events[i]["frame"])
    by_rally: dict[int, list[int]] = {}
    for i in order:
        by_rally.setdefault(events[i]["rally_idx"], []).append(i)
    sh = feats["shuttle_clean"]
    for ri, idxs in by_rally.items():
        sl = md.rally_slice(ri)
        xy = sh[sl]
        for k, i in enumerate(idxs):
            e = events[i]
            s = int(e["side"]); o = 1 - s
            f = X[i]
            row = int(md.rows_for_frames(np.array([e["frame"]]))[0])
            if row < 0:  # predicted frame outside the rally: clip to the rally
                row = int(np.clip(np.searchsorted(md.frame, e["frame"]), sl.start, sl.stop - 1))
            r = row - sl.start
            fwd = 1.0 if s == TOP else -1.0
            mx = -1.0 if s == TOP else 1.0
            # context
            f[col["is_top"]] = float(s == TOP)
            f[col["hit_index"]] = k
            f[col["is_first"]] = float(k == 0)
            f[col["dt_prev"]] = e["frame"] - events[idxs[k - 1]]["frame"] if k > 0 else np.nan
            f[col["dt_next"]] = events[idxs[k + 1]]["frame"] - e["frame"] if k + 1 < len(idxs) else np.nan
            # shuttle (image space, in hitter box heights)
            b = _box(md, row, s)
            bh = b[3] - b[1]
            if np.isfinite(bh) and bh > 1:
                va, vb, vl = _mean_velocity(xy, r + 1, r + 7), _mean_velocity(xy, r - 6, r), _mean_velocity(xy, r + 1, r + 13)
                f[col["sh_after_vx"]], f[col["sh_after_vfwd"]] = va[0] * mx / bh, va[1] * fwd / bh
                f[col["sh_after_speed"]] = np.linalg.norm(va) / bh
                f[col["sh_before_vx"]], f[col["sh_before_vfwd"]] = vb[0] * mx / bh, vb[1] * fwd / bh
                f[col["sh_before_speed"]] = np.linalg.norm(vb) / bh
                f[col["sh_long_vfwd"]] = vl[1] * fwd / bh
                near = [j for j in (r, r - 1, r + 1, r - 2, r + 2) if 0 <= j < len(xy) and not np.isnan(xy[j, 0])]
                if near:
                    p = xy[near[0]]
                    f[col["sh_rel_x"]] = (p[0] - (b[0] + b[2]) / 2) * mx / bh
                    f[col["sh_rel_y"]] = (p[1] - b[1]) / bh
                    end = min(r + 45, len(xy) if k + 1 == len(idxs) else events[idxs[k + 1]]["frame"] - md.frame[sl.start])
                    seg = xy[near[0]:max(end, near[0] + 1), 1]
                    if np.isfinite(seg).any():
                        f[col["sh_rise"]] = (p[1] - np.nanmin(seg)) / bh  # >0: shuttle rises in the image
            # court positions
            f[col["hit_depth"]], f[col["hit_x_own"]] = _court_feet(md, row, s, net_y, width)
            f[col["opp_depth"]], f[col["opp_x_own"]] = _court_feet(md, row, o, net_y, width)
            if k + 1 < len(idxs):
                n = events[idxs[k + 1]]
                nrow = int(md.rows_for_frames(np.array([n["frame"]]))[0])
                f[col["next_depth"]], f[col["next_x_own"]] = _court_feet(md, nrow, int(n["side"]), net_y, width)
            # pose of the hitter
            if md.pose_assign[row, s] > 0 and np.isfinite(bh) and bh > 1:
                kp = md.pose_kp_px[row, s].astype(np.float64)
                ok = kp[:, 2] >= min_score
                if ok[L_HIP] or ok[R_HIP]:
                    hip = kp[[j for j in (L_HIP, R_HIP) if ok[j]], :2].mean(axis=0)
                    for name, j in UPPER_BODY.items():
                        if ok[j]:
                            f[col[f"{name}_x"]] = (kp[j, 0] - hip[0]) * mx / bh
                            f[col[f"{name}_y"]] = (kp[j, 1] - hip[1]) / bh
                sh_y = [kp[j, 1] for j in (5, 6) if ok[j]]
                wr_y = [kp[j, 1] for j in (9, 10) if ok[j]]
                if sh_y and wr_y:
                    f[col["wrist_above_shoulder"]] = (np.mean(sh_y) - min(wr_y)) / bh
            f[col["wrist_speed"]] = feats["wspeed"][row, s]
    return X
