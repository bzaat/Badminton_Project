"""Movement measurements in court coordinates (metres). Measurements only - no judgement.

Player's own frame (used for zones and recovery positions):
    depth    distance from the net, 0 at the net, 6.70 at the player's own back boundary
    lateral  signed offset from the centre line, positive = the player's RIGHT-hand side
             (bottom player faces away from the camera: right = +x; top player faces it: right = -x)

Tracks are smoothed with a centred moving average inside runs of valid frames; gaps up to
`max_gap` frames are filled linearly first. Speeds come from central differences of the
smoothed track; distance is the summed step length between consecutive valid frames.
"""
from __future__ import annotations

import numpy as np

from .schema import TOP


def own_frame(xy: np.ndarray, side: int, net_y: float = 6.7, centre_x: float = 3.05) -> tuple[np.ndarray, np.ndarray]:
    """(depth, lateral) in the player's own frame for court points xy (..., 2)."""
    xy = np.asarray(xy, dtype=np.float64)
    depth = np.abs(xy[..., 1] - net_y)
    lateral = (centre_x - xy[..., 0]) if side == TOP else (xy[..., 0] - centre_x)
    return depth, lateral


def _runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index ranges of consecutive True values."""
    v = np.r_[False, valid, False].astype(int)
    d = np.diff(v)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def fill_gaps(xy: np.ndarray, valid: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation over invalid runs of length <= max_gap between valid frames."""
    xy = np.array(xy, dtype=np.float64, copy=True)
    valid = valid.copy()
    idx = np.flatnonzero(valid)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            w = (np.arange(1, b - a) / (b - a))[:, None]
            xy[a + 1:b] = xy[a] * (1 - w) + xy[b] * w
            valid[a + 1:b] = True
    xy[~valid] = np.nan
    return xy, valid


def smooth(xy: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average (odd window) within each valid run. Near run edges the window
    shrinks symmetrically (so a straight-line track stays exactly on the line)."""
    out = np.full_like(xy, np.nan, dtype=np.float64)
    h = max(window, 1) // 2
    for a, b in _runs(valid):
        seg = xy[a:b]
        n = b - a
        if h == 0 or n < 2:
            out[a:b] = seg
            continue
        c = np.cumsum(np.vstack([np.zeros((1, 2)), seg]), axis=0)
        i = np.arange(n)
        hi_ = np.minimum(h, np.minimum(i, n - 1 - i))
        lo, hi = i - hi_, i + hi_ + 1
        out[a:b] = (c[hi] - c[lo]) / (hi - lo)[:, None]
    return out


def speed_and_distance(xy_s: np.ndarray, valid: np.ndarray, fps: float) -> tuple[np.ndarray, float]:
    """Speed (m/s, NaN outside runs of >= 3 frames) and total distance (m) of a smoothed track."""
    speed = np.full(len(xy_s), np.nan)
    dist = 0.0
    for a, b in _runs(valid):
        seg = xy_s[a:b]
        if b - a >= 2:
            dist += float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())
        if b - a >= 3:
            v = np.gradient(seg, axis=0) * fps
            speed[a:b] = np.linalg.norm(v, axis=1)
    return speed, dist


ZONE_DEPTH = ("front", "mid", "rear")
ZONE_LAT = ("left", "centre", "right")
ZONE_NAMES = [f"{d}_{l}" for d in ZONE_DEPTH for l in ZONE_LAT]


def zone_index(depth: np.ndarray, lateral: np.ndarray, half_length: float = 6.7, singles_width: float = 5.18,
               margin: float = 0.5) -> np.ndarray:
    """3x3 zone of each point in the player's own half (0..8, row-major front->rear, left->right).
    Points more than `margin` outside the singles court give -1; points slightly outside are
    clipped into the nearest edge zone (players often stand on or just behind a line)."""
    d = np.asarray(depth, float); lat = np.asarray(lateral, float)
    w2 = singles_width / 2
    inside = (d >= -margin) & (d <= half_length + margin) & (np.abs(lat) <= w2 + margin) & ~np.isnan(d) & ~np.isnan(lat)
    d, lat = np.nan_to_num(d), np.nan_to_num(lat)
    di =np.clip((np.clip(d, 0, half_length - 1e-9) / (half_length / 3)).astype(int), 0, 2)
    li = np.clip(((np.clip(lat, -w2, w2 - 1e-9) + w2) / (singles_width / 3)).astype(int), 0, 2)
    return np.where(inside, di * 3 + li, -1)


def player_track(md, rows: slice, side: int, mcfg: dict, fps: float):
    """Smoothed own-frame track of one player in one rally.

    Returns dict with xy_s (n, 2) court metres, valid (n,), depth, lateral, speed (m/s), distance (m).
    """
    src_ok = md.foot_src[rows, side] == 0 if not mcfg.get("use_box_fallback", False) else md.foot_src[rows, side] >= 0
    valid = (md.court_flag[rows] == 0) & src_ok
    xy = md.foot_court[rows, side].astype(np.float64)
    d, lat = own_frame(xy, side)
    plausible = (d <= mcfg["max_depth_m"]) & (np.abs(lat) <= mcfg["max_lateral_m"])
    valid &= np.nan_to_num(plausible, nan=False).astype(bool)
    xy, valid = fill_gaps(xy, valid, int(mcfg["max_gap"]))
    xy_s = smooth(xy, valid, int(mcfg["smooth_window"]))
    speed, dist = speed_and_distance(xy_s, valid, fps)
    depth, lateral = own_frame(xy_s, side)
    return {"xy_s": xy_s, "valid": valid, "depth": depth, "lateral": lateral, "speed": speed, "distance": dist}
