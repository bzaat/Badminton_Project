"""Shuttle track cleaning: outlier removal and short-gap interpolation (image space, pixels).

TrackNet occasionally fires on something that is not the shuttle for one or a few frames. Such a
point disagrees with the trajectory on *both* sides, whereas a real hit point still agrees with the
incoming side (and the next points agree with the outgoing side). So a point is an outlier when
neither the linear extrapolation from its two left neighbours nor the one from its two right
neighbours lands within a tolerance of it.
"""
from __future__ import annotations

import numpy as np


def find_outliers(xy: np.ndarray, valid: np.ndarray, tol_px: float, speed_frac: float) -> np.ndarray:
    """Boolean mask of outlier points among `valid` ones (one pass).

    Tolerance for a prediction made with velocity v: max(tol_px, speed_frac * |v|).
    A valid point with no valid neighbour within +-2 frames is also an outlier.
    """
    n = len(xy)
    out = np.zeros(n, dtype=bool)
    for t in np.flatnonzero(valid):
        errs = []
        if t >= 2 and valid[t - 1] and valid[t - 2]:
            v = xy[t - 1] - xy[t - 2]
            errs.append(np.linalg.norm(xy[t] - (xy[t - 1] + v)) - max(tol_px, speed_frac * np.linalg.norm(v)))
        if t + 2 < n and valid[t + 1] and valid[t + 2]:
            v = xy[t + 1] - xy[t + 2]
            errs.append(np.linalg.norm(xy[t] - (xy[t + 1] + v)) - max(tol_px, speed_frac * np.linalg.norm(v)))
        if errs:
            out[t] = min(errs) > 0
            continue
        # fewer than two neighbours on each side: compare with the nearest neighbour only
        nb = [k for k in (t - 2, t - 1, t + 1, t + 2) if 0 <= k < n and valid[k]]
        if not nb:
            out[t] = True
        else:
            k = min(nb, key=lambda k: abs(k - t))
            out[t] = np.linalg.norm(xy[t] - xy[k]) > 3 * tol_px * abs(k - t)
    return out


def interpolate_gaps(xy: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill NaN runs of length <= max_gap that have valid points on both sides.
    Returns (filled xy, mask of filled frames)."""
    xy = np.array(xy, dtype=np.float64, copy=True)
    valid = ~np.isnan(xy[:, 0])
    filled = np.zeros(len(xy), dtype=bool)
    idx = np.flatnonzero(valid)
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            w = (np.arange(1, gap + 1) / (gap + 1))[:, None]
            xy[a + 1:b] = xy[a] * (1 - w) + xy[b] * w
            filled[a + 1:b] = True
    return xy, filled


def clean_track(xy: np.ndarray, tol_px: float = 40.0, speed_frac: float = 0.5, max_gap: int = 5,
                iters: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove outliers (up to `iters` passes), then fill short gaps.

    Returns (clean xy (n, 2) with NaN where unknown, outlier mask, interpolated mask).
    """
    xy = np.array(xy, dtype=np.float64, copy=True)
    valid = ~np.isnan(xy[:, 0])
    outlier = np.zeros(len(xy), dtype=bool)
    for _ in range(iters):
        new = find_outliers(xy, valid, tol_px, speed_frac)
        if not new.any():
            break
        outlier |= new
        valid &= ~new
    xy[~valid] = np.nan
    xy, filled = interpolate_gaps(xy, max_gap)
    return xy, outlier, filled
