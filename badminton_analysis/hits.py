"""Hit-event baseline: shuttle trajectory change + wrist proximity + wrist speed.

All signals are in image space (pixels), so the detector also works on frames where the court
calibration is invalid. Per frame t inside a rally:

    accel(t)     = |(p[t+d] - p[t]) - (p[t] - p[t-d])| / d      shuttle velocity change (px/frame^2)
    prox_s(t)    = min over +-prox_window frames and over the two wrists of player s of
                   |wrist - shuttle| / box height of s          (dimensionless)
    wspeed_s(t)  = max over +-wrist_window frames of the faster wrist speed of s / box height
    hitter(t)    = player with the smaller prox

Candidates are local maxima of accel (radius `peak_radius`, within a rally). A candidate becomes
a hit when accel >= A, prox_hitter <= P and wspeed_hitter >= S; the predicted frame is
t + offset. Thresholds and offset are tuned on training matches only (see task2.py).
If the hitter pose is missing, the proximity / wrist tests cannot be applied; the candidate is
then kept only if `keep_without_pose` (a tuned boolean).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d

from .schema import BOTTOM, TOP, MatchData
from .shuttle import clean_track

LEFT_WRIST, RIGHT_WRIST = 9, 10


@dataclass(frozen=True)
class HitParams:
    accel: float
    proximity: float = np.inf
    wrist_speed: float = 0.0
    offset: int = 0
    keep_without_pose: bool = True


def _box_height(md: MatchData, rows: slice) -> np.ndarray:
    """(n, 2) player box height, pose box first, manual box as fallback."""
    pb = md.pose_box_px[rows].astype(np.float64)
    mb = md.player_box_px[rows].astype(np.float64)
    h = pb[..., 3] - pb[..., 1]
    return np.where(np.isnan(h), mb[..., 3] - mb[..., 1], h)


def _window(fn, x: np.ndarray, r: int) -> np.ndarray:
    """Sliding min/max over +-r frames ignoring NaN (NaN if the whole window is NaN)."""
    n = len(x)
    out = np.full(n, np.nan)
    pad = np.r_[np.full(r, np.nan), x, np.full(r, np.nan)]
    stack = np.stack([pad[k:k + n] for k in range(2 * r + 1)])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = fn(stack, axis=0)
    return out


def frame_features(md: MatchData, cfg: dict, min_score: float = 0.3) -> dict[str, np.ndarray]:
    """Per-frame signals for the whole match (arrays aligned with md rows)."""
    F = md.n_frames
    feats = {k: np.full(F, np.nan) for k in ("accel", "turn_deg", "speed")}
    for k in ("prox", "wspeed"):
        feats[k] = np.full((F, 2), np.nan)
    feats["shuttle_clean"] = np.full((F, 2), np.nan)
    feats["outlier"] = np.zeros(F, bool)
    feats["interp"] = np.zeros(F, bool)
    d = int(cfg["derivative_step"])
    for ri in range(len(md.rallies)):
        sl = md.rally_slice(ri)
        n = sl.stop - sl.start
        if n < 2 * d + 1:
            continue
        xy, outl, interp = clean_track(md.shuttle_px[sl], cfg["outlier_px"], cfg["outlier_speed_frac"],
                                       cfg["max_interp_gap"])
        feats["shuttle_clean"][sl], feats["outlier"][sl], feats["interp"][sl] = xy, outl, interp
        v_in = np.full((n, 2), np.nan); v_out = np.full((n, 2), np.nan)
        v_in[d:] = xy[d:] - xy[:-d]
        v_out[:-d] = xy[d:] - xy[:-d]
        acc = np.linalg.norm(v_out - v_in, axis=1) / d
        feats["accel"][sl] = acc
        cosang = np.sum(v_in * v_out, axis=1) / (np.linalg.norm(v_in, axis=1) * np.linalg.norm(v_out, axis=1) + 1e-9)
        feats["turn_deg"][sl] = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        feats["speed"][sl] = np.linalg.norm(v_out, axis=1) / d
        # pose-based signals
        bh = _box_height(md, sl)
        kp = md.pose_kp_px[sl][:, :, [LEFT_WRIST, RIGHT_WRIST]].astype(np.float64)  # (n, 2, 2, 3)
        wr = np.where(kp[..., 2:3] >= min_score, kp[..., :2], np.nan)
        for s in (TOP, BOTTOM):
            dist = np.linalg.norm(wr[:, s] - xy[:, None, :], axis=-1)              # (n, 2 wrists)
            dmin = np.where(np.isnan(dist).all(axis=1), np.nan, np.nanmin(np.where(np.isnan(dist), np.inf, dist), axis=1))
            feats["prox"][sl, s] = _window(np.nanmin, dmin, int(cfg["prox_window"])) / bh[:, s]
            sp = np.full((n, 2), np.nan)
            sp[1:] = np.linalg.norm(np.diff(wr[:, s], axis=0), axis=-1)
            smax = np.where(np.isnan(sp).all(axis=1), np.nan, np.nanmax(np.where(np.isnan(sp), -np.inf, sp), axis=1))
            feats["wspeed"][sl, s] = _window(np.nanmax, smax, int(cfg["wrist_window"])) / bh[:, s]
    return feats


def candidates(md: MatchData, feats: dict, peak_radius: int) -> dict[str, np.ndarray]:
    """Local maxima of accel within each rally. Returns arrays over candidates:
    row, accel, prox (hitter), wspeed (hitter), hitter (0 top / 1 bottom / -1 unknown)."""
    rows = []
    for ri in range(len(md.rallies)):
        sl = md.rally_slice(ri)
        a = feats["accel"][sl]
        a0 = np.where(np.isnan(a), -np.inf, a)
        mx = maximum_filter1d(a0, size=2 * peak_radius + 1, mode="constant", cval=-np.inf)
        peak = np.isfinite(a0) & (a0 == mx)
        # plateau: keep the first frame of equal consecutive maxima
        idx = np.flatnonzero(peak)
        if len(idx):
            keep = np.r_[True, ~((np.diff(idx) <= peak_radius) & (a0[idx[1:]] == a0[idx[:-1]]))]
            rows.append(sl.start + idx[keep])
    r = np.concatenate(rows) if rows else np.zeros(0, int)
    prox = feats["prox"][r]
    known = ~np.isnan(prox).all(axis=1)
    hitter = np.where(known, np.nanargmin(np.where(np.isnan(prox), np.inf, prox), axis=1), -1)
    hp = np.where(known, prox[np.arange(len(r)), np.clip(hitter, 0, 1)], np.nan)
    hw = np.where(known, feats["wspeed"][r, np.clip(hitter, 0, 1)], np.nan)
    bh = _box_height(md, slice(0, md.n_frames))[r, np.clip(hitter, 0, 1)] if len(r) else np.zeros(0)
    return {"row": r, "accel": feats["accel"][r], "prox": hp, "wspeed": hw, "hitter": hitter,
            "turn_deg": feats["turn_deg"][r], "speed": feats["speed"][r], "box_h": np.where(known, bh, np.nan)}


FEATURE_NAMES = ("log_accel", "turn_deg", "log_speed", "prox", "prox_missing", "wspeed", "log_box_h")


def candidate_features(cand: dict) -> np.ndarray:
    """Design matrix (n, 7) for the learned scorer; missing pose -> neutral values + flag."""
    miss = np.isnan(cand["prox"]).astype(float)
    return np.c_[
        np.log1p(np.nan_to_num(cand["accel"])),
        np.nan_to_num(cand["turn_deg"]) / 180.0,
        np.log1p(np.nan_to_num(cand["speed"])),
        np.clip(np.nan_to_num(cand["prox"], nan=3.0), 0, 3.0),
        miss,
        np.clip(np.nan_to_num(cand["wspeed"], nan=0.0), 0, 0.5),
        np.log(np.nan_to_num(cand["box_h"], nan=150.0)),
    ]


def select(cand: dict, p: HitParams) -> np.ndarray:
    """Boolean mask of candidates accepted under parameters p."""
    ok = cand["accel"] >= p.accel
    has_pose = ~np.isnan(cand["prox"])
    pose_ok = (cand["prox"] <= p.proximity) & (np.nan_to_num(cand["wspeed"], nan=-1) >= p.wrist_speed)
    return ok & np.where(has_pose, pose_ok, p.keep_without_pose)


def predict(md: MatchData, cand: dict, p: HitParams) -> dict[str, np.ndarray]:
    """Predicted hits: frame (sorted), row, hitter side, accel score."""
    m = select(cand, p)
    rows = cand["row"][m]
    frames = md.frame[rows] + p.offset
    order = np.argsort(frames, kind="stable")
    return {"frame": frames[order], "row": rows[order], "hitter": cand["hitter"][m][order],
            "accel": cand["accel"][m][order]}


# --------------------------------------------------------------------------- pose-only baseline
def pose_only_candidates(md: MatchData, feats: dict, peak_radius: int) -> dict[str, np.ndarray]:
    """Baseline without the shuttle: local maxima of the larger normalized wrist speed."""
    rows, hit = [], []
    ws = feats["wspeed"]
    best = np.where(np.isnan(ws).all(axis=1), np.nan, np.nanmax(np.where(np.isnan(ws), -np.inf, ws), axis=1))
    for ri in range(len(md.rallies)):
        sl = md.rally_slice(ri)
        a0 = np.where(np.isnan(best[sl]), -np.inf, best[sl])
        mx = maximum_filter1d(a0, size=2 * peak_radius + 1, mode="constant", cval=-np.inf)
        idx = np.flatnonzero(np.isfinite(a0) & (a0 == mx))
        if len(idx):
            keep = np.r_[True, np.diff(idx) > peak_radius]
            rows.append(sl.start + idx[keep])
    r = np.concatenate(rows) if rows else np.zeros(0, int)
    hitter = np.nanargmax(np.where(np.isnan(ws[r]), -np.inf, ws[r]), axis=1) if len(r) else np.zeros(0, int)
    # reuse the same selection logic: "accel" slot holds the wrist-speed score
    return {"row": r, "accel": best[r], "prox": np.zeros(len(r)), "wspeed": best[r], "hitter": hitter}
