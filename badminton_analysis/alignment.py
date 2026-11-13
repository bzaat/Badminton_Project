"""Measure frame offsets between BFMD annotation sources, using the raw files only.

Pose was estimated on decoded video frames, so it is used as the reference clock:

* player_bbox vs pose: mean IoU between the manual box at raw frame f + s and the pose box at
  frame f, for shifts s. The best s means "manual frame f + s shows the same image as pose
  frame f", so player_bbox needs shift -s to be aligned to pose.
* shuttle vs pose: around each annotated hit, the closest approach between the shuttle and the
  hitter's nearer wrist. Shifting the shuttle track by the wrong amount makes the two tracks
  pass each other further apart, so the shift with the smallest median closest-approach
  distance is the best alignment. The same search also shows where the annotated hit frame
  sits relative to that closest approach.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .coords import box_iou, normalized_to_px
from .raw import iter_pose, load_hits, load_player_bbox, load_shuttle

LEFT_WRIST, RIGHT_WRIST = 9, 10


def _lookup(frames: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """values at `query` frames (NaN where the frame is absent). frames must be sorted."""
    out = np.full((len(query),) + values.shape[1:], np.nan)
    if len(frames) == 0:
        return out
    idx = np.clip(np.searchsorted(frames, query), 0, len(frames) - 1)
    ok = frames[idx] == query
    out[ok] = values[idx[ok]]
    return out


def collect_pose_by_side(ann_root: Path, match: str, width: int, height: int):
    """Pose frames with exactly 2 persons, split into top/bottom by bbox bottom edge.

    This geometric split is used here (rather than IoU with player_bbox) so the offset
    measurement does not depend on the thing being measured. Returns frames (P,),
    boxes (P, 2, 4) and keypoints in pixels (P, 2, 17, 3), index 0 = top.
    """
    frames, boxes, kps = [], [], []
    for _, f, persons in iter_pose(ann_root, match):
        if len(persons) != 2:
            continue
        persons = sorted(persons, key=lambda p: p[0][3])
        frames.append(f)
        boxes.append([p[0] for p in persons])
        kps.append([normalized_to_px(p[1], width, height) for p in persons])
    order = np.argsort(frames)
    return (np.asarray(frames, np.int64)[order], np.asarray(boxes)[order], np.asarray(kps)[order])


def bbox_shift_curve(pose_frames, pose_boxes, pb: dict, shifts=range(-3, 4)) -> dict[int, float]:
    """Mean IoU(pose box at f, manual box at f + s) over both sides, for each shift s."""
    out = {}
    for s in shifts:
        ious = []
        for k, side in enumerate(("top", "bottom")):
            fr, bx = pb[side]
            man = _lookup(fr, bx, pose_frames + s)
            ok = ~np.isnan(man[:, 0])
            iou = np.array([box_iou(a[None], b[None])[0, 0] for a, b in zip(pose_boxes[ok, k], man[ok])])
            ious.append(iou)
        out[s] = float(np.concatenate(ious).mean())
    return out


def shuttle_shift_curve(pose_frames, pose_kps, sh_frames, sh_xy, hits, shifts=range(-4, 5),
                        window: int = 6, min_score: float = 0.3):
    """Median closest-approach distance (px) between shuttle(f + s) and hitter wrist(f) within
    +-window frames of each hit, per shift s. Also returns, for the best shift, the
    distribution of (closest-approach frame - annotated hit frame)."""
    side_idx = {"top": 0, "bottom": 1}
    offs = np.arange(-window, window + 1)
    res, lags_by_shift = {}, {}
    for s in shifts:
        dmins, lags = [], []
        for h in hits:
            q = h["frame"] + offs
            kp = _lookup(pose_frames, pose_kps[:, side_idx[h["side"]]], q)  # (n, 17, 3)
            sh = _lookup(sh_frames, sh_xy, q + s)                          # (n, 2)
            wr = kp[:, [LEFT_WRIST, RIGHT_WRIST], :]
            wr_xy = np.where(wr[..., 2:3] >= min_score, wr[..., :2], np.nan)
            d = np.linalg.norm(wr_xy - sh[:, None, :], axis=-1)            # (n, 2)
            d = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
            if not np.isfinite(d).any():
                continue
            j = int(np.argmin(d))
            dmins.append(d[j])
            lags.append(int(offs[j]))
        res[s] = float(np.median(dmins)) if dmins else float("nan")
        lags_by_shift[s] = np.asarray(lags)
    best = min(res, key=lambda k: res[k])
    return res, best, lags_by_shift[best]


def measure_match(ann_root: Path, match: str, width: int, height: int) -> dict:
    pf, pbx, pkp = collect_pose_by_side(ann_root, match, width, height)
    pb = load_player_bbox(ann_root, match, width, height)
    shf, shxy = load_shuttle(ann_root, match, width, height)
    hits = load_hits(ann_root, match)
    bcurve = bbox_shift_curve(pf, pbx, pb)
    scurve, sbest, lags = shuttle_shift_curve(pf, pkp, shf, shxy, hits)
    return {
        "match": match,
        "bbox_iou_by_shift": bcurve,
        "bbox_best_shift": max(bcurve, key=lambda k: bcurve[k]),
        "shuttle_dist_by_shift": scurve,
        "shuttle_best_shift": sbest,
        "hit_lag_hist": {int(k): int(v) for k, v in zip(*np.unique(lags, return_counts=True))},
        "hit_lag_median": float(np.median(lags)) if len(lags) else float("nan"),
        "n_hits_used": int(len(lags)),
    }
