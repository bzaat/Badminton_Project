"""Coordinate conversions for the three conventions used inside BFMD.

| source                    | unit                                   |
|---------------------------|----------------------------------------|
| player_bbox, shuttle      | percent 0-100 of width/height, (x, y) = top-left, plus width/height |
| pose keypoints            | normalized 0-1 of width/height         |
| pose bbox, court corners  | pixels                                 |

Everything downstream works in pixels (image space) or metres (court space).
"""
from __future__ import annotations

import numpy as np


def percent_box_to_xyxy(x: float, y: float, w: float, h: float, width: int, height: int) -> np.ndarray:
    """Label Studio percent box (top-left x/y, width/height in %) -> pixel [x1, y1, x2, y2]."""
    x1 = x * width / 100.0
    y1 = y * height / 100.0
    return np.array([x1, y1, x1 + w * width / 100.0, y1 + h * height / 100.0], dtype=np.float64)


def percent_box_center(x: float, y: float, w: float, h: float, width: int, height: int) -> np.ndarray:
    """Centre of a percent box in pixels (used for the TrackNet shuttle position)."""
    b = percent_box_to_xyxy(x, y, w, h, width, height)
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


def normalized_to_px(xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalized (..., 2+) x/y in 0-1 -> pixels. Extra trailing columns (e.g. scores) are kept."""
    out = np.array(xy, dtype=np.float64, copy=True)
    out[..., 0] *= width
    out[..., 1] *= height
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N, 4) and b (M, 4), xyxy -> (N, M). NaN boxes give IoU 0."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = inter / union
    return np.nan_to_num(iou, nan=0.0)
