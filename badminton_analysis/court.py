"""Badminton court model and image <-> court homographies.

Court coordinates are metres. Origin = far-left corner of the outer (doubles) boundary as seen
from the main broadcast camera; x grows to the right, y grows towards the camera. So the top
(far) player lives at y < net_y and the bottom (near) player at y > net_y.

Dimensions follow the BWF Laws of Badminton: 13.40 m x 6.10 m (doubles), singles sidelines
0.46 m inside, short service lines 1.98 m from the net, doubles long service line 0.76 m inside
the back boundary. A homography only maps points on the ground plane; airborne points
(the shuttle, a wrist) must not be projected with it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")


@dataclass(frozen=True)
class CourtModel:
    length: float = 13.40
    width_doubles: float = 6.10
    width_singles: float = 5.18
    short_service_from_net: float = 1.98
    long_service_doubles_from_back: float = 0.76

    @classmethod
    def from_config(cls, court_cfg: dict) -> "CourtModel":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: float(v) for k, v in court_cfg.items() if k in keys})

    @property
    def net_y(self) -> float:
        return self.length / 2.0

    @property
    def singles_margin(self) -> float:
        return (self.width_doubles - self.width_singles) / 2.0

    def corners(self, which: str = "doubles_outer") -> np.ndarray:
        """Court-space corners (4, 2) in CORNER_ORDER for the line set the image corners mark."""
        W, L = self.width_doubles, self.length
        if which == "doubles_outer":
            x0, x1, y0, y1 = 0.0, W, 0.0, L
        elif which == "singles_outer":
            m = self.singles_margin
            x0, x1, y0, y1 = m, W - m, 0.0, L
        elif which == "doubles_long_service":
            d = self.long_service_doubles_from_back
            x0, x1, y0, y1 = 0.0, W, d, L - d
        else:
            raise ValueError(f"unknown corner definition {which!r}")
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)

    def lines(self) -> list[tuple[str, np.ndarray]]:
        """All painted lines plus the net, as (name, (2, 2) endpoints) in court metres."""
        W, L, n = self.width_doubles, self.length, self.net_y
        m, s, d = self.singles_margin, self.short_service_from_net, self.long_service_doubles_from_back
        seg = lambda x0, y0, x1, y1: np.array([[x0, y0], [x1, y1]], dtype=np.float64)  # noqa: E731
        return [
            ("doubles_sideline_left", seg(0, 0, 0, L)),
            ("doubles_sideline_right", seg(W, 0, W, L)),
            ("singles_sideline_left", seg(m, 0, m, L)),
            ("singles_sideline_right", seg(W - m, 0, W - m, L)),
            ("back_boundary_far", seg(0, 0, W, 0)),
            ("back_boundary_near", seg(0, L, W, L)),
            ("long_service_doubles_far", seg(0, d, W, d)),
            ("long_service_doubles_near", seg(0, L - d, W, L - d)),
            ("short_service_far", seg(0, n - s, W, n - s)),
            ("short_service_near", seg(0, n + s, W, n + s)),
            ("centre_far", seg(W / 2, 0, W / 2, n - s)),
            ("centre_near", seg(W / 2, n + s, W / 2, L)),
            ("net", seg(0, n, W, n)),
        ]


# --------------------------------------------------------------------------- homography
def homography_from_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Homography H (3, 3) with dst ~ H @ src, from N >= 4 point pairs (normalized DLT).

    With exactly 4 points (no 3 collinear) the fit is exact; with more it is least squares.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("need >= 4 matching point pairs")

    def normalizer(p: np.ndarray) -> np.ndarray:
        c = p.mean(axis=0)
        s = np.sqrt(2) / max(np.mean(np.linalg.norm(p - c, axis=1)), 1e-12)
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])

    Ts, Td = normalizer(src), normalizer(dst)
    ps = (Ts @ np.c_[src, np.ones(len(src))].T).T
    pd = (Td @ np.c_[dst, np.ones(len(dst))].T).T
    rows = []
    for (x, y, _), (u, v, _) in zip(ps, pd):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map points (..., 2) through H. NaN inputs give NaN outputs."""
    pts = np.asarray(pts, dtype=np.float64)
    flat = pts.reshape(-1, 2)
    hom = np.c_[flat, np.ones(len(flat))] @ H.T
    with np.errstate(invalid="ignore", divide="ignore"):
        out = hom[:, :2] / hom[:, 2:3]
    return out.reshape(pts.shape)


@dataclass
class CourtCalibration:
    """Static (main-camera) calibration of one match.

    corners_px: (4, 2) image corners in CORNER_ORDER
    net_px:     dict with left_base / right_base / left_top / right_top image points
    """

    corners_px: np.ndarray
    net_px: dict[str, np.ndarray]
    model: CourtModel = field(default_factory=CourtModel)
    corners_are: str = "doubles_outer"

    def __post_init__(self) -> None:
        self.corners_px = np.asarray(self.corners_px, dtype=np.float64).reshape(4, 2)
        court_pts = self.model.corners(self.corners_are)
        self.H_img2court = homography_from_points(self.corners_px, court_pts)
        self.H_court2img = np.linalg.inv(self.H_img2court)
        self.H_court2img /= self.H_court2img[2, 2]

    def to_court(self, px: np.ndarray) -> np.ndarray:
        return apply_homography(self.H_img2court, px)

    def to_image(self, m: np.ndarray) -> np.ndarray:
        return apply_homography(self.H_court2img, m)

    def net_base_court(self) -> np.ndarray:
        """Net post bases projected to the court (2, 2): should be (0, net_y) and (W, net_y)."""
        return self.to_court(np.stack([self.net_px["left_base"], self.net_px["right_base"]]))
