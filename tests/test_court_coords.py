"""Unit tests for coordinate conversions and court homographies (no data needed)."""
import numpy as np
import pytest

from badminton_analysis.coords import box_iou, normalized_to_px, percent_box_center, percent_box_to_xyxy
from badminton_analysis.court import (
    CourtCalibration,
    CourtModel,
    apply_homography,
    homography_from_points,
)

W, H = 1280, 720


# ----------------------------------------------------------------------------- coords
def test_percent_box_to_xyxy():
    b = percent_box_to_xyxy(50, 50, 10, 20, W, H)
    np.testing.assert_allclose(b, [640, 360, 768, 504])


def test_percent_box_center_matches_bfmd_formula():
    # BFMD README: centre = (x + w/2, y + h/2) in percent
    x, y, w, h = 67.8125, 31.3889, 0.8, 1.4222
    c = percent_box_center(x, y, w, h, W, H)
    np.testing.assert_allclose(c, [(x + w / 2) * W / 100, (y + h / 2) * H / 100])


def test_normalized_to_px_keeps_scores():
    kp = np.array([[0.5, 0.25, 0.9], [1.0, 1.0, 0.1]])
    out = normalized_to_px(kp, W, H)
    np.testing.assert_allclose(out, [[640, 180, 0.9], [1280, 720, 0.1]])
    assert kp[0, 0] == 0.5  # input not modified


def test_box_iou():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30], [np.nan] * 4])
    np.testing.assert_allclose(box_iou(a, b)[0], [1.0, 25 / 175, 0.0, 0.0])


# ----------------------------------------------------------------------------- court model
def test_court_dimensions_follow_bwf_laws():
    m = CourtModel()
    assert m.net_y == pytest.approx(6.70)
    assert m.singles_margin == pytest.approx(0.46)
    lines = dict(m.lines())
    np.testing.assert_allclose(lines["short_service_far"][:, 1], 6.70 - 1.98)
    np.testing.assert_allclose(lines["long_service_doubles_near"][:, 1], 13.40 - 0.76)
    np.testing.assert_allclose(lines["singles_sideline_right"][:, 0], 6.10 - 0.46)


@pytest.mark.parametrize("which", ["doubles_outer", "singles_outer", "doubles_long_service"])
def test_corner_definitions_are_rectangles(which):
    c = CourtModel().corners(which)
    assert c.shape == (4, 2)
    # TL, TR, BR, BL
    assert c[0, 1] == c[1, 1] and c[2, 1] == c[3, 1] and c[0, 0] == c[3, 0] and c[1, 0] == c[2, 0]


# ----------------------------------------------------------------------------- homography
def _random_homography(rng):
    H = np.eye(3) + rng.normal(scale=[[0.2, 0.2, 50], [0.2, 0.2, 50], [1e-4, 1e-4, 0]])
    return H / H[2, 2]


def test_homography_exact_from_four_points():
    rng = np.random.default_rng(0)
    for _ in range(20):
        Htrue = _random_homography(rng)
        src = np.array([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]])
        dst = apply_homography(Htrue, src)
        Hest = homography_from_points(src, dst)
        pts = rng.uniform([0, 0], [6.1, 13.4], size=(50, 2))
        np.testing.assert_allclose(apply_homography(Hest, pts), apply_homography(Htrue, pts), atol=1e-6)


def test_homography_least_squares_with_noise():
    rng = np.random.default_rng(1)
    Htrue = _random_homography(rng)
    src = rng.uniform(0, 10, size=(40, 2))
    dst = apply_homography(Htrue, src) + rng.normal(scale=1e-3, size=(40, 2))
    Hest = homography_from_points(src, dst)
    assert np.abs(apply_homography(Hest, src) - apply_homography(Htrue, src)).max() < 1e-2


def test_apply_homography_shapes_and_nan():
    H = np.eye(3)
    pts = np.full((3, 2, 2), np.nan)
    pts[0, 0] = [1, 2]
    out = apply_homography(H, pts)
    assert out.shape == (3, 2, 2)
    np.testing.assert_allclose(out[0, 0], [1, 2])
    assert np.isnan(out[1]).all()


def _synthetic_calibration():
    """A perspective view of the court built from a known camera homography."""
    model = CourtModel()
    H_c2i = np.array([[60.0, -8.0, 450.0], [0.0, 20.0, 290.0], [0.0, -0.035, 1.0]])
    corners_px = apply_homography(H_c2i, model.corners("doubles_outer"))
    net = apply_homography(H_c2i, np.array([[0, model.net_y], [model.width_doubles, model.net_y]]))
    calib = CourtCalibration(corners_px, {"left_base": net[0], "right_base": net[1]}, model)
    return calib, H_c2i


def test_calibration_round_trip():
    calib, _ = _synthetic_calibration()
    pts = np.random.default_rng(2).uniform([0, 0], [6.1, 13.4], size=(100, 2))
    np.testing.assert_allclose(calib.to_court(calib.to_image(pts)), pts, atol=1e-6)


def test_calibration_corners_and_net():
    calib, _ = _synthetic_calibration()
    np.testing.assert_allclose(calib.to_court(calib.corners_px), CourtModel().corners(), atol=1e-6)
    np.testing.assert_allclose(calib.net_base_court(), [[0, 6.7], [6.1, 6.7]], atol=1e-6)


def test_perspective_far_side_is_compressed():
    """In the image, 1 m on the far half must span fewer pixels than 1 m on the near half."""
    calib, _ = _synthetic_calibration()
    far = calib.to_image(np.array([[3.05, 1.0], [3.05, 2.0]]))
    near = calib.to_image(np.array([[3.05, 11.4], [3.05, 12.4]]))
    assert np.linalg.norm(far[1] - far[0]) < np.linalg.norm(near[1] - near[0])
