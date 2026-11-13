"""Regression tests on the real BFMD annotations (skipped when the package is not available).

Thresholds encode what Task 1 measured, so a code change that breaks the data layer shows up here.
"""
import json

import numpy as np
import pytest

from badminton_analysis.build import load_match, singles_matches
from badminton_analysis.config import load_config
from badminton_analysis.validate import serve_checks

cfg = load_config()
pytestmark = [pytest.mark.data,
              pytest.mark.skipif(not cfg.ann_root.is_dir(), reason=f"BFMD annotations not found at {cfg.ann_root}")]

MATCH = "KAPAL-API-Indonesia-Open-2025-Anders-Antonsen-DEN-3-vs.-Chou-Tien-Chen-TPE-6-F"


@pytest.fixture(scope="module")
def md():
    return load_match(cfg, MATCH)


def test_singles_filter_finds_the_12_matches():
    ms = singles_matches(cfg)
    assert len(ms) == 12 and MATCH in ms
    assert not any("2023" in m or "2024" in m for m in ms)  # doubles are from 2023/24


def test_schema_and_counts_match_raw(md):
    md.validate()
    raw_hits = json.load(open(cfg.ann_root / "hit_inferred" / f"{MATCH}.json"))["hits"]
    assert len(md.hits) == len(raw_hits) == 978
    assert len(md.rallies) == 77 and md.n_frames == 27054


def test_static_calibration_net_posts(md):
    err = md.calibration().net_base_court() - np.array([[0, 6.7], [6.1, 6.7]])
    assert np.abs(err[:, 0]).max() < 0.1   # posts on the doubles sidelines
    assert np.abs(err[:, 1]).max() < 0.35  # net line at mid-court


def test_serve_rules_hold(md):
    sc = serve_checks(md, md.calibration().model)
    assert sc["parity_ok"] / (sc["parity_ok"] + sc["parity_wrong"]) > 0.95
    d = sc["depth_median_m"]
    assert abs(d["server_top"] - d["server_bottom"]) < 0.4
    assert 2.0 < d["server_bottom"] < 4.5


def test_players_in_own_half(md):
    good = (md.court_flag == 0) & (md.foot_src == 0).all(axis=1)
    fy = md.foot_court[good, :, 1]
    assert (fy[:, 0] < 6.7).mean() > 0.98 and (fy[:, 1] > 6.7).mean() > 0.98
