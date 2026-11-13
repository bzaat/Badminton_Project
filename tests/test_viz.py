"""Smoke tests for the visualiser on the synthetic match."""
import numpy as np

from badminton_analysis.build import build_match
from badminton_analysis.schema import TOP
from badminton_analysis.viz import plot_rally, wrist_speed

from . import synthetic_bfmd as syn
from .test_build_synthetic import cfg, truth  # noqa: F401  (pytest fixtures)


def test_plot_rally_writes_png(cfg, tmp_path):  # noqa: F811
    md = build_match(cfg, syn.MATCH)[0]
    for ri in range(len(md.rallies)):
        out = plot_rally(md, ri, tmp_path / f"r{ri}.png")
        data = open(out, "rb").read()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 10_000


def test_wrist_speed_static_player_is_zero(cfg):  # noqa: F811
    md = build_match(cfg, syn.MATCH)[0]
    v = wrist_speed(md, md.rally_slice(0), TOP, 0.3)
    assert np.isnan(v[0])
    np.testing.assert_allclose(v[1:5], 0.0, atol=1e-9)  # synthetic player does not move
