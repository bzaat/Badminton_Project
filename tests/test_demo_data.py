"""Demo data helpers on the synthetic match (no Streamlit needed)."""
import numpy as np
import pandas as pd

from badminton_analysis.demo_data import load_outputs, rally_hits, rally_label, rally_tracks, shuttle_speed

from . import synthetic_bfmd as syn
from .test_build_synthetic import cfg, md, truth  # noqa: F401  (pytest fixtures)


def test_rally_hits_joins_detections_and_shot_predictions(md):  # noqa: F811
    ri = next(i for i, r in enumerate(md.rallies) if r["n_hits"] > 0)
    hits = [h for h in md.hits if h["rally_idx"] == ri]
    f0 = int(hits[0]["frame"])
    r = md.rallies[ri]
    t2 = pd.DataFrame({"match": [md.name, md.name, "OTHER"], "frame": [f0 + 1, r["start"], f0],
                       "hitter": ["top"] * 3, "score": [0.9] * 3, "matched_gt_frame": [f0, np.nan, f0]})
    t3 = pd.DataFrame({"match": [md.name], "frame": [f0], "player": ["x"], "player_in_training": [True],
                       "gt": [hits[0]["shot_type"]], "pred": [hits[0]["shot_type"]], "rally_flagged": [False]})
    table, fps = rally_hits(md, ri, t2, t3)
    assert len(table) == len(hits)
    first = table.iloc[0]
    assert first["detected"] and first["detection frame"] == f0 + 1 and bool(first["shot correct"])
    assert fps == [r["start"]]                                  # the unmatched detection, other match ignored
    if len(table) > 1:
        assert not table.iloc[1]["detected"] and table.iloc[1]["shot (predicted)"] == "—"


def test_rally_hits_without_outputs(md):  # noqa: F811
    ri = next(i for i, r in enumerate(md.rallies) if r["n_hits"] > 0)
    table, fps = rally_hits(md, ri, None, None)
    assert fps == [] and not table["detected"].any()


def test_tracks_speed_and_label(md):  # noqa: F811
    ri = 0
    tr = rally_tracks(md, ri)
    sl = md.rally_slice(ri)
    assert len(tr["frame"]) == sl.stop - sl.start and tr["top_m"].shape == (len(tr["frame"]), 2)
    v = shuttle_speed(np.array([[0, 0], [3, 4], [np.nan, np.nan], [6, 8]], float))
    assert np.isnan(v[0]) and v[1] == 5 and np.isnan(v[2]) and np.isnan(v[3])
    assert rally_label({"game": 1, "index": 0, "n_hits": 3, "winner": "A", "suspect": True}).startswith("Game 1 · rally 1")


def test_load_outputs_missing_dir(tmp_path):
    o = load_outputs(tmp_path)
    assert all(v is None for v in o.values())
