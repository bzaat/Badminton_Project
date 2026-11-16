"""Demo data helpers on the synthetic match (no Streamlit needed)."""
import numpy as np
import pandas as pd

from badminton_analysis.demo_data import (MOVEMENT_COLUMNS, load_outputs, rally_hits, rally_label, rally_tracks,
                                          results_summary, short_match_name, shuttle_speed)

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
    assert results_summary(o) is None


def test_results_summary_shapes_and_per_class_f1():
    cm = [[8, 2], [1, 9]]                       # rows annotated, columns predicted
    t2 = {"variants": {"logistic": {
        "micro": {"2": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "tp": 7, "fp": 2, "fn": 3}},
        "per_match": {"M": {"tol2": {"f1": 0.75, "tp": 7, "fp": 2, "fn": 3, "precision": 0.8}}}}}}
    t3 = {"schemes": {"fine": {"classes": ["a", "b"], "models": {"hgb": {"accuracy": 0.85, "macro_f1": 0.849}}}},
          "bootstrap_fine": {"hgb": {"macro_f1": [0.8, 0.9]}}, "confusion_fine": cm}
    mv = pd.DataFrame([{**{c: 1.0 for c in MOVEMENT_COLUMNS}, "match": "M", "player": "P", "zone_front_left": 0.2,
                        "extra_column": 5}])
    s = results_summary({"t2_metrics": t2, "t3_metrics": t3, "t4_summary": mv})
    assert s["hit"]["per_match"]["M"] == {"f1": 0.75, "tp": 7, "fp": 2, "fn": 3}
    # class a: precision 8/9, recall 8/10 -> F1 = 2*8/(9+10)
    assert abs(s["shot"]["per_class_f1"]["a"] - 16 / 19) < 1e-4
    row = s["movement"][0]
    assert row["zone_front_left"] == 0.2 and "extra_column" not in row      # only the listed columns


def test_short_match_name():
    assert short_match_name("VICTOR-China-Open-2025-Shi-Yu-Qi-CHN-3-vs.-Wang-Zheng-Xing-CHN-F") == \
        "China Open 2025 F: Shi Yu Qi vs Wang Zheng Xing"
    assert short_match_name("odd-name") == "odd-name"
