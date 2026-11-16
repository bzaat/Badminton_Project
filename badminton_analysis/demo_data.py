"""Data helpers for the Streamlit demo (demo/app.py). No Streamlit import here, so it is testable."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import SIDE_INDEX, MatchData


def load_outputs(output_dir: str | Path) -> dict:
    """Task 2/3/4 result files. Missing files give None (the demo then hides that part)."""
    out = Path(output_dir)

    def csv(p):
        return pd.read_csv(out / p) if (out / p).exists() else None

    def js(p):
        return json.loads((out / p).read_text()) if (out / p).exists() else None

    return {
        "t2_pred": csv("task2/predictions.csv"),
        "t3_pred": csv("task3/predictions.csv"),
        "t4_summary": csv("task4/player_match_summary.csv"),
        "t4_rally": csv("task4/rally_player_movement.csv"),
        "t2_metrics": js("task2/metrics.json"),
        "t3_metrics": js("task3/metrics.json"),
    }


def rally_hits(md: MatchData, rally_idx: int, t2_pred: pd.DataFrame | None, t3_pred: pd.DataFrame | None,
               tol: int = 2) -> tuple[pd.DataFrame, list[int]]:
    """Annotated hits of one rally with the Task 2 detection and Task 3 prediction for each.

    A detection counts for a hit when Task 2 matched it to that hit (`matched_gt_frame`), which
    is the one-to-one matching used in the evaluation at the primary tolerance. Returns the hit
    table and the frames of detections in the rally that matched no hit (false positives).
    """
    r = md.rallies[rally_idx]
    hits = [h for h in md.hits if h["rally_idx"] == rally_idx]
    det = pd.DataFrame(columns=["frame", "matched_gt_frame"])
    if t2_pred is not None:
        d = t2_pred[(t2_pred["match"] == md.name) & t2_pred["frame"].between(r["start"] - tol, r["end"] + tol)]
        det = d[["frame", "matched_gt_frame"]]
    by_gt = {int(g): int(f) for f, g in zip(det["frame"], det["matched_gt_frame"]) if pd.notna(g)}
    shot = {}
    if t3_pred is not None:
        s = t3_pred[t3_pred["match"] == md.name]
        shot = dict(zip(s["frame"].astype(int), s["pred"]))
    rows = []
    for i, h in enumerate(hits):
        f = int(h["frame"])
        rows.append({
            "#": i + 1, "frame": f, "side": h["side"], "player": h.get("player", ""),
            "shot (annotated)": h["shot_type"], "shot (predicted)": shot.get(f, "—"),
            "shot correct": shot.get(f) == h["shot_type"] if f in shot else None,
            "detected": f in by_gt, "detection frame": by_gt.get(f),
        })
    fps = sorted(int(f) for f, g in zip(det["frame"], det["matched_gt_frame"]) if pd.isna(g))
    return pd.DataFrame(rows), fps


def rally_tracks(md: MatchData, rally_idx: int) -> dict[str, np.ndarray]:
    """Per-frame arrays of one rally: frames, court positions (m) per side, shuttle (px)."""
    sl = md.rally_slice(rally_idx)
    return {
        "frame": md.frame[sl],
        "top_m": md.foot_court[sl, SIDE_INDEX["top"]].astype(float),
        "bottom_m": md.foot_court[sl, SIDE_INDEX["bottom"]].astype(float),
        "shuttle_px": md.shuttle_px[sl].astype(float),
        "court_flag": md.court_flag[sl],
    }


def shuttle_speed(px: np.ndarray) -> np.ndarray:
    """Shuttle image speed in px/frame (NaN where either frame is missing)."""
    v = np.linalg.norm(np.diff(px, axis=0), axis=1)
    return np.r_[np.nan, v]


def rally_label(r: dict) -> str:
    flags = []
    if r.get("suspect"):
        flags.append("suspect camera")
    if not r.get("starts_with_serve", True):
        flags.append("no serve")
    tail = f" · {', '.join(flags)}" if flags else ""
    return f"Game {r['game']} · rally {r['index'] + 1} · {r['n_hits']} hits · winner {r.get('winner') or '?'}{tail}"
