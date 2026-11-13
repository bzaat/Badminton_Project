"""Parsers for the raw BFMD annotation JSON (one function per annotation type).

All functions take `ann_root` = <BFMD_data>/annotations and a match name. They return raw frame
numbers exactly as stored (0-based decode order of the 30 fps broadcast video); any frame
shift is applied later in `build`. Pixel conversions use the video size from the caller.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .coords import percent_box_center, percent_box_to_xyxy
from .court import CORNER_ORDER

PERFRAME_FLAGS = {"valid": 0, "refit": 1, "no_court": 2}  # -1 = unknown
SIDES = ("top", "bottom")


def _read(path: Path) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_singles_matches(ann_root: Path) -> list[str]:
    """Matches with every annotation type Milestone 1 needs. In BFMD these are exactly the
    12 men's singles matches; doubles only exist in hits_doubles/ and are therefore excluded."""
    ann_root = Path(ann_root)
    need = {
        "hit_inferred": ".json", "metadata": ".json", "pose": ".json",
        "shuttle": ".json", "player_bbox": ".json", "court": "_court.json",
    }
    sets = []
    for sub, suffix in need.items():
        d = ann_root / sub
        names = {f[: -len(suffix)] for f in os.listdir(d) if f.endswith(suffix)} if d.is_dir() else set()
        sets.append(names)
    return sorted(set.intersection(*sets))


# --------------------------------------------------------------------------- metadata / rallies
@dataclass
class RawRally:
    game: int
    index: int          # 0-based rally index within the game (as used by hit_inferred)
    start: int          # first frame (inclusive)
    end: int            # last frame (inclusive)
    winner: str | None  # player name of the rally winner (see winner_from)
    score_after: dict[str, int] | None
    winner_from: str | None = None  # "metadata" (explicit field) or "score_diff" (score went up)


def load_metadata(ann_root: Path, match: str) -> dict:
    """Players, game1_top_player and all rallies (sorted by start frame).

    The explicit `winner` field of a score event is mostly null in BFMD (981 of 1,058), but
    `score_after_rally` is filled, so the winner is otherwise derived from which player's score
    went up versus the previous score event of the same game (0-0 at the start of a game).
    """
    md = _read(Path(ann_root) / "metadata" / f"{match}.json")
    md = md[0] if isinstance(md, list) else md
    names = {"player1": md["player1_name"], "player2": md["player2_name"]}
    rallies: list[RawRally] = []
    for gkey in sorted(md["games"], key=lambda s: int(s.replace("game", ""))):
        g = md["games"][gkey]
        gnum = int(gkey.replace("game", ""))
        events = {(e["rally_start"], e["rally_end"]): e for e in g.get("score_events", [])}
        prev = {"player1": 0, "player2": 0}
        starts = [r["start"] for r in g["rallies"]]
        if starts != sorted(starts):  # hit_inferred indexes rallies in stored order
            raise ValueError(f"{match} {gkey}: rallies not in time order")
        for i, r in enumerate(g["rallies"]):
            ev = events.get((r["start"], r["end"]))
            winner, source, score = None, None, None
            if ev:
                score = {k: int(v) for k, v in ev["score_after_rally"].items()}
                if ev.get("winner") in names:
                    winner, source = names[ev["winner"]], "metadata"
                else:
                    up = [k for k in ("player1", "player2") if score.get(k, 0) - prev.get(k, 0) == 1]
                    others_same = all(score.get(k, 0) == prev.get(k, 0) for k in ("player1", "player2") if k not in up)
                    if len(up) == 1 and others_same:
                        winner, source = names[up[0]], "score_diff"
                prev = score
            rallies.append(RawRally(
                game=gnum, index=i, start=int(r["start"]), end=int(r["end"]),
                winner=winner,
                score_after={names[k]: v for k, v in score.items()} if score else None,
                winner_from=source,
            ))
    rallies.sort(key=lambda r: r.start)
    return {
        "match_name": md["match_name"],
        "player1": names["player1"],
        "player2": names["player2"],
        "game1_top": names[md["game1_top_player"]],
        "match_winner": names.get(md.get("match_winner")),
        "rallies": rallies,
    }


def load_timeline(ann_root: Path, match: str) -> dict:
    """Manual timeline (shot_type/): rally intervals and single-frame events per label."""
    res = _read(Path(ann_root) / "shot_type" / f"{match}.json")["annotations"][0]["result"]
    intervals: dict[str, list[tuple[int, int]]] = {}
    for r in res:
        for lab in r["value"].get("timelinelabels", []):
            for rg in r["value"]["ranges"]:
                intervals.setdefault(lab, []).append((int(rg["start"]), int(rg["end"])))
    return {k: sorted(v) for k, v in intervals.items()}


def load_hits(ann_root: Path, match: str) -> list[dict]:
    """hit_inferred: [{frame, shot_type, side, player, game, rally}] sorted by frame."""
    hits = _read(Path(ann_root) / "hit_inferred" / f"{match}.json")["hits"]
    return sorted(hits, key=lambda h: h["frame"])


def load_caption_winners(ann_root: Path, match: str) -> dict[tuple[int, int], str]:
    """Rally winners recorded on the last shot of each rally in caption/ (an independent
    source used only to cross-check the score-derived winners). {(game, rally): name}."""
    p = Path(ann_root) / "caption" / f"{match}.json"
    if not p.exists():
        return {}
    shots = _read(p)["shots"]
    return {(s["game"], s["rally"]): s["rally_winner"] for s in shots if s.get("last_shot") and s.get("rally_winner")}


# --------------------------------------------------------------------------- tracks
def load_shuttle(ann_root: Path, match: str, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """TrackNet shuttle centres. Returns frames (K,) int64 and xy (K, 2) pixels.

    Entries with enabled=False or a (0, 0) position (TrackNet's "not visible") are dropped.
    """
    res = _read(Path(ann_root) / "shuttle" / f"{match}.json")["predictions"][0]["result"]
    frames, xy = [], []
    for track in res:
        for s in track["value"]["sequence"]:
            if not s.get("enabled", True) or (s["x"] <= 0 and s["y"] <= 0):
                continue
            frames.append(int(s["frame"]))
            xy.append(percent_box_center(s["x"], s["y"], s["width"], s["height"], width, height))
    order = np.argsort(frames, kind="stable")
    return np.asarray(frames, np.int64)[order], np.asarray(xy, np.float64).reshape(-1, 2)[order]


def load_player_bbox(ann_root: Path, match: str, width: int, height: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Manual player boxes per side: {"top": (frames (K,), boxes (K, 4) xyxy px), "bottom": ...}."""
    res = _read(Path(ann_root) / "player_bbox" / f"{match}.json")["annotations"][0]["result"]
    out: dict[str, tuple[list, list]] = {s: ([], []) for s in SIDES}
    for track in res:
        label = track["value"]["labels"][0].lower().replace("_player", "")
        if label not in out:
            continue
        for s in track["value"]["sequence"]:
            if s.get("enabled", True):
                out[label][0].append(int(s["frame"]))
                out[label][1].append(percent_box_to_xyxy(s["x"], s["y"], s["width"], s["height"], width, height))
    result = {}
    for side, (fr, bx) in out.items():
        fr_a = np.asarray(fr, np.int64)
        order = np.argsort(fr_a, kind="stable")
        result[side] = (fr_a[order], np.asarray(bx, np.float64).reshape(-1, 4)[order])
    return result


def iter_pose(ann_root: Path, match: str) -> Iterator[tuple[str, int, list[tuple[np.ndarray, np.ndarray]]]]:
    """Yield (clip_key, frame, persons) for every pose frame; persons = [(bbox_px (4,), kp (17, 3))].

    Keypoint x/y are still normalized 0-1 here (BFMD convention); score in the 3rd column.
    The JSON is ~130 MB per match, so only one match should be parsed at a time.
    """
    data = _read(Path(ann_root) / "pose" / f"{match}.json")["videos"]
    for key, clip in data.items():
        for fkey, persons in clip["frames"].items():
            yield key, int(fkey), [
                (np.asarray(p["bbox"], np.float64), np.asarray(p["keypoints"], np.float64)) for p in persons
            ]


# --------------------------------------------------------------------------- court
def load_court(ann_root: Path, match: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Static calibration: corners (4, 2) px in CORNER_ORDER and net points dict."""
    j = _read(Path(ann_root) / "court" / f"{match}_court.json")
    corners = np.array([j["court"][k] for k in CORNER_ORDER], np.float64)
    net = {k: np.asarray(v, np.float64) for k, v in j["net"].items()}
    return corners, net


def load_court_perframe(ann_root: Path, match: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """~1 Hz automatic court detections: frames (K,), corners (K, 4, 2) [TL, TR, BR, BL], flags (K,) int8."""
    j = _read(Path(ann_root) / "court_perframe" / f"{match}.json")
    fr = np.array([f["frame_idx"] for f in j["frames"]], np.int64)
    corners = np.array([f["corners"] if f.get("corners") else [[np.nan, np.nan]] * 4 for f in j["frames"]], np.float64)
    flags = np.array([PERFRAME_FLAGS.get(f["flag"], -1) for f in j["frames"]], np.int8)
    order = np.argsort(fr, kind="stable")
    return fr[order], corners[order], flags[order]
