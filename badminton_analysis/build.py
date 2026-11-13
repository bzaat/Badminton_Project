"""Raw BFMD JSON -> `MatchData`, and the on-disk cache (dataset/bfmd_m1/<match>.npz + .json).

Usage
    from badminton_analysis.config import load_config
    from badminton_analysis.build import load_match, singles_matches
    cfg = load_config()
    md = load_match(cfg, singles_matches(cfg)[0])   # builds the cache on first use
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from . import raw
from .config import Config
from .coords import box_iou, normalized_to_px
from .court import CourtCalibration, CourtModel
from .schema import ARRAY_SPEC, BOTTOM, N_KP, SCHEMA_VERSION, SIDE_INDEX, TOP, MatchData

log = logging.getLogger(__name__)
LEFT_ANKLE, RIGHT_ANKLE = 15, 16


def singles_matches(cfg: Config) -> list[str]:
    """The singles-only filter: matches that have every Milestone 1 annotation type."""
    return raw.list_singles_matches(cfg.ann_root)


def _settings(cfg: Config) -> dict:
    """Config values the cache content depends on (a change triggers a rebuild)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "frame_shift": {k: cfg.shift(k) for k in ("player_bbox", "shuttle", "pose")},
        "corners_are": cfg["court"]["corners_are"],
        "min_keypoint_score": float(cfg["pose"]["min_keypoint_score"]),
        "min_assign_iou": float(cfg["pose"]["min_assign_iou"]),
        "ankle_ground_offset": float(cfg["pose"].get("ankle_ground_offset", 0.0)),
        "box_ground_offset": float(cfg["pose"].get("box_ground_offset", 0.0)),
        "perframe_max_gap": int(cfg["court"]["perframe_max_gap"]),
        "max_wrong_half_frac": float(cfg["court"].get("max_wrong_half_frac", 0.2)),
        "size": [cfg.width, cfg.height],
    }


# --------------------------------------------------------------------------- pieces
def _rally_players(rallies: list[raw.RawRally], hits: list[dict]) -> list[dict]:
    """Top/bottom player per rally from the hit sides; rallies without hits copy a neighbour
    in the same game (the players only change ends between games and at 11 in game 3)."""
    by_rally: dict[tuple[int, int], dict[str, str]] = {}
    for h in hits:
        by_rally.setdefault((h["game"], h["rally"]), {})[h["side"]] = h["player"]
    out = []
    for i, r in enumerate(rallies):
        sides = dict(by_rally.get((r.game, r.index), {}))
        src = "hits" if sides else None
        if len(sides) < 2:
            for j in [i - 1, i + 1, i - 2, i + 2]:
                if 0 <= j < len(rallies) and rallies[j].game == r.game:
                    nb = by_rally.get((rallies[j].game, rallies[j].index), {})
                    if len(nb) == 2 and all(sides.get(k, v) == v for k, v in nb.items()):
                        sides = {**nb, **sides}
                        src = src or "neighbour"
                        break
        out.append({"top_player": sides.get("top"), "bottom_player": sides.get("bottom"), "players_from": src})
    return out


def _foot_point(kp_px: np.ndarray | None, box: np.ndarray | None, min_score: float,
                ankle_offset: float = 0.0, box_offset: float = 0.0) -> tuple[np.ndarray, int]:
    """Ground contact point in pixels.

    From confident ankles: their mean, moved down by ankle_offset * box height (the ankle joint
    is above the floor). Otherwise the bottom-centre of the box, moved by box_offset * height.
    Returns (point (2,), source code 0 ankles / 1 box / -1 none).
    """
    box_ok = box is not None and not np.isnan(box).any()
    box_h = float(box[3] - box[1]) if box_ok else 0.0
    if kp_px is not None:
        ank = kp_px[[LEFT_ANKLE, RIGHT_ANKLE]]
        good = ank[:, 2] >= min_score
        if good.any():
            pt = ank[good, :2].mean(axis=0)
            return np.array([pt[0], pt[1] + ankle_offset * box_h]), 0
    if box_ok:
        return np.array([(box[0] + box[2]) / 2.0, box[3] + box_offset * box_h]), 1
    return np.array([np.nan, np.nan]), -1


def _assign_persons(persons, manual: np.ndarray, min_iou: float, calib: CourtCalibration, net_y: float):
    """Map pose persons to (top, bottom). Returns {side_index: (person, method)}."""
    boxes = np.stack([p[0] for p in persons])
    iou = box_iou(boxes, manual)  # (P, 2); NaN manual boxes give 0
    P = len(persons)
    result: dict[int, tuple[int, int]] = {}
    if P >= 2:
        best, best_score = None, -1.0
        for i in range(P):
            for j in range(P):
                if i != j and iou[i, TOP] + iou[j, BOTTOM] > best_score:
                    best, best_score = (i, j), iou[i, TOP] + iou[j, BOTTOM]
        i, j = best
        if iou[i, TOP] >= min_iou and iou[j, BOTTOM] >= min_iou:
            return {TOP: (i, 1), BOTTOM: (j, 1)}
        order = np.argsort(boxes[:, 3])  # smaller bottom-y = further away = top
        return {TOP: (int(order[0]), 2), BOTTOM: (int(order[-1]), 2)}
    side = int(np.argmax(iou[0]))
    if iou[0, side] >= min_iou:
        result[side] = (0, 1)
    else:
        foot = np.array([(boxes[0, 0] + boxes[0, 2]) / 2.0, boxes[0, 3]])
        y = calib.to_court(foot)[1]
        result[TOP if y < net_y else BOTTOM] = (0, 2)
    return result


# --------------------------------------------------------------------------- build
def build_match(cfg: Config, match: str) -> tuple[MatchData, dict]:
    """Parse one match from the raw JSON. Returns (MatchData, build statistics)."""
    ann, W, H = cfg.ann_root, cfg.width, cfg.height
    st = _settings(cfg)
    stats: Counter = Counter()

    meta = raw.load_metadata(ann, match)
    timeline = raw.load_timeline(ann, match)
    hits_raw = raw.load_hits(ann, match)
    model = CourtModel.from_config(cfg["court"])
    corners, net = raw.load_court(ann, match)
    calib = CourtCalibration(corners, net, model, st["corners_are"])

    # ---- rallies and frame axis
    rr = meta["rallies"]
    tl_rallies = set(timeline.get("Rally", []))
    n_hits = Counter((h["game"], h["rally"]) for h in hits_raw)
    first_type: dict[tuple[int, int], str] = {}
    for h in hits_raw:
        first_type.setdefault((h["game"], h["rally"]), h["shot_type"])
    players = _rally_players(rr, hits_raw)
    rallies = []
    for r, p in zip(rr, players):
        rallies.append({
            "game": r.game, "index": r.index, "start": r.start, "end": r.end,
            "n_hits": n_hits.get((r.game, r.index), 0), "in_timeline": (r.start, r.end) in tl_rallies,
            "starts_with_serve": first_type.get((r.game, r.index)) in ("serve", "flick_serve"),
            "winner": r.winner, "winner_from": r.winner_from, "score_after": r.score_after, **p,
        })
    for a, b in zip(rallies, rallies[1:]):
        if b["start"] <= a["end"]:
            raise ValueError(f"{match}: overlapping rallies {a} / {b}")
    frame = np.concatenate([np.arange(r["start"], r["end"] + 1) for r in rallies]).astype(np.int64)
    rally_idx = np.concatenate(
        [np.full(r["end"] - r["start"] + 1, i) for i, r in enumerate(rallies)]).astype(np.int32)
    F = len(frame)

    def rows(frames: np.ndarray) -> np.ndarray:
        idx = np.clip(np.searchsorted(frame, frames), 0, F - 1)
        return np.where(frame[idx] == frames, idx, -1)

    arr = {k: np.full((F, *shape), np.nan if np.dtype(dt).kind == "f" else 0, dtype=dt)
           for k, (dt, shape) in ARRAY_SPEC.items()}
    arr["frame"], arr["rally_idx"] = frame, rally_idx
    arr["foot_src"][:] = -1
    arr["court_flag"][:] = -1

    # ---- hits
    key_to_idx = {(r["game"], r["index"]): i for i, r in enumerate(rallies)}
    hits = []
    for h in hits_raw:
        hits.append({**h, "rally_idx": key_to_idx[(h["game"], h["rally"])]})

    # ---- shuttle
    sf, sxy = raw.load_shuttle(ann, match, W, H)
    ri = rows(sf + st["frame_shift"]["shuttle"])
    ok = ri >= 0
    arr["shuttle_px"][ri[ok]] = sxy[ok]
    stats["shuttle_outside_rallies"] = int((~ok).sum())

    # ---- manual player boxes
    pb = raw.load_player_bbox(ann, match, W, H)
    for side, (fr, bx) in pb.items():
        ri = rows(fr + st["frame_shift"]["player_bbox"])
        ok = ri >= 0
        arr["player_box_px"][ri[ok], SIDE_INDEX[side]] = bx[ok]
        stats[f"player_box_outside_rallies_{side}"] = int((~ok).sum())

    # ---- pose (assigned to top/bottom) and feet
    min_score, min_iou = st["min_keypoint_score"], st["min_assign_iou"]
    seen = np.zeros(F, dtype=bool)
    for _, f, persons in raw.iter_pose(ann, match):
        r = rows(np.array([f + st["frame_shift"]["pose"]]))[0]
        if r < 0:
            stats["pose_frames_outside_rallies"] += 1
            continue
        if seen[r]:
            stats["pose_duplicate_frames"] += 1
        seen[r] = True
        if not persons:
            continue
        stats[f"pose_persons_{len(persons)}"] += 1
        manual = arr["player_box_px"][r].astype(np.float64)
        for s, (pi, method) in _assign_persons(persons, manual, min_iou, calib, model.net_y).items():
            box, kp = persons[pi]
            kp_px = normalized_to_px(kp, W, H)
            arr["pose_kp_px"][r, s] = kp_px[:N_KP]
            arr["pose_box_px"][r, s] = box
            arr["pose_assign"][r, s] = method
    # feet: from pose where available, else from the manual box
    for r in range(F):
        for s in (TOP, BOTTOM):
            has_pose = arr["pose_assign"][r, s] > 0
            kp = arr["pose_kp_px"][r, s].astype(np.float64) if has_pose else None
            box = arr["pose_box_px"][r, s] if has_pose else arr["player_box_px"][r, s]
            pt, src = _foot_point(kp, np.asarray(box, np.float64), min_score,
                                  st["ankle_ground_offset"], st["box_ground_offset"])
            arr["foot_px"][r, s] = pt
            arr["foot_src"][r, s] = src
    arr["foot_court"][:] = calib.to_court(arr["foot_px"].astype(np.float64))

    # ---- camera state from court_perframe
    cf, _, cflag = raw.load_court_perframe(ann, match)
    if len(cf):
        idx = np.clip(np.searchsorted(cf, frame), 1, len(cf) - 1)
        left, right = cf[idx - 1], cf[idx]
        nearest = np.where(np.abs(frame - left) <= np.abs(right - frame), idx - 1, idx)
        near_ok = np.abs(cf[nearest] - frame) <= st["perframe_max_gap"]
        arr["court_flag"][near_ok] = cflag[nearest[near_ok]]

    # ---- per-rally plausibility: is each player in their own half?
    fc_y = arr["foot_court"][..., 1]
    good = (arr["court_flag"] == 0) & (arr["foot_src"] >= 0).all(axis=1)
    wrong = (fc_y[:, TOP] > model.net_y) | (fc_y[:, BOTTOM] < model.net_y)
    n_good = np.bincount(rally_idx[good], minlength=len(rallies))
    n_wrong = np.bincount(rally_idx[good & wrong], minlength=len(rallies))
    for i, r in enumerate(rallies):
        frac = float(n_wrong[i] / n_good[i]) if n_good[i] else None
        r["main_camera_frames"] = int(n_good[i])
        r["main_camera_frac"] = round(float(n_good[i] / (r["end"] - r["start"] + 1)), 4)
        r["wrong_half_frac"] = None if frac is None else round(frac, 4)
        r["suspect"] = bool(frac is not None and frac > st["max_wrong_half_frac"])

    info = {
        "match": match, "player1": meta["player1"], "player2": meta["player2"],
        "game1_top": meta["game1_top"], "match_winner": meta["match_winner"], **st,
    }
    court = {"corners_px": corners.tolist(), "net_px": {k: v.tolist() for k, v in net.items()}}
    md = MatchData(info=info, rallies=rallies, hits=hits, court=court, arrays=arr)
    md.validate()
    return md, dict(stats)


# --------------------------------------------------------------------------- cache
def _paths(cfg: Config, match: str) -> tuple[Path, Path]:
    return cfg.cache_dir / f"{match}.npz", cfg.cache_dir / f"{match}.json"


def save_match(cfg: Config, md: MatchData, stats: dict | None = None) -> None:
    npz, js = _paths(cfg, md.name)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **md.arrays)
    with open(js, "w", encoding="utf-8") as f:
        json.dump({"info": md.info, "rallies": md.rallies, "hits": md.hits, "court": md.court,
                   "build_stats": stats or {}}, f, indent=1)


def load_match(cfg: Config, match: str, rebuild: bool = False) -> MatchData:
    """Load from cache; (re)build when missing, forced, or built with different settings."""
    npz, js = _paths(cfg, match)
    if not rebuild and npz.exists() and js.exists():
        with open(js, encoding="utf-8") as f:
            meta = json.load(f)
        want = _settings(cfg)
        if all(meta["info"].get(k) == v for k, v in want.items()):
            with np.load(npz) as z:
                arrays = {k: z[k] for k in z.files}
            md = MatchData(info=meta["info"], rallies=meta["rallies"], hits=meta["hits"],
                           court=meta["court"], arrays=arrays)
            md.validate()
            return md
        log.info("cache for %s was built with other settings, rebuilding", match)
    md, stats = build_match(cfg, match)
    save_match(cfg, md, stats)
    return md


def build_stats(cfg: Config, match: str) -> dict:
    """Build statistics stored next to a cached match (empty if not cached)."""
    js = _paths(cfg, match)[1]
    if not js.exists():
        return {}
    with open(js, encoding="utf-8") as f:
        return json.load(f).get("build_stats", {})
