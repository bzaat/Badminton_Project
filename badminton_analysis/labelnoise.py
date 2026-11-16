"""Hit label noise: re-annotate BFMD contact frames from video, blinded, and compare.

Workflow (one match with video):
    python -m badminton_analysis labelnoise-kit --match "China-Open-2025-Shi" --video <mp4> --out <dir>
        -> <dir>/index.html (annotation page), frames/, key.json (BFMD frames; the page never reads it)
    open <dir>/index.html in a browser, click the contact frame of every item, "Download results"
    python -m badminton_analysis labelnoise-eval --kit <dir> --annotations <results.json>
        -> outputs/labelnoise/{report.md, metrics.json, items.csv}

Blinding: every item shows a window of 2*HALF+1 frames around the BFMD hit frame, with the
window start shifted by a random -MAX_SHIFT..MAX_SHIFT frames, so the label is not at a fixed
position. The page only sees window-relative indices. A few items are shown twice (with a
different shift) to measure the annotator's own repeatability.
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import raw
from .config import Config

HALF = 12          # window = label - HALF .. label + HALF (before the random shift)
MAX_SHIFT = 4
CROP_H = 400       # crop images are resized to this height
FULL_W = 960       # full-frame images are resized to this width
TOOL_HTML = Path(__file__).with_name("labelnoise_tool.html")


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def sample_hits(hits: list[dict], n: int, n_repeat: int, seed: int = 0) -> list[dict]:
    """Stratified random sample by shot type (proportional, >= 2 per type), plus repeats.

    Returns items {id, frame, shot_type, side, game, rally, shift, repeat_of} in random order.
    """
    rng = np.random.default_rng(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for h in hits:
        by_type[h["shot_type"]].append(h)
    total = len(hits)
    floor = {t: min(2, len(v)) for t, v in by_type.items()}
    quota = {t: min(len(v), max(floor[t], round(n * len(v) / total))) for t, v in by_type.items()}
    while sum(quota.values()) > n:                       # trim the most over-represented stratum
        t = max((k for k in quota if quota[k] > floor[k]), key=lambda k: quota[k] - n * len(by_type[k]) / total)
        quota[t] -= 1
    while sum(quota.values()) < n:
        t = max((k for k in quota if quota[k] < len(by_type[k])), key=lambda k: n * len(by_type[k]) / total - quota[k])
        quota[t] += 1
    chosen = []
    for t in sorted(by_type):
        pool = by_type[t]
        idx = rng.choice(len(pool), size=min(quota[t], len(pool)), replace=False)
        chosen += [pool[i] for i in sorted(idx)]
    items = [dict(frame=int(h["frame"]), shot_type=h["shot_type"], side=h["side"], game=int(h["game"]),
                  rally=int(h["rally"]), repeat_of=None) for h in chosen]
    for i in rng.choice(len(items), size=min(n_repeat, len(items)), replace=False):
        items.append(dict(items[i], repeat_of=int(i)))   # index into the first-pass list, fixed below
    first_ids = [f"h{i:03d}" for i in range(len(chosen))]
    for it in items:
        it["shift"] = int(rng.integers(-MAX_SHIFT, MAX_SHIFT + 1))
        if it["repeat_of"] is not None:
            it["repeat_of"] = first_ids[it["repeat_of"]]
    order = rng.permutation(len(items))
    out = []
    for k, i in enumerate(order):
        it = dict(items[i])
        it["id"] = f"q{k:03d}"
        it["source_id"] = first_ids[i] if i < len(chosen) else it["repeat_of"]
        out.append(it)
    return out


def window(item: dict) -> np.ndarray:
    """Video frames shown for an item (the BFMD frame is at index HALF - shift)."""
    start = item["frame"] - HALF + item["shift"]
    return np.arange(start, start + 2 * HALF + 1)


# --------------------------------------------------------------------------- #
# Kit extraction
# --------------------------------------------------------------------------- #
CROP_SCALE = 2.6                 # crop side = CROP_SCALE x median hitter box height ...
CROP_MIN, CROP_MAX = 220, 480    # ... clipped to this range (px)


def _crop_box(boxes: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Fixed square crop on the hitter's median box over the window (racket above the head).

    The median over the whole window (not the box at the label frame) keeps the crop from
    leaking where the BFMD label is. None when no box is known.
    """
    ok = ~np.isnan(boxes).any(axis=1)
    if not ok.any():
        return None
    b = np.median(boxes[ok], axis=0)
    bh = b[3] - b[1]
    s = float(np.clip(CROP_SCALE * bh, CROP_MIN, CROP_MAX))
    cx, cy = (b[0] + b[2]) / 2, b[1] + 0.35 * bh          # a bit above the box centre
    x1 = int(np.clip(cx - s / 2, 0, width - s)); y1 = int(np.clip(cy - s / 2, 0, height - s))
    return x1, y1, x1 + int(s), y1 + int(s)


def make_kit(cfg: Config, match: str, video: str | Path, out: str | Path, n: int = 100,
             n_repeat: int = 10, seed: int = 0) -> dict:
    import cv2
    out = Path(out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    from .build import load_match
    from .schema import SIDE_INDEX
    md = load_match(cfg, match)
    hits = raw.load_hits(cfg.ann_root, match)
    items = sample_hits(hits, n, n_repeat, seed)

    need: dict[int, list[tuple[dict, int]]] = defaultdict(list)
    for it in items:
        fr = window(it)
        rows = md.rows_for_frames(fr)
        rows = rows[rows >= 0]
        boxes = md.player_box_px[rows, SIDE_INDEX[it["side"]]] if len(rows) else np.full((1, 4), np.nan)
        crop = _crop_box(boxes, cfg.width, cfg.height)
        hr = md.rows_for_frames(np.array([it["frame"]]))[0]
        it["court_flag"] = int(md.court_flag[hr]) if hr >= 0 else -1
        it["rally_suspect"] = bool(md.rallies[int(md.rally_idx[hr])]["suspect"]) if hr >= 0 else None
        # crop only when it is trustworthy: main camera (court valid) and a known hitter box
        it["crop"] = list(crop) if crop is not None else None
        it["default_view"] = "c" if crop is not None and it["court_flag"] == 0 and not it["rally_suspect"] else "f"
        for j, f in enumerate(fr):
            need[int(f)].append((it, j))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    last = max(need)
    f = 0
    written = 0
    while f <= last:                                     # sequential decode: no seek inaccuracy
        if f in need:
            ok, img = cap.read()
            if not ok:
                raise RuntimeError(f"video ended at frame {f}")
            assert img.shape[1] == cfg.width and img.shape[0] == cfg.height, img.shape
            full = cv2.resize(img, (FULL_W, round(FULL_W * cfg.height / cfg.width)), interpolation=cv2.INTER_AREA)
            for it, j in need[f]:
                x1, y1, x2, y2 = it["crop"] or (0, 0, cfg.height, cfg.height)   # placeholder crop
                crop = cv2.resize(img[y1:y2, x1:x2], (CROP_H, CROP_H), interpolation=cv2.INTER_CUBIC)
                d = out / "frames" / it["id"]
                d.mkdir(exist_ok=True)
                cv2.imwrite(str(d / f"c{j:02d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                cv2.imwrite(str(d / f"f{j:02d}.jpg"), full, [cv2.IMWRITE_JPEG_QUALITY, 85])
                written += 1
        else:
            if not cap.grab():
                raise RuntimeError(f"video ended at frame {f}")
        f += 1
    cap.release()

    n_win = 2 * HALF + 1
    manifest = [{"id": it["id"], "side": it["side"], "n": n_win, "view": it["default_view"]} for it in items]
    (out / "manifest.js").write_text("window.MANIFEST = " + json.dumps(manifest) + ";\n"
                                     f"window.KIT = {json.dumps({'match': match, 'seed': seed, 'n_items': len(items)})};\n")
    key = {"match": match, "seed": seed, "half": HALF, "max_shift": MAX_SHIFT, "items": items}
    (out / "key.json").write_text(json.dumps(key, indent=1))
    shutil.copy(TOOL_HTML, out / "index.html")
    return {"items": len(items), "first_pass": n, "repeats": n_repeat, "images": 2 * written,
            "shot_types": dict(Counter(it["shot_type"] for it in items if it["repeat_of"] is None)),
            "default_full_view": sum(it["default_view"] == "f" for it in items)}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _bootstrap(x: np.ndarray, fn, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def human_frames(key: dict, ann: dict) -> list[dict]:
    """Join annotations to the key. Returns one row per item with human video frame (or None)."""
    labels = ann["labels"]
    rows = []
    for it in key["items"]:
        a = labels.get(it["id"], {})
        idx = a.get("frame")
        hf = int(window(it)[idx]) if isinstance(idx, int) else None
        rows.append(dict(id=it["id"], source_id=it["source_id"], repeat=it["repeat_of"] is not None,
                         bfmd_frame=it["frame"], shot_type=it["shot_type"], side=it["side"],
                         game=it["game"], rally=it["rally"], status=a.get("status", "missing"),
                         confidence=a.get("confidence"), human_frame=hf,
                         main_view=it.get("default_view", "c") == "c",
                         offset=None if hf is None else hf - it["frame"]))
    return rows


def _within(off: np.ndarray) -> dict:
    return {f"within_{t}": float(np.mean(np.abs(off) <= t)) for t in (0, 1, 2, 3, 5)}


def evaluate(key: dict, ann: dict, predictions_csv: str | Path | None, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = human_frames(key, ann)
    first = [r for r in rows if not r["repeat"]]
    status = Counter(r["status"] for r in first)
    lab = [r for r in first if r["offset"] is not None]
    off = np.array([r["offset"] for r in lab], float)
    res: dict = {"match": key["match"], "items_first_pass": len(first), "status": dict(status),
                 "n_labelled": len(lab)}
    if len(off):
        res["offset_human_minus_bfmd"] = {
            "mean": float(off.mean()), "mean_ci95": _bootstrap(off, np.mean),
            "median": float(np.median(off)), "mad": float(np.median(np.abs(off - np.median(off)))),
            "mean_abs": float(np.abs(off).mean()), "mean_abs_ci95": _bootstrap(off, lambda x: np.abs(x).mean()),
            **_within(off),
            "within_2_ci95": _bootstrap(off, lambda x: np.mean(np.abs(x) <= 2)),
            "histogram": {int(k): int(v) for k, v in sorted(Counter(off.astype(int)).items())},
        }
        res["status_by_view"] = {v: dict(Counter(r["status"] for r in first if r["main_view"] == (v == "main")))
                                 for v in ("main", "other")}
        mv = np.array([r["offset"] for r in lab if r["main_view"]], float)
        if len(mv):
            res["offset_main_view_only"] = {"n": len(mv), "mean": float(mv.mean()),
                                            "mean_abs": float(np.abs(mv).mean()), **_within(mv)}
        sure = np.array([r["offset"] for r in lab if r["confidence"] == "sure"], float)
        if len(sure):
            res["offset_sure_only"] = {"n": len(sure), "mean_abs": float(np.abs(sure).mean()), **_within(sure)}
        per_type = {}
        for t in sorted({r["shot_type"] for r in lab}):
            o = np.array([r["offset"] for r in lab if r["shot_type"] == t], float)
            per_type[t] = {"n": len(o), "mean": float(o.mean()), "mean_abs": float(np.abs(o).mean()),
                           "within_2": float(np.mean(np.abs(o) <= 2))}
        res["per_shot_type"] = per_type

    # repeatability: same hit annotated twice
    by_src = {r["source_id"]: r for r in first}
    rep = []
    for r in rows:
        if r["repeat"] and r["human_frame"] is not None:
            o = by_src.get(r["source_id"])
            if o and o["human_frame"] is not None:
                rep.append(r["human_frame"] - o["human_frame"])
    if rep:
        rep_a = np.array(rep, float)
        res["repeatability"] = {"n_pairs": len(rep), "mean_abs_diff": float(np.abs(rep_a).mean()),
                                "diffs": [int(x) for x in rep], **_within(rep_a)}

    # Task 2 detector vs both references (nearest prediction within 8 frames of the hit)
    if predictions_csv and Path(predictions_csv).exists() and len(lab):
        with open(predictions_csv, newline="") as f:
            pf = np.array(sorted(int(p["frame"]) for p in csv.DictReader(f) if p["match"] == key["match"]))
        e_b, e_h = [], []
        for r in lab if len(pf) else []:
            ref = (r["bfmd_frame"] + r["human_frame"]) / 2   # neutral: favours neither reference
            j = int(np.argmin(np.abs(pf - ref)))
            if abs(pf[j] - ref) <= 8:
                e_b.append(pf[j] - r["bfmd_frame"]); e_h.append(pf[j] - r["human_frame"])
                r["task2_frame"] = int(pf[j])
        if e_b:
            e_b, e_h = np.array(e_b, float), np.array(e_h, float)
            res["task2_detector"] = {
                "n_with_prediction_within_8": len(e_b),
                "vs_bfmd": {"mean": float(e_b.mean()), "mean_abs": float(np.abs(e_b).mean()), **_within(e_b)},
                "vs_human": {"mean": float(e_h.mean()), "mean_abs": float(np.abs(e_h).mean()), **_within(e_h)},
            }

    with open(out_dir / "items.csv", "w", newline="") as f:
        cols = ["id", "source_id", "repeat", "shot_type", "side", "game", "rally", "bfmd_frame",
                "human_frame", "offset", "status", "confidence", "task2_frame"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=1))
    (out_dir / "report.md").write_text(_report(res))
    return res


def _report(r: dict) -> str:
    L = [f"# Hit label noise: {r['match']}", "",
         f"First-pass items: {r['items_first_pass']}; status: {r['status']}; labelled: {r['n_labelled']}", ""]
    o = r.get("offset_human_minus_bfmd")
    if o:
        L += ["## Human contact frame - BFMD hit frame", "",
              f"- mean {o['mean']:+.2f} (95% CI {o['mean_ci95'][0]:+.2f} .. {o['mean_ci95'][1]:+.2f}), "
              f"median {o['median']:+.0f}, MAD {o['mad']:.1f}",
              f"- mean |offset| {o['mean_abs']:.2f} (95% CI {o['mean_abs_ci95'][0]:.2f} .. {o['mean_abs_ci95'][1]:.2f})",
              "- within 0 / ±1 / ±2 / ±3 / ±5: " + " / ".join(f"{o[f'within_{t}']:.2f}" for t in (0, 1, 2, 3, 5))
              + f" (±2 CI {o['within_2_ci95'][0]:.2f} .. {o['within_2_ci95'][1]:.2f})",
              f"- histogram: {o['histogram']}", ""]
    if "status_by_view" in r:
        L += [f"Status by view (main camera with crop / other camera or unknown): {r['status_by_view']}", ""]
    if "offset_main_view_only" in r:
        s = r["offset_main_view_only"]
        L += [f"Main-camera items only (n={s['n']}): mean {s['mean']:+.2f}, mean |offset| {s['mean_abs']:.2f}, "
              f"within ±1 / ±2 {s['within_1']:.2f} / {s['within_2']:.2f}", ""]
    if "offset_sure_only" in r:
        s = r["offset_sure_only"]
        L += [f"Sure-only (n={s['n']}): mean |offset| {s['mean_abs']:.2f}, within ±2 {s['within_2']:.2f}", ""]
    if "repeatability" in r:
        p = r["repeatability"]
        L += ["## Annotator repeatability", "",
              f"{p['n_pairs']} hits labelled twice: mean |diff| {p['mean_abs_diff']:.2f}, within ±1 {p['within_1']:.2f}, "
              f"diffs {p['diffs']}", ""]
    if "per_shot_type" in r:
        L += ["## Per shot type", "", "| shot | n | mean | mean abs | within ±2 |", "|---|---|---|---|---|"]
        L += [f"| {t} | {v['n']} | {v['mean']:+.2f} | {v['mean_abs']:.2f} | {v['within_2']:.2f} |"
              for t, v in r["per_shot_type"].items()]
        L.append("")
    if "task2_detector" in r:
        d = r["task2_detector"]
        L += ["## Task 2 detector (nearest prediction within 8 frames)", "",
              f"n = {d['n_with_prediction_within_8']}", "", "| reference | mean | mean abs | ±1 | ±2 | ±3 |", "|---|---|---|---|---|---|"]
        for k in ("vs_bfmd", "vs_human"):
            v = d[k]
            L.append(f"| {k[3:]} | {v['mean']:+.2f} | {v['mean_abs']:.2f} | {v['within_1']:.2f} | {v['within_2']:.2f} | {v['within_3']:.2f} |")
        L.append("")
    return "\n".join(L)
