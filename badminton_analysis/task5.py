"""Task 5: compare TrackNetV3 shuttle predictions on a video clip with the BFMD shuttle track.

Important: BFMD's shuttle positions are themselves TrackNet predictions (an unspecified version),
not human labels. This comparison therefore measures AGREEMENT between two trackers, not
accuracy. Large disagreements are still informative (one of them is wrong), and the overlay
images let a human judge individual frames.

Workflow
    1. extract_clip(): copy a frame range of the broadcast video to its own .mp4
       (clip frame i == video frame start + i; TrackNetV3 needs an .mp4 input)
    2. run TrackNetV3 predict.py on the clip -> <clip>_ball.csv (Frame, Visibility, X, Y)
    3. compare() / run(): metrics + overlay images
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .build import load_match
from .config import Config


def pick_clip(md, n_rallies: int = 3, min_main_camera: float = 0.9, require_serve: bool = False) -> tuple[int, int]:
    """Frame range [start, end) covering the first `n_rallies` consecutive clean rallies of one
    game (not suspect, mostly main camera; optionally starting with an annotated serve),
    with a 1 s margin on both sides."""
    ok = [r["main_camera_frac"] >= min_main_camera and not r["suspect"] and (r["starts_with_serve"] or not require_serve)
          for r in md.rallies]
    for i in range(len(md.rallies) - n_rallies + 1):
        if all(ok[i:i + n_rallies]) and len({md.rallies[j]["game"] for j in range(i, i + n_rallies)}) == 1:
            return md.rallies[i]["start"] - 30, md.rallies[i + n_rallies - 1]["end"] + 30
    raise ValueError("no run of clean rallies found")


def extract_clip(video: str | Path, start: int, end: int, out: str | Path) -> int:
    """Write video frames [start, end) to `out` (.mp4, same size and fps). Returns frames written.
    Reads sequentially from `start` so frame indices match the decoder's numbering."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    for _ in range(end - start):
        ok, img = cap.read()
        if not ok:
            break
        wr.write(img)
        n += 1
    wr.release(); cap.release()
    return n


def load_tracknet_csv(path: str | Path) -> dict[str, np.ndarray]:
    """TrackNetV3 output: Frame (0-based in the clip), Visibility (0/1), X, Y (pixels)."""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return {"frame": np.array([int(r["Frame"]) for r in rows], np.int64),
            "visible": np.array([int(float(r["Visibility"])) == 1 for r in rows]),
            "xy": np.array([[float(r["X"]), float(r["Y"])] for r in rows], np.float64).reshape(-1, 2)}


def compare(md, pred: dict, clip_start: int) -> dict:
    """Agreement between TrackNetV3 (pred, clip frame numbering) and BFMD on rally frames."""
    vf = pred["frame"] + clip_start                       # video frame numbers
    rows = md.rows_for_frames(vf)
    inside = rows >= 0                                    # BFMD has shuttle data only inside rallies
    bf = md.shuttle_px[rows[inside]].astype(np.float64)
    b_vis = ~np.isnan(bf[:, 0])
    p_vis = pred["visible"][inside]
    both = b_vis & p_vis
    err = np.linalg.norm(pred["xy"][inside][both] - bf[both], axis=1)
    pct = lambda q: float(np.percentile(err, q)) if len(err) else None  # noqa: E731
    return {
        "frames_in_clip": int(len(vf)), "frames_in_rallies": int(inside.sum()),
        "both_visible": int(both.sum()), "only_bfmd": int((b_vis & ~p_vis).sum()),
        "only_tracknetv3": int((~b_vis & p_vis).sum()), "neither": int((~b_vis & ~p_vis).sum()),
        "visibility_agreement": float(np.mean(b_vis == p_vis)) if inside.any() else None,
        "error_px_median": pct(50), "error_px_mean": float(err.mean()) if len(err) else None,
        "error_px_p90": pct(90),
        "within_5px": float(np.mean(err <= 5)) if len(err) else None,
        "within_10px": float(np.mean(err <= 10)) if len(err) else None,
        "within_20px": float(np.mean(err <= 20)) if len(err) else None,
        "frames_over_50px": [int(f) for f in vf[inside][both][err > 50]][:50],
    }


def overlays(video: str | Path, md, pred: dict, clip_start: int, out_dir: Path, n: int = 6) -> list[str]:
    """Save frames with BFMD (yellow circle) and TrackNetV3 (magenta cross): the worst
    disagreements first, then evenly spaced frames where both see the shuttle."""
    import cv2
    from .viz import read_frames
    vf = pred["frame"] + clip_start
    rows = md.rows_for_frames(vf)
    cand = []
    for i, r in enumerate(rows):
        if r >= 0 and pred["visible"][i] and not np.isnan(md.shuttle_px[r, 0]):
            cand.append((float(np.linalg.norm(pred["xy"][i] - md.shuttle_px[r])), i))
    if not cand:
        return []
    cand.sort(reverse=True)
    picks = [i for _, i in cand[: n // 2]] + [i for _, i in cand[:: max(len(cand) // (n - n // 2), 1)][: n - n // 2]]
    imgs = read_frames(video, [int(vf[i]) for i in picks])
    paths = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in picks:
        f = int(vf[i]); img = imgs[f].copy(); r = rows[i]
        b = md.shuttle_px[r]
        cv2.circle(img, (int(b[0]), int(b[1])), 12, (0, 255, 255), 2)
        p = pred["xy"][i].astype(int)
        cv2.drawMarker(img, (int(p[0]), int(p[1])), (255, 0, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.putText(img, f"frame {f}  BFMD=yellow circle  TrackNetV3=magenta cross  dist={np.linalg.norm(pred['xy'][i] - b):.0f}px",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        path = out_dir / f"overlay_{f}.jpg"
        cv2.imwrite(str(path), img)
        paths.append(str(path))
    return paths


def run(cfg: Config, match: str, pred_csv: str | Path, clip_start: int, video: str | Path | None = None) -> dict:
    md = load_match(cfg, match)
    pred = load_tracknet_csv(pred_csv)
    res = {"match": match, "clip_start_frame": clip_start, "prediction_csv": str(pred_csv), **compare(md, pred, clip_start)}
    out = cfg.output_dir / "task5"
    out.mkdir(parents=True, exist_ok=True)
    if video:
        res["overlays"] = overlays(video, md, pred, clip_start, out)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    L = ["# Task 5: TrackNetV3 vs BFMD shuttle track (generated)", "",
         "BFMD's shuttle track is itself TrackNet output, so this is tracker-vs-tracker AGREEMENT, not accuracy.", "",
         f"Match {match}, clip starting at video frame {clip_start}: {res['frames_in_clip']} frames, "
         f"{res['frames_in_rallies']} inside annotated rallies.", "",
         "| | frames |", "|---|---|",
         f"| both see the shuttle | {res['both_visible']} |", f"| only BFMD | {res['only_bfmd']} |",
         f"| only TrackNetV3 | {res['only_tracknetv3']} |", f"| neither | {res['neither']} |", ""]
    if res["both_visible"]:
        L += [f"Visibility agreement {res['visibility_agreement']:.3f}. Position difference where both see it: "
              f"median {res['error_px_median']:.1f} px, mean {res['error_px_mean']:.1f}, p90 {res['error_px_p90']:.1f}; "
              f"within 5 / 10 / 20 px: {res['within_5px']:.3f} / {res['within_10px']:.3f} / {res['within_20px']:.3f}."]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return res
