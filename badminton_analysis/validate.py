"""Data-quality checks on built matches, written as JSON + a Markdown report.

The checks are measurements, not assertions: they report how the data actually looks so the
choices made in the config (frame shifts, corner definition, thresholds) can be justified.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .court import CourtModel
from .schema import BOTTOM, SIDE_INDEX, TOP, MatchData


def _pct(x: np.ndarray, q) -> list[float]:
    x = x[~np.isnan(x)]
    return [round(float(v), 3) for v in np.percentile(x, q)] if len(x) else []


def validate_match(md: MatchData, model: CourtModel | None = None, caption_winners: dict | None = None) -> dict:
    model = model or CourtModel()
    a = md.arrays
    F = md.n_frames
    hit_rows = md.rows_for_frames(np.array([h["frame"] for h in md.hits]))
    out: dict = {"match": md.name, "n_frames": F, "n_rallies": len(md.rallies), "n_hits": len(md.hits)}

    # ---- rallies and hits
    out["rallies_without_hits"] = [(r["game"], r["index"], r["start"], r["end"]) for r in md.rallies if r["n_hits"] == 0]
    mcf = np.array([r["main_camera_frac"] for r in md.rallies if r["n_hits"] > 0])
    out["rallies_main_camera_frac_lt_0.5"] = int((mcf < 0.5).sum())
    out["rallies_main_camera_frac_p10_p50"] = _pct(mcf, [10, 50])
    out["suspect_rallies"] = [(r["game"], r["index"], r["wrong_half_frac"]) for r in md.rallies if r.get("suspect")]
    out["rallies_not_in_timeline"] =[(r["game"], r["index"], r["start"], r["end"]) for r in md.rallies if not r["in_timeline"]]
    src = Counter(r.get("winner_from") for r in md.rallies)
    out["winner_sources"] = {str(k): v for k, v in src.items()}
    if caption_winners is not None:
        agree = disagree = 0
        for r in md.rallies:
            cw = caption_winners.get((r["game"], r["index"]))
            if cw and r["winner"]:
                agree += cw == r["winner"]; disagree += cw != r["winner"]
        out["winner_vs_caption"] = {"agree": agree, "disagree": disagree}
    out["shot_types"] = dict(Counter(h["shot_type"] for h in md.hits).most_common())
    same_side, small_gaps = [], []
    for h0, h1 in zip(md.hits, md.hits[1:]):
        if h0["rally_idx"] != h1["rally_idx"]:
            continue
        if h0["side"] == h1["side"]:
            same_side.append((h0["frame"], h1["frame"]))
        if h1["frame"] - h0["frame"] < 5:
            small_gaps.append((h0["frame"], h1["frame"]))
    out["consecutive_same_side_hits"] = same_side
    out["hit_gaps_below_5_frames"] = small_gaps
    gaps = [h1["frame"] - h0["frame"] for h0, h1 in zip(md.hits, md.hits[1:]) if h0["rally_idx"] == h1["rally_idx"]]
    out["hit_gap_frames_p1_p50"] = _pct(np.asarray(gaps, float), [1, 50])
    first_hits = {}
    for h in md.hits:
        first_hits.setdefault(h["rally_idx"], h)
    out["rallies_first_hit_not_serve"] = sum(1 for h in first_hits.values() if h["shot_type"] not in ("serve", "flick_serve"))

    # side naming vs metadata rule (games 1-2: ends swap between games)
    other = {md.info["player1"]: md.info["player2"], md.info["player2"]: md.info["player1"]}
    expected_top = {1: md.info["game1_top"], 2: other[md.info["game1_top"]]}
    out["rallies_top_player_disagrees_with_metadata_rule"] = [
        (r["game"], r["index"]) for r in md.rallies
        if r["game"] in expected_top and r["top_player"] and r["top_player"] != expected_top[r["game"]]]

    # ---- coverage
    out["coverage"] = {
        "shuttle_frames": round(float((~np.isnan(a["shuttle_px"][:, 0])).mean()), 4),
        "shuttle_at_hits": round(float((~np.isnan(a["shuttle_px"][hit_rows[hit_rows >= 0], 0])).mean()), 4),
        "player_box_top": round(float((~np.isnan(a["player_box_px"][:, TOP, 0])).mean()), 4),
        "player_box_bottom": round(float((~np.isnan(a["player_box_px"][:, BOTTOM, 0])).mean()), 4),
        "pose_top": round(float((a["pose_assign"][:, TOP] > 0).mean()), 4),
        "pose_bottom": round(float((a["pose_assign"][:, BOTTOM] > 0).mean()), 4),
        "hitter_pose_at_hits": round(float(np.mean([
            a["pose_assign"][r, SIDE_INDEX[h["side"]]] > 0 for h, r in zip(md.hits, hit_rows) if r >= 0])), 4),
    }
    out["pose_assign_counts"] = {
        side: {int(k): int(v) for k, v in zip(*np.unique(a["pose_assign"][:, i], return_counts=True))}
        for side, i in SIDE_INDEX.items()}
    out["foot_src_counts"] = {int(k): int(v) for k, v in zip(*np.unique(a["foot_src"], return_counts=True))}
    out["court_flag_counts"] = {int(k): int(v) for k, v in zip(*np.unique(a["court_flag"], return_counts=True))}

    # ---- court geometry
    calib = md.calibration(model)
    nb = calib.net_base_court()
    out["net_base_error_m"] = (nb - np.array([[0, model.net_y], [model.width_doubles, model.net_y]])).round(3).tolist()
    good = (a["court_flag"] == 0)
    geo = {}
    for side, i in SIDE_INDEX.items():
        ok = good & (a["foot_src"][:, i] == 0)  # ankle-based feet, main camera
        fc = a["foot_court"][ok, i].astype(np.float64)
        in_half = fc[:, 1] < model.net_y if i == TOP else fc[:, 1] > model.net_y
        outside = ((fc[:, 0] < -0.3) | (fc[:, 0] > model.width_doubles + 0.3) |
                   (fc[:, 1] < -0.3) | (fc[:, 1] > model.length + 0.3))
        geo[side] = {
            "n": int(len(fc)),
            "y_p0.5_p50_p99.5": _pct(fc[:, 1], [0.5, 50, 99.5]),
            "x_p0.5_p50_p99.5": _pct(fc[:, 0], [0.5, 50, 99.5]),
            "frac_in_own_half": round(float(in_half.mean()), 4) if len(fc) else None,
            "frac_outside_court_by_0.3m": round(float(outside.mean()), 4) if len(fc) else None,
        }
    out["feet_court_main_camera"] = geo
    out["serve_checks"] = serve_checks(md, model)
    return out


def serve_checks(md: MatchData, model: CourtModel) -> dict:
    """Independent checks of court mapping + side assignment from the rules of singles.

    * parity: at the serve, the server stands in the right service court when their own score
      is even, else the left. Bottom player faces away from the camera (their right = +x),
      top player faces the camera (their right = -x).
    * depth: distance from the net of server / receiver feet, separately for top and bottom.
      Both ends should give similar values if the ground point is unbiased.
    Only serves on main-camera frames (court_flag == 0) are used.
    """
    a = md.arrays
    first: dict[int, dict] = {}
    for h in md.hits:
        first.setdefault(h["rally_idx"], h)
    ok = wrong = 0
    depth: dict[str, list[float]] = {}
    score: dict[str, int] = {}
    game = None
    cx = model.width_doubles / 2
    for ri, r in enumerate(md.rallies):
        if r["game"] != game:
            game, score = r["game"], {md.info["player1"]: 0, md.info["player2"]: 0}
        h = first.get(ri)
        if h is not None and h["shot_type"] in ("serve", "flick_serve"):
            row = md.rows_for_frames(np.array([h["frame"]]))[0]
            s = SIDE_INDEX[h["side"]]
            if row >= 0 and a["court_flag"][row] == 0:
                for role, idx in (("server", s), ("receiver", 1 - s)):
                    if a["foot_src"][row, idx] == 0:
                        side = "top" if idx == TOP else "bottom"
                        depth.setdefault(f"{role}_{side}", []).append(abs(float(a["foot_court"][row, idx, 1]) - model.net_y))
                if a["foot_src"][row, s] == 0:
                    x = float(a["foot_court"][row, s, 0])
                    in_right = x > cx if s == BOTTOM else x < cx
                    if in_right == (score[h["player"]] % 2 == 0):
                        ok += 1
                    else:
                        wrong += 1
        if r["score_after"]:
            score = dict(r["score_after"])
    return {
        "parity_ok": ok, "parity_wrong": wrong,
        "depth_median_m": {k: round(float(np.median(v)), 3) for k, v in sorted(depth.items())},
        "depth_n": {k: len(v) for k, v in sorted(depth.items())},
    }


def write_report(results: list[dict], alignment: list[dict] | None, out_dir: Path) -> Path:
    """Write validation.json and validation.md; returns the Markdown path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "validation.json", "w", encoding="utf-8") as f:
        json.dump({"matches": results, "alignment": alignment}, f, indent=1, default=str)

    L = ["# Task 1 data validation (generated)", ""]
    tot = Counter()
    for r in results:
        tot["frames"] += r["n_frames"]; tot["rallies"] += r["n_rallies"]; tot["hits"] += r["n_hits"]
    L += [f"Matches: {len(results)} | rally frames: {tot['frames']} | rallies: {tot['rallies']} | hits: {tot['hits']}", ""]

    L += ["## Coverage (fraction of rally frames)", "",
          "| match | shuttle | shuttle@hit | box top | box bottom | pose top | pose bottom | hitter pose@hit |",
          "|---|---|---|---|---|---|---|---|"]
    for r in results:
        c = r["coverage"]
        L.append(f"| {r['match'][:48]} | {c['shuttle_frames']:.3f} | {c['shuttle_at_hits']:.3f} | {c['player_box_top']:.3f} | "
                 f"{c['player_box_bottom']:.3f} | {c['pose_top']:.3f} | {c['pose_bottom']:.3f} | {c['hitter_pose_at_hits']:.3f} |")

    L += ["", "## Rally / hit consistency", "",
          "| match | rallies w/o hits | not in timeline | same-side consecutive hits | gaps < 5 fr | first hit not serve | top player ≠ metadata rule | suspect rallies (game, idx) | rallies < 50% main camera |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        sus = ", ".join(f"g{g}r{i}" for g, i, _ in r["suspect_rallies"]) or "-"
        L.append(f"| {r['match'][:48]} | {len(r['rallies_without_hits'])} | {len(r['rallies_not_in_timeline'])} | "
                 f"{len(r['consecutive_same_side_hits'])} | {len(r['hit_gaps_below_5_frames'])} | "
                 f"{r['rallies_first_hit_not_serve']} | {len(r['rallies_top_player_disagrees_with_metadata_rule'])} | {sus} | {r['rallies_main_camera_frac_lt_0.5']} |")

    if all("winner_vs_caption" in r for r in results):
        ag = sum(r["winner_vs_caption"]["agree"] for r in results); dis = sum(r["winner_vs_caption"]["disagree"] for r in results)
        srcs = Counter()
        for r in results:
            srcs.update(r["winner_sources"])
        L += ["", f"Rally winner sources: {dict(srcs)}. Agreement with caption/ `rally_winner`: "
              f"{ag} agree, {dis} disagree."]

    L += ["", "## Court geometry (static calibration, corners = doubles outer boundary)", "",
          "Net-post bases projected to court metres minus expected (0, 6.70) / (6.10, 6.70).",
          "Feet: ankle midpoints on main-camera frames (court_perframe flag = valid).", "",
          "| match | net err L (x,y) | net err R (x,y) | top y p0.5/p50/p99.5 | bottom y p0.5/p50/p99.5 | top in far half | bottom in near half | top outside | bottom outside |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        g = r["feet_court_main_camera"]
        ne = r["net_base_error_m"]
        L.append(f"| {r['match'][:48]} | {ne[0]} | {ne[1]} | {g['top']['y_p0.5_p50_p99.5']} | {g['bottom']['y_p0.5_p50_p99.5']} | "
                 f"{g['top']['frac_in_own_half']} | {g['bottom']['frac_in_own_half']} | "
                 f"{g['top']['frac_outside_court_by_0.3m']} | {g['bottom']['frac_outside_court_by_0.3m']} |")

    L += ["", "## Serve checks (rules of singles; main-camera serves, ankle-based feet)", "",
          "Parity: server in the right service court on an even own score, left on odd.",
          "Depth: median distance of the feet from the net; top and bottom should agree.", "",
          "| match | parity ok / wrong | server top / bottom (m) | receiver top / bottom (m) |",
          "|---|---|---|---|"]
    ok = wrong = 0
    pooled: dict[str, list[float]] = {}
    for r in results:
        sc = r["serve_checks"]; d = sc["depth_median_m"]
        ok += sc["parity_ok"]; wrong += sc["parity_wrong"]
        for k, v in d.items():
            pooled.setdefault(k, []).append(v)
        L.append(f"| {r['match'][:48]} | {sc['parity_ok']} / {sc['parity_wrong']} | "
                 f"{d.get('server_top')} / {d.get('server_bottom')} | {d.get('receiver_top')} / {d.get('receiver_bottom')} |")
    acc = ok / max(ok + wrong, 1)
    L.append(f"| **all** | **{ok} / {wrong} ({acc:.1%})** | {np.median(pooled.get('server_top', [np.nan])):.2f} / "
             f"{np.median(pooled.get('server_bottom', [np.nan])):.2f} (median of match medians) | "
             f"{np.median(pooled.get('receiver_top', [np.nan])):.2f} / {np.median(pooled.get('receiver_bottom', [np.nan])):.2f} |")

    if alignment:
        L += ["", "## Frame alignment between sources (raw files, pose = reference)", "",
              "| match | bbox IoU @-1 / 0 / +1 | best bbox shift | shuttle-wrist px by shift -2..2 | best shuttle shift | hit lag median (closest approach - label) |",
              "|---|---|---|---|---|---|"]
        for al in alignment:
            b = al["bbox_iou_by_shift"]; s = al["shuttle_dist_by_shift"]
            L.append(f"| {al['match'][:48]} | {b[-1]:.3f} / {b[0]:.3f} / {b[1]:.3f} | {al['bbox_best_shift']} | "
                     f"{' / '.join(f'{s[k]:.1f}' for k in range(-2, 3))} | {al['shuttle_best_shift']} | {al['hit_lag_median']} |")
    path = out_dir / "validation.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path
