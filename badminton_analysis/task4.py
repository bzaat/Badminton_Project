"""Task 4: movement features in court coordinates -> CSVs + validation report.

Measurements only. Nothing here labels movement as good, bad or an "error".

Outputs (<output_dir>/task4/)
    rally_player_movement.csv   one row per rally x player: distance, speeds, zone occupancy
    shot_recovery.csv           one row per hit followed by an opponent hit in the same rally
    player_match_summary.csv    one row per match x player
    report.md, metrics.json, occupancy_heatmaps.png
"""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from .build import load_match, singles_matches
from .config import Config
from .movement import ZONE_NAMES, fill_gaps, own_frame, player_track, smooth, speed_and_distance, zone_index
from .schema import BOTTOM, SIDE_INDEX, TOP

log = logging.getLogger(__name__)
SIDE_NAME = {TOP: "top", BOTTOM: "bottom"}


def _included(r: dict, mc: dict) -> bool:
    return (r["main_camera_frac"] >= float(mc["min_main_camera_frac"])
            and not (mc["exclude_suspect"] and r["suspect"]) and r["n_hits"] > 0)


def _nan(x) -> float | str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def run(cfg: Config, matches: list[str] | None = None) -> dict:
    mc, fps = cfg["movement"], cfg.fps
    out = cfg.output_dir / "task4"
    out.mkdir(parents=True, exist_ok=True)
    matches = matches or singles_matches(cfg)
    rally_rows, shot_rows = [], []
    occ = defaultdict(list)                      # player -> list of (depth, lateral) samples
    speeds_side = defaultdict(lambda: {TOP: [], BOTTOM: []})
    all_speed = []
    dist_by_window = defaultdict(float)
    stationary = {w: {TOP: [], BOTTOM: []} for w in mc["windows_checked"]}
    n_rallies = n_used = 0

    for m in matches:
        md = load_match(cfg, m)
        log.info("movement: %s", m)
        for ri, r in enumerate(md.rallies):
            n_rallies += 1
            if not _included(r, mc):
                continue
            n_used += 1
            sl = md.rally_slice(ri)
            tracks = {s: player_track(md, sl, s, mc, fps) for s in (TOP, BOTTOM)}
            names = {TOP: r["top_player"], BOTTOM: r["bottom_player"]}
            for s, tr in tracks.items():
                v = tr["valid"]
                z = zone_index(tr["depth"][v], tr["lateral"][v])
                zfrac = {n: float(np.mean(z == k)) if v.any() else np.nan for k, n in enumerate(ZONE_NAMES)}
                sp = tr["speed"][v]
                rally_rows.append({
                    "match": m, "game": r["game"], "rally": r["index"], "player": names[s], "opponent": names[1 - s],
                    "side": SIDE_NAME[s], "rally_frames": sl.stop - sl.start, "valid_frames": int(v.sum()),
                    "valid_seconds": round(v.sum() / fps, 3), "distance_m": round(tr["distance"], 3),
                    "mean_speed_mps": _nan(np.nanmean(sp) if np.isfinite(sp).any() else np.nan),
                    "p95_speed_mps": _nan(np.nanpercentile(sp, 95) if np.isfinite(sp).any() else np.nan),
                    "max_speed_mps": _nan(np.nanmax(sp) if np.isfinite(sp).any() else np.nan),
                    **{f"zone_{k}": _nan(x) for k, x in zfrac.items()},
                    "outside_frac": _nan(float(np.mean(z < 0)) if v.any() else np.nan),
                })
                if names[s]:
                    occ[names[s]].append(np.c_[tr["depth"][v], tr["lateral"][v]])
                    if np.isfinite(sp).any():
                        speeds_side[names[s]][s].append(float(np.nanmean(sp)))
                all_speed.append(sp[np.isfinite(sp)])
                # distance sensitivity to the smoothing window (same valid frames)
                xy, vv = fill_gaps(md.foot_court[sl, s].astype(float), tr["valid"], 0)
                for w in mc["windows_checked"]:
                    dist_by_window[w] += speed_and_distance(smooth(xy, vv, int(w)), vv, fps)[1]
            # recovery after each shot: own hit -> opponent's next hit
            hits = [h for h in md.hits if h["rally_idx"] == ri]
            for h, nxt in zip(hits, hits[1:]):
                s = SIDE_INDEX[h["side"]]
                if SIDE_INDEX[nxt["side"]] == s:
                    continue
                a, b = h["frame"] - md.frame[sl.start], nxt["frame"] - md.frame[sl.start]
                tr = tracks[s]
                v = tr["valid"]
                row = {"match": m, "game": r["game"], "rally": r["index"], "frame": h["frame"], "player": h["player"],
                       "side": h["side"], "shot_type": h["shot_type"], "opponent_hit_frame": nxt["frame"],
                       "time_available_s": round((b - a) / fps, 3)}
                for tag, k in (("at_hit", a), ("at_opp_hit", b)):
                    row[f"depth_{tag}"] = _nan(tr["depth"][k]) if v[k] else ""
                    row[f"lateral_{tag}"] = _nan(tr["lateral"][k]) if v[k] else ""
                seg = slice(a, b + 1)
                ok = v[seg]
                row["path_m"] = _nan(float(np.nansum(np.linalg.norm(np.diff(tr["xy_s"][seg][ok], axis=0), axis=1)))) if ok.sum() > 1 and ok.all() else ""
                row["displacement_m"] = _nan(float(np.linalg.norm(tr["xy_s"][b] - tr["xy_s"][a]))) if v[a] and v[b] else ""
                sp = tr["speed"][seg]
                row["max_speed_mps"] = _nan(np.nanmax(sp)) if np.isfinite(sp).any() else ""
                row["valid_frac"] = round(float(ok.mean()), 3)
                shot_rows.append(row)
        # stationary check: server and receiver before a serve (laws: both stand still until the serve)
        first = {}
        for h in md.hits:
            first.setdefault(h["rally_idx"], h)
        for ri, h in first.items():
            r = md.rallies[ri]
            if h["shot_type"] not in ("serve", "flick_serve") or not _included(r, mc):
                continue
            sl = md.rally_slice(ri)
            hr = int(md.rows_for_frames(np.array([h["frame"]]))[0])
            lo = max(sl.start, hr - int(mc["stationary_frames_before_serve"]))
            if hr - lo < 5:
                continue
            for s in (TOP, BOTTOM):
                rows = slice(lo, hr)  # frames strictly before the serve
                valid = (md.court_flag[rows] == 0) & (md.foot_src[rows, s] == 0)
                if valid.sum() < 5:
                    continue
                xy = md.foot_court[rows, s].astype(float)
                for w in mc["windows_checked"]:
                    sp, _ = speed_and_distance(smooth(xy, valid, int(w)), valid, fps)
                    if np.isfinite(sp).any():
                        stationary[w][s].append(float(np.nanmedian(sp)))

    # ---- summaries
    summary = _player_match_summary(rally_rows, shot_rows)
    for name, rows in (("rally_player_movement.csv", rally_rows), ("shot_recovery.csv", shot_rows),
                       ("player_match_summary.csv", summary)):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    allsp = np.concatenate(all_speed) if all_speed else np.zeros(0)
    w0 = int(mc["smooth_window"])
    val = {
        "rallies_total": n_rallies, "rallies_used": n_used,
        "valid_frame_frac": float(np.sum([r["valid_frames"] for r in rally_rows]) / max(np.sum([r["rally_frames"] for r in rally_rows]), 1)),
        "stationary_median_speed_by_window": {int(w): {SIDE_NAME[s]: (float(np.median(v)) if v else None) for s, v in d.items()}
                                              for w, d in stationary.items()},
        "stationary_n": {SIDE_NAME[s]: len(v) for s, v in stationary[w0].items()},
        "total_distance_km_by_window": {int(w): round(d / 1000, 3) for w, d in dist_by_window.items()},
        "frac_speed_above_limit": float(np.mean(allsp > float(mc["implausible_speed_mps"]))) if len(allsp) else None,
        "speed_p50_p95_p99": [float(x) for x in np.percentile(allsp, [50, 95, 99])] if len(allsp) else None,
        "player_mean_speed_top_vs_bottom": {p: {"top": float(np.median(d[TOP])) if d[TOP] else None,
                                                "bottom": float(np.median(d[BOTTOM])) if d[BOTTOM] else None,
                                                "n_top": len(d[TOP]), "n_bottom": len(d[BOTTOM])}
                                            for p, d in sorted(speeds_side.items())},
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(val, f, indent=1)
    _report(val, summary, mc, out)
    _heatmaps(occ, out)
    return val


def _player_match_summary(rally_rows: list[dict], shot_rows: list[dict]) -> list[dict]:
    g = defaultdict(list)
    for r in rally_rows:
        if r["player"]:
            g[(r["match"], r["player"])].append(r)
    rec = defaultdict(list)
    for s in shot_rows:
        if s["depth_at_opp_hit"] != "":
            rec[(s["match"], s["player"])].append((s["depth_at_opp_hit"], s["lateral_at_opp_hit"]))
    out = []
    for (m, p), rows in sorted(g.items()):
        secs = sum(r["valid_seconds"] for r in rows)
        dist = sum(r["distance_m"] for r in rows)
        w = np.array([r["valid_frames"] for r in rows], float)
        zones = {}
        for z in ZONE_NAMES + ["outside"]:
            k = f"zone_{z}" if z != "outside" else "outside_frac"
            vals = np.array([r[k] if r[k] != "" else np.nan for r in rows], float)
            ok = ~np.isnan(vals)
            zones[f"{k}"] = round(float(np.sum(vals[ok] * w[ok]) / max(w[ok].sum(), 1)), 4)
        sp = np.array([r["mean_speed_mps"] for r in rows if r["mean_speed_mps"] != ""], float)
        p95 = np.array([r["p95_speed_mps"] for r in rows if r["p95_speed_mps"] != ""], float)
        rc = np.array(rec.get((m, p), []), float).reshape(-1, 2)
        out.append({"match": m, "player": p, "rallies": len(rows), "valid_seconds": round(secs, 1),
                    "distance_m": round(dist, 1), "distance_per_min_m": round(dist / secs * 60, 1) if secs else "",
                    "median_rally_mean_speed_mps": round(float(np.median(sp)), 3) if len(sp) else "",
                    "median_rally_p95_speed_mps": round(float(np.median(p95)), 3) if len(p95) else "",
                    "recovery_depth_median_m": round(float(np.median(rc[:, 0])), 3) if len(rc) else "",
                    "recovery_lateral_median_m": round(float(np.median(rc[:, 1])), 3) if len(rc) else "",
                    "recovery_n": len(rc), **zones})
    return out


def _report(val: dict, summary: list[dict], mc: dict, out: Path) -> None:
    w0 = int(mc["smooth_window"])
    L = ["# Task 4: movement measurements (generated)", "",
         "Measurements only (no judgement). Court metres from ankle-based ground points on main-camera frames; "
         f"rallies used: {val['rallies_used']} / {val['rallies_total']} (excluded: suspect or < {mc['min_main_camera_frac']:.0%} main camera); "
         f"valid frames in used rallies: {val['valid_frame_frac']:.1%}. Smoothing: centred moving average, {w0} frames.", "",
         "## Validation", "",
         "### Noise floor: speed while the laws require standing still (server and receiver before the serve)", "",
         "| smoothing window (frames) | top player median speed (m/s) | bottom player median speed (m/s) |", "|---|---|---|"]
    for w, d in val["stationary_median_speed_by_window"].items():
        f3 = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
        L.append(f"| {w}{' (used)' if int(w) == w0 else ''} | {f3(d['top'])} | {f3(d['bottom'])} |")
    L += [f"(n = {val['stationary_n']} pre-serve segments of 5-12 frames)", "",
          "### Sensitivity of total distance to the smoothing window", "",
          "| window | total distance (km, all players, used rallies) |", "|---|---|"]
    for w, d in val["total_distance_km_by_window"].items():
        L.append(f"| {w} | {d} |")
    p = val["speed_p50_p95_p99"]
    if p:
        L += ["", f"Speed distribution over valid frames: p50 {p[0]:.2f}, p95 {p[1]:.2f}, p99 {p[2]:.2f} m/s; "
              f"frames above {mc['implausible_speed_mps']} m/s: {val['frac_speed_above_limit']:.3%}."]
    L += ["",
          "### Same player at both ends (players change ends every game)", "",
          "| player | median rally mean speed, far side (m/s) | near side (m/s) | rallies far / near |", "|---|---|---|---|"]
    for pl, d in val["player_mean_speed_top_vs_bottom"].items():
        if d["top"] is not None and d["bottom"] is not None:
            L.append(f"| {pl} | {d['top']:.3f} | {d['bottom']:.3f} | {d['n_top']} / {d['n_bottom']} |")
    L += ["", "## Per player and match", "",
          "| match | player | rallies | distance (m) | m per min of play | median rally mean speed | median rally p95 speed | recovery depth / lateral (median, m) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in summary:
        L.append(f"| {r['match'][:40]} | {r['player']} | {r['rallies']} | {r['distance_m']} | {r['distance_per_min_m']} | "
                 f"{r['median_rally_mean_speed_mps']} | {r['median_rally_p95_speed_mps']} | {r['recovery_depth_median_m']} / {r['recovery_lateral_median_m']} |")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def _heatmaps(occ: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    players = sorted(occ, key=lambda p: -sum(len(a) for a in occ[p]))
    n = len(players)
    cols = min(n, 5)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 4.2 * rows), squeeze=False, constrained_layout=True)
    for ax, p in zip(axes.flat, players):
        a = np.vstack(occ[p])
        H, xe, ye = np.histogram2d(a[:, 1], a[:, 0], bins=[26, 34], range=[[-3.05, 3.05], [0, 8.0]])
        ax.imshow((H / H.sum()).T, origin="upper", extent=[-3.05, 3.05, 8.0, 0], cmap="magma", aspect="equal")
        for x in (-2.59, 2.59, -3.05, 3.05):
            ax.axvline(x, color="w", lw=0.5, alpha=0.6)
        for y in (0, 1.98, 6.7 - 0.76, 6.7):
            ax.axhline(y, color="w", lw=0.5, alpha=0.6)
        ax.set_title(f"{p}\n{len(a) / 30 / 60:.0f} min of play", fontsize=9)
        ax.set_xlabel("lateral (m), + = player's right"); ax.set_ylabel("distance from net (m)")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Court occupancy in each player's own half (net at the top), all matches", fontsize=11)
    fig.savefig(out / "occupancy_heatmaps.png", dpi=110)
    plt.close(fig)
