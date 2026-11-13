"""Task 2: hit-event baseline with leave-one-match-out (LOMO) tuning and evaluation.

For every held-out match, all parameters (thresholds, suppression radius, frame offset, and for
the learned variant the logistic-regression weights) are chosen on the other 11 matches only,
then applied unchanged to the held-out match. Reported numbers are always on unseen matches.

Variants
    shuttle_only       accel peaks of the shuttle track + accel threshold
    shuttle+proximity  ... + shuttle must be near a wrist of the nearer player
    full               ... + that wrist must be moving (wrist speed threshold)
    logistic           accel peaks scored by a logistic regression over 7 candidate features
    pose_only          peaks of wrist speed only (no shuttle) - a weak reference
Outputs go to <output_dir>/task2/.
"""
from __future__ import annotations

import csv
import itertools
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import raw
from .build import load_match, singles_matches
from .config import Config
from .events_eval import match_events, prf
from .hits import (FEATURE_NAMES, HitParams, candidate_features, candidates, frame_features,
                   pose_only_candidates, select)
from .schema import SIDE_INDEX

log = logging.getLogger(__name__)

VARIANTS = ("shuttle_only", "shuttle+proximity", "full", "logistic", "pose_only")


class MatchCase:
    """Everything Task 2 needs about one match, computed once."""

    def __init__(self, cfg: Config, match: str, radii: list[int]):
        hc = cfg["hits"]
        self.md = md = load_match(cfg, match)
        self.feats = frame_features(md, hc, float(cfg["pose"]["min_keypoint_score"]))
        self.cand = {r: candidates(md, self.feats, r) for r in radii}
        self.cand_pose = pose_only_candidates(md, self.feats, int(hc["peak_radius"]))
        self.X = {r: candidate_features(c) for r, c in self.cand.items()}
        self.gt = np.array([h["frame"] for h in md.hits], dtype=np.int64)
        self.gt_side = np.array([SIDE_INDEX[h["side"]] for h in md.hits])
        tl = raw.load_timeline(cfg.ann_root, match)
        self.bounce = np.array(sorted(s for lab in ("bounce", "net_hit") for s, _ in tl.get(lab, [])), dtype=np.int64)

    def labels(self, r: int, tol: int) -> np.ndarray:
        f = self.md.frame[self.cand[r]["row"]]
        idx = np.clip(np.searchsorted(self.gt, f), 1, max(len(self.gt) - 1, 1))
        d = np.minimum(np.abs(f - self.gt[idx - 1]), np.abs(self.gt[np.minimum(idx, len(self.gt) - 1)] - f))
        return d <= tol


def _predict(case: MatchCase, cand: dict, mask: np.ndarray, offset: int, score: np.ndarray) -> dict:
    rows = cand["row"][mask]
    frames = case.md.frame[rows] + offset
    o = np.argsort(frames, kind="stable")
    return {"frame": frames[o], "row": rows[o], "hitter": cand["hitter"][mask][o], "score": score[mask][o]}


def _f1(c: np.ndarray) -> np.ndarray:
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    return np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)


# --------------------------------------------------------------------------- rule-based variants
def _rule_grid(hc: dict, variant: str) -> list[tuple[int, HitParams]]:
    g = hc["grid"]
    if variant == "pose_only":
        pg = hc["pose_only_grid"]
        return [(-1, HitParams(accel=a, offset=o)) for a, o in itertools.product(pg["score"], pg["offset"])]
    prox = [np.inf] if variant == "shuttle_only" else [float(x) for x in g["proximity"]]
    wrist = [0.0] if variant != "full" else [float(x) for x in g["wrist_speed"]]
    keep = [True] if variant == "shuttle_only" else [bool(x) for x in g["keep_without_pose"]]
    return [(int(r), HitParams(accel=float(a), proximity=p, wrist_speed=w, offset=int(o), keep_without_pose=k))
            for r, a, p, w, o, k in itertools.product(g["peak_radius"], g["accel"], prox, wrist, g["offset"], keep)]


def _rule_pred(case: MatchCase, r: int, p: HitParams) -> dict:
    cand = case.cand_pose if r < 0 else case.cand[r]
    return _predict(case, cand, select(cand, p), p.offset, cand["accel"])


def _run_rule_variant(cases: dict, variant: str, hc: dict, prim: int):
    grid = _rule_grid(hc, variant)
    C = {m: np.array([match_events(c.gt, _rule_pred(c, r, p)["frame"], prim).counts for r, p in grid])
         for m, c in cases.items()}
    chosen, preds = {}, {}
    for m in cases:
        train = sum(C[o] for o in cases if o != m)
        k = int(np.argmax(_f1(train)))
        r, p = grid[k]
        chosen[m] = {"peak_radius": r, **{kk: (None if isinstance(v, float) and np.isinf(v) else v) for kk, v in p.__dict__.items()},
                     "train_f1": float(_f1(train)[k])}
        preds[m] = _rule_pred(cases[m], r, p)
    return len(grid), chosen, preds


# --------------------------------------------------------------------------- learned variant
def _run_logistic(cases: dict, hc: dict, prim: int):
    lg = hc["logistic_grid"]
    radii = [int(r) for r in hc["grid"]["peak_radius"]]
    combos = list(itertools.product(lg["threshold"], lg["offset"]))
    chosen, preds, coefs = {}, {}, []
    for m in cases:
        train = [o for o in cases if o != m]
        best = None
        for r in radii:
            X = np.vstack([cases[o].X[r] for o in train])
            y = np.concatenate([cases[o].labels(r, prim) for o in train])
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            model.fit(X, y)
            probs = {o: model.predict_proba(cases[o].X[r])[:, 1] for o in cases}
            tot = np.zeros((len(combos), 3), np.int64)
            for o in train:
                c = cases[o]
                for k, (thr, off) in enumerate(combos):
                    tot[k] += match_events(c.gt, _predict(c, c.cand[r], probs[o] >= thr, off, probs[o])["frame"], prim).counts
            k = int(np.argmax(_f1(tot)))
            if best is None or _f1(tot)[k] > best[0]:
                best = (_f1(tot)[k], r, combos[k], probs[m], model)
        f1, r, (thr, off), pm, model = best
        lr = model[-1]
        coefs.append(lr.coef_[0])
        chosen[m] = {"peak_radius": r, "threshold": thr, "offset": off, "train_f1": float(f1)}
        preds[m] = _predict(cases[m], cases[m].cand[r], pm >= thr, off, pm)
    coef = np.array(coefs)
    return len(radii) * len(combos), chosen, preds, {n: (float(coef[:, i].mean()), float(coef[:, i].std()))
                                                     for i, n in enumerate(FEATURE_NAMES)}


# --------------------------------------------------------------------------- failure analysis
def _failures(case: MatchCase, pred: dict, tol: int) -> tuple[list[dict], list[dict]]:
    md, gt = case.md, case.gt
    res = match_events(gt, pred["frame"], tol)
    raw_sh = ~np.isnan(md.shuttle_px[:, 0])
    rows_gt = md.rows_for_frames(gt)
    cand = case.cand[min(case.cand)]  # densest candidate set: "was there any trajectory peak at all?"
    cand_frames = md.frame[cand["row"]]
    first = {}
    for h in md.hits:
        first.setdefault(h["rally_idx"], h["frame"])
    last = {}
    for h in md.hits:
        last[h["rally_idx"]] = h["frame"]
    fns = []
    for i in res.fn:
        g, row, h = gt[i], rows_gt[i], md.hits[i]
        if raw_sh[max(row - 2, 0):row + 3].sum() < 3:
            why = "shuttle_missing"
        elif np.any(np.abs(pred["frame"] - g) <= 8):
            why = "prediction_nearby_but_off_or_taken"
        elif not np.any(np.abs(cand_frames - g) <= tol + 1):
            why = "no_trajectory_peak"
        else:
            why = "peak_scored_below_threshold"
        r = md.rallies[h["rally_idx"]]
        fns.append({"match": md.name, "frame": int(g), "shot_type": h["shot_type"], "side": h["side"],
                    "game": h["game"], "rally": h["rally"], "reason": why,
                    "first_hit_of_rally": bool(first[h["rally_idx"]] == h["frame"]),
                    "rally_flagged": bool(r["suspect"] or r["main_camera_frac"] < 0.5)})
    matched_gt = {g for g, _ in res.pairs}
    fps = []
    for j in res.fp:
        f = int(pred["frame"][j])
        ri = int(md.rally_idx[pred["row"][j]])
        if len(case.bounce) and np.min(np.abs(case.bounce - f)) <= 5:
            why = "near_annotated_bounce_or_net_hit"
        elif f > last.get(ri, -1) + tol:
            why = "after_last_hit_of_rally"
        elif f < first.get(ri, 1 << 60) - tol:
            why = "before_first_hit_of_rally"
        elif any(abs(int(gt[g]) - f) <= 8 for g in matched_gt):
            why = "duplicate_near_detected_hit"
        else:
            why = "other_mid_rally"
        fps.append({"match": md.name, "frame": f, "reason": why, "score": round(float(pred["score"][j]), 3),
                    "shuttle_outlier_within_3": bool(case.feats["outlier"][max(pred["row"][j] - 3, 0):pred["row"][j] + 4].any())})
    return fns, fps


# --------------------------------------------------------------------------- main
def run(cfg: Config, matches: list[str] | None = None) -> dict:
    hc = cfg["hits"]
    tols = [int(t) for t in hc["tolerances"]]
    prim = int(hc["primary_tolerance"])
    out_dir = cfg.output_dir / "task2"
    out_dir.mkdir(parents=True, exist_ok=True)
    matches = matches or singles_matches(cfg)
    radii = [int(r) for r in hc["grid"]["peak_radius"]]
    dev = [m for m in matches if any(d in m for d in hc.get("dev_matches", []))]
    cases = {}
    for m in matches:
        log.info("features: %s", m)
        cases[m] = MatchCase(cfg, m, radii)

    res: dict = {"tolerances": tols, "primary_tolerance": prim, "dev_matches": dev, "variants": {}}
    all_preds = {}
    for variant in VARIANTS:
        log.info("variant %s", variant)
        if variant == "logistic":
            n, chosen, preds, coef = _run_logistic(cases, hc, prim)
        else:
            n, chosen, preds = _run_rule_variant(cases, variant, hc, prim)
            coef = None
        all_preds[variant] = preds
        per_match, sums, sums_dev, sums_new = {}, {t: np.zeros(3, np.int64) for t in tols}, {t: np.zeros(3, np.int64) for t in tols}, {t: np.zeros(3, np.int64) for t in tols}
        for m in matches:
            row = {"params": chosen[m]}
            for t in tols:
                c = np.array(match_events(cases[m].gt, preds[m]["frame"], t).counts)
                sums[t] += c
                (sums_dev if m in dev else sums_new)[t] += c
                row[f"tol{t}"] = prf(*c)
            per_match[m] = row
        f1s = {t: [per_match[m][f"tol{t}"]["f1"] for m in matches] for t in tols}
        res["variants"][variant] = {
            "n_parameter_sets": n, "micro": {t: prf(*sums[t]) for t in tols},
            "micro_dev": {t: prf(*sums_dev[t]) for t in tols}, "micro_not_dev": {t: prf(*sums_new[t]) for t in tols},
            "macro_f1_mean": {t: float(np.mean(f1s[t])) for t in tols}, "macro_f1_std": {t: float(np.std(f1s[t])) for t in tols},
            "per_match": per_match, "logistic_coef_mean_std": coef,
        }

    best = max(("full", "logistic"), key=lambda v: res["variants"][v]["micro"][prim]["f1"])
    res["analysed_variant"] = best
    res["analysis"] = _analyse(cases, all_preds[best], prim, out_dir)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    _write_report(res, out_dir)
    _plots(res, out_dir)
    return res


def _analyse(cases: dict, preds: dict, prim: int, out_dir: Path) -> dict:
    fns, fps, dts, side_ok, pred_rows = [], [], [], [], []
    sub = {"clean": np.zeros(3, np.int64), "flagged": np.zeros(3, np.int64)}
    shot_rec, shot_tot = Counter(), Counter()
    for m, case in cases.items():
        pred = preds[m]
        fn, fp = _failures(case, pred, prim)
        fns += fn; fps += fp
        dts += [int(pred["frame"][j] - case.gt[i]) for i, j in match_events(case.gt, pred["frame"], 5).pairs]
        rp = match_events(case.gt, pred["frame"], prim)
        matched = dict(rp.pairs)
        inv = {j: i for i, j in rp.pairs}
        side_ok += [int(pred["hitter"][j]) == int(case.gt_side[i]) for i, j in rp.pairs]
        for i, h in enumerate(case.md.hits):
            shot_tot[h["shot_type"]] += 1
            shot_rec[h["shot_type"]] += i in matched
        flag = np.array([bool(r["suspect"] or r["main_camera_frac"] < 0.5 or not r["starts_with_serve"]) for r in case.md.rallies])
        g_flag = flag[[h["rally_idx"] for h in case.md.hits]]
        p_flag = flag[case.md.rally_idx[pred["row"]]]
        for name, gm, pm in (("clean", ~g_flag, ~p_flag), ("flagged", g_flag, p_flag)):
            sub[name] += np.array(match_events(case.gt[gm], pred["frame"][pm], prim).counts)
        for j in range(len(pred["frame"])):
            pred_rows.append({"match": m, "frame": int(pred["frame"][j]),
                              "hitter": ["top", "bottom"][int(pred["hitter"][j])] if pred["hitter"][j] >= 0 else "",
                              "score": round(float(pred["score"][j]), 3),
                              "matched_gt_frame": int(case.gt[inv[j]]) if j in inv else ""})
    for name, rows in (("false_negatives.csv", fns), ("false_positives.csv", fps), ("predictions.csv", pred_rows)):
        if rows:
            with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)
    no_serve = {k: v for k, v in shot_tot.items() if k not in ("serve", "flick_serve")}
    return {
        "fn_reasons": dict(Counter(f["reason"] for f in fns).most_common()),
        "fp_reasons": dict(Counter(f["reason"] for f in fps).most_common()),
        "fn_first_hit_of_rally": int(sum(f["first_hit_of_rally"] for f in fns)),
        "fp_with_shuttle_outlier_within_3": int(sum(f["shuttle_outlier_within_3"] for f in fps)),
        "timing_pred_minus_gt_within5": {int(k): int(v) for k, v in zip(*np.unique(dts, return_counts=True))},
        "hitter_side_accuracy": float(np.mean(side_ok)) if side_ok else None,
        "recall_by_shot_type": {k: {"recall": shot_rec[k] / shot_tot[k], "n": shot_tot[k]} for k in sorted(shot_tot, key=lambda k: -shot_tot[k])},
        "recall_excluding_serves": sum(shot_rec[k] for k in no_serve) / max(sum(no_serve.values()), 1),
        "subset_clean_vs_flagged": {k: prf(*v) for k, v in sub.items()},
    }


def _fmt(m: dict) -> str:
    return f"{m['precision']:.3f} / {m['recall']:.3f} / **{m['f1']:.3f}**"


def _write_report(res: dict, out_dir: Path) -> None:
    tols, prim = res["tolerances"], res["primary_tolerance"]
    L = ["# Task 2: hit-event baseline (generated)", "",
         f"Leave-one-match-out over {len(res['variants']['full']['per_match'])} matches: every parameter is chosen on the other matches "
         f"(max F1 at ±{prim} frames) and evaluated on the held-out match. Micro = pooled counts; macro = mean ± std of per-match F1.",
         f"Design exploration used only the dev matches {[m[:40] for m in res['dev_matches']]}; the 'not-dev' column is "
         "on matches never looked at while designing.", "",
         "| variant | " + " | ".join(f"P / R / F1 @±{t}" for t in tols) + f" | macro F1 @±{prim} | F1 @±{prim} dev / not-dev |",
         "|---|" + "---|" * (len(tols) + 2)]
    for v, r in res["variants"].items():
        L.append(f"| {v} | " + " | ".join(_fmt(r["micro"][t]) for t in tols)
                 + f" | {r['macro_f1_mean'][prim]:.3f} ± {r['macro_f1_std'][prim]:.3f} | "
                 f"{r['micro_dev'][prim]['f1']:.3f} / {r['micro_not_dev'][prim]['f1']:.3f} |")
    lc = res["variants"]["logistic"]["logistic_coef_mean_std"]
    L += ["", "Logistic regression coefficients (standardized features, mean ± std over the folds): "
          + ", ".join(f"{k} {m:+.2f}±{s:.2f}" for k, (m, s) in lc.items())]
    v = res["analysed_variant"]
    L += ["", f"## {v}: per held-out match (±{prim})", "",
          "| match | P | R | F1 | TP | FP | FN | chosen parameters |", "|---|---|---|---|---|---|---|---|"]
    for m, r in res["variants"][v]["per_match"].items():
        t = r[f"tol{prim}"]
        ps = ", ".join(f"{k}={val}" for k, val in r["params"].items() if k != "train_f1")
        L.append(f"| {m[:50]} | {t['precision']:.3f} | {t['recall']:.3f} | {t['f1']:.3f} | {t['tp']} | {t['fp']} | {t['fn']} | {ps} |")
    a = res["analysis"]
    L += ["", f"## Failure analysis ({v}, held-out predictions, ±{prim})", "",
          f"- False negatives by reason: {a['fn_reasons']} (first hit of a rally: {a['fn_first_hit_of_rally']})",
          f"- False positives by reason: {a['fp_reasons']} (with a removed shuttle outlier within ±3 frames: {a['fp_with_shuttle_outlier_within_3']})",
          f"- Hitter side correct on matched hits: {a['hitter_side_accuracy']:.4f}",
          f"- Recall excluding serve / flick_serve: {a['recall_excluding_serves']:.3f}",
          f"- Timing (pred − GT) of pairs matched within ±5: {a['timing_pred_minus_gt_within5']}",
          "- Clean rallies vs flagged (suspect / <50 % main camera / no serve): "
          + ", ".join(f"{k}: P {x['precision']:.3f} R {x['recall']:.3f} F1 {x['f1']:.3f} (n_gt={x['tp'] + x['fn']})" for k, x in a["subset_clean_vs_flagged"].items()),
          "", "| shot type | recall | n |", "|---|---|---|"]
    for k, x in a["recall_by_shot_type"].items():
        L.append(f"| {k} | {x['recall']:.3f} | {x['n']} |")
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def _plots(res: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prim = res["primary_tolerance"]
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)
    names = list(res["variants"])
    matches = list(res["variants"]["full"]["per_match"])
    x = np.arange(len(matches)); w = 0.8 / len(names)
    for k, v in enumerate(names):
        ax[0].bar(x + k * w, [res["variants"][v]["per_match"][m][f"tol{prim}"]["f1"] for m in matches], w, label=v)
    ax[0].set_xticks(x + 0.4 - w / 2, [m.split("-2025-")[0][:9] + "\n" + m.split("-2025-")[1][:24] for m in matches], rotation=90, fontsize=7)
    ax[0].set_ylabel(f"F1 @ ±{prim} (held-out)"); ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8, ncol=3)
    ax[0].set_title("Hit detection per held-out match")
    tl = res["analysis"]["timing_pred_minus_gt_within5"]
    ax[1].bar([int(k) for k in tl], list(tl.values()), color="#1a73e8")
    ax[1].set_xlabel("predicted − annotated frame"); ax[1].set_ylabel("hits matched within ±5")
    ax[1].set_title(f"Timing error ({res['analysed_variant']})")
    fig.savefig(out_dir / "task2_summary.png", dpi=110)
    plt.close(fig)
