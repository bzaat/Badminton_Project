"""Task 3: shot-type baseline with leave-one-match-out evaluation.

Main setting: classify the ANNOTATED hits (isolates the stroke-type question). Secondary:
classify the hits detected in Task 2 (end-to-end), with the same per-fold models.
Model hyperparameters are fixed in the config (no tuning), so nothing is selected on test data.
"""
from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .build import load_match, singles_matches
from .config import Config
from .events_eval import match_events
from .hits import frame_features
from .schema import SIDE_INDEX
from .shots import FEATURE_GROUPS, FEATURE_NAMES, FINE_CLASSES, event_features, merged_label

log = logging.getLogger(__name__)


def make_model(name: str, sc: dict, n_iter: int | None = None):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             LogisticRegression(max_iter=4000, C=float(sc["logreg_C"]), class_weight="balanced"))
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=n_iter or int(sc["hgb_max_iter"]),
                                              learning_rate=float(sc["hgb_learning_rate"]),
                                              max_leaf_nodes=int(sc["hgb_max_leaf_nodes"]),
                                              class_weight="balanced", random_state=0)
    raise ValueError(name)


class ShotCase:
    def __init__(self, cfg: Config, match: str):
        self.md = md = load_match(cfg, match)
        self.feats = frame_features(md, cfg["hits"], float(cfg["pose"]["min_keypoint_score"]))
        ev = [{"frame": h["frame"], "side": SIDE_INDEX[h["side"]], "rally_idx": h["rally_idx"]} for h in md.hits]
        self.X = event_features(md, self.feats, ev, float(cfg["pose"]["min_keypoint_score"]))
        self.y = np.array([h["shot_type"] for h in md.hits])
        self.player = np.array([h["player"] for h in md.hits])
        self.frames = np.array([h["frame"] for h in md.hits])
        flag = [bool(r["suspect"] or r["main_camera_frac"] < 0.5 or not r["starts_with_serve"]) for r in md.rallies]
        self.flagged = np.array([flag[h["rally_idx"]] for h in md.hits])


def _metrics(y: np.ndarray, p: np.ndarray, classes: list[str]) -> dict:
    pr, rc, f1, n = precision_recall_fscore_support(y, p, labels=classes, zero_division=0)
    return {"accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, labels=classes, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, p, labels=classes, average="weighted", zero_division=0)),
            "n": int(len(y)),
            "per_class": {c: {"precision": float(a), "recall": float(b), "f1": float(d), "n": int(e)}
                          for c, a, b, d, e in zip(classes, pr, rc, f1, n)}}


def _lomo(cases: dict, model_name: str, sc: dict, cols: list[int], labels_of, keep_models: bool = False, n_iter=None):
    preds, models = {}, {}
    for m in cases:
        tr = [o for o in cases if o != m]
        X = np.vstack([cases[o].X[:, cols] for o in tr])
        y = np.concatenate([labels_of(cases[o].y) for o in tr])
        clf = make_model(model_name, sc, n_iter).fit(X, y)
        preds[m] = clf.predict(cases[m].X[:, cols])
        if keep_models:
            models[m] = clf
    return preds, models


def run(cfg: Config, matches: list[str] | None = None) -> dict:
    sc = cfg["shots"]
    out = cfg.output_dir / "task3"
    out.mkdir(parents=True, exist_ok=True)
    matches = matches or singles_matches(cfg)
    cases = {}
    for m in matches:
        log.info("features: %s", m)
        cases[m] = ShotCase(cfg, m)
    all_cols = list(range(len(FEATURE_NAMES)))
    schemes = {"fine": (lambda y: y, FINE_CLASSES),
               "merged": (lambda y: np.array([merged_label(v) for v in y]), sorted({merged_label(c) for c in FINE_CLASSES}))}
    res: dict = {"n_hits": int(sum(len(c.y) for c in cases.values())), "schemes": {}}
    main_preds, main_models = None, None
    fine_preds: dict[str, dict] = {}
    for scheme, (lab, classes) in schemes.items():
        res["schemes"][scheme] = {"classes": classes, "models": {}}
        for model_name in sc["models"]:
            log.info("%s / %s", scheme, model_name)
            keep = scheme == "fine" and model_name == sc["main_model"]
            preds, models = _lomo(cases, model_name, sc, all_cols, lab, keep_models=keep)
            y_all = np.concatenate([lab(cases[m].y) for m in matches])
            p_all = np.concatenate([preds[m] for m in matches])
            per_match = {m: _metrics(lab(cases[m].y), preds[m], classes) for m in matches}
            r = _metrics(y_all, p_all, classes)
            r["macro_f1_per_match_mean"] = float(np.mean([v["macro_f1"] for v in per_match.values()]))
            r["macro_f1_per_match_std"] = float(np.std([v["macro_f1"] for v in per_match.values()]))
            r["accuracy_per_match"] = {m: v["accuracy"] for m, v in per_match.items()}
            r["macro_f1_per_match"] = {m: v["macro_f1"] for m, v in per_match.items()}
            res["schemes"][scheme]["models"][model_name] = r
            if scheme == "fine":
                fine_preds[model_name] = preds
            if keep:
                main_preds, main_models = preds, models
                res["confusion_fine"] = confusion_matrix(y_all, p_all, labels=classes).tolist()

    res["bootstrap_fine"] = _bootstrap(cases, fine_preds, int(sc.get("bootstrap", 2000)))

    # ---- feature-group ablation (fine labels, main model with fewer iterations for speed)
    abl = {}
    groups = {g: [FEATURE_NAMES.index(n) for n in names] for g, names in FEATURE_GROUPS.items()}
    settings = {"all": all_cols, **{f"only_{g}": cols for g, cols in groups.items()}}
    settings.update({f"all_minus_{g}": [c for c in all_cols if c not in cols] for g, cols in groups.items()})
    for name, cols in settings.items():
        log.info("ablation %s", name)
        preds, _ = _lomo(cases, sc["main_model"], sc, cols, lambda y: y, n_iter=int(sc["ablation_max_iter"]))
        y_all = np.concatenate([cases[m].y for m in matches]); p_all = np.concatenate([preds[m] for m in matches])
        abl[name] = {"macro_f1": float(f1_score(y_all, p_all, labels=FINE_CLASSES, average="macro", zero_division=0)),
                     "accuracy": float(accuracy_score(y_all, p_all))}
    res["ablation_fine"] = abl

    # ---- per player, seen vs unseen in training
    players_by_match = {m: set(cases[m].player) for m in matches}
    rows, per_player = [], defaultdict(lambda: {"y": [], "p": []})
    seen = {"seen": {"y": [], "p": []}, "unseen": {"y": [], "p": []}}
    for m in matches:
        others = set().union(*(players_by_match[o] for o in matches if o != m))
        c = cases[m]
        for i in range(len(c.y)):
            key = "seen" if c.player[i] in others else "unseen"
            per_player[c.player[i]]["y"].append(c.y[i]); per_player[c.player[i]]["p"].append(main_preds[m][i])
            seen[key]["y"].append(c.y[i]); seen[key]["p"].append(main_preds[m][i])
            rows.append({"match": m, "frame": int(c.frames[i]), "player": c.player[i], "player_in_training": key == "seen",
                         "gt": c.y[i], "pred": main_preds[m][i], "rally_flagged": bool(c.flagged[i])})
    n_matches = Counter(p for m in matches for p in players_by_match[m])
    res["per_player"] = {p: {**{k: v for k, v in _metrics(np.array(d["y"]), np.array(d["p"]), FINE_CLASSES).items() if k != "per_class"},
                             "matches": n_matches[p]} for p, d in sorted(per_player.items(), key=lambda kv: -len(kv[1]["y"]))}
    res["seen_vs_unseen_player"] = {k: {kk: vv for kk, vv in _metrics(np.array(d["y"]), np.array(d["p"]), FINE_CLASSES).items() if kk != "per_class"}
                                    for k, d in seen.items() if d["y"]}
    fl = np.concatenate([cases[m].flagged for m in matches])
    y_all = np.concatenate([cases[m].y for m in matches]); p_all = np.concatenate([main_preds[m] for m in matches])
    res["clean_vs_flagged_rallies"] = {k: {kk: vv for kk, vv in _metrics(y_all[mask], p_all[mask], FINE_CLASSES).items() if kk != "per_class"}
                                       for k, mask in (("clean", ~fl), ("flagged", fl))}

    # ---- end-to-end on Task 2 detections
    res["end_to_end"] = _end_to_end(cfg, cases, main_models, out)
    with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    _report(res, sc, out)
    _plot_confusion(res, out)
    return res


def _bootstrap(cases: dict, preds: dict[str, dict], n: int, seed: int = 0) -> dict:
    """95 % CIs of pooled macro-F1 / accuracy by resampling whole matches with replacement,
    plus the paired difference between the two learned models (same resamples)."""
    rng = np.random.default_rng(seed)
    ms = list(cases)
    names = [m for m in preds if m != "majority"]
    stats = {k: {"macro_f1": [], "accuracy": []} for k in names}
    diff = []
    for _ in range(n):
        pick = rng.choice(len(ms), len(ms), replace=True)
        y = np.concatenate([cases[ms[i]].y for i in pick])
        vals = {}
        for k in names:
            p = np.concatenate([preds[k][ms[i]] for i in pick])
            vals[k] = f1_score(y, p, labels=FINE_CLASSES, average="macro", zero_division=0)
            stats[k]["macro_f1"].append(vals[k]); stats[k]["accuracy"].append(float(np.mean(y == p)))
        if len(names) == 2:
            diff.append(vals[names[1]] - vals[names[0]])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    out = {k: {m: ci(v) for m, v in d.items()} for k, d in stats.items()}
    if diff:
        out[f"diff_macro_f1_{names[1]}_minus_{names[0]}"] = {"ci95": ci(diff), "p_le_0": float(np.mean(np.array(diff) <= 0))}
    out["n_resamples"] = n
    return out


def _end_to_end(cfg: Config, cases: dict, models: dict, out: Path) -> dict | None:
    path = cfg.output_dir / "task2" / "predictions.csv"
    if not path.exists():
        return None
    det = defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["hitter"]:
            det[r["match"]].append((int(r["frame"]), SIDE_INDEX[r["hitter"]]))
    tol = int(cfg["hits"]["primary_tolerance"])
    n_gt = n_det = n_matched = n_correct = 0
    ys, ps = [], []
    for m, c in cases.items():
        if m not in det or m not in models:
            continue
        md = c.md
        ev = []
        for f, s in sorted(det[m]):
            row = md.rows_for_frames(np.array([f]))[0]
            if row < 0:
                row = int(np.clip(np.searchsorted(md.frame, f), 0, md.n_frames - 1))
            ev.append({"frame": f, "side": s, "rally_idx": int(md.rally_idx[row])})
        X = event_features(md, c.feats, ev, float(cfg["pose"]["min_keypoint_score"]))
        pred = models[m].predict(X)
        res = match_events(c.frames, np.array([e["frame"] for e in ev]), tol)
        n_gt += len(c.y); n_det += len(ev); n_matched += len(res.pairs)
        for gi, pj in res.pairs:
            ys.append(c.y[gi]); ps.append(pred[pj])
            n_correct += c.y[gi] == pred[pj]
    if not ys:
        return None
    return {"tolerance": tol, "n_gt": n_gt, "n_detected": n_det, "n_matched": n_matched,
            "accuracy_on_matched": n_correct / n_matched,
            "macro_f1_on_matched": float(f1_score(ys, ps, labels=FINE_CLASSES, average="macro", zero_division=0)),
            "gt_detected_and_correct": n_correct / n_gt,
            "detections_correct": n_correct / n_det}


def _report(res: dict, sc: dict, out: Path) -> None:
    nf = len(next(iter(res["schemes"].values()))["models"][sc["main_model"]]["accuracy_per_match"])
    L = ["# Task 3: shot-type baseline (generated)", "",
         f"{res['n_hits']} annotated hits, leave-one-match-out ({nf} folds). Fixed hyperparameters (config `shots`), "
         f"class_weight=balanced. Pooled = all held-out predictions; per-match = mean ± std over the {nf} folds.", ""]
    for scheme, s in res["schemes"].items():
        L += [f"## {scheme} labels ({len(s['classes'])} classes)", "",
              "| model | accuracy | macro F1 (pooled) | macro F1 per match | weighted F1 |", "|---|---|---|---|---|"]
        for mname, r in s["models"].items():
            L.append(f"| {mname} | {r['accuracy']:.3f} | **{r['macro_f1']:.3f}** | {r['macro_f1_per_match_mean']:.3f} ± {r['macro_f1_per_match_std']:.3f} | {r['weighted_f1']:.3f} |")
        L.append("")
    b = res["bootstrap_fine"]
    L += [f"Match-level bootstrap ({b['n_resamples']} resamples of the 12 held-out matches), 95 % CI, fine labels: "
          + "; ".join(f"{k}: macro F1 [{v['macro_f1'][0]:.3f}, {v['macro_f1'][1]:.3f}], accuracy [{v['accuracy'][0]:.3f}, {v['accuracy'][1]:.3f}]"
                      for k, v in b.items() if isinstance(v, dict) and "macro_f1" in v)]
    for k, v in b.items():
        if k.startswith("diff_"):
            L.append(f"Paired difference {k[5:]}: CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}], share of resamples ≤ 0: {v['p_le_0']:.3f}")
    L.append("")
    main = res["schemes"]["fine"]["models"][sc["main_model"]]
    L += [f"## Per class ({sc['main_model']}, fine labels, pooled held-out)", "", "| class | precision | recall | F1 | n |", "|---|---|---|---|---|"]
    for c, v in main["per_class"].items():
        L.append(f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['n']} |")
    L += ["", f"## Feature-group ablation ({sc['main_model']}, {sc['ablation_max_iter']} iterations, fine labels)", "",
          "| features | macro F1 | accuracy |", "|---|---|---|"]
    for k, v in res["ablation_fine"].items():
        L.append(f"| {k} | {v['macro_f1']:.3f} | {v['accuracy']:.3f} |")
    L += ["", "## Per player (fine labels, held-out)", "",
          "Players repeat across matches (Shi Yu Qi is in 7 of 12). A player is 'unseen' only when their single match is held out.", "",
          "| player | matches | hits | accuracy | macro F1 |", "|---|---|---|---|---|"]
    for p, v in res["per_player"].items():
        L.append(f"| {p} | {v['matches']} | {v['n']} | {v['accuracy']:.3f} | {v['macro_f1']:.3f} |")
    L += ["", "| hitter in training data? | hits | accuracy | macro F1 |", "|---|---|---|---|"]
    for k, v in res["seen_vs_unseen_player"].items():
        L.append(f"| {k} | {v['n']} | {v['accuracy']:.3f} | {v['macro_f1']:.3f} |")
    L += ["", "| rallies | hits | accuracy | macro F1 |", "|---|---|---|---|"]
    for k, v in res["clean_vs_flagged_rallies"].items():
        L.append(f"| {k} | {v['n']} | {v['accuracy']:.3f} | {v['macro_f1']:.3f} |")
    e = res.get("end_to_end")
    if e:
        L += ["", f"## End-to-end on Task 2 detections (±{e['tolerance']})", "",
              f"- detections {e['n_detected']}, annotated hits {e['n_gt']}, matched {e['n_matched']}",
              f"- accuracy on matched detections {e['accuracy_on_matched']:.3f}, macro F1 {e['macro_f1_on_matched']:.3f}",
              f"- annotated hits both detected and correctly classified: {e['gt_detected_and_correct']:.3f}",
              f"- detections that are a real hit with the correct class: {e['detections_correct']:.3f}"]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def _plot_confusion(res: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(res["confusion_fine"], dtype=float)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(FINE_CLASSES)), FINE_CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(FINE_CLASSES)), FINE_CLASSES)
    for i in range(len(FINE_CLASSES)):
        for j in range(len(FINE_CLASSES)):
            if cm[i, j]:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("annotated")
    ax.set_title("Shot types, held-out (row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.savefig(out / "confusion_fine.png", dpi=110)
    plt.close(fig)
