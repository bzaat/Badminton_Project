"""Hit label-noise kit: sampling, blinding and evaluation (no video needed)."""
import json
from collections import Counter

import numpy as np
import pytest

from badminton_analysis import labelnoise as ln


def _hits():
    types = ["net_shot"] * 50 + ["lift"] * 30 + ["smash"] * 15 + ["serve"] * 4 + ["net_kill"] * 1
    return [dict(frame=1000 + 40 * i, shot_type=t, side="top" if i % 2 else "bottom", game=1, rally=i // 8)
            for i, t in enumerate(types)]


def test_sample_is_stratified_with_repeats_and_random_shift():
    items = ln.sample_hits(_hits(), n=20, n_repeat=4, seed=0)
    first = [it for it in items if it["repeat_of"] is None]
    reps = [it for it in items if it["repeat_of"] is not None]
    assert len(first) == 20 and len(reps) == 4
    c = Counter(it["shot_type"] for it in first)
    assert c["serve"] == 2 and c["net_kill"] == 1          # small strata: >= 2 where possible
    assert c["net_shot"] > c["lift"] > c["smash"]
    assert len({it["id"] for it in items}) == 24
    assert len({it["source_id"] for it in first}) == 20
    src = {it["source_id"]: it for it in first}
    for r in reps:                                        # a repeat shows the same hit
        assert src[r["repeat_of"]]["frame"] == r["frame"] and r["source_id"] == r["repeat_of"]
    shifts = [it["shift"] for it in items]
    assert min(shifts) >= -ln.MAX_SHIFT and max(shifts) <= ln.MAX_SHIFT and len(set(shifts)) > 3
    assert ln.sample_hits(_hits(), 20, 4, seed=0) == items  # deterministic


def test_window_contains_label_at_shifted_index():
    for s in range(-ln.MAX_SHIFT, ln.MAX_SHIFT + 1):
        w = ln.window({"frame": 500, "shift": s})
        assert len(w) == 2 * ln.HALF + 1 and w[ln.HALF - s] == 500


def test_evaluate_offsets_repeatability_and_detector(tmp_path):
    items = ln.sample_hits(_hits(), n=20, n_repeat=4, seed=1)
    key = {"match": "M", "items": items}
    labels = {}
    for k, it in enumerate(items):
        if k == 0:
            labels[it["id"]] = {"status": "no_hit"}
            continue
        off = 1 if it["repeat_of"] is None else 2            # repeats one frame later than first pass
        human = it["frame"] + off
        labels[it["id"]] = {"status": "contact", "frame": int(np.where(ln.window(it) == human)[0][0]),
                            "confidence": "sure"}
    preds = tmp_path / "pred.csv"
    rows = ["match,frame,hitter,score,matched_gt_frame"] + [f"M,{it['frame'] + 1},top,0.9," for it in items]
    preds.write_text("\n".join(rows) + "\nOTHER,5,top,0.9,\n")
    r = ln.evaluate(key, {"labels": labels}, preds, tmp_path / "out")
    o = r["offset_human_minus_bfmd"]
    first_ids = [it["id"] for it in items if it["repeat_of"] is None]
    expected_n = 20 - (items[0]["id"] in first_ids)
    assert r["n_labelled"] == expected_n
    assert o["mean"] == pytest.approx(1.0) and o["within_0"] == 0 and o["within_1"] == 1
    rep = r["repeatability"]
    assert all(d == 1 for d in rep["diffs"])
    d = r["task2_detector"]
    assert d["vs_human"]["within_0"] == 1.0 and d["vs_bfmd"]["mean"] == pytest.approx(1.0)
    assert (tmp_path / "out" / "report.md").read_text().startswith("# Hit label noise")
    json.loads((tmp_path / "out" / "metrics.json").read_text())
