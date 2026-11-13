"""Command line: python -m badminton_analysis <command> [options]

    list                      singles matches found in the BFMD package
    build    [--match S]      build / refresh the per-match cache (dataset/bfmd_m1)
    validate [--match S]      data checks + frame-alignment measurements -> outputs/task1/
    viz --match S --game G --rally R   rally figure -> outputs/task1/viz/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from .config import load_config


def _select(cfg, sub: str | None) -> list[str]:
    from .build import singles_matches
    ms = singles_matches(cfg)
    if sub:
        ms = [m for m in ms if sub.lower() in m.lower()]
    if not ms:
        sys.exit(f"no singles match matches {sub!r}")
    return ms


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m badminton_analysis", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML config (default configs/default.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    b = sub.add_parser("build"); b.add_argument("--match"); b.add_argument("--force", action="store_true")
    v = sub.add_parser("validate"); v.add_argument("--match"); v.add_argument("--no-alignment", action="store_true")
    z = sub.add_parser("viz")
    z.add_argument("--match", required=True); z.add_argument("--game", type=int, required=True)
    z.add_argument("--rally", type=int, required=True, help="0-based rally index within the game")
    z.add_argument("--out", default=None)
    sub.add_parser("viz-samples", help="one typical rally per match + every suspect rally")
    t2 = sub.add_parser("task2", help="hit detection: LOMO tuning + evaluation -> outputs/task2/")
    t2.add_argument("--match", default=None, help="restrict to matches containing this (for quick tests)")
    t3 = sub.add_parser("task3", help="shot types: LOMO evaluation -> outputs/task3/")
    t3.add_argument("--match", default=None)
    t4 = sub.add_parser("task4", help="movement measurements -> outputs/task4/")
    t4.add_argument("--match", default=None)
    sub.add_parser("all", help="Milestone 1 end to end: build, validate, viz-samples, task2, task3, task4")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(a.config)

    if a.cmd == "all":
        t = time.time()
        for step in (["build"], ["validate", "--no-alignment"], ["viz-samples"], ["task2"], ["task3"], ["task4"]):
            print(f"\n===== {' '.join(step)} =====", flush=True)
            main((["--config", a.config] if a.config else []) + step)
        print(f"\nMilestone 1 done in {time.time() - t:.0f}s. Reports: {cfg.output_dir}/task{{1,2,3,4}}/ and REPORT.md")
        return 0
    if a.cmd == "list":
        for m in _select(cfg, None):
            print(m)
    elif a.cmd == "build":
        from .build import load_match
        for m in _select(cfg, a.match):
            t = time.time()
            md = load_match(cfg, m, rebuild=a.force)
            print(f"{m}: {md.n_frames} frames, {len(md.rallies)} rallies, {len(md.hits)} hits ({time.time() - t:.1f}s)")
    elif a.cmd == "validate":
        from .alignment import measure_match
        from .build import load_match
        from .court import CourtModel
        from .validate import validate_match, write_report
        model = CourtModel.from_config(cfg["court"])
        results, align = [], []
        for m in _select(cfg, a.match):
            from .raw import load_caption_winners
            results.append(validate_match(load_match(cfg, m), model, load_caption_winners(cfg.ann_root, m)))
            if not a.no_alignment:
                al = measure_match(cfg.ann_root, m, cfg.width, cfg.height)
                al.pop("hit_lag_hist", None)
                align.append(al)
            print(f"validated {m}", flush=True)
        path = write_report(results, align or None, cfg.output_dir / "task1")
        print(f"report: {path}")
    elif a.cmd == "viz":
        from .build import load_match
        from .viz import plot_rally
        m = _select(cfg, a.match)
        if len(m) > 1:
            sys.exit("ambiguous --match:\n  " + "\n  ".join(m))
        md = load_match(cfg, m[0])
        ri = md.find_rally(a.game, a.rally)
        out = a.out or str(cfg.output_dir / "task1" / "viz" / f"{m[0]}_g{a.game}_r{a.rally:03d}.png")
        print(json.dumps({"figure": plot_rally(md, ri, out)}))
    elif a.cmd == "viz-samples":
        import numpy as np
        from .build import load_match
        from .viz import plot_rally
        out_dir = cfg.output_dir / "task1" / "viz"
        n = 0
        for m in _select(cfg, None):
            md = load_match(cfg, m)
            ok = [i for i, r in enumerate(md.rallies) if not r["suspect"] and r["n_hits"] > 0]
            hits = np.array([md.rallies[i]["n_hits"] for i in ok])
            typical = ok[int(np.argmin(np.abs(hits - np.median(hits))))]  # rally with the median hit count
            picks = [(typical, "typical")] + [(i, "SUSPECT") for i, r in enumerate(md.rallies) if r["suspect"]]
            for i, tag in picks:
                r = md.rallies[i]
                plot_rally(md, i, out_dir / f"{tag}_{m}_g{r['game']}_r{r['index']:03d}.png")
                n += 1
        print(f"wrote {n} figures to {out_dir}")
    elif a.cmd == "task2":
        from .task2 import run
        t = time.time()
        run(cfg, _select(cfg, a.match) if a.match else None)
        print(f"task2 done in {time.time() - t:.0f}s -> {cfg.output_dir / 'task2' / 'report.md'}")
    elif a.cmd == "task3":
        from .task3 import run as run3
        t = time.time()
        run3(cfg, _select(cfg, a.match) if a.match else None)
        print(f"task3 done in {time.time() - t:.0f}s -> {cfg.output_dir / 'task3' / 'report.md'}")
    elif a.cmd == "task4":
        from .task4 import run as run4
        t = time.time()
        run4(cfg, _select(cfg, a.match) if a.match else None)
        print(f"task4 done in {time.time() - t:.0f}s -> {cfg.output_dir / 'task4' / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
