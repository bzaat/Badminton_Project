"""Task 4 runner on the synthetic match: outputs exist and static players measure ~0 movement."""
import csv

from badminton_analysis.task4 import run

from . import synthetic_bfmd as syn
from .test_build_synthetic import cfg, truth  # noqa: F401  (pytest fixtures)


def test_task4_outputs_on_synthetic(cfg):  # noqa: F811
    cfg.raw["movement"]["min_main_camera_frac"] = 0.5
    val = run(cfg, [syn.MATCH])
    out = cfg.output_dir / "task4"
    rows = list(csv.DictReader(open(out / "rally_player_movement.csv")))
    assert val["rallies_used"] == 1 and val["rallies_total"] == 2     # rally 1 is not on the main camera
    assert {r["side"] for r in rows} == {"top", "bottom"}
    assert all(float(r["distance_m"]) < 1e-6 for r in rows)             # synthetic players stand still
    top = next(r for r in rows if r["side"] == "top")
    assert top["player"] == "Alpha A" and float(top["zone_rear_centre"]) > 0.99  # 4.7 m from net: rear third (4.47-6.7 m), centre
    shots = list(csv.DictReader(open(out / "shot_recovery.csv")))
    assert len(shots) == 1 and shots[0]["shot_type"] == "serve" and float(shots[0]["displacement_m"]) < 1e-6
    assert (out / "report.md").exists() and (out / "occupancy_heatmaps.png").exists()
