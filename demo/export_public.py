"""Write demo/public_results.json: aggregate results for the public demo (no frame-level BFMD data).

    uv run python demo/export_public.py

Reads the Milestone 1 outputs (`python -m badminton_analysis all`). The file holds only numbers that
REPORT.md already publishes: hit-detection scores per tolerance and per match, shot-type scores,
the confusion matrix and per-match movement summaries.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from badminton_analysis.config import load_config  # noqa: E402
from badminton_analysis.demo_data import load_outputs, results_summary  # noqa: E402

OUT = Path(__file__).with_name("public_results.json")


def main() -> int:
    summary = results_summary(load_outputs(load_config().output_dir))
    if summary is None:
        sys.exit("Task 2/3 outputs not found: run `python -m badminton_analysis all` first.")
    OUT.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB): "
          f"{len(summary['hit']['per_match'])} matches, {len(summary['movement'])} player-match rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
