"""CLI smoke tests: the entry point imports and every subcommand is registered."""
import subprocess
import sys

from badminton_analysis.config import REPO_ROOT


def test_cli_help_lists_all_commands():
    out = subprocess.run([sys.executable, "-m", "badminton_analysis", "--help"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True).stdout
    for cmd in ("list", "build", "validate", "viz", "viz-samples", "task2", "task3", "task4", "all"):
        assert cmd in out


def test_cli_list_runs_on_synthetic_data(tmp_path):
    from . import synthetic_bfmd as syn
    syn.write(tmp_path / "BFMD_data")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"base: {REPO_ROOT / 'configs' / 'default.yaml'}\npaths:\n  bfmd_root: {tmp_path / 'BFMD_data'}\n")
    out = subprocess.run([sys.executable, "-m", "badminton_analysis", "--config", str(cfg), "list"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True).stdout
    assert out.strip() == syn.MATCH
