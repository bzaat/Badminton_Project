# CLAUDE.md — project context

Badminton match analytics on the BFMD dataset (rallies, hits, shots, player movement, shuttle
tracking). Results are in `REPORT.md`; data-licence notes are in `AUDIT.md`.

## Tech stack

- **Language:** Python ≥ 3.10 (`.python-version` = 3.10); the Streamlit Cloud demo also runs 3.12–3.14.
- **Package manager:** [uv](https://docs.astral.sh/uv/) with `pyproject.toml` + `uv.lock`.
  Dependency groups: `dev` (pytest), `demo` (streamlit, plotly).
- **Libraries:** numpy, pandas, scipy, scikit-learn, opencv-python, matplotlib, pyyaml,
  torch/torchvision, tqdm.
- **Main package:** `badminton_analysis/` (CLI: `uv run python -m badminton_analysis <cmd>`,
  e.g. `build`, `validate`, `task2`…`task5-eval`, `all`). Config: `configs/default.yaml`
  (`BFMD_ROOT` env var overrides the data path).
- **Demo:** `demo/app.py` (Streamlit). `demo/requirements.txt` uses version ranges, not exact
  pins: an exact pin with no wheel for Cloud's Python breaks the whole install.
- **Tests:** pytest in `tests/` (`testpaths`/`pythonpath` set in `pyproject.toml`). Tests marked
  `data` need the local BFMD package and are skipped without it; `tests/synthetic_bfmd.py`
  builds synthetic fixtures.

## Commands

```bash
uv sync                      # install deps (incl. dev group)
uv run pytest -q             # full test suite: must be 100% green
uv run pytest tests/test_x.py -q
```

## Rules

- **Tests:** keep `uv run pytest -q` fully green; add or update tests for every behaviour you
  change. Never delete, skip or weaken a test to make it pass.
- **Clean code:** small focused functions, descriptive names, type hints on new functions,
  docstrings where behaviour isn't obvious, no dead code, no debug prints, no new dependencies
  without a reason.
- **Data safety:** never commit data, weights or media (`dataset/`, `outputs/`, `*.pt`, `*.mp4`, …;
  see `.gitignore`). The BFMD licence is unverified (`AUDIT.md`).
- **Secrets:** never commit API keys, tokens or `.env` files.
- **Commits:** Conventional Commits, `type(scope): summary`, with type one of
  `feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert`, imperative mood, ≤ 72 chars,
  e.g. `fix(demo): relax requirement pins for Python 3.14`.
