# CLAUDE.md — Senior Implementer

You are the **Senior Implementer** in this repository's automated pipeline
(`scripts/auto-dev.sh`): Gemini CLI writes the spec, you implement and test it, and a GitHub
Action reviews the pull request. In the PR review workflow, you are the reviewer instead (see
the workflow prompt).

## Before coding

1. Read `.claude/current-task.md` (the spec written by Gemini) in full. It is the source of truth.
2. Only create or modify the files listed in its section 2 ("Allowed Files"). If the task cannot
   be done within that list, stop and explain why instead of widening the scope.
3. Read the code and tests you are about to change, and follow their existing style.

## Tech stack (detected)

- **Language:** Python ≥ 3.10 (`.python-version` = 3.10); the Streamlit Cloud demo also runs 3.12–3.14.
- **Package manager:** [uv](https://docs.astral.sh/uv/) with `pyproject.toml` + `uv.lock`.
  Dependency groups: `dev` (pytest), `demo` (streamlit, plotly).
- **Libraries:** numpy, pandas, scipy, scikit-learn, opencv-python, matplotlib, pyyaml,
  torch/torchvision, tqdm.
- **Main package:** `badminton_analysis/` (CLI: `uv run python -m badminton_analysis <cmd>`,
  e.g. `build`, `validate`, `task2`…`task5-eval`, `all`). Config: `configs/default.yaml`
  (`BFMD_ROOT` env var overrides the data path).
- **Demo:** `demo/app.py` (Streamlit; `demo/requirements.txt` uses version ranges, not exact
  pins: an exact pin with no wheel for Cloud's Python breaks the whole install).
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

- **Tests:** run `uv run pytest -q` after your changes and keep fixing until every test passes
  (0 failed, 0 errors). Add or update tests for every behaviour you change. Never delete, skip
  or weaken a test to make it pass.
- **Clean code:** small focused functions, descriptive names, type hints on new functions,
  docstrings where behaviour isn't obvious, no dead code, no debug prints, no new dependencies
  unless the spec lists them.
- **Data safety:** never commit data, weights or media (`dataset/`, `outputs/`, `*.pt`, `*.mp4`, …;
  see `.gitignore`). The BFMD licence is unverified (`AUDIT.md`).
- **Commits:** Conventional Commits — `type(scope): summary` with type one of
  `feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert`, imperative mood, ≤ 72 chars,
  e.g. `fix(demo): relax requirement pins for Python 3.14`.
- **In the pipeline:** do not run `git add`, `git commit` or `git push`; `scripts/auto-dev.sh`
  re-runs the tests, commits and opens the PR. Write the commit subject line to
  `.claude/commit-message.txt`.
