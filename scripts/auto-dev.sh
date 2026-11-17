#!/usr/bin/env bash
# Closed-loop AI development: Gemini CLI plans -> Claude Code builds/tests -> commit, push, PR.
# The PR is then reviewed by .github/workflows/ai-pr-review.yml.
#
# Usage: scripts/auto-dev.sh "Describe the task"
# Env:   BASE_BRANCH (default: current branch), GEMINI_MODEL (optional), CLAUDE_MODEL (optional)
set -euo pipefail

die() { echo "auto-dev: ERROR: $*" >&2; exit 1; }
log() { echo "auto-dev: $*"; }

# ---------- 0. tools on PATH (nvm-installed gemini/claude, uv in ~/.local/bin) ----------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v gemini >/dev/null || ! command -v claude >/dev/null; then
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  # shellcheck disable=SC1091
  [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
fi
for t in git gh gemini claude uv; do
  command -v "$t" >/dev/null || die "'$t' not found on PATH"
done

# ---------- 1. arguments and preconditions ----------
TASK="${1:-}"
[ -n "$TASK" ] || die "usage: $0 \"task description\""

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || die "not inside a git repository"
cd "$ROOT"

[ -z "$(git status --porcelain)" ] || die "working tree is not clean; commit or stash first"
BASE="${BASE_BRANCH:-$(git branch --show-current)}"
[ -n "$BASE" ] || die "detached HEAD; set BASE_BRANCH"

SPEC=".claude/current-task.md"
MSG_FILE=".claude/commit-message.txt"
TEST_CMD=(uv run pytest -q)

BRANCH="ai/task-$(date +%Y%m%d-%H%M%S)"
git checkout -q -b "$BRANCH"
log "on branch $BRANCH (base: $BASE)"
trap 'rc=$?; [ "$rc" -eq 0 ] || echo "auto-dev: stopped (exit $rc). Branch $BRANCH is left as-is for inspection (git checkout $BASE to go back)." >&2' EXIT

mkdir -p .claude
rm -f "$SPEC" "$MSG_FILE"

# ---------- 2. Gemini: read-only research + spec ----------
log "Gemini: writing the spec to $SPEC ..."
GEMINI_PROMPT="You are the Tech Lead & Researcher described in GEMINI.md. Follow it strictly.
Task: ${TASK}

Inspect this repository (structure, CLAUDE.md, pyproject.toml, the relevant modules and tests),
search the web for current documentation if the task involves external libraries or APIs, and
then print ONLY the Markdown technical specification with the four sections required by
GEMINI.md: 1. Objective, 2. Allowed Files to Create/Modify, 3. Detailed Implementation Plan
per Module/Function, 4. Verification & Test Commands. Do not modify any file."
GEMINI_ARGS=(--approval-mode plan -o text)
[ -n "${GEMINI_MODEL:-}" ] && GEMINI_ARGS+=(-m "$GEMINI_MODEL")
gemini "${GEMINI_ARGS[@]}" -p "$GEMINI_PROMPT" > "$SPEC" || die "gemini failed"

[ -s "$SPEC" ] && grep -q '[^[:space:]]' "$SPEC" || die "Gemini produced an empty spec ($SPEC)"
for s in "Objective" "Allowed Files" "Implementation Plan" "Verification"; do
  grep -qi "^##.*$s" "$SPEC" || die "spec is missing the '$s' section; see $SPEC"
done
[ -z "$(git status --porcelain)" ] || die "Gemini changed files in the repo (it must be read-only): $(git status --porcelain | head -5)"
log "spec ready ($(wc -l < "$SPEC") lines)"

# ---------- 3. Claude Code: implement + test until green ----------
log "Claude: implementing the spec ..."
CLAUDE_PROMPT="You are the Senior Implementer described in CLAUDE.md. Follow it strictly.
1. Read ${SPEC} completely; it is the spec for this task: ${TASK}
2. Implement it, creating/modifying only the files listed in its section 2.
3. Run '${TEST_CMD[*]}' (and the commands in the spec's section 4) and keep fixing the code
   until every test passes. Never skip, delete or weaken tests.
4. Do not run git add/commit/push. When done, write a single Conventional Commits subject line
   (e.g. 'feat(shots): add rally-level F1') to ${MSG_FILE}.
Finish with a short summary of what changed and the final test result."
CLAUDE_ARGS=(
  --allowedTools "Edit" "Write" "Read" "Glob" "Grep"
  "Bash(git status)" "Bash(git status:*)" "Bash(git diff)" "Bash(git diff:*)"
  "Bash(uv run pytest)" "Bash(uv run pytest:*)" "Bash(uv run python -m pytest:*)"
)
[ -n "${CLAUDE_MODEL:-}" ] && CLAUDE_ARGS+=(--model "$CLAUDE_MODEL")
claude -p "$CLAUDE_PROMPT" "${CLAUDE_ARGS[@]}" || die "claude failed"

# ---------- 4. independent test gate ----------
log "re-running tests: ${TEST_CMD[*]}"
"${TEST_CMD[@]}" || die "tests fail after Claude's changes; not committing"

# ---------- 5. commit, push, PR ----------
git add -A
git reset -q -- "$SPEC" "$MSG_FILE" 2>/dev/null || true   # spec/message never go into the commit
git diff --cached --quiet && die "Claude made no changes; nothing to commit"

CC_RE='^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([a-z0-9._/-]+\))?!?: .{1,72}$'
SUBJECT="$(head -n1 "$MSG_FILE" 2>/dev/null | tr -d '\r' || true)"
if ! [[ "$SUBJECT" =~ $CC_RE ]]; then
  log "no valid Conventional Commits subject from Claude ('$SUBJECT'); using a fallback"
  SUBJECT="feat: $(echo "$TASK" | tr '\n' ' ' | cut -c1-60)"
fi

git commit -q -F - <<EOF
$SUBJECT

Task: $TASK

Spec by Gemini CLI, implemented and tested by Claude Code (scripts/auto-dev.sh).

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
log "committed: $SUBJECT"

git push -u origin "$BRANCH"

BODY="$(mktemp)"
{
  echo "## Task"; echo; echo "$TASK"; echo
  echo "## Spec (Gemini CLI)"; echo
  echo "<details><summary>.claude/current-task.md</summary>"; echo
  cat "$SPEC"; echo; echo "</details>"; echo
  echo "## Verification"; echo
  echo "\`${TEST_CMD[*]}\` passed locally before push (re-run by auto-dev.sh, not just by Claude)."
  echo; echo "🤖 Generated with [Claude Code](https://claude.com/claude-code) via \`scripts/auto-dev.sh\`"
} > "$BODY"
gh pr create --base "$BASE" --head "$BRANCH" --title "$SUBJECT" --body-file "$BODY"
rm -f "$BODY"
log "done. The AI PR review workflow will comment on the PR."
