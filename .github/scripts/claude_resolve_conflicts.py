"""Claude-assisted git rebase conflict resolver for CI workflows.

Used as a recovery step when a workflow's normal rebase + push retry
loop has exhausted its attempts. This script:

1. Reconstructs the conflict by running `git pull --rebase` once more.
2. Detects each unmerged file (with `<<<<<<<`/`=======`/`>>>>>>>`
   markers in place).
3. Sends each conflicting file to Claude with surrounding context
   describing what the file represents and which side to prefer.
4. Writes the resolved content back and `git add`s it.
5. `git rebase --continue` to finalise.
6. Pushes.

If Claude returns garbage (no JSON, missing `resolved` field, etc.)
the script aborts the rebase and exits 1 — the caller's CI step
fails loudly so we don't push corrupt content.

Design notes:
- One Sonnet call per conflicting file. Small per-file context keeps
  each call fast (<30s typical).
- Files are categorised by path so Claude knows what's safe to
  overwrite vs what needs careful merging. The "policy" string is
  injected into the prompt.
- Resolution is bounded to text files under 256KB. Anything larger
  bails immediately to keep the worst-case cost predictable.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("claude_resolve_conflicts")


_MAX_FILE_BYTES = 256 * 1024
_PROMPT_CHAR_BUDGET = 60_000


# Path-glob → resolution policy. The policy is a short instruction
# Claude reads to decide which side to favour on ambiguous merges.
_POLICIES: list[tuple[str, str]] = [
    (
        "docs/news/index.html",
        "Generated artifact — fully regenerated from briefs on every "
        "run. Prefer the LOCAL side; the remote's version is older "
        "than this run's output. If you spot a stale element on the "
        "local side that the remote clearly fixed, keep the local "
        "structure but adopt the remote's fix surgically.",
    ),
    (
        "docs/news/*.html",
        "Generated news page. Prefer LOCAL; this run regenerated it.",
    ),
    (
        "docs/news/*.json",
        "Generated state. Prefer LOCAL.",
    ),
    (
        "docs/data.json",
        "Generated dashboard data. Prefer LOCAL since this run wrote it.",
    ),
    (
        "state/daily_news/*.json",
        "Per-date brief state file. If both sides have the same date, "
        "they should have the same content; prefer LOCAL.",
    ),
    (
        "state/predictions/*",
        "Per-strategy prediction files. Each line is keyed by ticker + "
        "date; on conflict, prefer the union (keep both sides' rows) "
        "and drop duplicates by (strategy_id, ticker, prediction_date).",
    ),
    (
        "state/ledger.jsonl",
        "Trade ledger. Each row is keyed by trade_id. On conflict, "
        "prefer the row with more populated fields (e.g. exit_date "
        "set, pnl_gbp computed) over an emptier counterpart for the "
        "same trade_id. Concatenate non-conflicting rows.",
    ),
    (
        "state/predictions.jsonl",
        "Append-only predictions log. Union of both sides; dedupe by "
        "(strategy_id, ticker, prediction_date). Keep the side with "
        "richer fields (reflection, actual_class) on dupes.",
    ),
    (
        "*.py",
        "Source code. Read both sides carefully and produce a "
        "merge that preserves both intents where possible. If the "
        "two sides made conflicting changes to the same function, "
        "preserve the LOCAL side's logic — that's this run's "
        "intent — but lift any error-handling or doc improvements "
        "from the remote.",
    ),
    (
        "*",
        "Unknown file type. Preserve both intents where you can; "
        "if forced to pick, prefer the LOCAL side.",
    ),
]


def _policy_for(path: str) -> str:
    """Pick the first matching policy by path."""
    from fnmatch import fnmatch
    for pattern, policy in _POLICIES:
        if fnmatch(path, pattern):
            return policy
    return "Preserve both intents where possible; if forced to pick, prefer LOCAL."


def _unmerged_files() -> list[str]:
    """List files currently in conflict (diff --name-only --diff-filter=U)."""
    r = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        capture_output=True, text=True, check=False,
    )
    if r.returncode != 0:
        return []
    return [line.strip() for line in r.stdout.splitlines() if line.strip()]


def _trigger_conflict_state() -> list[str]:
    """If we're not already mid-rebase, pull once to recreate the
    conflict. Returns the list of unmerged files; empty if there's
    actually no conflict (i.e. the failure was elsewhere)."""
    already = _unmerged_files()
    if already:
        return already
    # Fetch + rebase deliberately, expecting conflict. Hard 60-sec
    # timeout on the fetch so an unreachable remote can't hang the
    # workflow indefinitely (the runner's overall timeout would
    # eventually trip but that's much later and much noisier).
    subprocess.run(
        ["git", "fetch", "origin", "main"],
        check=False, timeout=60,
    )
    r = subprocess.run(
        ["git", "rebase", "origin/main"],
        capture_output=True, text=True, check=False, timeout=60,
    )
    if r.returncode == 0:
        # Rebase succeeded cleanly — no conflict to resolve.
        return []
    return _unmerged_files()


def _resolve_one(path: str) -> bool:
    """Send one conflicting file to Claude, write the resolved bytes
    back, and `git add` it. Returns True on success."""
    p = Path(path)
    if not p.exists():
        log.warning("%s: file no longer exists — skipping", path)
        return False
    try:
        raw = p.read_bytes()
    except OSError as e:
        log.warning("%s: read failed: %s — skipping", path, e)
        return False
    if len(raw) > _MAX_FILE_BYTES:
        log.warning(
            "%s: %d bytes exceeds %d cap — skipping (manual resolution required)",
            path, len(raw), _MAX_FILE_BYTES,
        )
        return False
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        log.warning("%s: not utf-8 — skipping", path)
        return False
    if "<<<<<<<" not in text:
        # Already resolved somehow (maybe deleted on one side); just stage it.
        subprocess.run(["git", "add", "--", path], check=False)
        return True

    policy = _policy_for(path)
    log.info("%s: invoking Claude (%d bytes, policy: %s…)", path, len(raw), policy[:60])

    prompt = _build_prompt(path, text, policy)
    if len(prompt) > _PROMPT_CHAR_BUDGET:
        log.warning("%s: prompt exceeds budget (%d chars) — skipping", path, len(prompt))
        return False

    from trading_bot.llm.claude_code import ClaudeCodeError, run_claude_for_json
    try:
        response = run_claude_for_json(prompt, model="sonnet", timeout_seconds=180)
    except ClaudeCodeError as e:
        log.error("%s: Claude resolution failed: %s", path, e)
        return False

    if not isinstance(response, dict):
        log.error("%s: Claude returned non-dict response", path)
        return False

    resolved = response.get("resolved")
    if not isinstance(resolved, str) or not resolved.strip():
        log.error("%s: Claude response missing or empty `resolved` field", path)
        return False
    if "<<<<<<<" in resolved or ">>>>>>>" in resolved:
        log.error("%s: Claude's resolution still contains conflict markers", path)
        return False

    rationale = str(response.get("rationale") or "").strip()
    if rationale:
        log.info("%s: %s", path, rationale[:200])

    # Trailing newline conventions — preserve whatever the original
    # file had so diffs stay minimal.
    if text.endswith("\n") and not resolved.endswith("\n"):
        resolved += "\n"

    p.write_text(resolved, encoding="utf-8")
    subprocess.run(["git", "add", "--", path], check=False)
    return True


def _build_prompt(path: str, text: str, policy: str) -> str:
    return f"""You are resolving a git rebase conflict in a CI
workflow's recovery step. The file below contains conflict markers
(`<<<<<<<`, `=======`, `>>>>>>>`). The LOCAL side is the work this
CI run produced; the REMOTE side is what landed on `main` since this
run started.

## File

Path: `{path}`

## Resolution policy

{policy}

## Conflicting content

```
{text}
```

## Output

Return JSON ONLY, no prose:

```json
{{
  "resolved": "<the entire resolved file content, with NO conflict markers anywhere>",
  "rationale": "<one short sentence on what you preferred and why>"
}}
```

The `resolved` value must be the complete file contents. Don't
truncate or summarise — include every line that should appear in
the final file. Strip every `<<<<<<<`, `=======`, and `>>>>>>>`
marker; the result must apply cleanly.
"""


def main() -> int:
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        log.error("CLAUDE_CODE_OAUTH_TOKEN missing — cannot run Claude resolution")
        return 2

    conflicts = _trigger_conflict_state()
    if not conflicts:
        log.info("No conflicts to resolve — the failure must have been elsewhere.")
        return 0

    log.info("Conflicting files (%d): %s", len(conflicts), conflicts)

    failures: list[str] = []
    for path in conflicts:
        if not _resolve_one(path):
            failures.append(path)

    if failures:
        log.error("Unresolved conflicts in: %s", failures)
        subprocess.run(["git", "rebase", "--abort"], check=False)
        return 1

    # Continue the rebase, then push.
    cont = subprocess.run(
        ["git", "rebase", "--continue"],
        env={**os.environ, "GIT_EDITOR": "true"},
        capture_output=True, text=True, check=False,
    )
    if cont.returncode != 0:
        log.error("git rebase --continue failed: %s", cont.stderr[:400])
        subprocess.run(["git", "rebase", "--abort"], check=False)
        return 1

    push = subprocess.run(["git", "push", "origin", "HEAD:main"], capture_output=True, text=True, check=False)
    if push.returncode != 0:
        log.error("Final push failed: %s", push.stderr[:400])
        return 1

    log.info("Conflict resolution + push succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
