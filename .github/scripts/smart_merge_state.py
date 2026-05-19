"""Workflow helper: merge our local JSONL state additions on top of remote.

When the runner's `git push` fails because main has drifted, we:
1. Save our local JSONL files (which have new appended rows).
2. `git reset --hard origin/main` to pick up everyone else's changes.
3. Run this script — it re-applies our additions by line-union (dedup by
   primary key per file).
4. Rebuild dashboard. Commit. Push.

Idempotent and conflict-free for append-only JSONL files: union-dedup
respects whoever wrote which row first.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


# (path, primary-key field for dedup; None = position-based, no dedup)
TARGETS = [
    ("state/ledger.jsonl", "trade_id"),
    ("state/predictions.jsonl", None),
    ("state/macro/predictions.jsonl", "prediction_id"),
]


def merge_file(repo_root: Path, save_root: Path, rel_path: str, key: str | None) -> int:
    """Merge the saved local copy into the repo's (just-reset) version.
    Returns the number of new lines added from local."""
    repo_file = repo_root / rel_path
    save_file = save_root / rel_path
    if not save_file.exists():
        return 0

    remote_lines: list[str] = []
    if repo_file.exists():
        for line in repo_file.read_text().splitlines():
            line = line.strip()
            if line:
                remote_lines.append(line)

    local_lines: list[str] = []
    for line in save_file.read_text().splitlines():
        line = line.strip()
        if line:
            local_lines.append(line)

    if key is None:
        merged = remote_lines + local_lines
    else:
        # Local first, then remote. Dedupe keeps the FIRST occurrence
        # of each primary key — so a row edited in place locally (e.g.
        # state/ledger.jsonl rows getting their exit_date / pnl_gbp
        # populated by the exit phase) wins over the stale remote
        # version that pre-dates this run. Earlier ordering (remote
        # first) silently dropped every trade closure on any exit cron
        # that raced with another push and fell into this fallback.
        seen: set[str] = set()
        merged: list[str] = []
        for line in local_lines + remote_lines:
            try:
                k = json.loads(line).get(key)
            except json.JSONDecodeError:
                continue
            if k is None:
                merged.append(line)
                continue
            if k in seen:
                continue
            seen.add(k)
            merged.append(line)

    repo_file.parent.mkdir(parents=True, exist_ok=True)
    repo_file.write_text("\n".join(merged) + "\n" if merged else "")
    new_lines = len(merged) - len(remote_lines)
    return max(new_lines, 0)


def main(repo_root_str: str, save_root_str: str) -> int:
    repo_root = Path(repo_root_str)
    save_root = Path(save_root_str)
    total_added = 0
    for rel_path, key in TARGETS:
        added = merge_file(repo_root, save_root, rel_path, key)
        print(f"  {rel_path}: +{added} from local")
        total_added += added
    print(f"Total new rows merged: {total_added}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: smart_merge_state.py <repo-root> <local-save-root>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
