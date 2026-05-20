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


# (path, primary-key for dedup):
#   str            → dedup on a single field
#   tuple[str,...] → composite key built from named fields
#   None           → no dedup (use only when the file is genuinely
#                    append-only with no logical primary key)
TARGETS = [
    ("state/ledger.jsonl", "trade_id"),
    # predictions.jsonl is logically keyed by (strategy_id, region,
    # ticker, prediction_date). Without dedup, every smart-merge
    # fallback would concatenate both sides → duplicate rows for
    # every overlapping (strategy, ticker, date), which biased the
    # downstream IC calculations until this fix landed.
    ("state/predictions.jsonl", ("strategy_id", "region", "ticker", "prediction_date")),
    ("state/macro/predictions.jsonl", "prediction_id"),
]


def _composite_key(rec: dict, key_spec):
    """Build the dedup key. Returns None if any required field is
    missing — those rows fall through to append-without-dedup."""
    if isinstance(key_spec, str):
        return rec.get(key_spec)
    if isinstance(key_spec, (list, tuple)):
        parts: list = []
        for k in key_spec:
            v = rec.get(k)
            if v is None:
                return None
            parts.append(v)
        return tuple(parts)
    return None


def merge_file(repo_root: Path, save_root: Path, rel_path: str, key) -> int:
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
        seen: set = set()
        merged: list[str] = []
        for line in local_lines + remote_lines:
            try:
                k = _composite_key(json.loads(line), key)
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
