"""Trim `state/predictions.jsonl` into gzipped monthly cold-storage shards.

Why this exists
---------------
The prediction ledger is append-only and was growing ~120k rows (~60 MB)
a month, accelerating. In August 2026 it crossed GitHub's hard 100 MB
per-file limit, and from then on *every* pipeline push was rejected by
the remote pre-receive hook:

    remote: error: File state/predictions.jsonl is 102.35 MB;
            this exceeds GitHub's file size limit of 100.00 MB

The runs themselves succeeded — they just couldn't persist their state,
which is why the dashboard showed no predictions after 2026-08-21.

What it does
------------
Moves rows older than the hot window out of `predictions.jsonl` and into
`state/predictions_archive/YYYY-MM.jsonl.gz` (~8x compression), then
records the cutoff in `watermark.json`. Readers go through
`trading_bot.state.predictions_archive.iter_predictions()`, which walks
the shards and the hot file transparently — no history is lost.

Two independent bounds, whichever bites first:
  --keep-days   drop rows older than N days out of the hot file
  --max-bytes   hard size ceiling; if the hot file is still over it,
                whole days are moved oldest-first until it fits. This is
                the safety net that keeps us under 100 MB even if daily
                prediction volume keeps climbing. The floor is one day —
                the newest day always stays hot so same-day reads and the
                reflection write-back keep working, which means a single
                day larger than --max-bytes would still exceed it (at
                ~2 MB/day that is ~30x away).

Idempotent: running twice in a row is a no-op the second time.

Usage
-----
    python scripts/trim_predictions_ledger.py
    python scripts/trim_predictions_ledger.py --keep-days 45
    python scripts/trim_predictions_ledger.py --dry-run
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_bot.state import paths  # noqa: E402
from trading_bot.state import predictions_archive as pa  # noqa: E402

log = logging.getLogger("trim_predictions_ledger")

# 30 days of hot rows is ~60 MB at Aug-2026 volume. Windowed consumers
# (metrics 14d, tool_attribution 60d) reach further back than this, but
# they read through iter_predictions() which spans the archive, so the
# hot window only has to be big enough to keep the common path cheap.
_DEFAULT_KEEP_DAYS = 30
# Hard ceiling for the hot file. Well under GitHub's 100 MB so a busy
# week between trims can't push us over.
_DEFAULT_MAX_BYTES = 60_000_000

# Same logical primary key smart_merge_state.py dedups on.
_KEY_FIELDS = ("strategy_id", "region", "ticker", "prediction_date")


def _key(rec: dict):
    return tuple(rec.get(f) for f in _KEY_FIELDS)


def _encode(rec: dict) -> str:
    return json.dumps(rec)


def _sort_key(rec: dict):
    return (
        rec.get("prediction_date") or "",
        rec.get("strategy_id") or "",
        rec.get("region") or "",
        rec.get("ticker") or "",
    )


def _row_date(rec: dict) -> str | None:
    """The row's shard/window key, or None if it has no usable date."""
    d = rec.get("prediction_date")
    return d if isinstance(d, str) and len(d) >= 7 else None


def _read_hot() -> list[dict]:
    p = paths.predictions_path()
    if not p.exists():
        return []
    rows = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("Skipping malformed ledger line")
    return rows


def _byte_size(rows: list[dict]) -> int:
    return sum(len(_encode(r)) + 1 for r in rows)


def _write_gz(path: Path, rows: list[dict]) -> None:
    """Deterministic gzip (mtime=0) so an unchanged shard re-serialises to
    identical bytes and doesn't churn a new blob into git every run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(_encode(r) + "\n" for r in rows).encode()
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
            gz.write(payload)


def _merge_into_shard(month: str, new_rows: list[dict]) -> tuple[int, int]:
    """Append new rows to a month's shard, deduped on the primary key.
    Returns (rows_added, total_rows_in_shard)."""
    existing = list(pa.iter_shard(month))
    seen = {_key(r) for r in existing}
    added: list[dict] = []
    for r in new_rows:
        k = _key(r)
        if k in seen:
            continue
        seen.add(k)
        added.append(r)
    merged = sorted(existing + added, key=_sort_key)
    _write_gz(pa.shard_path(month), merged)
    return len(added), len(merged)


def plan_split(
    rows: list[dict], *, keep_days: int, max_bytes: int, today: date
) -> tuple[list[dict], list[dict], str | None]:
    """Partition rows into (keep_hot, to_archive, archived_before).

    Rows are split on whole `prediction_date` boundaries so the watermark
    stays a clean "everything before this date is archived" statement —
    a date is never half in the hot file and half in the archive.
    """
    cutoff = (today - timedelta(days=keep_days)).isoformat()

    by_date: dict[str, list[dict]] = defaultdict(list)
    undated: list[dict] = []
    for r in rows:
        d = _row_date(r)
        # Rows with no usable date can't be shard-keyed — keep them hot.
        (by_date[d] if d else undated).append(r)

    archive_dates = {d for d in by_date if d < cutoff}

    # Size ceiling: move whole days, oldest first, until the hot file fits.
    if max_bytes > 0:
        kept_dates = sorted(d for d in by_date if d not in archive_dates)
        size = _byte_size(undated) + sum(_byte_size(by_date[d]) for d in kept_dates)
        # Stop at the newest day — never archive the whole hot file.
        while size > max_bytes and len(kept_dates) > 1:
            oldest = kept_dates.pop(0)
            archive_dates.add(oldest)
            size -= _byte_size(by_date[oldest])

    # Both lists preserve the ledger's original append order. That matters:
    # rewriting the hot file in sorted order would rewrite every line and
    # defeat git's delta compression, turning each daily trim into a fresh
    # ~50 MB blob in history. Order-preserving means the trimmed file is
    # "yesterday's file minus a prefix", which deltas cheaply.
    keep = [r for r in rows if _row_date(r) not in archive_dates]
    to_archive = [r for r in rows if _row_date(r) in archive_dates]

    kept_dates = [d for d in (_row_date(r) for r in keep) if d]
    archived_before = min(kept_dates) if kept_dates else (cutoff if to_archive else None)
    return keep, to_archive, archived_before


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="trim_predictions_ledger")
    ap.add_argument("--keep-days", type=int, default=_DEFAULT_KEEP_DAYS)
    ap.add_argument("--max-bytes", type=int, default=_DEFAULT_MAX_BYTES)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    hot_path = paths.predictions_path()
    rows = _read_hot()
    if not rows:
        log.info("No rows in %s — nothing to trim", hot_path)
        return 0

    before_bytes = _byte_size(rows)
    keep, to_archive, archived_before = plan_split(
        rows, keep_days=args.keep_days, max_bytes=args.max_bytes, today=date.today()
    )

    log.info(
        "Ledger %s: %d rows (%.1f MB) → keep %d (%.1f MB), archive %d",
        hot_path, len(rows), before_bytes / 1e6,
        len(keep), _byte_size(keep) / 1e6, len(to_archive),
    )

    if not to_archive:
        log.info("Hot file already within bounds — no-op")
        return 0

    by_month: dict[str, list[dict]] = defaultdict(list)
    for r in to_archive:
        m = pa.month_of(r)
        if m:
            by_month[m].append(r)

    if args.dry_run:
        for month in sorted(by_month):
            log.info("  [dry-run] %s ← %d rows", pa.shard_path(month).name, len(by_month[month]))
        log.info("  [dry-run] watermark archived_before=%s", archived_before)
        return 0

    total_added = 0
    for month in sorted(by_month):
        added, total = _merge_into_shard(month, by_month[month])
        total_added += added
        log.info("  %s: +%d rows (%d total, %.1f MB gz)",
                 pa.shard_path(month).name, added, total,
                 pa.shard_path(month).stat().st_size / 1e6)

    # Rewrite the hot file only after every shard is safely on disk, so an
    # interrupted run can't lose rows (worst case it re-archives them,
    # which the shard dedup absorbs).
    hot_path.write_text("".join(_encode(r) + "\n" for r in keep))

    # The watermark must never move backwards — smart_merge uses it to
    # reject resurrected rows, and a regression would let them back in.
    prev = pa.read_watermark()
    if archived_before and (prev is None or archived_before > prev):
        pa.write_watermark(archived_before, rows_archived=total_added)
    elif prev:
        log.info("Watermark stays at %s (computed %s)", prev, archived_before)

    log.info(
        "Done: %s is now %.1f MB (was %.1f MB), %d rows archived",
        hot_path, hot_path.stat().st_size / 1e6, before_bytes / 1e6, total_added,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
