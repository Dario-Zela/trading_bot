"""One-shot re-grade of predictions whose actual_return_pct was filled
against a stale OHLCV cache bar (the wrong trading day's open→close).

Root cause: the Grade-predictions cron runs at 05:00Z but
ohlcv-daily-update runs at 06:10Z, so on 2026-06-09 (and every prior
Monday with the same ordering) the grader served bars from Friday's
cache via the default 3-day grace window. Momentum predictions are
correlated with the prior day's move, which inflated UK-EU IC scores
to >+0.4 — the very signal evolution uses to pick promotion targets.

This script:
  1. Walks `state/predictions.jsonl`.
  2. For each row with `actual_return_pct` set, re-fetches the bar for
     the prediction's `prediction_date` (max_grace_days=0, the same
     fix applied to the live grader).
  3. If the fetched bar's `bar_date` matches the prediction date and
     the recomputed return differs from the stored one, overwrites
     `actual_return_pct` + `actual_class`.
  4. If the fetched bar doesn't exist or doesn't match the prediction
     date, leaves the stored value alone (worst case it stays wrong;
     we'd rather not delete and then have IC re-compute over a
     smaller, biased sample).

Usage:
    python scripts/regrade_predictions.py [--since YYYY-MM-DD] [--region uk-eu]
                                          [--dry-run]

Defaults: --since today-30, no region filter, dry-run off.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

from trading_bot.state.paths import predictions_path
from trading_bot.tools.history import get_history
from trading_bot.meta.reflection import _classify_outcome


log = logging.getLogger("regrade_predictions")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--since", default=(date.today() - timedelta(days=30)).isoformat(),
                   help="Only re-grade predictions with prediction_date >= this (YYYY-MM-DD).")
    p.add_argument("--region", default=None, help="Filter to one region (e.g. uk-eu).")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would change; don't write the file back.")
    args = p.parse_args()

    since = date.fromisoformat(args.since)
    path = Path(predictions_path())
    if not path.exists():
        log.error("predictions.jsonl not found at %s", path)
        return 1

    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    log.info("Loaded %d total prediction rows", len(rows))

    # Group rows that need re-grading by (date, region) → ticker set, so we
    # can batch one yfinance call per date+region combination.
    candidates: list[dict] = []
    for r in rows:
        if r.get("actual_return_pct") is None:
            continue
        d = r.get("prediction_date") or ""
        if not d or d < since.isoformat():
            continue
        if args.region and r.get("region") != args.region:
            continue
        candidates.append(r)
    log.info("Candidates for re-grade: %d", len(candidates))

    # Bucket by date so we can do one batched fetch per day.
    by_date: dict[str, set[str]] = {}
    for r in candidates:
        by_date.setdefault(r["prediction_date"], set()).add(r["ticker"])

    # Fetch fresh bars per date.
    fresh_bars: dict[str, dict[str, "Bar"]] = {}
    for d_iso, tickers in sorted(by_date.items()):
        d_obj = date.fromisoformat(d_iso)
        log.info("Fetching %d ticker(s) for %s ...", len(tickers), d_iso)
        bars = get_history(
            sorted(tickers), lookback_days=1, end_date=d_obj, max_grace_days=0,
        )
        # Keep only bars whose bar_date matches the requested day.
        kept: dict[str, "Bar"] = {}
        for tkr, bar_list in bars.items():
            if not bar_list:
                continue
            b = bar_list[-1]
            if b.bar_date != d_obj:
                continue
            kept[tkr] = b
        fresh_bars[d_iso] = kept
        log.info("  → %d/%d tickers have a valid bar dated %s",
                 len(kept), len(tickers), d_iso)

    # Apply corrections.
    n_unchanged = 0
    n_changed = 0
    n_no_bar = 0
    for r in candidates:
        d_iso = r["prediction_date"]
        tkr = r["ticker"]
        b = fresh_bars.get(d_iso, {}).get(tkr)
        if b is None:
            n_no_bar += 1
            continue
        if not b.open or b.open <= 0:
            n_no_bar += 1
            continue
        true_pct = round((b.close / b.open - 1.0) * 100.0, 2)
        stored_pct = float(r["actual_return_pct"])
        if abs(true_pct - stored_pct) < 0.01:
            n_unchanged += 1
            continue
        if not args.dry_run:
            r["actual_return_pct"] = true_pct
            r["actual_class"] = _classify_outcome(true_pct)
        n_changed += 1
        if n_changed <= 20:
            log.info("  %s %s %-12s  was %+8.2f%% → now %+8.2f%%",
                     d_iso, r.get("region","?"), tkr, stored_pct, true_pct)

    log.info("=" * 60)
    log.info("Re-grade summary: %d corrected, %d unchanged, %d missing-bar (kept as-is)",
             n_changed, n_unchanged, n_no_bar)

    if args.dry_run:
        log.info("Dry-run; predictions.jsonl NOT written.")
        return 0

    if n_changed == 0:
        log.info("No changes to write.")
        return 0

    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)
    log.info("Rewrote %s with %d corrected rows.", path, n_changed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
