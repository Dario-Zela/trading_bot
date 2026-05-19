"""Generate proper LLM reflections for back-filled rows the daily reflect
cron missed:

1. **Trades closed by recover_t212_strands**: the recovery script writes
   a generic "this row was back-filled" placeholder into outcome_notes
   and risks_observed. We replace it with a real Sonnet reflection.

2. **Untraded predictions for any day that's been graded but not
   reflected**: catches days where the prediction-reflection pass
   either didn't run (older pipelines) or failed. The weekly evolution
   agent reads predictions.jsonl heavily and benefits from pre-baked
   one-line reflections on every miss.

Idempotent — already-reflected rows are skipped. Triggered via the
recover-t212-strands workflow's `reflect` mode.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from datetime import date

from trading_bot.meta.reflection import (
    _apply_reflections,
    _reflect_strategy,
    reflect_predictions_on_day,
)
from trading_bot.state.paths import ledger_path, predictions_path


# Cap how many (date, region) pairs we'll process in one run. The
# Sonnet calls take ~1-3 minutes per strategy-chunk so a single
# workflow run can't realistically cover unbounded history. Today's
# uk-eu + us is the default; bump via the env var to back-fill more.
_DAYS_LIMIT = int(os.environ.get("REFLECT_DAYS_LIMIT", "2") or "2")


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("reflect_recovered")


_PLACEHOLDER_MARKER = "Recovered post-hoc by recover_t212_strands.py"


def _iter_records():
    p = ledger_path()
    if not p.exists():
        return
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    targets: list[dict] = []
    for rec in _iter_records():
        if rec.get("exit_reason") != "recovered":
            continue
        notes = rec.get("outcome_notes") or ""
        if _PLACEHOLDER_MARKER not in notes:
            # Already has a real reflection — skip
            continue
        targets.append(rec)

    if not targets:
        log.info("No recovered trades with placeholder reflections — nothing to do")
        return 0

    by_strategy: dict[str, list[dict]] = defaultdict(list)
    for t in targets:
        by_strategy[t["strategy_id"]].append(t)

    log.info(
        "Re-reflecting on %d recovered trade(s) across %d strategy/strategies",
        len(targets), len(by_strategy),
    )

    total_updated = 0
    for sid, trades in by_strategy.items():
        # `_reflect_strategy` keys by trade_id and uses a single Sonnet
        # call covering the day's basket; one call per strategy keeps
        # cost down even when one strategy has 5 recovered trades.
        any_date = trades[0].get("exit_date")
        try:
            on_date = date.fromisoformat(any_date) if any_date else date.today()
        except (TypeError, ValueError):
            on_date = date.today()

        try:
            notes = _reflect_strategy(sid, trades, on_date)
        except Exception as e:
            log.error("Reflection failed for %s: %s", sid, e)
            continue

        updated = _apply_reflections(trades, notes)
        log.info("Reflected %d/%d trades for %s", updated, len(trades), sid)
        total_updated += updated

    log.info("Trade reflection done — %d total trade(s) updated", total_updated)

    # Also catch up on any graded-but-unreflected predictions. Walk
    # the prediction file once to find candidate dates+regions, then
    # call the per-day reflector for each. Bounded by the
    # `reflection` field guard inside, so it's cheap to re-invoke.
    days_regions: set[tuple[str, str]] = set()
    pp = predictions_path()
    if pp.exists():
        for line in pp.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("was_traded"):
                continue
            if r.get("actual_class") is None:
                continue
            if (r.get("reflection") or "").strip():
                continue
            pdate = r.get("prediction_date")
            region = r.get("region")
            if pdate and region:
                days_regions.add((pdate, region))

    if not days_regions:
        log.info("Prediction reflection — no eligible days/regions to catch up")
        return 0

    # Most recent first so today's gets processed before any older backfill.
    pairs = sorted(days_regions, reverse=True)
    processed = pairs[:_DAYS_LIMIT]
    deferred = pairs[_DAYS_LIMIT:]
    log.info(
        "Prediction reflection — %d (date, region) pair(s) eligible; "
        "processing newest %d this run (limit REFLECT_DAYS_LIMIT=%d)",
        len(pairs), len(processed), _DAYS_LIMIT,
    )
    if deferred:
        log.info(
            "Deferred for a future run: %s",
            ", ".join(f"{d}/{r}" for d, r in deferred[:8])
            + (" …" if len(deferred) > 8 else ""),
        )

    total_preds = 0
    for d_iso, region in processed:
        try:
            d = date.fromisoformat(d_iso)
        except ValueError:
            continue
        try:
            n = reflect_predictions_on_day(d, region=region)
        except Exception as e:
            log.error("Prediction reflection failed for %s/%s: %s", d_iso, region, e)
            continue
        log.info("  %s/%s — %d prediction(s) reflected", d_iso, region, n)
        total_preds += n
    log.info("Prediction reflection done — %d row(s) updated", total_preds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
