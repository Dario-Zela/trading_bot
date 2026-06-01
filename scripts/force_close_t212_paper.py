"""One-shot force-close of every open T212-paper position.

Differs from `clear_slot()` (which marks ledger rows cancelled with P&L=0)
by reusing the regular exit_scheduled close-and-fill machinery so real
exit prices and P&L are recorded. Bypasses `filter_due_for_exit` so
positions still inside their hold window are also closed.

Use this when you want to wipe the T212 paper portfolio cleanly before
hitting the platform-level reset, with accurate exit prices captured
for the evaluation logs.

Usage (via the force-close-t212 workflow):
    python scripts/force_close_t212_paper.py
"""
from __future__ import annotations

import logging
import sys
from datetime import date

from trading_bot.executor import trading212_demo
from trading_bot.executor.trading212_demo import Trading212DemoExecutor
from trading_bot.state.ledger import read_open_trades


log = logging.getLogger("force_close_t212_paper")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    # Collect every (strategy_id, region) that has an open T212-paper row.
    # The executor's exit_scheduled is keyed by (strategy_id, region), so
    # we drive it once per distinct pair found in the ledger — including
    # lingerers from strategies that were later demoted/changed (e.g.
    # RHM.DE on news-reactive after that sleeve moved back to shadow).
    open_rows = [
        r for r in _read_all_open()
        if r.get("tier") == "trading212-paper"
    ]
    if not open_rows:
        log.info("No open T212-paper trades — nothing to close.")
        return 0

    keys: set[tuple[str, str]] = {
        (r["strategy_id"], r["region"]) for r in open_rows
    }
    log.info("Open T212-paper trades: %d across %d (strategy, region) pair(s)", len(open_rows), len(keys))
    for sid, region in sorted(keys):
        n = sum(1 for r in open_rows if r["strategy_id"] == sid and r["region"] == region)
        log.info("  %s @ %s : %d open", sid, region, n)

    # Currently slot 1 is the only slot wired (per strategy configs). If
    # more slots come online, list them in load_slots() below.
    slot = 1
    log.info("Instantiating T212 slot %d executor", slot)
    executor = Trading212DemoExecutor(slot)

    # Monkey-patch filter_due_for_exit inside the executor's module
    # namespace so every open trade is treated as due. The function is
    # imported as a module-level name (line 39 of trading212_demo.py),
    # so this override is what the executor will see.
    original_filter = trading212_demo.filter_due_for_exit
    trading212_demo.filter_due_for_exit = lambda trades, on_date: list(trades)
    log.info("Patched filter_due_for_exit → identity (force-close mode)")

    today = date.today()
    total_closed = 0
    total_pnl_gbp = 0.0
    try:
        for sid, region in sorted(keys):
            log.info("Closing all open T212-paper for %s @ %s ...", sid, region)
            try:
                closed = executor.exit_scheduled(
                    strategy_id=sid,
                    region=region,
                    on_date=today,
                )
            except Exception as e:
                log.error("exit_scheduled failed for %s @ %s: %s", sid, region, e)
                continue
            for row in closed:
                pnl = row.get("pnl_gbp") or 0.0
                pct = row.get("pnl_pct")
                pct_s = f"{pct:+.2f}%" if pct is not None else "  n/a "
                log.info(
                    "  closed %s @ %.4f  PnL=£%+.2f (%s)  reason=%s",
                    row.get("ticker"), row.get("exit_price") or 0.0,
                    pnl, pct_s, row.get("exit_reason"),
                )
                total_pnl_gbp += pnl
            total_closed += len(closed)
    finally:
        trading212_demo.filter_due_for_exit = original_filter
        log.info("Restored filter_due_for_exit")

    log.info("=" * 60)
    log.info("Force-close complete: %d position(s) closed, total realised P&L £%+.2f", total_closed, total_pnl_gbp)
    log.info("Ledger updated. Safe to trigger T212 platform-level reset now.")
    return 0


def _read_all_open() -> list[dict]:
    """Read every open ledger row across ALL strategies. read_open_trades
    requires (sid, region) — but we need the inverse here: find the
    distinct keys *from* the open rows. So we read the ledger directly."""
    import json
    from pathlib import Path
    from trading_bot.state.paths import ledger_path

    p = Path(ledger_path())
    if not p.exists():
        return []
    out = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("exit_date") is None:
                out.append(r)
    return out


if __name__ == "__main__":
    sys.exit(main())
