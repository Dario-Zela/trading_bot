"""Phase 8D — midday trailing-stop adjustment for both brokers.

Walks the stop up on any position in profit:
- **Alpaca**: PATCH the existing bracket-stop child to the new price
  (modifying via PATCH is supported).
- **T212**: cancel-and-replace pattern — DELETE the existing stop
  order for the symbol (if any), POST a fresh stop at the new
  price. T212's Invest API doesn't expose a PATCH on stops.

Run by `.github/workflows/midday-trail.yml` at ~17:00 UK
(mid-US-session, ~UK after-close so both brokers' positions are
visible). Doesn't modify the underlying ledger — the trail
adjustment lives entirely broker-side; if a stop fires the existing
exit pipeline picks up the close via order history.

Usage:
    python scripts/midday_trail.py
    python scripts/midday_trail.py --activation 1.5 --trail 1.0 --brokers alpaca
"""
from __future__ import annotations

import argparse
import logging
import sys

from trading_bot.executor.alpaca_trail import (
    DEFAULT_ACTIVATION_PCT, DEFAULT_TRAIL_PCT,
    format_log as alpaca_log, trail_alpaca_slots,
)
from trading_bot.executor.t212_trail import (
    format_log as t212_log, trail_t212_slots,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walk broker stops up on positions in profit.")
    parser.add_argument("--activation", type=float, default=DEFAULT_ACTIVATION_PCT,
                        help=f"Profit % required before we start trailing (default {DEFAULT_ACTIVATION_PCT}%)")
    parser.add_argument("--trail", type=float, default=DEFAULT_TRAIL_PCT,
                        help=f"Distance below current price to set the new stop (default {DEFAULT_TRAIL_PCT}%)")
    parser.add_argument("--slots", nargs="*", type=int,
                        help="Specific slot numbers to check; defaults to 1-3 for each broker.")
    parser.add_argument("--brokers", nargs="*", choices=["alpaca", "t212"],
                        default=["alpaca", "t212"],
                        help="Which brokers to scan (default: both).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    exit_code = 0

    if "alpaca" in args.brokers:
        ap_actions = trail_alpaca_slots(
            slots=args.slots,
            activation_pct=args.activation, trail_pct=args.trail,
        )
        print("\nTrailing-stop pass — Alpaca paper")
        print("=" * 60)
        print(alpaca_log(ap_actions))
        ap_applied = sum(1 for a in ap_actions if a.status == "applied")
        ap_skipped = sum(1 for a in ap_actions if a.status == "skipped")
        ap_failed = sum(1 for a in ap_actions if a.status == "failed")
        print(f"alpaca: {ap_applied} applied, {ap_skipped} skipped, {ap_failed} failed")
        if ap_failed > 0:
            exit_code = 1

    if "t212" in args.brokers:
        t_actions = trail_t212_slots(
            slots=args.slots,
            activation_pct=args.activation, trail_pct=args.trail,
        )
        print("\nTrailing-stop pass — Trading 212 demo")
        print("=" * 60)
        print(t212_log(t_actions))
        t_placed = sum(1 for a in t_actions if a.status == "placed")
        t_tightened = sum(1 for a in t_actions if a.status == "tightened")
        t_skipped = sum(1 for a in t_actions if a.status == "skipped")
        t_failed = sum(1 for a in t_actions if a.status == "failed")
        print(f"t212: {t_placed} placed (new), {t_tightened} tightened, {t_skipped} skipped, {t_failed} failed")
        if t_failed > 0:
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
