"""Phase 11I — CLI for the technicals-only backtest.

Usage:
    python scripts/backtest.py --strategy control-rule-based \\
        --start 2025-01-01 --end 2025-03-01 [--limit-days 5]
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from trading_bot.meta.backtest import run_backtest, write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a technicals-only backtest.")
    parser.add_argument("--strategy", required=True, help="Strategy ID (e.g. control-rule-based)")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--limit-days", type=int, help="For LLM strategies — cap days to keep cost manageable.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    report = run_backtest(
        args.strategy,
        start_date=start, end_date=end,
        limit_days=args.limit_days,
    )

    out = write_report(report)
    print(f"\n=== Backtest summary ===")
    print(f"Strategy: {report.strategy_id} ({report.region})")
    print(f"Window:   {report.start_date} → {report.end_date}  ({report.n_days} trading days)")
    print(f"Trades:   {report.n_trades}")
    print(f"Total P&L: {report.total_pnl_pct:+.2f}%")
    print(f"Avg P&L:   {report.avg_pnl_pct:+.2f}% per trade")
    print(f"Hit rate:  {report.hit_rate * 100:.1f}%")
    print(f"W/L:       {report.win_loss_ratio:.2f}")
    print(f"\nReport written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
