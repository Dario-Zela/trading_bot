from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from datetime import date

from trading_bot.executor import ShadowExecutor
from trading_bot.executor.base import Executor
from trading_bot.notify.email import render_daily_summary, send_summary_email
from trading_bot.state import read_open_trades
from trading_bot.strategy.base import Strategy
from trading_bot.strategy.registry import load_active_strategies


log = logging.getLogger(__name__)


def _executor_for_tier(tier: str) -> Executor:
    if tier == "shadow":
        return ShadowExecutor()
    raise NotImplementedError(
        f"Executor for tier '{tier}' is not implemented in Wave 1 — only 'shadow' is supported"
    )


def run_entry(region: str, on_date: date) -> dict[str, list[dict]]:
    strategies = load_active_strategies(region=region)
    if not strategies:
        log.info("No active strategies for region %s", region)
        return {}

    entries: dict[str, list[dict]] = {}
    for strategy in strategies:
        intents = strategy.select_picks(on_date)
        log.info("%s: %d picks", strategy.config.id, len(intents))

        executor = _executor_for_tier(strategy.config.tier)
        executor.enter(
            intents,
            strategy_id=strategy.config.id,
            region=strategy.config.region,
            capital_gbp=strategy.config.capital_gbp,
            on_date=on_date,
        )

        opened = read_open_trades(
            strategy_id=strategy.config.id,
            region=strategy.config.region,
            on_date=on_date,
        )
        entries[strategy.config.id] = opened
    return entries


def run_exit(region: str, on_date: date) -> dict[str, list[dict]]:
    strategies = load_active_strategies(region=region)
    if not strategies:
        log.info("No active strategies for region %s", region)
        return {}

    exits: dict[str, list[dict]] = defaultdict(list)
    for strategy in strategies:
        executor = _executor_for_tier(strategy.config.tier)
        closed = executor.exit_scheduled(
            strategy_id=strategy.config.id,
            region=strategy.config.region,
            on_date=on_date,
        )
        if closed:
            exits[strategy.config.id].extend(closed)
    return dict(exits)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trading_bot.pipeline")
    parser.add_argument("mode", choices=["entry", "exit"])
    parser.add_argument("--region", default="us", choices=["us", "uk-eu"])
    parser.add_argument("--date", help="ISO date (defaults to today)")
    parser.add_argument("--email", action="store_true", help="Send summary email after exit")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    on_date = date.fromisoformat(args.date) if args.date else date.today()

    if args.mode == "entry":
        entries = run_entry(args.region, on_date)
        log.info("Entry complete: %d strategies acted, %d total positions opened",
                 len(entries), sum(len(v) for v in entries.values()))
        return 0

    # exit mode
    exits = run_exit(args.region, on_date)
    log.info("Exit complete: %d strategies closed, %d total positions",
             len(exits), sum(len(v) for v in exits.values()))

    if args.email:
        subject, body_text, body_html = render_daily_summary(
            run_date=on_date,
            region=args.region,
            entries={},  # exits already include the entry data; no need to double-list
            exits=exits,
        )
        try:
            send_summary_email(subject=subject, body_text=body_text, body_html=body_html)
        except Exception as e:
            log.error("Email send failed: %s", e)
            # Don't fail the whole run on email failure
    return 0


if __name__ == "__main__":
    sys.exit(main())
