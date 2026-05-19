from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from datetime import date

from trading_bot.executor import AlpacaPaperExecutor, ShadowExecutor, Trading212DemoExecutor
from trading_bot.executor.base import Executor
from trading_bot.notify.email import render_daily_summary, send_summary_email
from trading_bot.state import read_open_trades
from trading_bot.strategy.base import Strategy, StrategyConfig
from trading_bot.strategy.registry import load_active_strategies


log = logging.getLogger(__name__)


def _executor_for_strategy(config: StrategyConfig) -> Executor:
    if config.tier == "shadow":
        return ShadowExecutor()
    if config.tier == "alpaca-paper":
        if config.alpaca_slot is None:
            raise ValueError(
                f"Strategy {config.id}: tier=alpaca-paper requires an alpaca_slot in config"
            )
        return AlpacaPaperExecutor(slot=config.alpaca_slot)
    if config.tier == "trading212-paper":
        if config.t212_slot is None:
            raise ValueError(
                f"Strategy {config.id}: tier=trading212-paper requires a t212_slot in config"
            )
        return Trading212DemoExecutor(slot=config.t212_slot)
    raise NotImplementedError(
        f"Executor for tier '{config.tier}' is not implemented yet — "
        "valid tiers are 'shadow', 'alpaca-paper', 'trading212-paper'"
    )


_MAX_PARALLEL_STRATEGIES = 4


# Target market-local times that orders should hit.
#
# Entry targets: the cron fires 30 min before these (see
# setup_cron_jobs.py) so the multi-stage analysis can finish; the
# `_sleep_until_target` guard in `run_entry` then holds any
# orders until the actual target if we finished early.
#
# Exit targets: the cron fires AT these times directly, no lead,
# and the exit pipeline submits immediately. Post-trade work
# (reflection, missed-movers, dashboard, email) happens after.
_MARKET_TARGETS: dict[tuple[str, str], tuple[str, int, int]] = {
    ("us", "entry"):    ("America/New_York",  9, 35),
    ("us", "exit"):     ("America/New_York", 15, 30),
    ("uk-eu", "entry"): ("Europe/London",     8, 35),
    ("uk-eu", "exit"):  ("Europe/London",    16,  0),
}
_MAX_SLEEP_SECONDS = 60 * 60   # 1h hard cap — if cron is mis-set, fail loud rather than block all day


def _sleep_until_target(region: str, mode: str) -> None:
    """If the analysis phase finished before the target market-local
    time, sleep until then so orders fire on schedule even on a fast
    pipeline run. No-op if we're already past target or if no target
    is configured for the (region, mode) pair."""
    spec = _MARKET_TARGETS.get((region, mode))
    if not spec:
        return
    tz_name, hh, mm = spec
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        log.warning("zoneinfo unavailable — skipping sleep guard")
        return
    import time as _time
    from datetime import datetime as _dt, time as _t
    tz = ZoneInfo(tz_name)
    now = _dt.now(tz)
    target = _dt.combine(now.date(), _t(hh, mm), tzinfo=tz)
    if now >= target:
        return
    sleep_s = (target - now).total_seconds()
    if sleep_s > _MAX_SLEEP_SECONDS:
        log.warning(
            "Pipeline %s/%s finished %.0fs before target %s — exceeds %ds cap, "
            "running NOW (cron likely mis-set)",
            region, mode, sleep_s, target.strftime("%H:%M %Z"), _MAX_SLEEP_SECONDS,
        )
        return
    log.info(
        "Pipeline %s/%s finished early; sleeping %.0fs until %s target",
        region, mode, sleep_s, target.strftime("%H:%M %Z"),
    )
    _time.sleep(sleep_s)


def run_entry(region: str, on_date: date) -> dict[str, list[dict]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from trading_bot.tools.calendar import is_market_open_on
    if not is_market_open_on(on_date, region):
        log.info("Market closed in region=%s on %s — skipping entry", region, on_date.isoformat())
        return {}

    strategies = load_active_strategies(region=region)
    if not strategies:
        log.info("No active strategies for region %s", region)
        return {}

    # Phase 8F — kill switch. Check yesterday's live-tier P&L; halt
    # new entries on live tiers if it breached the threshold. Shadow
    # strategies still run (no real money at risk; we want the data).
    #
    # `_yesterday_live_pnl` sums across regions, so the denominator
    # MUST also span every active live-tier strategy — not just this
    # region's. Otherwise a region with small live capital divided by
    # the cross-region loss spuriously trips the halt.
    from trading_bot.state.halt import (
        LIVE_TIERS, evaluate_and_set_halt, is_halted,
    )
    all_active = load_active_strategies(region=None)
    live_capital = sum(s.config.capital_gbp for s in all_active if s.config.tier in LIVE_TIERS)
    if live_capital > 0:
        evaluate_and_set_halt(on_date, total_live_capital_gbp=live_capital)
    halted, halt_rec = is_halted()
    if halted:
        log.error("Kill switch ENGAGED — skipping live-tier strategies (%s)",
                  (halt_rec.reason if halt_rec else "no record"))
        strategies = [s for s in strategies if s.config.tier not in LIVE_TIERS]
        if not strategies:
            return {}

    # Phase 1: select_picks() in parallel. Each strategy makes its own
    # Claude Code subprocess call; we let up to N run concurrently.
    # yfinance history is process-cached so the first finisher pays the
    # full ~60s universe fetch and the rest hit the warm cache.
    intents_by_id: dict[str, tuple] = {}  # sid -> (strategy, intents)
    log.info(
        "Entry phase: fanning out %d strategies (max %d concurrent)",
        len(strategies), _MAX_PARALLEL_STRATEGIES,
    )
    with ThreadPoolExecutor(max_workers=_MAX_PARALLEL_STRATEGIES) as pool:
        futures = {pool.submit(strategy.select_picks, on_date): strategy for strategy in strategies}
        for fut in as_completed(futures):
            strategy = futures[fut]
            try:
                intents = fut.result()
                log.info("%s: %d picks", strategy.config.id, len(intents))
                intents_by_id[strategy.config.id] = (strategy, intents)
            except Exception as e:
                log.exception("Strategy %s failed in select_picks: %s", strategy.config.id, e)
                intents_by_id[strategy.config.id] = (strategy, [])

    # Phase 10E — hold orders until the actual market-target time if
    # the LLM analysis phase finished early. Caps the wait at 1h.
    _sleep_until_target(region, "entry")

    # Phase 12G — enforce max_positions ACROSS days + dedup tickers
    # against currently-open positions. Same-day round-trips zero out
    # the ledger nightly so `max_positions` was naturally a daily cap;
    # with multi-day holds, prior sessions' open trades stack on top of
    # today's entries. Two adjustments:
    #   1. Drop intents whose ticker is already held by this strategy
    #      (don't double up on a multi-day position).
    #   2. Trim what's left so total open ≤ cfg.max_positions.
    for sid, (strategy, intents) in list(intents_by_id.items()):
        if not intents:
            continue
        cfg = strategy.config
        currently_open = read_open_trades(strategy_id=sid, region=cfg.region)
        held_tickers = {t.get("ticker") for t in currently_open if t.get("ticker")}

        if held_tickers:
            kept: list = []
            for i in intents:
                if i.ticker in held_tickers:
                    log.info(
                        "%s: skipping %s — already held in an open multi-day position",
                        sid, i.ticker,
                    )
                    continue
                kept.append(i)
            intents = kept

        slots_remaining = max(0, cfg.max_positions - len(currently_open))
        if slots_remaining < len(intents):
            log.info(
                "%s: %d currently open + %d new picks > %d cap — trimming today's "
                "intents to %d",
                sid, len(currently_open), len(intents),
                cfg.max_positions, slots_remaining,
            )
            intents = intents[:slots_remaining]
        intents_by_id[sid] = (strategy, intents)

    # Phase 2: enter() sequentially. Broker API calls — we want
    # deterministic rate against Alpaca / T212 (already throttled per
    # request). Sequential here keeps the order in logs stable too.
    entries: dict[str, list[dict]] = {}
    for sid, (strategy, intents) in intents_by_id.items():
        try:
            executor = _executor_for_strategy(strategy.config)
            executor.enter(
                intents,
                strategy_id=sid,
                region=strategy.config.region,
                capital_gbp=strategy.config.capital_gbp,
                on_date=on_date,
            )
            opened = read_open_trades(
                strategy_id=sid,
                region=strategy.config.region,
                on_date=on_date,
            )
            entries[sid] = opened
        except Exception as e:
            log.exception("Strategy %s failed in enter phase: %s", sid, e)
    return entries


def run_exit(region: str, on_date: date) -> dict[str, list[dict]]:
    from trading_bot.tools.calendar import is_market_open_on
    if not is_market_open_on(on_date, region):
        log.info(
            "Market closed in region=%s on %s — skipping exit (any open positions roll over to next session)",
            region, on_date.isoformat(),
        )
        return {}

    strategies = load_active_strategies(region=region)
    if not strategies:
        log.info("No active strategies for region %s", region)
        return {}

    # Exits have no pre-trade analysis to overlap with broker latency,
    # so the cron is scheduled to fire AT the target market-local time
    # and the close goes out immediately. The reflection + missed-
    # movers + dashboard rebuild that follow are all post-trade and
    # don't need to land before any market deadline. (Entries are the
    # opposite: cron fires 30 min early to give the multi-stage LLM
    # analysis room, then `_sleep_until_target` aligns the actual
    # order submission with the market target.)

    exits: dict[str, list[dict]] = defaultdict(list)
    for strategy in strategies:
        try:
            executor = _executor_for_strategy(strategy.config)
            closed = executor.exit_scheduled(
                strategy_id=strategy.config.id,
                region=strategy.config.region,
                on_date=on_date,
            )
            if closed:
                exits[strategy.config.id].extend(closed)
        except Exception as e:
            log.exception("Strategy %s failed in exit phase: %s", strategy.config.id, e)

    # Phase 8E — real per-trade LLM reflection. Replaces the templated
    # outcome_notes / risks_observed with a Haiku call per trade. Runs
    # in parallel here at the pipeline level (rather than per-executor)
    # so we can batch the full day's exits in one fan-out.
    try:
        _run_per_trade_reflection(exits, on_date)
    except Exception as e:
        log.warning("Per-trade reflection failed (non-fatal): %s", e)

    # Post-exit: scan the day's biggest movers across the union of
    # strategy universes and identify which we missed + why. Non-fatal —
    # the analysis writes its own state file and is consumed by the
    # daily news brief + weekly evolution agent.
    try:
        from trading_bot.meta.missed_movers import analyze_missed_movers
        report = analyze_missed_movers(on_date, region)
        log.info(
            "missed-movers: %d movers analysed for %s (summary: %s)",
            len(report.top_movers), region, report.summary[:160],
        )
    except Exception as e:
        log.warning("missed-movers analysis failed (non-fatal): %s", e)

    return dict(exits)


def _run_per_trade_reflection(exits: dict[str, list[dict]], on_date: date) -> None:
    """Take all exits from this run, fetch any available context (today's
    bars + news for each ticker), call the reflection agent in parallel,
    and rewrite the affected ledger rows."""
    from trading_bot.meta.trade_reflection import reflect_batch
    from trading_bot.state.ledger import mark_trade_exited
    from trading_bot.tools.news import get_recent_news

    # Flatten + dedupe across strategies
    all_trades: list[dict] = []
    seen_ids: set[str] = set()
    for sid, trades in exits.items():
        for t in trades:
            tid = t.get("trade_id")
            if not tid or tid in seen_ids:
                continue
            if t.get("exit_reason") in ("cancelled", "cleared"):
                continue   # nothing to reflect on
            seen_ids.add(tid)
            all_trades.append(t)
    if not all_trades:
        return

    tickers = sorted({t.get("ticker") for t in all_trades if t.get("ticker")})
    news_by_ticker: dict[str, list] = {}
    try:
        raw = get_recent_news(tickers, days=2, limit=5)
        news_by_ticker = {tk: [{"timestamp": n.timestamp, "headline": n.headline, "summary": n.summary} for n in items] for tk, items in raw.items()}
    except Exception as e:
        log.debug("Reflection news fetch failed (continuing without): %s", e)

    log.info("Per-trade reflection: %d trades to score", len(all_trades))
    reflections = reflect_batch(all_trades, news_by_ticker=news_by_ticker)
    for trade in all_trades:
        tid = trade.get("trade_id")
        if tid not in reflections:
            continue
        outcome, risks = reflections[tid]
        try:
            mark_trade_exited(
                trade_id=tid,
                exit_date=on_date,
                exit_price=float(trade.get("exit_price") or 0),
                pnl_gbp=float(trade.get("pnl_gbp") or 0),
                pnl_pct=float(trade.get("pnl_pct") or 0),
                exit_reason=trade.get("exit_reason", "scheduled"),
                outcome_notes=outcome,
                risks_observed=risks,
                fees_gbp=float(trade.get("fees_gbp") or 0),
                fees_breakdown=trade.get("fees_breakdown") or {},
            )
        except Exception as e:
            log.warning("Failed to rewrite reflection for %s: %s", tid, e)


def run_clear_slot(slot: int) -> None:
    executor = AlpacaPaperExecutor(slot=slot)
    executor.clear_slot()


def run_reflect(region: str, on_date: date) -> int:
    from trading_bot.meta.reflection import (
        grade_predictions,
        reflect_on_day,
        reflect_predictions_on_day,
    )

    n_graded = grade_predictions(on_date, region=region)
    log.info("Graded %d predictions with actual returns", n_graded)
    n_trades = reflect_on_day(on_date, region=region)
    # Pre-compute per-prediction reflection text on every untraded
    # pick so the weekly evolution agent doesn't have to rationalise
    # 100s of misses from raw rationale + actual_class. Cheap — one
    # Sonnet call per strategy, ~20 tickers per call.
    try:
        n_preds = reflect_predictions_on_day(on_date, region=region)
        log.info("Reflected on %d untraded predictions", n_preds)
    except Exception as e:
        log.warning("Prediction reflection failed (non-fatal): %s", e)
    return n_trades


def run_weekly_macro_cmd(on_date: date) -> None:
    from trading_bot.meta.macro import run_weekly_macro

    summary = run_weekly_macro(on_date)
    log.info("Weekly macro run summary: %s", summary)


def run_weekly_evolution_cmd(on_date: date) -> None:
    from trading_bot.meta.evolution import run_weekly_evolution

    summary = run_weekly_evolution(on_date)
    log.info("Weekly evolution run summary: %s", summary)


def run_daily_news_brief_cmd(on_date: date) -> None:
    from trading_bot.meta.daily_news import run_daily_news_brief

    summary = run_daily_news_brief(on_date)
    log.info("Daily news brief summary: %s", summary)


def run_grade_predictions_cmd(on_date: date) -> None:
    from trading_bot.meta.grade_predictions import grade_predictions_cli

    grade_predictions_cli(on_date)


def run_summary(region: str, on_date: date) -> None:
    """Read today's exits from the ledger and send the summary email.
    Runs after exit + reflect so the email reflects any LLM-updated
    outcome_notes / risks_observed."""
    import json
    from pathlib import Path
    from collections import defaultdict

    from trading_bot.state.paths import ledger_path

    target = on_date.isoformat()
    exits: dict[str, list[dict]] = defaultdict(list)
    path: Path = ledger_path()
    if path.exists():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("exit_date") != target:
                    continue
                if region is not None and r.get("region") != region:
                    continue
                exits[r["strategy_id"]].append(r)

    subject, body_text, body_html = render_daily_summary(
        run_date=on_date,
        region=region,
        entries={},
        exits=dict(exits),
    )
    try:
        send_summary_email(subject=subject, body_text=body_text, body_html=body_html)
    except Exception as e:
        log.error("Email send failed: %s", e)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trading_bot.pipeline")
    parser.add_argument(
        "mode",
        choices=[
            "entry", "exit", "clear-slot", "reflect", "summary",
            "weekly-macro", "weekly-evolution", "daily-news-brief",
            "grade-predictions",
            "t212-reconcile-orphans",
        ],
    )
    parser.add_argument("--region", default="us", choices=["us", "uk-eu"])
    parser.add_argument("--strategy", help="strategy_id to attribute reconciled orphans to")
    parser.add_argument("--date", help="ISO date (defaults to today)")
    parser.add_argument("--email", action="store_true", help="Send summary email after exit")
    parser.add_argument("--slot", type=int, help="Alpaca slot number (used by clear-slot)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.mode == "clear-slot":
        if args.slot is None:
            parser.error("clear-slot requires --slot N")
        run_clear_slot(args.slot)
        log.info("Slot %d cleared", args.slot)
        return 0

    if args.mode == "t212-reconcile-orphans":
        if not args.strategy:
            parser.error("t212-reconcile-orphans requires --strategy <id>")
        from datetime import date as _date
        target_date = _date.fromisoformat(args.date) if args.date else _date.today()
        executor = Trading212DemoExecutor(slot=1)
        recovered = executor.reconcile_orphans(
            attribute_to_strategy=args.strategy,
            region=args.region,
            on_date=target_date,
        )
        log.info("T212 reconcile: recovered %d orphan position(s): %s",
                 len(recovered), [r["ticker"] for r in recovered])
        return 0

    on_date = date.fromisoformat(args.date) if args.date else date.today()

    if args.mode == "weekly-macro":
        run_weekly_macro_cmd(on_date)
        return 0

    if args.mode == "weekly-evolution":
        run_weekly_evolution_cmd(on_date)
        return 0

    if args.mode == "daily-news-brief":
        run_daily_news_brief_cmd(on_date)
        return 0

    if args.mode == "grade-predictions":
        run_grade_predictions_cmd(on_date)
        return 0

    if args.mode == "reflect":
        n = run_reflect(args.region, on_date)
        log.info("Reflection complete: %d trades updated", n)
        return 0

    if args.mode == "summary":
        run_summary(args.region, on_date)
        log.info("Summary email dispatched for %s region", args.region)
        return 0

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
