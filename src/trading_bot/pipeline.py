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

    # Per-trade reflection is handled later, in `run_reflect`, by the
    # Sonnet-based `meta/reflection.py:reflect_on_day` pass. That pass
    # writes one prompt per strategy summarising the whole day's
    # basket, which gives richer context than per-trade Haiku calls
    # and lets the LLM see basket-level patterns. The Haiku pass that
    # used to live here was overwritten by the Sonnet pass anyway —
    # ~1 minute of wasted compute every exit cron — so it's gone.

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
            "weekly-macro", "weekly-evolution", "weekly-external-research",
            "weekly-backtest-pass",
            "daily-news-brief",
            "grade-predictions",
            "t212-reconcile-orphans",
            "ohlcv-prune",
            "ohlcv-daily-update",
            "ohlcv-backfill",
            "ohlcv-stooq-fill-gaps",
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

    if args.mode == "weekly-external-research":
        from trading_bot.meta.external_research import run_external_research
        summary = run_external_research(on_date)
        log.info("Weekly external research summary: %s", summary)
        return 0

    if args.mode == "weekly-backtest-pass":
        from trading_bot.meta.backtest import run_weekly_backtest_pass
        summary = run_weekly_backtest_pass(on_date)
        log.info(
            "Weekly backtest pass: %d strategies replayed (window=%d days)",
            len(summary.get("strategies") or []),
            int(summary.get("window_days") or 0),
        )
        return 0

    if args.mode == "ohlcv-prune":
        # 1-year rolling cutoff. Called from weekly-evolution after the
        # main evolution pass. Bounds the local SQLite OHLCV cache size
        # by dropping bars older than 365 days. Idempotent.
        from trading_bot.tools.ohlcv_store import prune_old, row_count, store_size_bytes
        deleted = prune_old(365)
        log.info(
            "ohlcv prune: deleted %d bars; remaining %d rows; db size %.1f MB",
            deleted, row_count(), store_size_bytes() / 1_048_576,
        )
        return 0

    if args.mode == "ohlcv-daily-update":
        # Post-close warm-up: fetch today's bar for every ticker active
        # across all strategies' universes, write back to the local store
        # so tomorrow's morning entry pipeline reads from cache instead
        # of yfinance. Idempotent — re-running is safe (INSERT OR REPLACE).
        from trading_bot.strategy.registry import load_active_strategies
        from trading_bot.tools.universe import get_universe
        from trading_bot.tools.history import get_history
        from trading_bot.tools.ohlcv_store import row_count
        active = load_active_strategies(region=None)
        all_tickers: set[str] = set()
        for s in active:
            try:
                all_tickers.update(get_universe(s.config.universe))
            except Exception as e:
                log.warning("ohlcv-daily-update: skipping %s (%s)", s.config.universe, e)
        log.info("ohlcv-daily-update: %d unique tickers across %d active strategies",
                 len(all_tickers), len(active))
        # Fetching with a 5-day lookback (not 1) makes the write-back
        # backfill any gaps from a missed run. get_history writes to the
        # store internally.
        if all_tickers:
            n_before = row_count()
            get_history(sorted(all_tickers), lookback_days=5, end_date=on_date)
            n_after = row_count()
            log.info("ohlcv-daily-update: cache rows %d → %d (Δ %d)",
                     n_before, n_after, n_after - n_before)
        return 0

    if args.mode == "ohlcv-stooq-fill-gaps":
        # Second-pass backfill: walk every active strategy's universe,
        # find tickers MISSING from the local cache (yfinance failures
        # — wrong format, rebranded, no coverage), and fetch them from
        # Stooq as the fallback source. Idempotent: re-running is safe.
        from datetime import timedelta
        from trading_bot.strategy.registry import load_active_strategies
        from trading_bot.tools.universe import get_universe
        from trading_bot.tools.ohlcv_store import (
            read_bars_bulk, write_bars, row_count, StoredBar,
        )
        from trading_bot.tools.stooq import fetch_history_bulk
        active = load_active_strategies(region=None)
        all_tickers: set[str] = set()
        for s in active:
            try:
                all_tickers.update(get_universe(s.config.universe))
            except Exception as e:
                log.warning("stooq-fill-gaps: skipping %s (%s)", s.config.universe, e)

        # Identify tickers that have NO bars in the local store for the
        # trailing 70-day window (i.e. yfinance failed entirely for them).
        end = on_date
        start = end - timedelta(days=70 * 2 + 5)
        hits = read_bars_bulk(sorted(all_tickers), start, end)
        missing = sorted(all_tickers - set(hits.keys()))
        log.info(
            "stooq-fill-gaps: %d/%d tickers missing from cache after yfinance pass",
            len(missing), len(all_tickers),
        )
        if not missing:
            return 0

        n_before = row_count()
        stooq_results = fetch_history_bulk(missing, lookback_days=70, end_date=end)
        rows: list[StoredBar] = []
        for tkr, bars in stooq_results.items():
            for b in bars:
                rows.append(StoredBar(
                    ticker=tkr, bar_date=b["bar_date"],
                    open=b["open"], high=b["high"], low=b["low"],
                    close=b["close"], volume=b["volume"],
                ))
        if rows:
            write_bars(rows)
        n_after = row_count()
        log.info(
            "stooq-fill-gaps: %d/%d missing tickers recovered from Stooq; "
            "cache rows %d → %d (Δ %d)",
            len(stooq_results), len(missing), n_before, n_after, n_after - n_before,
        )
        return 0

    if args.mode == "ohlcv-backfill":
        # One-time bulk backfill: fetch 70 trading days of OHLCV for
        # every ticker across all active strategies' universes. Run
        # this once after switching to the t212_isa universes so the
        # first morning entry doesn't pay a ~12-min cold-cache hit.
        from trading_bot.strategy.registry import load_active_strategies
        from trading_bot.tools.universe import get_universe
        from trading_bot.tools.history import get_history
        from trading_bot.tools.ohlcv_store import row_count, store_size_bytes
        active = load_active_strategies(region=None)
        all_tickers: set[str] = set()
        for s in active:
            try:
                all_tickers.update(get_universe(s.config.universe))
            except Exception as e:
                log.warning("ohlcv-backfill: skipping %s (%s)", s.config.universe, e)
        log.info("ohlcv-backfill: %d unique tickers, fetching 70-day history",
                 len(all_tickers))
        if all_tickers:
            n_before = row_count()
            get_history(sorted(all_tickers), lookback_days=70, end_date=on_date)
            n_after = row_count()
            log.info("ohlcv-backfill: cache rows %d → %d (Δ %d); db size %.1f MB",
                     n_before, n_after, n_after - n_before,
                     store_size_bytes() / 1_048_576)
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
