"""Phase 11I — technicals-only backtest framework.

Walk a date range; for each trading day, invoke the strategy's
`select_picks(on_date)` against historical data (yfinance already
supports `end_date` everywhere), simulate fills at *the next day's*
close, record P&L, and aggregate.

Bounded scope:
- Rule-based + momentum-stub strategies replay fully (deterministic).
- LLM strategies can replay too, but they'll hit the live Claude
  Code subprocess per day → expensive + slow. The `--limit-days N`
  flag exists for that mode so you can sanity-check on 5 days
  before committing to a year.
- News / macro context is NOT replayed — they don't have historical
  archives. The LLM sees today's macro stub when called in backtest
  mode. Acknowledged limitation; results are conservative.

Outputs a Backtest dataclass + writes a markdown report to
`state/diagnostics/backtest_<strategy>_<start>_<end>.md`.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

from trading_bot.state.paths import STATE_ROOT
from trading_bot.tools.history import get_history


log = logging.getLogger(__name__)


@dataclass
class _BackTrade:
    """One simulated trade in the backtest."""
    date: str
    ticker: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    allocation_pct: float


@dataclass
class BacktestReport:
    strategy_id: str
    region: str
    start_date: str
    end_date: str
    n_days: int
    n_trades: int
    total_pnl_pct: float                    # additive return %
    avg_pnl_pct: float
    hit_rate: float
    win_loss_ratio: float                   # |avg_win| / |avg_loss|
    by_day_pnl: dict[str, float] = field(default_factory=dict)
    trades: list[_BackTrade] = field(default_factory=list)


def run_backtest(
    strategy_id: str,
    *,
    start_date: date,
    end_date: date,
    limit_days: int | None = None,
    region: str | None = None,
) -> BacktestReport:
    """Replay a strategy over the given date range. Returns a
    BacktestReport. Caller is responsible for writing the markdown.

    When `region` is supplied, the strategy's config is overridden
    with that region's `runs_in` entry — tier, universe, slot, and
    the region itself. Multi-region strategies otherwise default to
    their top-level region (usually "us"), which means a UK-EU
    backtest would silently run against the US universe."""
    from trading_bot.strategy.registry import load_strategy_config, _strategies_dir
    from trading_bot.strategy.control_rule_based import ControlRuleBased
    from trading_bot.strategy.momentum_stub import MomentumTraderStub
    from trading_bot.strategy.llm_strategy import LLMStrategy
    import yaml

    cfg = load_strategy_config(strategy_id)
    if region:
        # Pull the runs_in entry for this region and apply its
        # overrides to the cfg the strategy will see. Falls back to
        # the top-level config if the strategy is single-region or
        # doesn't have a matching entry.
        try:
            raw = yaml.safe_load(
                (_strategies_dir() / strategy_id / "config.yaml").read_text()
            )
        except Exception as e:
            log.warning("backtest: could not re-read %s config for region override: %s",
                        strategy_id, e)
            raw = {}
        runs_in = raw.get("runs_in") if isinstance(raw, dict) else None
        if isinstance(runs_in, list):
            for entry in runs_in:
                if not isinstance(entry, dict):
                    continue
                if entry.get("region") == region:
                    cfg.region = region
                    cfg.tier = entry.get("tier", cfg.tier)
                    cfg.universe = entry.get("universe", cfg.universe)
                    aslot = entry.get("alpaca_slot")
                    tslot = entry.get("t212_slot")
                    if aslot is not None:
                        cfg.alpaca_slot = int(aslot)
                    if tslot is not None:
                        cfg.t212_slot = int(tslot)
                    break
        else:
            # Single-region config — just verify the region matches
            # what the caller asked for so we don't silently backtest
            # the wrong universe.
            if cfg.region != region:
                log.warning(
                    "backtest: %s is single-region (%s) but caller asked for %s — using %s",
                    strategy_id, cfg.region, region, cfg.region,
                )

    impl = cfg.implementation
    if impl == "rule_based":
        strat = ControlRuleBased(cfg)
    elif impl == "momentum_stub":
        strat = MomentumTraderStub(cfg)
    elif impl == "llm":
        strat = LLMStrategy(cfg)
    else:
        raise ValueError(f"Unknown strategy implementation: {impl!r}")

    trading_days = _trading_days_in_range(start_date, end_date, region=cfg.region)
    if limit_days:
        trading_days = trading_days[:limit_days]

    log.info(
        "Backtest %s (%s): %d trading days from %s to %s",
        strategy_id, cfg.region, len(trading_days), start_date, end_date,
    )

    trades: list[_BackTrade] = []
    by_day: dict[str, float] = {}

    for d in trading_days:
        try:
            intents = strat.select_picks(d)
        except Exception as e:
            log.warning("Backtest %s on %s failed: %s", strategy_id, d, e)
            by_day[d.isoformat()] = 0.0
            continue
        if not intents:
            by_day[d.isoformat()] = 0.0
            continue

        # For each pick, look up entry (today's close) and exit (next
        # day's close) from yfinance. Backtests simulate the
        # close-to-close return as the "scheduled exit at close" model.
        tickers = [i.ticker for i in intents]
        # Lookback 5 covers small holidays; we just need the next bar after `d`.
        hist = get_history(tickers, lookback_days=5, end_date=d + timedelta(days=5))
        day_pnl_pct = 0.0
        for intent in intents:
            bars = hist.get(intent.ticker) or []
            # Entry = first bar on-or-after d, exit = the next bar. Using
            # >= d (not == d) means a data gap or a holiday the approximate
            # calendar missed enters at the next available close instead of
            # silently dropping the pick. bars are chronological ascending.
            entry_idx = next((i for i, b in enumerate(bars) if b.bar_date >= d), None)
            if entry_idx is None or entry_idx + 1 >= len(bars):
                continue
            entry_price = float(bars[entry_idx].close)
            exit_price = float(bars[entry_idx + 1].close)
            if entry_price <= 0:
                continue
            pnl_pct = (exit_price / entry_price - 1.0) * 100.0
            # Weight by allocation_pct for the daily aggregation
            alloc_frac = intent.allocation_pct / 100.0
            day_pnl_pct += pnl_pct * alloc_frac
            trades.append(_BackTrade(
                date=d.isoformat(),
                ticker=intent.ticker,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_pct=round(pnl_pct, 3),
                allocation_pct=intent.allocation_pct,
            ))
        by_day[d.isoformat()] = round(day_pnl_pct, 3)

    # Aggregate
    if not trades:
        return BacktestReport(
            strategy_id=strategy_id, region=cfg.region,
            start_date=start_date.isoformat(), end_date=end_date.isoformat(),
            n_days=len(trading_days), n_trades=0,
            total_pnl_pct=0.0, avg_pnl_pct=0.0, hit_rate=0.0, win_loss_ratio=0.0,
            by_day_pnl=by_day, trades=[],
        )
    pnl_pcts = [t.pnl_pct for t in trades]
    wins = [p for p in pnl_pcts if p > 0]
    losses = [p for p in pnl_pcts if p < 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss else 0.0

    return BacktestReport(
        strategy_id=strategy_id, region=cfg.region,
        start_date=start_date.isoformat(), end_date=end_date.isoformat(),
        n_days=len(trading_days), n_trades=len(trades),
        total_pnl_pct=round(sum(by_day.values()), 3),
        avg_pnl_pct=round(sum(pnl_pcts) / len(pnl_pcts), 3),
        hit_rate=round(len(wins) / len(trades), 3),
        win_loss_ratio=round(win_loss_ratio, 3),
        by_day_pnl=by_day,
        trades=trades,
    )


def run_weekly_backtest_pass(today: date, window_days: int = 14) -> dict:
    """Replay every active strategy over the trailing `window_days`
    using TODAY's code + prompts. Writes a per-strategy JSON report
    to `state/backtest/<iso-week>/<strategy>.json` plus a roll-up
    `state/backtest/<iso-week>/summary.json` the evolution prompt
    reads.

    Caveat: LLM strategies hit Claude live during the replay, which
    means tools like WebSearch see TODAY's web — that's lookahead
    bias. The result still has signal as a "what would current code
    do on yesterday's bars" sanity check, but it's not a clean
    out-of-sample backtest. The summary file records this caveat so
    downstream readers don't over-interpret.

    Triggered weekly from the evolution workflow; safe to dispatch
    on demand.
    """
    iso_year, iso_week, _ = today.isocalendar()
    out_dir = STATE_ROOT / "backtest" / f"{iso_year}-W{iso_week:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover every active strategy / region pair to backtest. We
    # call into the registry's loader rather than re-walking the yaml
    # ourselves, so the multi-region expansion stays consistent.
    from trading_bot.strategy.registry import load_active_strategies
    active = load_active_strategies(region=None)
    if not active:
        log.warning("backtest pass: no active strategies")
        return {"window_days": window_days, "strategies": []}

    # `window_days` is trading-days targeted; add headroom for two
    # weekends + possible holidays so the calendar window actually
    # yields the intended trading-day count after filtering.
    start = today - timedelta(days=window_days + 7)
    end = today - timedelta(days=1)                   # backtest stops yesterday so we can resolve next-day fills

    results: list[dict] = []
    # Per-strategy day cap. LLM strategies hit Claude once per trading
    # day with tools — at ~2-3 min per call, a 14-day walk for one
    # strategy can take 30-40 min. With 6 LLM strategies × 2 regions
    # that's 6-8 hours, well past the GitHub Actions workflow timeout.
    # Cap LLM walks at 5 trading days (enough to detect prompt-vs-live
    # divergence for the evolution agent's "is the cost gate bleeding
    # edge?" signal). Rule-based strategies are cheap, get the full
    # window. Override via the BACKTEST_LLM_DAYS env var if you want
    # a fuller replay (set to 0 to skip LLM strategies entirely).
    llm_day_cap = int(os.environ.get("BACKTEST_LLM_DAYS", "5"))

    for strat in active:
        sid = strat.config.id
        region = strat.config.region
        impl = strat.config.implementation
        cap = None
        if impl == "llm":
            if llm_day_cap <= 0:
                log.info("backtest pass: skipping %s/%s (LLM, BACKTEST_LLM_DAYS=0)", sid, region)
                results.append({
                    "strategy_id": sid, "region": region,
                    "skipped": "LLM backtest disabled via BACKTEST_LLM_DAYS=0",
                })
                continue
            cap = llm_day_cap
        log.info("backtest pass: %s/%s window %s → %s (limit_days=%s)", sid, region, start, end, cap)
        try:
            report = run_backtest(
                sid,
                start_date=start,
                end_date=end,
                limit_days=cap,
                region=region,   # crucial for multi-region strategies
            )
        except Exception as e:
            log.warning("backtest pass: %s failed: %s", sid, e)
            results.append({
                "strategy_id": sid, "region": region,
                "error": str(e)[:240],
            })
            continue

        # Drop the per-trade detail when serialising the roll-up so
        # the evolution prompt doesn't drown in noise. The full
        # report still lives in the markdown the existing harness writes.
        results.append({
            "strategy_id": sid,
            "region": region,
            "window": f"{start.isoformat()} → {end.isoformat()}",
            "n_days": report.n_days,
            "n_trades": report.n_trades,
            "total_pnl_pct": report.total_pnl_pct,
            "avg_pnl_pct": report.avg_pnl_pct,
            "hit_rate": report.hit_rate,
            "win_loss_ratio": report.win_loss_ratio,
        })

        # Per-strategy markdown
        write_report(report)

    summary = {
        "generated_for_iso_week": f"{iso_year}-W{iso_week:02d}",
        "window_days": window_days,
        "lookahead_bias_caveat": (
            "LLM strategies fetched live web context during replay (WebSearch "
            "sees today's news). Treat the numbers as a structural sanity "
            "check, not a clean out-of-sample test."
        ),
        "strategies": results,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("backtest pass: wrote %s (%d strategies)", summary_path, len(results))
    return summary


def latest_backtest_summary() -> dict | None:
    """Read the most recent weekly backtest summary, if any."""
    d = STATE_ROOT / "backtest"
    if not d.exists():
        return None
    week_dirs = sorted([p for p in d.iterdir() if p.is_dir()])
    if not week_dirs:
        return None
    path = week_dirs[-1] / "summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _trading_days_in_range(start: date, end: date, region: str) -> list[date]:
    """Approximate trading-day filter using the existing calendar tool.
    Excludes weekends + known holidays for the region."""
    from trading_bot.tools.calendar import is_market_open_on
    out: list[date] = []
    d = start
    while d <= end:
        if is_market_open_on(d, region):
            out.append(d)
        d += timedelta(days=1)
    return out


def write_report(report: BacktestReport) -> Path:
    """Write a markdown report under state/diagnostics/."""
    d = STATE_ROOT / "diagnostics"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"backtest_{report.strategy_id}_{report.start_date}_{report.end_date}.md"
    with p.open("w") as f:
        f.write(f"# Backtest — {report.strategy_id} ({report.region})\n\n")
        f.write(f"**Window:** {report.start_date} → {report.end_date}  \n")
        f.write(f"**Trading days:** {report.n_days}  \n")
        f.write(f"**Trades:** {report.n_trades}  \n\n")
        f.write("## Aggregate\n\n")
        f.write(f"- Total P&L: **{report.total_pnl_pct:+.2f}%** (sum of daily weighted returns)\n")
        f.write(f"- Avg P&L per trade: {report.avg_pnl_pct:+.2f}%\n")
        f.write(f"- Hit rate: {report.hit_rate * 100:.1f}%\n")
        f.write(f"- Win/loss ratio: {report.win_loss_ratio:.2f}\n\n")
        f.write("## Limitations\n\n")
        f.write("- Technicals only — no news / macro replay.\n")
        f.write("- Fills modelled at next-day close. No slippage / fees.\n")
        f.write("- Single-region; multi-region strategies need separate runs.\n\n")
        # Sample trades
        if report.trades:
            f.write("## First 20 trades\n\n")
            f.write("| Date | Ticker | Entry | Exit | P&L % | Alloc % |\n")
            f.write("|---|---|---:|---:|---:|---:|\n")
            for t in report.trades[:20]:
                f.write(f"| {t.date} | {t.ticker} | ${t.entry_price:.2f} | "
                        f"${t.exit_price:.2f} | {t.pnl_pct:+.2f}% | {t.allocation_pct:.1f}% |\n")
    return p
