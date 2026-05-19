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
) -> BacktestReport:
    """Replay a strategy over the given date range. Returns a
    BacktestReport. Caller is responsible for writing the markdown."""
    from trading_bot.strategy.registry import load_strategy_config
    from trading_bot.strategy.control_rule_based import ControlRuleBased
    from trading_bot.strategy.momentum_stub import MomentumTraderStub
    from trading_bot.strategy.llm_strategy import LLMStrategy

    cfg = load_strategy_config(strategy_id)
    impl = cfg.implementation
    # Direct instantiation (skips runs_in expansion — we backtest one
    # (strategy, region) at a time, region comes from the top-level
    # cfg.region field).
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
            # Find the bar whose date == d (entry) and the next one (exit)
            entry_idx = next((i for i, b in enumerate(bars) if str(b.bar_date) == d.isoformat()), None)
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
