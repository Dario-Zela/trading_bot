"""Phase 11I — backtest framework for the weekly evolution agent.

Two paths, picked by strategy implementation:

- Rule-based + momentum-stub strategies: `run_backtest` walks a date
  range, invokes `select_picks(on_date)` against historical data, and
  simulates fills at the next-day close. Deterministic and cheap.

- LLM strategies: `grade_from_live_predictions` reads
  `state/predictions.jsonl` and aggregates the picks the strategy
  *actually* emitted during the window, scored by the live
  prediction-grader. No Claude calls at backtest time. No lookahead
  bias — these are the bets the strategy genuinely made, not what a
  re-invoked LLM would pick today with a backdated label.

We deliberately do *not* replay LLM strategies any more. Re-asking
present-day Claude "what would you have picked last Tuesday?" mostly
measures today-Claude with hindsight (the prompt-time tools fetch
current data; the underlying model is the current weights), not the
strategy's historical edge. The live-predictions path uses the real
picks that actually traded, so the resulting metrics are what the
evolution agent should be making decisions on.

Outputs a BacktestReport dataclass + a markdown summary under
`state/diagnostics/backtest_<strategy>_<start>_<end>.md`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

from trading_bot.state.paths import STATE_ROOT, predictions_path
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


def grade_from_live_predictions(
    strategy_id: str, region: str, start_date: date, end_date: date,
) -> BacktestReport:
    """Aggregate the strategy's actually-emitted live predictions in
    [start_date, end_date] from `state/predictions.jsonl`. Returns a
    BacktestReport with the same shape as `run_backtest` so the
    weekly summary schema is uniform across LLM and rule-based
    strategies.

    Each record in predictions.jsonl carries `actual_return_pct`,
    populated by the prediction-grader when the holding period
    resolves. Picks not yet graded (last 1–2 days) are skipped.
    Entry/exit prices are not tracked in predictions, so the synthesized
    _BackTrade entries leave those at 0.0 — the markdown writer prints
    "—" for them.
    """
    path = predictions_path()
    trades: list[_BackTrade] = []
    by_day: dict[str, float] = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("strategy_id") != strategy_id:
                    continue
                if r.get("region") != region:
                    continue
                d_str = r.get("prediction_date")
                if not d_str:
                    continue
                try:
                    d = date.fromisoformat(d_str)
                except ValueError:
                    continue
                if d < start_date or d > end_date:
                    continue
                actual = r.get("actual_return_pct")
                if actual is None:
                    continue
                pnl = float(actual)
                trades.append(_BackTrade(
                    date=d_str,
                    ticker=r.get("ticker", ""),
                    entry_price=0.0,
                    exit_price=0.0,
                    pnl_pct=round(pnl, 3),
                    allocation_pct=0.0,
                ))
                by_day[d_str] = round(by_day.get(d_str, 0.0) + pnl, 3)

    if not trades:
        return BacktestReport(
            strategy_id=strategy_id, region=region,
            start_date=start_date.isoformat(), end_date=end_date.isoformat(),
            n_days=len(by_day), n_trades=0,
            total_pnl_pct=0.0, avg_pnl_pct=0.0, hit_rate=0.0, win_loss_ratio=0.0,
            by_day_pnl=by_day, trades=[],
        )
    pnls = [t.pnl_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    wl = abs(avg_win / avg_loss) if avg_loss else 0.0
    return BacktestReport(
        strategy_id=strategy_id, region=region,
        start_date=start_date.isoformat(), end_date=end_date.isoformat(),
        n_days=len(by_day), n_trades=len(trades),
        total_pnl_pct=round(sum(pnls), 3),
        avg_pnl_pct=round(sum(pnls) / len(pnls), 3),
        hit_rate=round(len(wins) / len(trades), 3),
        win_loss_ratio=round(wl, 3),
        by_day_pnl=by_day,
        trades=trades,
    )


def run_weekly_backtest_pass(today: date, window_days: int = 14) -> dict:
    """Produce per-strategy 14-day metrics for the evolution agent.

    Routing by implementation:
    - rule_based / momentum_stub → `run_backtest` (cheap deterministic
      replay; LLM not invoked).
    - llm → `grade_from_live_predictions` (reads picks the strategy
      actually emitted live during the window; no Claude calls, no
      lookahead bias).

    Writes a roll-up `state/backtest/<iso-week>/summary.json` the
    evolution prompt reads, plus per-strategy markdown under
    `state/diagnostics/`. Runs in seconds; safe to dispatch on demand.
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
    end = today - timedelta(days=1)                   # stop yesterday so we can resolve next-day fills

    results: list[dict] = []

    for strat in active:
        sid = strat.config.id
        region = strat.config.region
        impl = strat.config.implementation
        log.info("backtest pass: %s/%s impl=%s window %s → %s", sid, region, impl, start, end)
        try:
            if impl == "llm":
                report = grade_from_live_predictions(
                    sid, region, start_date=start, end_date=end,
                )
                source = "live-predictions"
            else:
                report = run_backtest(
                    sid, start_date=start, end_date=end, region=region,
                )
                source = "replay"
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
            "source": source,
        })

        # Per-strategy markdown
        write_report(report, source=source)

    summary = {
        "generated_for_iso_week": f"{iso_year}-W{iso_week:02d}",
        "window_days": window_days,
        "methodology_note": (
            "LLM strategies are scored on the picks they actually emitted live "
            "during the window (from state/predictions.jsonl, graded by the "
            "live prediction-grader). Rule-based and momentum-stub strategies "
            "are replayed deterministically against historical bars. No LLM "
            "is re-invoked at backtest time, so the numbers are the strategy's "
            "real performance, free of replay/lookahead bias."
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


def write_report(report: BacktestReport, *, source: str = "replay") -> Path:
    """Write a markdown report under state/diagnostics/.

    `source` is "replay" (rule-based / momentum-stub deterministic walk)
    or "live-predictions" (LLM strategies scored on real emitted picks).
    The Limitations section adapts so the human reader knows which
    methodology produced the numbers.
    """
    d = STATE_ROOT / "diagnostics"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"backtest_{report.strategy_id}_{report.start_date}_{report.end_date}.md"
    with p.open("w") as f:
        f.write(f"# Backtest — {report.strategy_id} ({report.region})\n\n")
        f.write(f"**Window:** {report.start_date} → {report.end_date}  \n")
        f.write(f"**Source:** {source}  \n")
        f.write(f"**Trading days:** {report.n_days}  \n")
        f.write(f"**Trades:** {report.n_trades}  \n\n")
        f.write("## Aggregate\n\n")
        f.write(f"- Total P&L: **{report.total_pnl_pct:+.2f}%** (sum of per-trade returns)\n")
        f.write(f"- Avg P&L per trade: {report.avg_pnl_pct:+.2f}%\n")
        f.write(f"- Hit rate: {report.hit_rate * 100:.1f}%\n")
        f.write(f"- Win/loss ratio: {report.win_loss_ratio:.2f}\n\n")
        f.write("## Methodology\n\n")
        if source == "live-predictions":
            f.write(
                "- Scored from `state/predictions.jsonl` — the picks this "
                "strategy actually emitted live during the window, graded by "
                "the live prediction-grader.\n"
                "- No replay, no LLM re-invocation, no lookahead bias.\n"
                "- Includes only resolved predictions; the last 1–2 days may "
                "still be ungraded.\n\n"
            )
        else:
            f.write(
                "- Deterministic replay of `select_picks(on_date)` against "
                "historical bars.\n"
                "- Fills modelled at next-day close. No slippage / fees.\n"
                "- Single-region; multi-region strategies need separate runs.\n\n"
            )
        # Sample trades
        if report.trades:
            f.write("## First 20 trades\n\n")
            if source == "live-predictions":
                # Entry/exit prices aren't carried in predictions; print just
                # date / ticker / realised P&L.
                f.write("| Date | Ticker | P&L % |\n")
                f.write("|---|---|---:|\n")
                for t in report.trades[:20]:
                    f.write(f"| {t.date} | {t.ticker} | {t.pnl_pct:+.2f}% |\n")
            else:
                f.write("| Date | Ticker | Entry | Exit | P&L % | Alloc % |\n")
                f.write("|---|---|---:|---:|---:|---:|\n")
                for t in report.trades[:20]:
                    f.write(f"| {t.date} | {t.ticker} | ${t.entry_price:.2f} | "
                            f"${t.exit_price:.2f} | {t.pnl_pct:+.2f}% | {t.allocation_pct:.1f}% |\n")
    return p
