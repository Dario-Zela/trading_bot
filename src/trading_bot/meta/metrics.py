"""Per-strategy rolling-window metrics for the evolution agent.

Reads ledger.jsonl + predictions.jsonl and computes the numbers that drive
promote/demote/tune decisions in trading_bot.meta.evolution.

Metrics are keyed by (strategy_id, region) because each strategy can run
independently across regions (us / uk-eu / asia) with its own tier and
performance profile. The evolution agent then makes per-(sid, region)
promote/demote calls.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from trading_bot.state.paths import ledger_path, predictions_path


log = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    strategy_id: str
    region: str
    window_days: int
    window_start: str
    window_end: str

    # Trade-derived
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    total_pnl_gbp: float = 0.0
    avg_pnl_pct: float = 0.0
    hit_rate: float = 0.0       # n_wins / n_trades
    max_drawdown_pct: float = 0.0   # peak-to-trough on cumulative P&L
    pnl_per_week: list[float] = field(default_factory=list)

    # Prediction-derived
    n_predictions: int = 0
    n_predictions_graded: int = 0
    ic: float | None = None     # spearman-rank correlation predicted vs actual
    top_minus_bottom_decile_spread: float | None = None  # mean actual return: top decile - bottom decile by predicted

    def is_alive(self) -> bool:
        """True if we have enough data to make decisions about this strategy."""
        return self.n_trades >= 5 or self.n_predictions_graded >= 30


def compute_metrics(
    strategy_id: str,
    region: str,
    *,
    window_days: int = 14,
    end_date: date | None = None,
) -> StrategyMetrics:
    """Compute metrics for a single (strategy, region) pair over the window.

    Phase 11B — if the strategy has a `last_tune_date` (set by the
    evolution agent on `tune` actions), the window starts from that
    date so pre-tune trades don't dilute the post-tune signal.
    """
    end = end_date or date.today()
    start = end - timedelta(days=window_days)
    # Apply the last-tune reset if it's MORE recent than the window start
    tune_iso = _last_tune_date(strategy_id)
    if tune_iso:
        try:
            tune_date = date.fromisoformat(tune_iso)
            if tune_date > start:
                start = tune_date
        except (TypeError, ValueError):
            pass

    m = StrategyMetrics(
        strategy_id=strategy_id,
        region=region,
        window_days=window_days,
        window_start=start.isoformat(),
        window_end=end.isoformat(),
    )

    trades_window = _read_trades(strategy_id, region, start, end)
    _fill_trade_metrics(m, trades_window)

    preds_window = _read_predictions(strategy_id, region, start, end)
    _fill_prediction_metrics(m, preds_window)

    return m


def compute_all_metrics(
    *,
    window_days: int = 14,
    end_date: date | None = None,
) -> dict[tuple[str, str], StrategyMetrics]:
    """Compute metrics for every (strategy_id, region) pair seen in the window.

    Returns {(strategy_id, region): StrategyMetrics}. The evolution agent
    iterates over this so it can make region-specific promote/demote calls.
    """
    end = end_date or date.today()
    start = end - timedelta(days=window_days)
    keys: set[tuple[str, str]] = set()
    for rec in _iter_lines(ledger_path()):
        sid = rec.get("strategy_id")
        region = rec.get("region") or "us"
        if sid and _in_window(rec.get("entry_date"), start, end):
            keys.add((sid, region))
    for rec in _iter_lines(predictions_path()):
        sid = rec.get("strategy_id")
        region = rec.get("region") or "us"
        if sid and _in_window(rec.get("prediction_date"), start, end):
            keys.add((sid, region))

    return {
        (sid, region): compute_metrics(sid, region, window_days=window_days, end_date=end)
        for (sid, region) in sorted(keys)
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _last_tune_date(strategy_id: str) -> str | None:
    """Read `last_tune_date` from the strategy's config.yaml.
    Returns None if the field is absent (strategy not yet tuned)."""
    try:
        from trading_bot.strategy.registry import load_strategy_config
        cfg = load_strategy_config(strategy_id)
    except Exception:
        return None
    return getattr(cfg, "last_tune_date", None)


def _iter_lines(path: Path):
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _in_window(date_str: str | None, start: date, end: date) -> bool:
    if not date_str:
        return False
    try:
        d = date.fromisoformat(date_str)
    except (TypeError, ValueError):
        return False
    return start <= d <= end


def _read_trades(strategy_id: str, region: str, start: date, end: date) -> list[dict]:
    out = []
    for rec in _iter_lines(ledger_path()):
        if rec.get("strategy_id") != strategy_id:
            continue
        if (rec.get("region") or "us") != region:
            continue
        if not rec.get("exit_date"):
            continue
        if not _in_window(rec.get("entry_date"), start, end):
            continue
        # Skip phantom trades (cancelled / cleared) — they're not real P&L
        if rec.get("exit_reason") in ("cancelled", "cleared"):
            continue
        out.append(rec)
    return out


def _read_predictions(strategy_id: str, region: str, start: date, end: date) -> list[dict]:
    out = []
    for rec in _iter_lines(predictions_path()):
        if rec.get("strategy_id") != strategy_id:
            continue
        if (rec.get("region") or "us") != region:
            continue
        if not _in_window(rec.get("prediction_date"), start, end):
            continue
        out.append(rec)
    return out


def _fill_trade_metrics(m: StrategyMetrics, trades: list[dict]) -> None:
    if not trades:
        return
    pnl_pcts: list[float] = []
    pnl_gbps: list[float] = []
    n_wins = n_losses = 0
    for t in trades:
        pnl_gbp = float(t.get("pnl_gbp") or 0.0)
        pnl_pct = float(t.get("pnl_pct") or 0.0)
        pnl_gbps.append(pnl_gbp)
        pnl_pcts.append(pnl_pct)
        if pnl_gbp > 0:
            n_wins += 1
        elif pnl_gbp < 0:
            n_losses += 1

    m.n_trades = len(trades)
    m.n_wins = n_wins
    m.n_losses = n_losses
    m.total_pnl_gbp = round(sum(pnl_gbps), 2)
    m.avg_pnl_pct = round(sum(pnl_pcts) / len(pnl_pcts), 3)
    m.hit_rate = round(n_wins / m.n_trades, 3) if m.n_trades else 0.0

    # Max drawdown on the equity curve through the window
    sorted_trades = sorted(trades, key=lambda t: t.get("exit_date") or "")
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in sorted_trades:
        cum += float(t.get("pnl_gbp") or 0.0)
        peak = max(peak, cum)
        dd = cum - peak  # negative when below peak
        max_dd = min(max_dd, dd)
    # Express as pct of strategy capital — approximate using £10k default
    m.max_drawdown_pct = round((max_dd / 10000.0) * 100.0, 2)


def _fill_prediction_metrics(m: StrategyMetrics, preds: list[dict]) -> None:
    if not preds:
        return
    m.n_predictions = len(preds)
    graded = [
        p for p in preds
        if p.get("actual_return_pct") is not None and p.get("predicted_return_pct") is not None
    ]
    m.n_predictions_graded = len(graded)
    if len(graded) < 4:
        return

    # Spearman rank correlation
    pred_ranks = _rank([float(p["predicted_return_pct"]) for p in graded])
    actual_ranks = _rank([float(p["actual_return_pct"]) for p in graded])
    n = len(graded)
    mean_p = sum(pred_ranks) / n
    mean_a = sum(actual_ranks) / n
    num = sum((pr - mean_p) * (ar - mean_a) for pr, ar in zip(pred_ranks, actual_ranks))
    den_p = sum((pr - mean_p) ** 2 for pr in pred_ranks) ** 0.5
    den_a = sum((ar - mean_a) ** 2 for ar in actual_ranks) ** 0.5
    if den_p > 0 and den_a > 0:
        m.ic = round(num / (den_p * den_a), 3)

    # Top vs bottom decile actual-return spread
    sorted_by_pred = sorted(graded, key=lambda p: float(p["predicted_return_pct"]), reverse=True)
    decile = max(1, len(sorted_by_pred) // 10)
    top = sorted_by_pred[:decile]
    bottom = sorted_by_pred[-decile:]
    top_avg = sum(float(p["actual_return_pct"]) for p in top) / len(top)
    bot_avg = sum(float(p["actual_return_pct"]) for p in bottom) / len(bottom)
    m.top_minus_bottom_decile_spread = round(top_avg - bot_avg, 3)


def _rank(values: list[float]) -> list[float]:
    """Average rank, handling ties."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-indexed
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks
