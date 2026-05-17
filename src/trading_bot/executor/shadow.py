from __future__ import annotations

import uuid
from datetime import date

from trading_bot.executor.base import Executor, TradeIntent
from trading_bot.state import (
    TradeRecord,
    append_trade,
    mark_trade_exited,
    read_open_trades,
)
from trading_bot.tools import get_history


_TIER = "shadow"


class ShadowExecutor(Executor):
    """Records trades on paper using real market prices. No orders placed anywhere.

    Entry: looks up the most recent close as the assumed fill price (Wave 1 simplification —
    we'll switch to the day's open price once we run intraday and have access to that bar).
    Exit: looks up the close on the exit date, computes P&L.
    Stops / take-profits: simulated at end-of-day by checking the intra-day high/low against
    the trigger prices and recording the trigger fill if hit.
    """

    def enter(
        self,
        intents: list[TradeIntent],
        *,
        strategy_id: str,
        region: str,
        capital_gbp: float,
        on_date: date,
    ) -> None:
        if not intents:
            return

        tickers = [i.ticker for i in intents]
        history = get_history(tickers, lookback_days=2, end_date=on_date)

        for intent in intents:
            bars = history.get(intent.ticker)
            if not bars:
                # No price data available; skip this intent. Wave 1 doesn't have
                # a richer fallback — we just don't record the trade.
                continue
            entry_price = bars[-1].close
            allocation_gbp = capital_gbp * (intent.allocation_pct / 100.0)
            quantity = allocation_gbp / entry_price if entry_price > 0 else 0.0

            record = TradeRecord(
                trade_id=str(uuid.uuid4()),
                strategy_id=strategy_id,
                region=region,
                tier=_TIER,
                ticker=intent.ticker,
                side="long",
                entry_date=on_date.isoformat(),
                entry_price=entry_price,
                quantity=quantity,
                allocation_pct=intent.allocation_pct,
                stop_loss_pct=intent.stop_loss_pct,
                take_profit_pct=intent.take_profit_pct,
                thesis=intent.thesis,
            )
            append_trade(record)

    def exit_scheduled(
        self,
        *,
        strategy_id: str,
        region: str,
        on_date: date,
    ) -> list[dict]:
        open_trades = read_open_trades(strategy_id=strategy_id, region=region, on_date=on_date)
        if not open_trades:
            return []

        tickers = list({t["ticker"] for t in open_trades})
        history = get_history(tickers, lookback_days=1, end_date=on_date)

        closed: list[dict] = []
        for trade in open_trades:
            bars = history.get(trade["ticker"])
            if not bars:
                continue
            bar = bars[-1]
            entry_price = float(trade["entry_price"])
            quantity = float(trade["quantity"])

            exit_price, exit_reason = _resolve_exit(
                entry_price=entry_price,
                day_high=bar.high,
                day_low=bar.low,
                day_close=bar.close,
                stop_loss_pct=trade.get("stop_loss_pct"),
                take_profit_pct=trade.get("take_profit_pct"),
            )

            pnl_gbp = (exit_price - entry_price) * quantity
            pnl_pct = (exit_price / entry_price - 1.0) * 100.0 if entry_price > 0 else 0.0

            mark_trade_exited(
                trade_id=trade["trade_id"],
                exit_date=on_date,
                exit_price=exit_price,
                pnl_gbp=pnl_gbp,
                pnl_pct=pnl_pct,
                exit_reason=exit_reason,
            )

            closed.append(
                {
                    **trade,
                    "exit_date": on_date.isoformat(),
                    "exit_price": exit_price,
                    "pnl_gbp": pnl_gbp,
                    "pnl_pct": pnl_pct,
                    "exit_reason": exit_reason,
                }
            )
        return closed


def _resolve_exit(
    *,
    entry_price: float,
    day_high: float,
    day_low: float,
    day_close: float,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
) -> tuple[float, str]:
    """Decide the exit fill price for a single position given the day's range.

    Conservative simulation:
    - If both stop and take-profit could have triggered intra-day, assume the worst
      (stop fires first). This avoids over-stating shadow P&L.
    - If only one triggered, that's the exit.
    - Otherwise, exit at the day's close.
    """
    stop_price = entry_price * (1 + stop_loss_pct / 100.0) if stop_loss_pct is not None else None
    target_price = (
        entry_price * (1 + take_profit_pct / 100.0) if take_profit_pct is not None else None
    )

    stop_hit = stop_price is not None and day_low <= stop_price
    target_hit = target_price is not None and day_high >= target_price

    if stop_hit and target_hit:
        return stop_price, "stop"  # type: ignore[return-value]
    if stop_hit:
        return stop_price, "stop"  # type: ignore[return-value]
    if target_hit:
        return target_price, "take_profit"  # type: ignore[return-value]
    return day_close, "scheduled"
