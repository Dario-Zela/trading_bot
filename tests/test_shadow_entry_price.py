"""Shadow same-day round-trips were silently writing pnl=0.0.

The bug: `_prefilter` style — entry used `bars[-1].close` for the
entry price, exit used the same `.close` for the same bar, so on
1-day holds entry_price == exit_price and the simulator wrote
pnl=0 regardless of what the price actually did during the session
(observed across every shadow exit on 2026-05-26 — CODX, INTC,
MDB, the control microcaps all came back pnl=0.0).

Fix: when today's bar is available at entry time, use bar.open as
the entry price so entry (open) and exit (close) sit on different
sides of the day's range.
"""
from __future__ import annotations

import uuid
from datetime import date

import pytest

from trading_bot.executor.shadow import ShadowExecutor
from trading_bot.executor.base import TradeIntent
from trading_bot.tools.history import Bar


def _bar(d: date, *, open_: float, close: float, high: float | None = None, low: float | None = None) -> Bar:
    return Bar(
        ticker="FAKE",
        bar_date=d,
        open=open_,
        high=high if high is not None else max(open_, close),
        low=low if low is not None else min(open_, close),
        close=close,
        volume=1_000_000,
    )


def _intent(ticker: str = "FAKE", hold_days: int = 1) -> TradeIntent:
    return TradeIntent(
        ticker=ticker, allocation_pct=10.0,
        stop_loss_pct=-3.0, take_profit_pct=3.0,
        thesis="test", hold_days=hold_days,
    )


def test_shadow_entry_uses_today_open_when_today_bar_available(monkeypatch, tmp_path):
    """Today's bar present at entry → entry_price = today.open, NOT
    today.close. This is the fix for the same-day-round-trip pnl=0 bug."""
    today = date(2026, 5, 27)
    # yfinance has today's bar already (entry pipeline ran after open)
    fake_bars = {"FAKE": [_bar(today, open_=100.0, close=103.0)]}
    monkeypatch.setattr(
        "trading_bot.executor.shadow.get_history",
        lambda tickers, lookback_days, end_date: fake_bars,
    )
    # Stub price-sanity so the anomaly check doesn't drop our entry
    monkeypatch.setattr(
        "trading_bot.tools.price_sanity.is_price_anomalous",
        lambda close, bars: (False, ""),
    )
    # Capture the trade record without actually writing the ledger
    written: list = []
    monkeypatch.setattr(
        "trading_bot.executor.shadow.append_trade",
        lambda record: written.append(record),
    )

    ShadowExecutor().enter(
        [_intent()], strategy_id="x", region="us",
        capital_gbp=1000.0, on_date=today,
    )

    assert len(written) == 1
    # Entry price = today's open, NOT today's close
    assert written[0].entry_price == 100.0


def test_shadow_entry_uses_prev_close_when_today_bar_missing(monkeypatch):
    """When yfinance hasn't published today's bar (pre-market entry),
    fall back to yesterday's close — the standard last-available-price
    pattern. Verifies the conditional doesn't break the legacy path."""
    today = date(2026, 5, 27)
    yesterday = date(2026, 5, 26)
    fake_bars = {"FAKE": [_bar(yesterday, open_=98.0, close=100.0)]}
    monkeypatch.setattr(
        "trading_bot.executor.shadow.get_history",
        lambda tickers, lookback_days, end_date: fake_bars,
    )
    monkeypatch.setattr(
        "trading_bot.tools.price_sanity.is_price_anomalous",
        lambda close, bars: (False, ""),
    )
    written: list = []
    monkeypatch.setattr(
        "trading_bot.executor.shadow.append_trade",
        lambda record: written.append(record),
    )

    ShadowExecutor().enter(
        [_intent()], strategy_id="x", region="us",
        capital_gbp=1000.0, on_date=today,
    )

    assert len(written) == 1
    # Yesterday's close, since today's bar wasn't available
    assert written[0].entry_price == 100.0
