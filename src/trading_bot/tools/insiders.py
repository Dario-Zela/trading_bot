"""Insider-trading tool — recent insider buy/sell pattern from yfinance.

yfinance's `.insider_purchases` and `.insider_transactions` parse from Yahoo's
data feed, which mirrors SEC Form 4 filings. Less authoritative than EDGAR
directly but sufficient for "are insiders buying or selling?" signal.

Returns a compact summary (counts + net direction) rather than raw rows so
LLM prompts stay bounded.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class InsiderSummary:
    ticker: str
    lookback_days: int
    n_buys: int
    n_sells: int
    total_buy_value_usd: float | None
    total_sell_value_usd: float | None
    net_signal: str  # "net buying" / "net selling" / "balanced" / "no activity"


def get_insider_trades(ticker: str, days: int = 60) -> InsiderSummary:
    n_buys = 0
    n_sells = 0
    buy_value = 0.0
    sell_value = 0.0

    cutoff = datetime.utcnow() - timedelta(days=days)

    try:
        df = yf.Ticker(ticker).insider_transactions
    except Exception as e:
        log.debug("insider_transactions fetch failed for %s: %s", ticker, e)
        df = None

    if isinstance(df, pd.DataFrame) and not df.empty:
        for _, row in df.iterrows():
            start = row.get("Start Date") or row.get("Date")
            if start is not None:
                try:
                    dt = pd.to_datetime(start)
                    if dt.to_pydatetime() < cutoff:
                        continue
                except Exception:
                    pass
            txn = str(row.get("Transaction", "")).lower()
            value = row.get("Value")
            try:
                value_f = float(value) if value is not None else 0.0
            except (TypeError, ValueError):
                value_f = 0.0
            if "buy" in txn or "purchase" in txn:
                n_buys += 1
                buy_value += abs(value_f)
            elif "sale" in txn or "sell" in txn or "disposition" in txn:
                n_sells += 1
                sell_value += abs(value_f)

    if n_buys == 0 and n_sells == 0:
        signal = "no activity"
    elif buy_value > sell_value * 1.5:
        signal = "net buying"
    elif sell_value > buy_value * 1.5:
        signal = "net selling"
    else:
        signal = "balanced"

    return InsiderSummary(
        ticker=ticker,
        lookback_days=days,
        n_buys=n_buys,
        n_sells=n_sells,
        total_buy_value_usd=buy_value if buy_value > 0 else None,
        total_sell_value_usd=sell_value if sell_value > 0 else None,
        net_signal=signal,
    )
