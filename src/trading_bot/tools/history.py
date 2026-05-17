from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class Bar:
    ticker: str
    bar_date: date
    open: float
    high: float
    low: float
    close: float
    volume: int

    @property
    def return_pct(self) -> float:
        return (self.close / self.open - 1.0) * 100.0


def get_history(
    tickers: Iterable[str],
    lookback_days: int = 5,
    end_date: date | None = None,
) -> dict[str, list[Bar]]:
    """Fetch daily OHLCV for the given tickers over the lookback window.

    Wave 1 uses yfinance (no auth, batched). yfinance returns trading days only.
    Returns a dict {ticker: [Bar, ...]} in chronological order, most-recent last.
    Tickers with no data are omitted from the result.
    """
    tickers = list(tickers)
    if not tickers:
        return {}

    end = end_date or date.today()
    # Pad the lookback so we always cover requested trading days even across weekends/holidays
    period = max(lookback_days * 2 + 5, 10)

    df = yf.download(
        tickers=tickers,
        period=f"{period}d",
        end=pd.Timestamp(end) + pd.Timedelta(days=1),
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    return _flatten(df, tickers, lookback_days)


def _flatten(df: pd.DataFrame, tickers: list[str], lookback_days: int) -> dict[str, list[Bar]]:
    out: dict[str, list[Bar]] = {}
    is_single = len(tickers) == 1

    for ticker in tickers:
        sub = df if is_single else df.get(ticker)
        if sub is None or sub.empty:
            continue
        sub = sub.dropna().tail(lookback_days)
        bars: list[Bar] = []
        for ts, row in sub.iterrows():
            bars.append(
                Bar(
                    ticker=ticker,
                    bar_date=_to_date(ts),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                )
            )
        if bars:
            out[ticker] = bars
    return out


def _to_date(ts) -> date:
    if isinstance(ts, pd.Timestamp):
        return ts.date()
    if isinstance(ts, datetime):
        return ts.date()
    return date.fromisoformat(str(ts)[:10])
