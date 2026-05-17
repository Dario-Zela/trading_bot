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
    """Build {ticker: [Bar]} from yfinance's mixed-shape output.

    yfinance returns different layouts depending on universe / number of
    tickers / version: single-ticker is flat columns; multi-ticker can be
    MultiIndex (ticker, price) or (price, ticker). We normalise to flat
    Open/High/Low/Close/Volume columns before iterating.
    """
    out: dict[str, list[Bar]] = {}
    is_single = len(tickers) == 1

    for ticker in tickers:
        sub = _slice_ticker(df, ticker, is_single)
        if sub is None or sub.empty:
            continue
        sub = sub.dropna().tail(lookback_days)
        bars: list[Bar] = []
        for ts in sub.index:
            try:
                bars.append(
                    Bar(
                        ticker=ticker,
                        bar_date=_to_date(ts),
                        open=float(sub.at[ts, "Open"]),
                        high=float(sub.at[ts, "High"]),
                        low=float(sub.at[ts, "Low"]),
                        close=float(sub.at[ts, "Close"]),
                        volume=int(sub.at[ts, "Volume"]),
                    )
                )
            except (KeyError, ValueError, TypeError):
                # Single missing bar shouldn't kill the whole ticker
                continue
        if bars:
            out[ticker] = bars
    return out


def _slice_ticker(df: pd.DataFrame, ticker: str, is_single: bool) -> pd.DataFrame | None:
    """Extract one ticker's per-day OHLCV with a flat column index."""
    if is_single:
        sub = df
    elif isinstance(df.columns, pd.MultiIndex):
        # Try ticker-at-level-0 (the group_by='ticker' layout) first, then
        # fall back to ticker-at-level-1.
        level_0 = df.columns.get_level_values(0)
        level_1 = df.columns.get_level_values(1)
        if ticker in level_0:
            sub = df[ticker]
        elif ticker in level_1:
            sub = df.xs(ticker, axis=1, level=1)
        else:
            return None
    else:
        return None
    if isinstance(sub.columns, pd.MultiIndex):
        # Flatten — innermost level should be the OHLCV name.
        sub = sub.copy()
        sub.columns = [c[-1] if isinstance(c, tuple) else c for c in sub.columns]
    return sub


def _to_date(ts) -> date:
    if isinstance(ts, pd.Timestamp):
        return ts.date()
    if isinstance(ts, datetime):
        return ts.date()
    return date.fromisoformat(str(ts)[:10])
