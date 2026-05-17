"""Sector-strength tool — ranks the 11 SPDR sector ETFs by relative return.

Used by sector-rotator (its core decision signal) and macro-aligned (to
confirm/contradict the macro view's sector calls).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from trading_bot.tools.universe import _US_ETFS_SECTOR


_SECTOR_LABELS = {
    "XLF": "Financials",
    "XLE": "Energy",
    "XLK": "Technology",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}


@dataclass(frozen=True)
class SectorRanking:
    ticker: str
    label: str
    return_1d_pct: float | None
    return_5d_pct: float | None
    return_20d_pct: float | None


def get_sector_strength() -> list[SectorRanking]:
    """Return the 11 SPDR sector ETFs ranked by 5-day return (best → worst).
    Caller can re-rank by any window."""
    df = yf.download(
        _US_ETFS_SECTOR, period="40d", progress=False, threads=True, auto_adjust=False
    )
    out: list[SectorRanking] = []
    if not isinstance(df.columns, pd.MultiIndex):
        return out
    for ticker in _US_ETFS_SECTOR:
        try:
            close = df["Close"][ticker].dropna()
        except KeyError:
            continue
        if len(close) < 2:
            continue
        out.append(
            SectorRanking(
                ticker=ticker,
                label=_SECTOR_LABELS.get(ticker, ticker),
                return_1d_pct=_pct(close, 1),
                return_5d_pct=_pct(close, 5),
                return_20d_pct=_pct(close, 20),
            )
        )
    out.sort(key=lambda x: (x.return_5d_pct if x.return_5d_pct is not None else -999), reverse=True)
    return out


def _pct(series: pd.Series, periods: int) -> float | None:
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    if len(series) <= periods:
        return None
    past = float(series.iloc[-(periods + 1)])
    if past <= 0:
        return None
    return (float(series.iloc[-1]) / past - 1.0) * 100.0
