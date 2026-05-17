"""Cross-asset tools — yield curve, credit spreads, dollar, commodities.

All yfinance-backed for the simplest possible implementation. The data
shape returned by each function is a small dataclass / dict that LLMStrategy
can render compactly into a prompt.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd
import yfinance as yf


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# get_yield_curve
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class YieldCurve:
    as_of: str
    y3m: float | None  # 3-month T-bill yield (%)
    y2y: float | None
    y5y: float | None
    y10y: float | None
    y30y: float | None

    @property
    def spread_2s10s(self) -> float | None:
        if self.y2y is None or self.y10y is None:
            return None
        return self.y10y - self.y2y

    @property
    def spread_3m10y(self) -> float | None:
        if self.y3m is None or self.y10y is None:
            return None
        return self.y10y - self.y3m


_YIELD_TICKERS = {
    "y3m": "^IRX",
    "y5y": "^FVX",
    "y10y": "^TNX",
    "y30y": "^TYX",
}
# 2Y has no widely-tracked yfinance ticker; approximate with iShares 2Y ETF
# converted via implied YTM. Simpler: omit if not available.


def get_yield_curve() -> YieldCurve:
    """Snapshot of US Treasury yields by tenor (3M, 5Y, 10Y, 30Y)."""
    df = yf.download(
        tickers=list(_YIELD_TICKERS.values()),
        period="5d",
        progress=False,
        threads=True,
        auto_adjust=False,
    )
    out: dict[str, float | None] = {k: None for k in ["y3m", "y2y", "y5y", "y10y", "y30y"]}
    as_of = date.today().isoformat()
    if isinstance(df.columns, pd.MultiIndex):
        for key, ticker in _YIELD_TICKERS.items():
            try:
                series = df["Close"][ticker].dropna()
                if not series.empty:
                    out[key] = float(series.iloc[-1])
                    as_of = str(series.index[-1].date())
            except KeyError:
                continue
    return YieldCurve(as_of=as_of, **out)


# ---------------------------------------------------------------------------
# get_credit_spreads
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CreditSpreads:
    as_of: str
    hyg_close: float | None
    lqd_close: float | None
    hyg_5d_return_pct: float | None
    lqd_5d_return_pct: float | None
    hy_vs_ig_5d_diff: float | None  # HY minus IG performance — proxy for spread widening


def get_credit_spreads() -> CreditSpreads:
    """HY (HYG) vs IG (LQD) snapshot. Real CDX spreads are paywalled; ETF
    proxies are the public-data approximation."""
    df = yf.download(
        tickers=["HYG", "LQD"], period="10d", progress=False, threads=True, auto_adjust=False
    )
    out = CreditSpreads(
        as_of=date.today().isoformat(),
        hyg_close=None, lqd_close=None,
        hyg_5d_return_pct=None, lqd_5d_return_pct=None,
        hy_vs_ig_5d_diff=None,
    )
    if not isinstance(df.columns, pd.MultiIndex):
        return out
    try:
        hyg = df["Close"]["HYG"].dropna()
        lqd = df["Close"]["LQD"].dropna()
    except KeyError:
        return out
    if hyg.empty or lqd.empty:
        return out
    hyg_5d = _pct_return(hyg, 5)
    lqd_5d = _pct_return(lqd, 5)
    diff = None
    if hyg_5d is not None and lqd_5d is not None:
        diff = hyg_5d - lqd_5d
    return CreditSpreads(
        as_of=str(hyg.index[-1].date()),
        hyg_close=float(hyg.iloc[-1]),
        lqd_close=float(lqd.iloc[-1]),
        hyg_5d_return_pct=hyg_5d,
        lqd_5d_return_pct=lqd_5d,
        hy_vs_ig_5d_diff=diff,
    )


# ---------------------------------------------------------------------------
# get_dollar_index
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DollarIndex:
    as_of: str
    level: float | None
    return_5d_pct: float | None
    return_20d_pct: float | None


def get_dollar_index() -> DollarIndex:
    """DXY snapshot via yfinance's ICE DX-Y.NYB. UUP ETF is a fallback when
    the index ticker is flaky."""
    for ticker in ("DX-Y.NYB", "UUP"):
        try:
            df = yf.download(ticker, period="30d", progress=False, auto_adjust=False)
            if df.empty or "Close" not in df:
                continue
            close = df["Close"].dropna()
            if close.empty:
                continue
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            return DollarIndex(
                as_of=str(close.index[-1].date()),
                level=float(close.iloc[-1]),
                return_5d_pct=_pct_return(close, 5),
                return_20d_pct=_pct_return(close, 20),
            )
        except Exception as e:
            log.debug("Dollar index fetch via %s failed: %s", ticker, e)
            continue
    return DollarIndex(as_of=date.today().isoformat(), level=None, return_5d_pct=None, return_20d_pct=None)


# ---------------------------------------------------------------------------
# get_commodity_prices
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CommoditySnapshot:
    name: str
    ticker: str
    close: float | None
    return_5d_pct: float | None
    return_20d_pct: float | None


_COMMODITY_PROXIES = [
    ("Gold", "GLD"),
    ("Silver", "SLV"),
    ("Oil (USO)", "USO"),
    ("Natural gas (UNG)", "UNG"),
    ("Agriculture (DBA)", "DBA"),
    ("Base metals (DBB)", "DBB"),
]


def get_commodity_prices() -> list[CommoditySnapshot]:
    tickers = [t for _, t in _COMMODITY_PROXIES]
    df = yf.download(tickers, period="30d", progress=False, threads=True, auto_adjust=False)
    out: list[CommoditySnapshot] = []
    if not isinstance(df.columns, pd.MultiIndex):
        return out
    for name, ticker in _COMMODITY_PROXIES:
        try:
            close = df["Close"][ticker].dropna()
        except KeyError:
            out.append(CommoditySnapshot(name=name, ticker=ticker, close=None, return_5d_pct=None, return_20d_pct=None))
            continue
        if close.empty:
            out.append(CommoditySnapshot(name=name, ticker=ticker, close=None, return_5d_pct=None, return_20d_pct=None))
            continue
        out.append(
            CommoditySnapshot(
                name=name, ticker=ticker,
                close=float(close.iloc[-1]),
                return_5d_pct=_pct_return(close, 5),
                return_20d_pct=_pct_return(close, 20),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _pct_return(series: pd.Series, periods: int) -> float | None:
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    series = series.dropna()
    if len(series) <= periods:
        return None
    past = float(series.iloc[-(periods + 1)])
    now = float(series.iloc[-1])
    if past <= 0:
        return None
    return (now / past - 1.0) * 100.0
