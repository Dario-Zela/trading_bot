"""Technicals tool — RSI, MACD, ATR, moving averages, volume profile, recent returns.

Pure Python from yfinance daily OHLCV. Intentionally library-light (no pandas-ta /
TA-Lib dependency) so the runtime install in CI stays small and predictable. The
formulas are textbook; if anything subtle changes (e.g., Wilder's smoothing vs
EMA for RSI), it's localised to one function and easy to audit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd

from trading_bot.tools.history import get_history

log = logging.getLogger(__name__)


@dataclass
class Technicals:
    ticker: str
    as_of: str  # ISO date of the most recent bar used
    close: float
    rsi_14: float | None
    macd_line: float | None
    macd_signal: float | None
    macd_histogram: float | None
    atr_14: float | None
    sma_20: float | None
    sma_50: float | None
    above_sma_20: bool | None
    above_sma_50: bool | None
    volume: int
    avg_volume_20: float | None
    volume_ratio: float | None
    return_5d_pct: float | None
    return_20d_pct: float | None
    # Phase 11F — relative strength vs the benchmark (SPY for US, IWM
    # for small-cap US, XLE/XLK/etc when known sector). Positive means
    # the ticker outperformed the benchmark by this many percent over
    # the window. None when the benchmark history couldn't be fetched.
    rel_strength_5d: float | None = None
    rel_strength_20d: float | None = None
    benchmark: str | None = None


def _pick_benchmark(ticker: str) -> str:
    """Map a ticker to its closest benchmark for relative-strength
    calculation. Keeps things simple: SPY for US shares (the LLM
    knows it shouldn't expect sector-level granularity), FTSE all-share
    proxy for UK names. Sector-ETFs as benchmarks would be nicer
    but they require a per-ticker sector lookup."""
    t = (ticker or "").upper()
    if t.endswith(".L"):
        return "ISF.L"        # FTSE 100 ETF on LSE — broad UK proxy
    if any(s in t for s in (".DE", ".PA", ".AS", ".MI", ".MC")):
        return "EXSA.DE"      # STOXX Europe 600 — broad EU proxy
    return "SPY"


def get_technicals(
    tickers: str | Iterable[str],
    end_date: date | None = None,
) -> dict[str, Technicals]:
    """Compute a fixed set of technical indicators for each ticker.

    Returns {ticker: Technicals}. Tickers with insufficient history (less than
    ~55 trading days for the 50-day SMA + Wilder warmup) are omitted.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = list(tickers)
    if not tickers:
        return {}

    # 60 trading days covers 50-day SMA + warmup for RSI/MACD/ATR
    history = get_history(tickers, lookback_days=70, end_date=end_date)

    # Phase 11F — fetch benchmark history once per benchmark + reuse.
    benchmarks_needed = sorted({_pick_benchmark(t) for t in tickers})
    bench_history: dict[str, list] = {}
    if benchmarks_needed:
        try:
            bench_history = get_history(benchmarks_needed, lookback_days=70, end_date=end_date)
        except Exception as e:
            log.warning("rel-strength benchmark fetch failed: %s", e)

    out: dict[str, Technicals] = {}
    for ticker, bars in history.items():
        if len(bars) < 30:
            continue  # Not enough data for meaningful indicators
        df = _bars_to_df(bars)
        tech = _compute(ticker, df)
        # Compute rel-strength against the ticker's benchmark
        bench = _pick_benchmark(ticker)
        bench_bars = bench_history.get(bench)
        rel5, rel20 = _rel_strength(bars, bench_bars)
        tech.rel_strength_5d = rel5
        tech.rel_strength_20d = rel20
        tech.benchmark = bench
        out[ticker] = tech
    return out


def _rel_strength(ticker_bars, bench_bars) -> tuple[float | None, float | None]:
    """Return (5d_rel, 20d_rel) — ticker return minus benchmark return
    over each window. None when benchmark history is missing or too short."""
    if not ticker_bars or not bench_bars or len(bench_bars) < 21 or len(ticker_bars) < 21:
        return None, None
    def _ret(bars, periods: int) -> float | None:
        if len(bars) < periods + 1:
            return None
        past = float(bars[-(periods + 1)].close)
        if past <= 0:
            return None
        return (float(bars[-1].close) / past - 1.0) * 100.0
    t5, t20 = _ret(ticker_bars, 5), _ret(ticker_bars, 20)
    b5, b20 = _ret(bench_bars, 5), _ret(bench_bars, 20)
    rel5 = round(t5 - b5, 2) if (t5 is not None and b5 is not None) else None
    rel20 = round(t20 - b20, 2) if (t20 is not None and b20 is not None) else None
    return rel5, rel20


def _bars_to_df(bars) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": [b.bar_date for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        }
    )
    return df


def _compute(ticker: str, df: pd.DataFrame) -> Technicals:
    last = df.iloc[-1]
    close = float(last["close"])

    rsi = _rsi(df["close"], 14)
    macd_line, macd_signal, macd_hist = _macd(df["close"], 12, 26, 9)
    atr = _atr(df, 14)
    sma_20 = _safe_last(df["close"].rolling(20).mean())
    sma_50 = _safe_last(df["close"].rolling(50).mean())
    avg_vol_20 = _safe_last(df["volume"].rolling(20).mean())
    volume = int(last["volume"])

    def _ret(periods: int) -> float | None:
        if len(df) < periods + 1:
            return None
        past = float(df.iloc[-(periods + 1)]["close"])
        if past <= 0:
            return None
        return (close / past - 1.0) * 100.0

    return Technicals(
        ticker=ticker,
        as_of=str(last["date"]),
        close=close,
        rsi_14=rsi,
        macd_line=macd_line,
        macd_signal=macd_signal,
        macd_histogram=macd_hist,
        atr_14=atr,
        sma_20=sma_20,
        sma_50=sma_50,
        above_sma_20=(close > sma_20) if sma_20 is not None else None,
        above_sma_50=(close > sma_50) if sma_50 is not None else None,
        volume=volume,
        avg_volume_20=avg_vol_20,
        volume_ratio=(volume / avg_vol_20) if avg_vol_20 and avg_vol_20 > 0 else None,
        return_5d_pct=_ret(5),
        return_20d_pct=_ret(20),
    )


def _safe_last(series: pd.Series) -> float | None:
    v = series.iloc[-1]
    if pd.isna(v):
        return None
    return float(v)


def _rsi(closes: pd.Series, period: int = 14) -> float | None:
    """Wilder's RSI using EMA smoothing of gains/losses."""
    if len(closes) < period + 1:
        return None
    delta = closes.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing: alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    last_avg_gain = float(avg_gain.iloc[-1])
    last_avg_loss = float(avg_loss.iloc[-1])
    if last_avg_loss == 0:
        return 100.0
    rs = last_avg_gain / last_avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(
    closes: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9
) -> tuple[float | None, float | None, float | None]:
    if len(closes) < slow + signal_period:
        return None, None, None
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return (
        float(macd_line.iloc[-1]),
        float(signal_line.iloc[-1]),
        float(histogram.iloc[-1]),
    )


def _atr(df: pd.DataFrame, period: int = 14) -> float | None:
    if len(df) < period + 1:
        return None
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    last = atr.iloc[-1]
    if pd.isna(last):
        return None
    return float(last)
