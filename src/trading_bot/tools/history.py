from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

import pandas as pd
import yfinance as yf


log = logging.getLogger(__name__)


# Yahoo Finance rate-limits aggressively on shared CI IPs. Chunking the
# universe into batches of this size with a small inter-batch sleep keeps
# us well under the rate cap. Tuned to give ~95%+ success on full FTSE350.
_BATCH_SIZE = 40
_BATCH_SLEEP_S = 1.5

# Process-lifetime cache for get_history. With per-strategy LLM calls
# fanned out in parallel (run_entry), every strategy in a region tends
# to request the same universe at the same end_date. The chunked
# yfinance download takes ~60s; caching means only the first caller
# pays it. Key: (sorted-tickers-tuple, lookback_days, end_date.iso).
_HISTORY_CACHE: dict = {}


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
    cache_key = (tuple(sorted(tickers)), lookback_days, end.isoformat())
    cached = _HISTORY_CACHE.get(cache_key)
    if cached is not None:
        # Return a shallow copy so a caller mutating one ticker's bar list
        # doesn't corrupt the cache. Bar objects themselves are frozen.
        return {k: list(v) for k, v in cached.items()}

    # Pad the lookback so we always cover requested trading days even across weekends/holidays
    period = max(lookback_days * 2 + 5, 10)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)

    # Chunk the request to avoid yfinance rate-limiting on shared CI IPs.
    # `threads=False` per-batch (yfinance multithreads aggressively when
    # tickers > 1; on Yahoo's shared rate-limit pool that blows up fast).
    out: dict[str, list[Bar]] = {}
    for i in range(0, len(tickers), _BATCH_SIZE):
        chunk = tickers[i : i + _BATCH_SIZE]
        df = yf.download(
            tickers=chunk,
            period=f"{period}d",
            end=end_ts,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        out.update(_flatten(df, chunk, lookback_days))
        # Sleep between chunks (not after the last one)
        if i + _BATCH_SIZE < len(tickers):
            time.sleep(_BATCH_SLEEP_S)

    log.info(
        "yfinance history: %d/%d tickers returned bars (lookback=%dd)",
        len(out), len(tickers), lookback_days,
    )
    _HISTORY_CACHE[cache_key] = out
    return {k: list(v) for k, v in out.items()}


def _flatten(df: pd.DataFrame, tickers: list[str], lookback_days: int) -> dict[str, list[Bar]]:
    """Build {ticker: [Bar]} from yfinance's mixed-shape output.

    yfinance returns different layouts depending on universe / number of
    tickers / version: single-ticker is flat columns; multi-ticker can be
    MultiIndex (ticker, price) or (price, ticker). We normalise to flat
    Open/High/Low/Close/Volume columns before iterating.

    Currency normalisation: LSE-listed tickers (`.L`) are quoted in pence,
    not pounds. We divide OHLC by 100 so downstream sizing math is in £.
    Volume stays in raw share count.
    """
    out: dict[str, list[Bar]] = {}
    is_single = len(tickers) == 1

    for ticker in tickers:
        sub = _slice_ticker(df, ticker, is_single)
        if sub is None or sub.empty:
            continue
        sub = sub.dropna().tail(lookback_days)
        # LSE quotes in pence, divide by 100 to get £.
        price_scale = 0.01 if ticker.endswith(".L") else 1.0
        bars: list[Bar] = []
        for ts in sub.index:
            try:
                bars.append(
                    Bar(
                        ticker=ticker,
                        bar_date=_to_date(ts),
                        open=float(sub.at[ts, "Open"]) * price_scale,
                        high=float(sub.at[ts, "High"]) * price_scale,
                        low=float(sub.at[ts, "Low"]) * price_scale,
                        close=float(sub.at[ts, "Close"]) * price_scale,
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
