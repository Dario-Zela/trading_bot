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

    Read path:
      1. In-process cache (sorted-tuple keyed) — handles repeated calls
         from parallel strategies fanning out for the same universe.
      2. SQLite OHLCV store (state/ohlcv.db) — local cache that grows
         organically as the bot fetches. Skips yfinance entirely for
         tickers whose full window we already have.
      3. yfinance batched download — only invoked for tickers MISSING
         from the local store, or whose stored coverage doesn't reach
         the requested end_date. Results written back to the store on
         the way through.

    Returns a dict {ticker: [Bar, ...]} in chronological order,
    most-recent last. Tickers with no data anywhere are omitted.
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

    # Pad the lookback so we always cover requested trading days even
    # across weekends/holidays. The store read uses the same padded
    # window so coverage matches what yfinance would have returned.
    period = max(lookback_days * 2 + 5, 10)
    start_date = end - pd.Timedelta(days=period).to_pytimedelta()

    out: dict[str, list[Bar]] = {}
    needs_fetch: list[str] = []

    # 2. Local SQLite store. Read in bulk for all requested tickers.
    try:
        from trading_bot.tools.ohlcv_store import read_bars_bulk, write_bars, StoredBar
        store_hits = read_bars_bulk(tickers, start_date, end)
    except Exception as e:
        log.warning("OHLCV store read failed (falling through to yfinance): %s", e)
        store_hits = {}
        StoredBar = None       # type: ignore

    # Decide which tickers to fetch from yfinance: any that the store
    # doesn't have OR whose latest stored bar is more than 1 trading day
    # before `end`. The +3-day grace accounts for weekends/holidays.
    grace = pd.Timedelta(days=3)
    for tkr in tickers:
        rows = store_hits.get(tkr, [])
        if not rows:
            needs_fetch.append(tkr)
            continue
        latest = rows[-1].bar_date
        if pd.Timestamp(end) - pd.Timestamp(latest) > grace:
            needs_fetch.append(tkr)
            continue
        # Store coverage is good enough — promote to output as Bar objects.
        out[tkr] = [
            Bar(ticker=tkr, bar_date=r.bar_date, open=r.open, high=r.high,
                low=r.low, close=r.close, volume=r.volume)
            for r in rows[-lookback_days:]
        ]

    log.info(
        "history cache: %d/%d tickers served from local store, %d need yfinance",
        len(out), len(tickers), len(needs_fetch),
    )

    # 3. yfinance fallback for misses. Chunk batched to avoid rate limits.
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    fresh_from_yf: dict[str, list[Bar]] = {}
    for i in range(0, len(needs_fetch), _BATCH_SIZE):
        chunk = needs_fetch[i : i + _BATCH_SIZE]
        df = yf.download(
            tickers=chunk,
            period=f"{period}d",
            end=end_ts,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        fresh_from_yf.update(_flatten(df, chunk, lookback_days))
        # Sleep between chunks (not after the last one)
        if i + _BATCH_SIZE < len(needs_fetch):
            time.sleep(_BATCH_SLEEP_S)

    if fresh_from_yf:
        log.info(
            "yfinance history: %d/%d tickers returned bars (lookback=%dd)",
            len(fresh_from_yf), len(needs_fetch), lookback_days,
        )
        out.update(fresh_from_yf)
        # Write back to the local store so subsequent runs hit the cache.
        if StoredBar is not None:
            try:
                stored_rows = [
                    StoredBar(ticker=tkr, bar_date=b.bar_date, open=b.open,
                              high=b.high, low=b.low, close=b.close, volume=b.volume)
                    for tkr, bars in fresh_from_yf.items()
                    for b in bars
                ]
                from trading_bot.tools.ohlcv_store import write_bars as _wb
                n = _wb(stored_rows)
                log.debug("OHLCV store write-back: %d bars cached", n)
            except Exception as e:
                log.warning("OHLCV store write-back failed (non-fatal): %s", e)

    # 4. Stooq fallback for whatever yfinance still missed. Short-
    #    circuits cleanly if STOOQ_API_KEY isn't set in the env. Tickers
    #    yfinance failed on (rebranded epics, slash-in-name lines like
    #    AGM/A, smaller European listings yfinance doesn't index) often
    #    succeed here. Same write-back-to-store contract.
    stooq_misses = [t for t in needs_fetch if t not in out]
    if stooq_misses:
        try:
            from trading_bot.tools.stooq import fetch_history_bulk
            stooq_results = fetch_history_bulk(
                stooq_misses, lookback_days=lookback_days, end_date=end,
            )
        except Exception as e:
            log.warning("Stooq fallback failed (non-fatal): %s", e)
            stooq_results = {}
        if stooq_results:
            log.info(
                "stooq fallback: recovered %d/%d tickers yfinance missed",
                len(stooq_results), len(stooq_misses),
            )
            from trading_bot.tools.ohlcv_store import write_bars as _wb
            new_rows: list[StoredBar] = [] if StoredBar is not None else None
            for tkr, bars in stooq_results.items():
                converted = [
                    Bar(ticker=tkr, bar_date=b["bar_date"], open=b["open"],
                        high=b["high"], low=b["low"], close=b["close"],
                        volume=b["volume"])
                    for b in bars
                ]
                out[tkr] = converted
                if StoredBar is not None:
                    new_rows.extend(
                        StoredBar(ticker=tkr, bar_date=b.bar_date, open=b.open,
                                  high=b.high, low=b.low, close=b.close,
                                  volume=b.volume)
                        for b in converted
                    )
            if StoredBar is not None and new_rows:
                try:
                    _wb(new_rows)
                except Exception as e:
                    log.warning("OHLCV store write-back (stooq) failed: %s", e)

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
