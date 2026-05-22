from __future__ import annotations

import json
import logging
import os
import tempfile
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
    *,
    max_grace_days: int = 3,
) -> dict[str, list[Bar]]:
    """Fetch daily OHLCV for the given tickers over the lookback window.

    Read path:
      1. In-process cache (sorted-tuple keyed) — handles repeated calls
         from parallel strategies fanning out for the same universe.
      2. SQLite OHLCV store (state/ohlcv.db) — local cache. Served
         from cache when its latest bar is within `max_grace_days` of
         `end_date`.
      3. yfinance batched download — only invoked for tickers MISSING
         from the local store, or whose stored coverage doesn't reach
         `end_date` within the grace window. Results written back to
         the store per-batch (so partial progress survives a kill).

    `max_grace_days` controls cache-vs-fetch tradeoff:
      - 3 (default) — tolerant; strategies' mid-day get_history calls
        accept up-to-3-day-old bars rather than slow-fetch each time
      - 0 — strict; ohlcv-daily-update uses this to FORCE a fetch for
        any ticker whose latest stored bar isn't end_date itself.
        Needed for the cache-freshness pre-step before morning entry,
        otherwise grace serves stale data and the cache drifts.

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
    # doesn't have OR whose latest stored bar is more than `max_grace_days`
    # behind `end`. Default 3-day grace accounts for weekends/holidays;
    # callers can pass 0 to force a refresh whenever cache isn't at end_date.
    grace = pd.Timedelta(days=max(0, max_grace_days))
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
    # Write-back to the SQLite store happens PER BATCH, not at the end —
    # so a workflow timeout / cancel mid-loop preserves whatever's
    # already been fetched. The end-of-function "all-at-once" write was
    # what caused the warmup workflow's 90-min timeout to lose ALL
    # progress despite ~8k tickers being downloaded.
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    fresh_from_yf: dict[str, list[Bar]] = {}
    _write_bars = None
    if StoredBar is not None:
        from trading_bot.tools.ohlcv_store import write_bars as _write_bars  # noqa: F401
    n_batches = (len(needs_fetch) + _BATCH_SIZE - 1) // _BATCH_SIZE
    # Heartbeat every PROGRESS_EVERY batches so the log streams progress
    # rather than going silent for 30-45 min. Stdout flushes after each
    # log call, so the next gh-log fetch sees the line.
    PROGRESS_EVERY = max(1, n_batches // 20)   # ~20 heartbeats per run
    _start_t = time.time()
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
        batch_results = _flatten(df, chunk, lookback_days)
        fresh_from_yf.update(batch_results)
        # Per-batch write-back. Any later failure / interrupt leaves
        # everything fetched so far safely on disk.
        if _write_bars is not None and batch_results:
            try:
                rows = [
                    StoredBar(ticker=tkr, bar_date=b.bar_date, open=b.open,
                              high=b.high, low=b.low, close=b.close, volume=b.volume)
                    for tkr, bars in batch_results.items()
                    for b in bars
                ]
                _write_bars(rows)
            except Exception as e:
                log.warning("OHLCV store write-back (batch %d) failed: %s", i // _BATCH_SIZE, e)
        # Progress heartbeat — every PROGRESS_EVERY batches log
        # cumulative state. Lets the operator see the loop is alive
        # (no silent 30-minute stretches).
        batch_idx = (i // _BATCH_SIZE) + 1
        if batch_idx % PROGRESS_EVERY == 0 or batch_idx == n_batches:
            elapsed = time.time() - _start_t
            log.info(
                "yfinance progress: batch %d/%d (%d tickers requested, %d fetched so far, %.0fs elapsed)",
                batch_idx, n_batches, min(i + _BATCH_SIZE, len(needs_fetch)),
                len(fresh_from_yf), elapsed,
            )
        # Sleep between chunks (not after the last one)
        if i + _BATCH_SIZE < len(needs_fetch):
            time.sleep(_BATCH_SLEEP_S)

    if fresh_from_yf:
        log.info(
            "yfinance history: %d/%d tickers returned bars (lookback=%dd)",
            len(fresh_from_yf), len(needs_fetch), lookback_days,
        )
        out.update(fresh_from_yf)

    # 4. Stooq fallback for whatever yfinance still missed. Short-
    #    circuits cleanly if STOOQ_API_KEY isn't set in the env. Tickers
    #    yfinance failed on (rebranded epics, slash-in-name lines like
    #    AGM/A, smaller European listings yfinance doesn't index) often
    #    succeed here. Same write-back-to-store contract.
    stooq_misses = [t for t in needs_fetch if t not in out]
    if stooq_misses:
        # Per-ticker on_result callback so each Stooq response writes to
        # the local store the moment it lands — preserves progress even
        # if a workflow timeout fires mid-fetch.
        from trading_bot.tools.ohlcv_store import write_bars as _wb_stooq
        def _persist_one(tkr: str, bars: list[dict]) -> None:
            if not bars or StoredBar is None:
                return
            try:
                _wb_stooq([
                    StoredBar(ticker=tkr, bar_date=b["bar_date"],
                              open=b["open"], high=b["high"], low=b["low"],
                              close=b["close"], volume=b["volume"])
                    for b in bars
                ])
            except Exception as e:
                log.debug("Stooq per-ticker write-back failed for %s: %s", tkr, e)
        try:
            from trading_bot.tools.stooq import fetch_history_bulk
            stooq_results = fetch_history_bulk(
                stooq_misses, lookback_days=lookback_days, end_date=end,
                on_result=_persist_one,
            )
        except Exception as e:
            log.warning("Stooq fallback failed (non-fatal): %s", e)
            stooq_results = {}
        if stooq_results:
            log.info(
                "stooq fallback: recovered %d/%d tickers yfinance missed",
                len(stooq_results), len(stooq_misses),
            )
            # Per-ticker write-back already happened via on_result above —
            # just hydrate the in-memory `out` dict for the caller.
            for tkr, bars in stooq_results.items():
                out[tkr] = [
                    Bar(ticker=tkr, bar_date=b["bar_date"], open=b["open"],
                        high=b["high"], low=b["low"], close=b["close"],
                        volume=b["volume"])
                    for b in bars
                ]

    _HISTORY_CACHE[cache_key] = out
    return {k: list(v) for k, v in out.items()}


# --- LSE quote-currency detection ------------------------------------
# yfinance quotes ordinary LSE shares in pence ('GBp'), which must be
# divided by 100 to reach pounds. But many LSE-listed ETF / bond / UCITS
# lines quote in whole 'GBP', 'USD', or 'EUR' and must NOT be divided.
# Keying the /100 on the '.L' suffix alone silently mangled those lines
# (IGLS.L and HYGG.L are GBP, IHYG.L is EUR, VWRL.L is GBP, CSPX.L is USD,
# ...) by 100x — and the bad price then mis-sized the position. The trade
# currency from fees.yf_ticker_classify can't disambiguate this: it returns
# "GBP" for both pence-quoted SHEL.L and pound-quoted IGLS.L. So we read the
# real quote currency from yfinance once per ticker and cache it both in
# process and in a sidecar JSON (the universe is bounded, so the file warms
# once and then costs nothing). On any lookup failure we assume pence — the
# correct default for the overwhelming majority of the LSE universe — so a
# transient yfinance hiccup can't suddenly 100x an ordinary share.
from trading_bot.state.paths import STATE_ROOT  # noqa: E402

_LSE_CCY_PATH = STATE_ROOT / "lse_quote_ccy.json"
_LSE_CCY: dict[str, str] | None = None


def _load_lse_ccy() -> dict[str, str]:
    global _LSE_CCY
    if _LSE_CCY is None:
        try:
            _LSE_CCY = json.loads(_LSE_CCY_PATH.read_text())
        except (OSError, ValueError):
            _LSE_CCY = {}
    return _LSE_CCY


def _save_lse_ccy(cache: dict[str, str]) -> None:
    # Atomic, race-tolerant: a unique temp file per writer + os.replace means
    # concurrent select_picks threads can't corrupt the destination. A lost
    # write just costs a re-fetch — it's a cache, not source of truth.
    try:
        STATE_ROOT.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(STATE_ROOT), suffix=".lse_ccy.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(cache, f, sort_keys=True)
            os.replace(tmp, _LSE_CCY_PATH)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    except OSError as e:
        log.debug("could not persist LSE currency cache: %s", e)


def _lse_quote_is_pence(ticker: str) -> bool:
    """True if yfinance quotes this LSE ticker in pence (GBp), so its OHLC
    must be divided by 100 to reach pounds. Ordinary shares are pence;
    GBP/USD/EUR-quoted ETF & bond lines are not. Result cached per ticker."""
    cache = _load_lse_ccy()
    ccy = cache.get(ticker)
    if ccy is None:
        resolved: str | None = None
        try:
            tk = yf.Ticker(ticker)
            c = getattr(tk.fast_info, "currency", None)
            if not c:
                # fast_info omits currency for many ETP / leveraged-tracker
                # lines (Leverage Shares, Ossiam, etc.); the heavier .info
                # carries it. Worth the extra call — these are exactly the
                # non-pence USD/EUR lines we must NOT divide.
                c = (tk.info or {}).get("currency")
            if c:
                resolved = str(c)
        except Exception as e:
            log.debug("LSE currency lookup failed for %s; assuming pence: %s", ticker, e)
        # Safe default: most of the LSE universe is pence, so a total lookup
        # failure keeps the historical /100 rather than risking a 100x blow-up.
        ccy = resolved or "GBp"
        cache[ticker] = ccy
        _save_lse_ccy(cache)
    return ccy == "GBp"


def _flatten(df: pd.DataFrame, tickers: list[str], lookback_days: int) -> dict[str, list[Bar]]:
    """Build {ticker: [Bar]} from yfinance's mixed-shape output.

    yfinance returns different layouts depending on universe / number of
    tickers / version: single-ticker is flat columns; multi-ticker can be
    MultiIndex (ticker, price) or (price, ticker). We normalise to flat
    Open/High/Low/Close/Volume columns before iterating.

    Currency normalisation: ordinary LSE shares (`.L`) are quoted by
    yfinance in pence, so we divide OHLC by 100 to get £. But LSE-listed
    ETF/bond/UCITS lines often quote in whole GBP/USD/EUR and must NOT be
    divided — `_lse_quote_is_pence` checks the real quote currency rather
    than assuming every `.L` is pence. Volume stays in raw share count.
    """
    out: dict[str, list[Bar]] = {}
    is_single = len(tickers) == 1

    for ticker in tickers:
        sub = _slice_ticker(df, ticker, is_single)
        if sub is None or sub.empty:
            continue
        sub = sub.dropna().tail(lookback_days)
        # Pence-correction only for genuinely pence-quoted (.L) lines; GBP/
        # USD/EUR-quoted ETF & bond lines on LSE must not be divided.
        price_scale = 0.01 if (ticker.endswith(".L") and _lse_quote_is_pence(ticker)) else 1.0
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
        # Flatten to the OHLCV field name. Don't assume it's the innermost
        # level: a single-ticker download can come back as (Price, Ticker)
        # *or* (Ticker, Price), so picking c[-1] blindly yields ticker-named
        # columns and every "Open"/"Close" lookup then KeyErrors (silently
        # dropping the ticker). Pick whichever level holds the price field.
        _PRICE_FIELDS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        sub = sub.copy()
        sub.columns = [
            next((p for p in c if p in _PRICE_FIELDS), c[-1]) if isinstance(c, tuple) else c
            for c in sub.columns
        ]
    return sub


def _to_date(ts) -> date:
    if isinstance(ts, pd.Timestamp):
        return ts.date()
    if isinstance(ts, datetime):
        return ts.date()
    return date.fromisoformat(str(ts)[:10])
