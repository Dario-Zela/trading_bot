"""Stooq adapter — backup OHLCV source for tickers yfinance can't find.

Stooq offers per-symbol daily CSV downloads covering virtually every
developed-market exchange the bot trades on:

  https://stooq.com/q/d/l/?s=<stooq_ticker>&d1=<YYYYMMDD>&d2=<YYYYMMDD>&i=d&apikey=...

Stooq now requires a free API key (captcha-gated) on the CSV download
endpoint. Set `STOOQ_API_KEY` in the environment (locally + as a
GitHub Actions secret for CI). Without it, this module short-circuits
and the get_history fallback path silently degrades — callers just
see "no Stooq results" instead of failures.

Returns CSV (Date,Open,High,Low,Close,Volume). Used as a fallback in
`tools.history.get_history` when yfinance fails on a ticker — most
commonly because:

  * Ticker contains a slash (AGM/A → yfinance dies; Stooq has agm-a.us)
  * Recently-rebranded ticker (T212's legacy symbol diverges from yf's
    modern one and yf can't find either)
  * Smaller European listings yf doesn't cover but Stooq does

yfinance-ticker → Stooq-ticker convention:

  bare US ticker (AAPL)   → aapl.us
  BARC.L (LSE)            → barc.uk
  SAP.DE (Xetra)          → sap.de
  TTE.PA (Paris)          → tte.fr
  ASML.AS (Amsterdam)     → asml.nl
  ENI.MI (Milan)          → eni.it
  IBE.MC (Madrid)         → ibe.es
  NESN.SW (Swiss)         → nesn.ch
  NHY.OL (Oslo)           → nhy.no
  TELIA.ST (Stockholm)    → telia.se
  NOKIA.HE (Helsinki)     → nokia.fi

For US class-share names we normalise dashes back to nothing (Stooq:
brk-b.us; yfinance: BRK-B). Mapping is one-way: yfinance is the
canonical ticker representation throughout the bot.
"""
from __future__ import annotations

import csv
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Iterable

import requests


log = logging.getLogger(__name__)


_BASE_URL = "https://stooq.com/q/d/l/"
_USER_AGENT = "trading-bot/0.1 (research; +https://github.com/Dario-Zela/trading_bot)"
_TIMEOUT_S = 12
# Stooq's free tier throttles per-IP — burst >3 concurrent connections
# in a short window puts the IP into a SYN_SENT lockout for several
# hours. Override via env vars so workflows running on fresh CI IPs
# can dial up; local re-runs from a throttled IP stay polite.
#   STOOQ_MAX_PARALLEL    — concurrent workers (default 3)
#   STOOQ_REQUEST_SPACING — inter-request sleep per worker (default 0.3s)
_MAX_PARALLEL = int(os.environ.get("STOOQ_MAX_PARALLEL", "3"))
_REQUEST_SPACING_S = float(os.environ.get("STOOQ_REQUEST_SPACING", "0.3"))


# yfinance suffix → Stooq country code
_YF_SUFFIX_TO_STOOQ = {
    ".L":  "uk",
    ".DE": "de",
    ".F":  "de",       # Frankfurt — same as Xetra in Stooq's index
    ".PA": "fr",
    ".AS": "nl",
    ".MI": "it",
    ".MC": "es",
    ".SW": "ch",
    ".OL": "no",
    ".ST": "se",
    ".HE": "fi",
    ".CO": "dk",       # Copenhagen
    ".IL": "uk",       # International Order Book listings — LSE-domiciled
    ".TO": "ca",       # Toronto
    ".AX": "au",       # Australia
}


def yf_to_stooq(yf_ticker: str) -> str | None:
    """Convert a yfinance ticker to its Stooq equivalent. Returns None
    for tickers we can't map (unknown exchange suffix)."""
    if not yf_ticker:
        return None
    t = yf_ticker.strip()
    # US ticker — bare, no suffix. yfinance uses dash for class shares
    # (BRK-B); Stooq concatenates (brk-b.us is correct as-is actually).
    if "." not in t:
        return f"{t.lower()}.us"
    # Suffix lookup
    for suffix, country in _YF_SUFFIX_TO_STOOQ.items():
        if t.endswith(suffix):
            base = t[: -len(suffix)]
            return f"{base.lower()}.{country}"
    return None


def _parse_csv(text: str) -> list[tuple[date, float, float, float, float, int]]:
    """Parse Stooq's CSV response. Tolerates the BOM + header line + the
    occasional 'No data' or HTML-error response (returns empty list)."""
    if not text or text.startswith("<") or "No data" in text[:100]:
        return []
    rows: list[tuple[date, float, float, float, float, int]] = []
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if not header or not header[0].lower().lstrip("﻿").startswith("date"):
        return []
    for row in reader:
        if len(row) < 6:
            continue
        try:
            d = date.fromisoformat(row[0])
            o = float(row[1])
            h = float(row[2])
            lo = float(row[3])
            c = float(row[4])
            v = int(float(row[5])) if row[5] else 0
        except (ValueError, IndexError):
            continue
        rows.append((d, o, h, lo, c, v))
    return rows


def _get_api_key() -> str | None:
    return os.environ.get("STOOQ_API_KEY") or None


def fetch_history(
    yf_ticker: str,
    *,
    lookback_days: int = 70,
    end_date: date | None = None,
    session: requests.Session | None = None,
) -> list[dict] | None:
    """Fetch daily OHLCV from Stooq for one ticker. Returns list of
    {bar_date, open, high, low, close, volume} dicts or None on failure
    / no coverage. Caller is responsible for converting to the Bar
    dataclass downstream. Short-circuits to None if STOOQ_API_KEY isn't
    set — callers treat that as "no Stooq coverage available"."""
    api_key = _get_api_key()
    if not api_key:
        return None
    stooq_t = yf_to_stooq(yf_ticker)
    if not stooq_t:
        return None
    end = end_date or date.today()
    # Pad lookback to ride over weekends/holidays (same convention as yfinance path)
    start = end - timedelta(days=max(lookback_days * 2 + 5, 10))
    params = {
        "s": stooq_t,
        "d1": start.strftime("%Y%m%d"),
        "d2": end.strftime("%Y%m%d"),
        "i": "d",
        "apikey": api_key,
    }
    headers = {"User-Agent": _USER_AGENT}
    try:
        sess = session or requests
        resp = sess.get(_BASE_URL, params=params, headers=headers, timeout=_TIMEOUT_S)
        if resp.status_code != 200:
            return None
        rows = _parse_csv(resp.text)
    except (requests.RequestException, ValueError) as e:
        log.debug("stooq fetch failed for %s (stooq=%s): %s", yf_ticker, stooq_t, e)
        return None
    if not rows:
        return None
    # Only return bars within the requested window
    cutoff = end - timedelta(days=lookback_days * 2 + 5)
    out = [
        {"bar_date": d, "open": o, "high": h, "low": lo, "close": c, "volume": v}
        for (d, o, h, lo, c, v) in rows
        if d >= cutoff and d <= end
    ]
    return out or None


def fetch_history_bulk(
    yf_tickers: Iterable[str],
    *,
    lookback_days: int = 70,
    end_date: date | None = None,
    on_result: "callable | None" = None,
) -> dict[str, list[dict]]:
    """Parallel-fetch Stooq history for many tickers. Returns
    {yfinance_ticker: [bar_dict, ...]} for tickers Stooq has coverage on.
    Tickers without coverage are silently omitted.

    `on_result(ticker, bars)` fires for each successful fetch the moment
    it lands (before all parallel workers complete). Use this to
    persist results incrementally — a workflow timeout mid-fetch then
    preserves everything that completed beforehand instead of losing
    the whole batch.
    """
    tickers = [t for t in yf_tickers if t]
    if not tickers:
        return {}
    if not _get_api_key():
        log.warning("stooq bulk fetch skipped: STOOQ_API_KEY not set in environment")
        return {}

    out: dict[str, list[dict]] = {}
    sess = requests.Session()
    sess.headers["User-Agent"] = _USER_AGENT

    def _one(tkr: str):
        bars = fetch_history(tkr, lookback_days=lookback_days, end_date=end_date, session=sess)
        # Throttle between calls — Stooq isn't strict but burst-friendly only
        time.sleep(_REQUEST_SPACING_S)
        return tkr, bars

    log.info("stooq bulk fetch: %d tickers via %d-way parallel", len(tickers), _MAX_PARALLEL)
    with ThreadPoolExecutor(max_workers=_MAX_PARALLEL) as pool:
        futs = {pool.submit(_one, t): t for t in tickers}
        for fut in as_completed(futs):
            try:
                tkr, bars = fut.result()
                if bars:
                    out[tkr] = bars
                    if on_result is not None:
                        try:
                            on_result(tkr, bars)
                        except Exception as e:
                            log.warning("stooq on_result callback failed for %s: %s", tkr, e)
            except Exception as e:
                t = futs[fut]
                log.debug("stooq future failed for %s: %s", t, e)
    log.info("stooq bulk fetch: returned %d/%d tickers with bars", len(out), len(tickers))
    return out
