"""Trading212 instrument metadata + ticker translation.

Our strategies internally use yfinance ticker syntax (`VOD.L`, `SAP.DE`,
`AAPL`). Trading212's API uses its own instrument format (`VOD_LON_EQ`,
`SAPd_EQ`, `AAPL_US_EQ`). This module fetches T212's full instrument list
once per process, caches it on disk for cross-run reuse, and exposes a
translator that maps yfinance → T212 ticker.

Cache location: state/t212_instruments.json. Auto-refreshes when older
than 7 days. Cache is local to the runner; the GH-Actions runner re-fetches
on first use each run, which is fine — the endpoint is fast and unmetered.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

from trading_bot.state.paths import STATE_ROOT
from trading_bot.t212_slot import T212Creds


log = logging.getLogger(__name__)

_CACHE_TTL_S = 7 * 24 * 3600  # 7 days

# Map yfinance ticker suffix → T212 exchange substring in the instrument
# ticker. T212 embeds the exchange in the middle of its ticker string,
# e.g. VOD_LON_EQ. The substring is what we match against.
_SUFFIX_TO_T212_EXCHANGE = {
    ".L": "LON",   # London Stock Exchange
    ".DE": "FRA",  # Xetra / Frankfurt (T212 uses lowercase d sometimes; FRA is the more reliable match)
    ".PA": "PAR",  # Euronext Paris
    ".AS": "AMS",  # Euronext Amsterdam
    ".BR": "BRU",  # Euronext Brussels
    ".LS": "LIS",  # Euronext Lisbon
    ".MI": "MIL",  # Borsa Italiana
    ".MC": "MAD",  # BME Madrid
    ".ST": "STO",  # Nasdaq Stockholm
    ".HE": "HEL",  # Nasdaq Helsinki
    ".CO": "CPH",  # Nasdaq Copenhagen
}


def _cache_path() -> Path:
    return STATE_ROOT / "t212_instruments.json"


def fetch_instruments(creds: T212Creds, *, force_refresh: bool = False) -> list[dict[str, Any]]:
    """Return the full T212 instrument list, hitting the cache when fresh."""
    cache = _cache_path()
    if not force_refresh and cache.exists():
        age = time.time() - cache.stat().st_mtime
        if age < _CACHE_TTL_S:
            try:
                return json.loads(cache.read_text())
            except json.JSONDecodeError:
                log.warning("Cached T212 instrument list is corrupted — refetching")

    response = requests.get(
        f"{creds.base_url}/equity/metadata/instruments",
        headers={"Authorization": creds.auth_header()},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    log.info("Fetched %d T212 instruments and cached at %s", len(data), cache)
    return data


def yfinance_to_t212(
    yf_ticker: str,
    instruments: list[dict[str, Any]],
) -> str | None:
    """Translate one yfinance ticker to its T212 instrument ticker.

    Matches by `shortName` (the human-readable ticker T212 uses internally)
    combined with an exchange substring derived from the yfinance suffix.
    Returns None if no match — caller should skip the trade and log.
    """
    if not yf_ticker:
        return None

    # Split into stem + exchange suffix
    if "." in yf_ticker:
        stem, _, suffix = yf_ticker.rpartition(".")
        suffix = "." + suffix
        exch_substr = _SUFFIX_TO_T212_EXCHANGE.get(suffix)
        if exch_substr is None:
            log.debug("No T212 exchange mapping for suffix %s", suffix)
            return None
    else:
        # No suffix → US listing
        stem = yf_ticker
        exch_substr = "US"

    # yfinance uses '-' for share classes (BRK-B), T212 uses dot or letter suffixes.
    # Try the literal first, then a few common rewrites.
    candidates_stem = [stem, stem.replace("-", "."), stem.replace("-", "")]

    for cand in candidates_stem:
        for inst in instruments:
            short = (inst.get("shortName") or "").upper()
            ticker = (inst.get("ticker") or "").upper()
            if short != cand.upper():
                continue
            if exch_substr in ticker:
                return inst["ticker"]
    return None


def build_translator(creds: T212Creds) -> "Translator":
    """Convenience constructor — fetches once, returns a stateful translator."""
    return Translator(fetch_instruments(creds))


class Translator:
    """Caches the instrument list and a per-call lookup memo."""

    def __init__(self, instruments: list[dict[str, Any]]):
        self.instruments = instruments
        self._memo: dict[str, str | None] = {}

    def translate(self, yf_ticker: str) -> str | None:
        if yf_ticker in self._memo:
            return self._memo[yf_ticker]
        result = yfinance_to_t212(yf_ticker, self.instruments)
        self._memo[yf_ticker] = result
        return result

    def get_instrument(self, t212_ticker: str) -> dict[str, Any] | None:
        for inst in self.instruments:
            if inst.get("ticker") == t212_ticker:
                return inst
        return None
