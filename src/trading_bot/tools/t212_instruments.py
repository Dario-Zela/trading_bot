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

# T212 uses two parallel ticker conventions:
#   1. Single-letter suffix appended to the shortName: `VODl_EQ` (LSE),
#      `ASMLa_EQ` (Amsterdam), `SAPd_EQ` (Xetra), `MCp_EQ` (Paris). This
#      covers the bulk of European exchanges.
#   2. Country code in the middle: `UCB_BE_EQ` (Brussels), `STN_US_EQ`
#      (NYSE). Used for Belgium, Portugal, Austria, Canada, US.
#
# Mapping derived empirically from a full T212 instrument dump (~17k rows).
_SUFFIX_TO_T212_LETTER = {
    ".L":  "l",   # London (LSE) — 4293 tickers
    ".DE": "d",   # Deutsche Börse / Xetra — 3259
    ".PA": "p",   # Euronext Paris — 493
    ".AS": "a",   # Euronext Amsterdam — 273
    ".MC": "e",   # BME Madrid (España) — 158
    ".MI": "m",   # Borsa Italiana (Milan) — 319
    ".ST": "s",   # Nasdaq Stockholm — 467
    ".HE": "h",   # Nasdaq Helsinki (unverified count)
    ".CO": "c",   # Nasdaq Copenhagen (unverified count)
}

_SUFFIX_TO_T212_COUNTRY = {
    ".BR": "BE",  # Euronext Brussels — 102
    ".LS": "PT",  # Euronext Lisbon — 28
    ".VI": "AT",  # Wiener Börse Vienna — 67
    ".TO": "CA",  # Toronto — 537
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

    Matching strategy:
    1. Find every T212 instrument whose `shortName` matches the yfinance
       stem (case-insensitive, with hyphen/dot share-class rewrites).
    2. Among those, pick the one whose ticker matches the exchange
       convention implied by the yfinance suffix:
       - `XYZl_EQ` for `.L`, `XYZd_EQ` for `.DE`, etc. (single-letter form)
       - `XYZ_BE_EQ` for `.BR`, `XYZ_US_EQ` for no suffix (country form)
    Returns None if no match — caller should skip the trade and log.
    """
    if not yf_ticker:
        return None

    # Determine the expected T212 ticker pattern from the yfinance suffix.
    t212_letter: str | None = None
    t212_country: str | None = None
    if "." in yf_ticker:
        stem, _, suffix = yf_ticker.rpartition(".")
        suffix = "." + suffix
        t212_letter = _SUFFIX_TO_T212_LETTER.get(suffix)
        t212_country = _SUFFIX_TO_T212_COUNTRY.get(suffix)
        if t212_letter is None and t212_country is None:
            log.debug("No T212 exchange mapping for suffix %s", suffix)
            return None
    else:
        stem = yf_ticker
        t212_country = "US"

    # yfinance uses '-' for US share classes (BRK-B); T212 sometimes uses
    # '.' or no separator. Try a few common rewrites of the stem.
    candidates_stem = [stem, stem.replace("-", "."), stem.replace("-", "")]
    candidates_stem_upper = [c.upper() for c in candidates_stem]

    for inst in instruments:
        short = (inst.get("shortName") or "").upper()
        if short not in candidates_stem_upper:
            continue
        ticker = inst.get("ticker") or ""
        if not ticker.endswith("_EQ"):
            continue
        stem_in_ticker = ticker[:-3]  # strip "_EQ"

        if t212_letter is not None:
            # Single-letter convention: VODl, ASMLa, SAPd, etc. We can't
            # require stem == f"{short}{letter}" because T212 keeps legacy
            # ticker codes after company renames (e.g., shortName=HBR but
            # ticker stays PMOl_EQ from Premier Oil). The exchange letter
            # is lowercase in T212's convention, and shortNames are all
            # uppercase, so a case-sensitive endswith disambiguates safely:
            # `AMSEL` (hypothetical ticker) ends with "L" not "l".
            if stem_in_ticker.endswith(t212_letter):
                return ticker
        if t212_country is not None:
            # Country-code convention: VOD_US, UCB_BE.
            if stem_in_ticker.endswith(f"_{t212_country}"):
                return ticker
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
