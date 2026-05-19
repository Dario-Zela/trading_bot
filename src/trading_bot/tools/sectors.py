"""Phase 9D — cached sector lookup per ticker.

yfinance's `Ticker(t).info["sector"]` is slow + rate-limited, so we
cache results in `state/ticker_sectors.json`. The dashboard build
calls `bulk_lookup(tickers)` once per render; misses are fetched
lazily.

The mapping covers cases that yfinance returns; non-equities (ETFs,
crypto, FX) get `None`. Callers should treat `None` as "unknown"
rather than mapping to a placeholder bucket.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from trading_bot.state.paths import STATE_ROOT


log = logging.getLogger(__name__)


def _cache_path() -> Path:
    p = STATE_ROOT / "ticker_sectors.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_cache() -> dict[str, str | None]:
    p = _cache_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def _save_cache(cache: dict[str, str | None]) -> None:
    _cache_path().write_text(json.dumps(cache, indent=2, sort_keys=True))


def bulk_lookup(tickers: list[str]) -> dict[str, str | None]:
    """Return {ticker: sector_or_None} for every ticker. Uses the cache
    + fetches misses one at a time (with a small delay between calls
    to be polite to yfinance)."""
    cache = _load_cache()
    out: dict[str, str | None] = {}
    misses: list[str] = []
    for t in tickers:
        if t in cache:
            out[t] = cache[t]
        else:
            misses.append(t)

    if misses:
        log.info("sectors: %d cache misses, fetching", len(misses))
        try:
            import yfinance as yf
        except ImportError:
            log.warning("yfinance unavailable — sector lookup degraded")
            return out
        for t in misses:
            sector = None
            try:
                info = yf.Ticker(t).info or {}
                sector = info.get("sector") or None
            except Exception as e:
                log.debug("sector fetch %s failed: %s", t, e)
            cache[t] = sector
            out[t] = sector
            time.sleep(0.15)
        _save_cache(cache)

    return out


def get_sector(ticker: str) -> str | None:
    """One-shot lookup; goes through bulk_lookup so caching applies."""
    return bulk_lookup([ticker]).get(ticker)
