"""FX rates: convert non-GBP currencies to GBP for sizing and P&L.

T212 quotes instruments in their listing exchange's currency — EUR for
Xetra/Amsterdam/Paris, USD for US listings, GBX (pence) for LSE. Our
ledger and capital allocations are all in GBP. Without an FX layer:
  - Sizing math under-allocates: £3000 / €30 = 100 shares instead of ~118
  - Recorded fill prices in foreign currency get summed as £ — wildly wrong P&L

This module fetches end-of-day spot rates from yfinance (FX pairs like
EURGBP=X) and caches them in-process per run. For daily-cron strategies
EOD rates are precise enough; intraday FX drift on the bot's exposure
sizes (£100s per position) is negligible vs the trading edge we're after.

GBX is treated as 1/100 GBP (pence), not a currency lookup.
"""
from __future__ import annotations

import logging
from datetime import date
from functools import lru_cache

import yfinance as yf


log = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def _fetch_rate(pair: str) -> float | None:
    """Fetch the latest close for a yfinance FX pair like EURGBP=X."""
    try:
        df = yf.download(
            pair, period="5d", progress=False, threads=False, auto_adjust=False
        )
        if df is None or df.empty:
            return None
        # yfinance returns a multi-indexed DataFrame for single-ticker downloads
        # in newer versions; flatten if needed.
        close_col = None
        if "Close" in df.columns:
            close_col = df["Close"]
        elif hasattr(df.columns, "get_level_values") and "Close" in df.columns.get_level_values(0):
            close_col = df["Close"]
        if close_col is None:
            return None
        # Drop nans, take the most recent value
        close_col = close_col.dropna()
        if close_col.empty:
            return None
        v = close_col.iloc[-1]
        # If still a Series (multi-ticker), take its scalar value
        if hasattr(v, "item"):
            v = v.item() if v.size == 1 else float(v.iloc[0])
        return float(v)
    except Exception as e:
        log.warning("FX fetch failed for %s: %s", pair, e)
        return None


def to_gbp_multiplier(currency: str) -> float | None:
    """Return the multiplier that converts an amount in `currency` to GBP.

    - GBP / "" → 1.0 (no-op)
    - GBX → 0.01 (pence → pounds, no API call)
    - EUR/USD/etc → spot rate from yfinance (e.g., EURGBP=X ≈ 0.85)

    Returns None if the rate can't be fetched. Callers should log + fall
    back to native-currency recording rather than inventing a number.
    """
    if not currency:
        return 1.0
    cur = currency.upper()
    if cur == "GBP":
        return 1.0
    if cur == "GBX":
        return 0.01
    # Foreign currency: fetch from yfinance
    pair = f"{cur}GBP=X"
    rate = _fetch_rate(pair)
    if rate is None or rate <= 0:
        return None
    return rate


def convert_to_gbp(amount: float, currency: str) -> float | None:
    """Convert an amount expressed in `currency` to GBP. Returns None if
    the FX rate is unavailable."""
    mult = to_gbp_multiplier(currency)
    if mult is None:
        return None
    return amount * mult


def convert_from_gbp(amount_gbp: float, target_currency: str) -> float | None:
    """Convert a GBP amount to the target currency (used for sizing
    allocations against foreign-currency instruments)."""
    mult = to_gbp_multiplier(target_currency)
    if mult is None or mult == 0:
        return None
    return amount_gbp / mult
