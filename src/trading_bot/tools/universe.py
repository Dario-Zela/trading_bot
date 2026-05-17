from __future__ import annotations

from io import StringIO

import pandas as pd
import requests


_SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_FTSE100_WIKI_URL = "https://en.wikipedia.org/wiki/FTSE_100_Index"
_USER_AGENT = "trading-bot/0.1 (research; +https://github.com/Dario-Zela/trading_bot)"

# Static lists — these change rarely, easier to keep in-tree than fetch.
# Update if SPDR rebalances sector mappings (very infrequent).
_US_ETFS_SECTOR = [
    "XLF",   # Financials
    "XLE",   # Energy
    "XLK",   # Technology
    "XLV",   # Health Care
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLI",   # Industrials
    "XLU",   # Utilities
    "XLB",   # Materials
    "XLRE",  # Real Estate
    "XLC",   # Communication Services
]

_US_ETFS_BOND = ["TLT", "IEF", "SHY", "HYG", "LQD"]
_US_ETFS_COMMODITY = ["GLD", "SLV", "USO", "DBA", "DBB"]


def get_universe(universe_id: str) -> list[str]:
    """Return a ticker list for the named universe.

    Supported:
    - US equities: 'sp500'
    - US ETFs: 'us_etfs_sector', 'us_etfs_bond', 'us_etfs_commodity'
    - UK equities: 'ftse100'
    """
    if universe_id == "sp500":
        return _fetch_sp500()
    if universe_id == "ftse100":
        return _fetch_ftse100()
    if universe_id == "us_etfs_sector":
        return list(_US_ETFS_SECTOR)
    if universe_id == "us_etfs_bond":
        return list(_US_ETFS_BOND)
    if universe_id == "us_etfs_commodity":
        return list(_US_ETFS_COMMODITY)
    raise ValueError(f"Unknown universe: {universe_id}")


def _fetch_sp500() -> list[str]:
    # Fetch via requests so we use certifi's CA bundle (more reliable than
    # pandas' internal urllib call, especially on macOS python.org builds).
    response = requests.get(_SP500_WIKI_URL, headers={"User-Agent": _USER_AGENT}, timeout=15)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    df = tables[0]
    # Wikipedia uses dot-format (BRK.B); yfinance wants dash-format (BRK-B)
    return sorted(df["Symbol"].str.replace(".", "-", regex=False).tolist())


def _fetch_ftse100() -> list[str]:
    """Scrape the FTSE 100 constituent list from Wikipedia. The table column
    that holds the EPIC ticker varies by Wikipedia revision; we look for any
    column named 'EPIC', 'Ticker', or 'Symbol'."""
    response = requests.get(_FTSE100_WIKI_URL, headers={"User-Agent": _USER_AGENT}, timeout=15)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))

    for df in tables:
        cols_lower = {str(c).lower().strip(): c for c in df.columns}
        ticker_col = (
            cols_lower.get("epic")
            or cols_lower.get("ticker")
            or cols_lower.get("symbol")
            or cols_lower.get("code")
        )
        if ticker_col is None:
            continue
        tickers = df[ticker_col].astype(str).str.strip().str.upper()
        # yfinance wants .L suffix for LSE-listed names
        tickers = [t if t.endswith(".L") else f"{t}.L" for t in tickers if t and t.lower() != "nan"]
        # De-dupe + filter out malformed entries
        seen = set()
        out = []
        for t in tickers:
            if t in seen:
                continue
            seen.add(t)
            if "." in t.replace(".L", "") or " " in t:
                continue
            out.append(t)
        if out:
            return sorted(out)
    raise RuntimeError("Could not find an FTSE 100 ticker column in any Wikipedia table")
