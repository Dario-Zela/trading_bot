from __future__ import annotations

from io import StringIO

import pandas as pd
import requests


_SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
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

    Supported: 'sp500', 'us_etfs_sector', 'us_etfs_bond', 'us_etfs_commodity'.
    Future waves: broker-catalog filters, news-driven dynamic universes.
    """
    if universe_id == "sp500":
        return _fetch_sp500()
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
