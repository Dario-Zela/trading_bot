from __future__ import annotations

from io import StringIO

import pandas as pd
import requests


_SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_USER_AGENT = "trading-bot/0.1 (research; +https://github.com/Dario-Zela/trading_bot)"


def get_universe(universe_id: str) -> list[str]:
    """Return a ticker list for the named universe.

    Wave 1 supports only 'sp500'. Later waves will add ETF universes
    ('us_etfs_sector', 'us_etfs_bond', 'us_etfs_commodity'), broker-catalog
    filters, and news-driven dynamic universes.
    """
    if universe_id == "sp500":
        return _fetch_sp500()
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
