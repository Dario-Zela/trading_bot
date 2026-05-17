from __future__ import annotations

from io import StringIO

import pandas as pd
import requests


_SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_FTSE100_WIKI_URL = "https://en.wikipedia.org/wiki/FTSE_100_Index"
_DAX40_WIKI_URL = "https://en.wikipedia.org/wiki/DAX"
_CAC40_WIKI_URL = "https://en.wikipedia.org/wiki/CAC_40"
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

# iShares STOXX Europe 600 sector ETFs. Listed in Frankfurt (.DE) — these
# are the EU analogues of US SPDR sector ETFs.
_EU_ETFS_SECTOR = [
    "EXH1.DE",  # Banks
    "EXH4.DE",  # Consumer Goods
    "EXH5.DE",  # Industrial
    "EXH8.DE",  # Utilities
    "EXH9.DE",  # Travel & Leisure / Consumer Services
    "EXSA.DE",  # Basic Resources
    "EXV1.DE",  # Insurance
    "EXV3.DE",  # Technology
    "EXV4.DE",  # Telecom
    "EXV6.DE",  # Health Care
    "EXV9.DE",  # Oil & Gas
]

# European bond ETFs — UK Gilts, Bunds, EUR credit. Sufficient for the
# bond-cycle strategy's duration + credit decisions in the EU regime.
_EU_ETFS_BOND = [
    "IGLT.L",   # UK Gilts (all maturities)
    "IGLS.L",   # UK Gilts (short)
    "IGLO.L",   # UK Gilts (long)
    "EUNH.DE",  # EUR govt 1-3Y
    "IBGM.L",   # EUR govt 7-10Y
    "IEAC.L",   # EUR investment-grade corp
    "IHYG.L",   # EUR high yield
]

# AEX 25 — hand-curated list of Amsterdam-listed Euronext names. Small and
# stable (the AEX index has 25 components), so easier to hardcode than scrape.
_AEX25 = [
    "AALB.AS",  # Aalberts
    "ABN.AS",   # ABN Amro
    "AD.AS",    # Ahold Delhaize
    "ADYEN.AS", # Adyen
    "AGN.AS",   # Aegon
    "AKZA.AS",  # AkzoNobel
    "ASM.AS",   # ASM International
    "ASML.AS",  # ASML
    "ASRNL.AS", # ASR Nederland
    "BESI.AS",  # BE Semiconductor
    "DSFIR.AS", # DSM-Firmenich
    "EXO.AS",   # Exor
    "HEIA.AS",  # Heineken
    "IMCD.AS",  # IMCD
    "INGA.AS",  # ING Group
    "KPN.AS",   # KPN
    "MT.AS",    # ArcelorMittal
    "NN.AS",    # NN Group
    "PHIA.AS",  # Philips
    "PRX.AS",   # Prosus
    "RAND.AS",  # Randstad
    "REN.AS",   # Relx
    "SHELL.AS", # Shell
    "UMG.AS",   # Universal Music Group
    "UNA.AS",   # Unilever
    "WKL.AS",   # Wolters Kluwer
]


def get_universe(universe_id: str) -> list[str]:
    """Return a ticker list for the named universe.

    Supported:
    - US equities: 'sp500'
    - US ETFs: 'us_etfs_sector', 'us_etfs_bond', 'us_etfs_commodity'
    - UK equities: 'ftse100'
    - EU equities: 'dax40' (Frankfurt), 'cac40' (Paris), 'aex25' (Amsterdam),
                   'eu_blue_chips' (union of the three)
    """
    if universe_id == "sp500":
        return _fetch_sp500()
    if universe_id == "ftse100":
        return _fetch_ftse100()
    if universe_id == "dax40":
        return _fetch_dax40()
    if universe_id == "cac40":
        return _fetch_cac40()
    if universe_id == "aex25":
        return list(_AEX25)
    if universe_id == "eu_blue_chips":
        combined = set(_fetch_dax40()) | set(_fetch_cac40()) | set(_AEX25)
        return sorted(combined)
    if universe_id == "uk_eu_blue_chips":
        # FTSE 100 + DAX 40 + CAC 40 + AEX 25 — the full UK + EU pipeline universe
        combined = (
            set(_fetch_ftse100())
            | set(_fetch_dax40())
            | set(_fetch_cac40())
            | set(_AEX25)
        )
        return sorted(combined)
    if universe_id == "us_etfs_sector":
        return list(_US_ETFS_SECTOR)
    if universe_id == "us_etfs_bond":
        return list(_US_ETFS_BOND)
    if universe_id == "us_etfs_commodity":
        return list(_US_ETFS_COMMODITY)
    if universe_id == "eu_etfs_sector":
        return list(_EU_ETFS_SECTOR)
    if universe_id == "eu_etfs_bond":
        return list(_EU_ETFS_BOND)
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


def _fetch_dax40() -> list[str]:
    """Scrape DAX 40 constituents from Wikipedia. Append .DE for yfinance."""
    return _scrape_eu_index(_DAX40_WIKI_URL, suffix=".DE", existing_suffix=".DE")


def _fetch_cac40() -> list[str]:
    """Scrape CAC 40 constituents from Wikipedia. Append .PA for yfinance."""
    return _scrape_eu_index(_CAC40_WIKI_URL, suffix=".PA", existing_suffix=".PA")


def _scrape_eu_index(url: str, *, suffix: str, existing_suffix: str) -> list[str]:
    """Common Wikipedia-scrape pattern for DAX / CAC etc.

    Looks at every table on the page, finds the one with a ticker-like column,
    extracts tickers, and appends the appropriate yfinance suffix if not
    already present.
    """
    response = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=15)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))

    candidates = ("ticker symbol", "ticker", "symbol", "code", "isin", "epic")
    for df in tables:
        cols_lower = {str(c).lower().strip(): c for c in df.columns}
        col = None
        for k in candidates:
            if k in cols_lower:
                col = cols_lower[k]
                break
        if col is None:
            continue

        tickers = df[col].astype(str).str.strip()
        # Heuristic: a valid ticker is 1-5 alphanumerics; ISINs are 12 chars
        # alphanumeric and don't fit our suffix pattern, so skip them.
        cleaned = []
        for t in tickers:
            t = t.upper().split()[0] if t else ""
            if not t or t.lower() == "nan":
                continue
            # Skip rows that look like ISINs (12 chars starting with 2 letters)
            if len(t) == 12 and t[:2].isalpha() and t[2:].isalnum():
                continue
            # Cap at 8 chars to filter out company-name fall-throughs
            if len(t) > 8:
                continue
            # If the ticker already has a market suffix (any '.X' tail),
            # leave it as-is. Otherwise append the index's default suffix.
            if "." not in t:
                t = f"{t}{suffix}"
            cleaned.append(t)

        # Dedupe + check we have at least ~20 (sanity floor)
        seen: set[str] = set()
        out = []
        for t in cleaned:
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
        if len(out) >= 20:
            return sorted(out)

    raise RuntimeError(f"Could not find a ticker column in any Wikipedia table at {url}")
