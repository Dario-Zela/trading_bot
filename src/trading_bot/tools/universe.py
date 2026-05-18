from __future__ import annotations

from io import StringIO

import pandas as pd
import requests


_SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_SP400_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
_SP600_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
_FTSE100_WIKI_URL = "https://en.wikipedia.org/wiki/FTSE_100_Index"
_FTSE250_WIKI_URL = "https://en.wikipedia.org/wiki/FTSE_250_Index"
_DAX40_WIKI_URL = "https://en.wikipedia.org/wiki/DAX"
_CAC40_WIKI_URL = "https://en.wikipedia.org/wiki/CAC_40"
_HANGSENG_WIKI_URL = "https://en.wikipedia.org/wiki/Hang_Seng_Index"

# Curated Japanese large-caps. The English Wikipedia Nikkei 225 page lacks a
# constituents table (the JP Wikipedia has one but parsing it cleanly with
# pandas.read_html is brittle), and the index changes only ~2-3 names/year,
# so a pinned list is the right trade-off for shadow trading. Mostly
# Nikkei 225 + TOPIX-100 overlap names with good ADR liquidity. Refresh
# this list once a year or when membership changes are flagged in lessons.
_JP_LARGE_CAPS = [
    "1605.T",   # INPEX
    "1925.T",   # Daiwa House
    "2502.T",   # Asahi Group
    "2503.T",   # Kirin
    "2802.T",   # Ajinomoto
    "2914.T",   # Japan Tobacco
    "3382.T",   # Seven & i
    "3402.T",   # Toray Industries
    "3407.T",   # Asahi Kasei
    "4063.T",   # Shin-Etsu Chemical
    "4452.T",   # Kao
    "4502.T",   # Takeda Pharmaceutical
    "4503.T",   # Astellas Pharma
    "4519.T",   # Chugai Pharma
    "4523.T",   # Eisai
    "4543.T",   # Terumo
    "4568.T",   # Daiichi Sankyo
    "4661.T",   # Oriental Land
    "4901.T",   # Fujifilm
    "5108.T",   # Bridgestone
    "5401.T",   # Nippon Steel
    "5713.T",   # Sumitomo Metal Mining
    "6098.T",   # Recruit Holdings
    "6273.T",   # SMC Corp
    "6301.T",   # Komatsu
    "6326.T",   # Kubota
    "6367.T",   # Daikin Industries
    "6501.T",   # Hitachi
    "6502.T",   # Toshiba
    "6503.T",   # Mitsubishi Electric
    "6594.T",   # Nidec
    "6701.T",   # NEC
    "6702.T",   # Fujitsu
    "6752.T",   # Panasonic
    "6758.T",   # Sony Group
    "6857.T",   # Advantest
    "6861.T",   # Keyence
    "6902.T",   # Denso
    "6920.T",   # Lasertec
    "6954.T",   # Fanuc
    "6981.T",   # Murata Manufacturing
    "7011.T",   # Mitsubishi Heavy Industries
    "7201.T",   # Nissan Motor
    "7203.T",   # Toyota Motor
    "7267.T",   # Honda Motor
    "7269.T",   # Suzuki Motor
    "7270.T",   # Subaru
    "7733.T",   # Olympus
    "7741.T",   # HOYA
    "7751.T",   # Canon
    "7832.T",   # Bandai Namco
    "7974.T",   # Nintendo
    "8001.T",   # Itochu
    "8002.T",   # Marubeni
    "8031.T",   # Mitsui & Co
    "8035.T",   # Tokyo Electron
    "8053.T",   # Sumitomo Corp
    "8058.T",   # Mitsubishi Corp
    "8267.T",   # Aeon
    "8306.T",   # Mitsubishi UFJ Financial
    "8316.T",   # Sumitomo Mitsui Financial
    "8411.T",   # Mizuho Financial
    "8591.T",   # Orix
    "8604.T",   # Nomura Holdings
    "8725.T",   # MS&AD Insurance
    "8766.T",   # Tokio Marine
    "8801.T",   # Mitsui Fudosan
    "8802.T",   # Mitsubishi Estate
    "9020.T",   # JR East
    "9022.T",   # JR Central
    "9432.T",   # NTT
    "9433.T",   # KDDI
    "9434.T",   # SoftBank Corp
    "9501.T",   # TEPCO
    "9613.T",   # NTT Data
    "9983.T",   # Fast Retailing (Uniqlo)
    "9984.T",   # SoftBank Group
]
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
    - US equities: 'sp500', 'sp400', 'sp600', 'sp1500' (500+400+600 combined)
    - US ETFs: 'us_etfs_sector', 'us_etfs_bond', 'us_etfs_commodity'
    - UK equities: 'ftse100', 'ftse250', 'ftse350' (100+250 combined)
    - EU equities: 'dax40', 'cac40', 'aex25', 'eu_blue_chips' (DAX+CAC+AEX)
    - UK+EU combined: 'uk_eu_blue_chips' (FTSE100+DAX+CAC+AEX),
                       'uk_eu_extended' (FTSE350+DAX+CAC+AEX, ~450 names)
    - Asia equities: 'jp_large_caps' (curated TSE), 'hangseng' (HKEX),
                     'asia_blue_chips' (jp_large_caps + hangseng, ~150 names)
    """
    if universe_id == "sp500":
        return _fetch_sp500()
    if universe_id == "sp400":
        return _fetch_sp400()
    if universe_id == "sp600":
        return _fetch_sp600()
    if universe_id == "sp1500":
        combined = set(_fetch_sp500()) | set(_fetch_sp400()) | set(_fetch_sp600())
        return sorted(combined)
    if universe_id == "ftse100":
        return _fetch_ftse100()
    if universe_id == "ftse250":
        return _fetch_ftse250()
    if universe_id == "ftse350":
        combined = set(_fetch_ftse100()) | set(_fetch_ftse250())
        return sorted(combined)
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
        combined = (
            set(_fetch_ftse100())
            | set(_fetch_dax40())
            | set(_fetch_cac40())
            | set(_AEX25)
        )
        return sorted(combined)
    if universe_id == "uk_eu_extended":
        combined = (
            set(_fetch_ftse100())
            | set(_fetch_ftse250())
            | set(_fetch_dax40())
            | set(_fetch_cac40())
            | set(_AEX25)
        )
        return sorted(combined)
    if universe_id == "jp_large_caps":
        return list(_JP_LARGE_CAPS)
    if universe_id == "hangseng":
        return _fetch_hangseng()
    if universe_id == "asia_blue_chips":
        combined = set(_JP_LARGE_CAPS) | set(_fetch_hangseng())
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


def _fetch_sp400() -> list[str]:
    response = requests.get(_SP400_WIKI_URL, headers={"User-Agent": _USER_AGENT}, timeout=15)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    df = tables[0]
    return sorted(df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist())


def _fetch_sp600() -> list[str]:
    response = requests.get(_SP600_WIKI_URL, headers={"User-Agent": _USER_AGENT}, timeout=15)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    df = tables[0]
    return sorted(df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist())


def _fetch_ftse250() -> list[str]:
    """FTSE 250 (UK mid-cap, ~250 LSE names). Same pattern as FTSE 100."""
    response = requests.get(_FTSE250_WIKI_URL, headers={"User-Agent": _USER_AGENT}, timeout=15)
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
        tickers = [t if t.endswith(".L") else f"{t}.L" for t in tickers if t and t.lower() != "nan"]
        seen = set()
        out = []
        for t in tickers:
            if t in seen or " " in t or len(t) > 8:
                continue
            seen.add(t)
            out.append(t)
        if len(out) >= 100:  # FTSE 250 sanity floor
            return sorted(out)
    raise RuntimeError("Could not find an FTSE 250 ticker column in any Wikipedia table")


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


def _fetch_hangseng() -> list[str]:
    """Hang Seng Index — Hong Kong large-caps. Tickers like 0700.HK
    (Tencent). HKEX uses 4-digit zero-padded codes. Wikipedia renders the
    ticker column as "SEHK: 700" so we strip the prefix before padding."""
    response = requests.get(_HANGSENG_WIKI_URL, headers={"User-Agent": _USER_AGENT}, timeout=15)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    for df in tables:
        cols_lower = {str(c).lower().strip(): c for c in df.columns}
        ticker_col = (
            cols_lower.get("stock code")
            or cols_lower.get("ticker")
            or cols_lower.get("ticker symbol")
            or cols_lower.get("symbol")
            or cols_lower.get("code")
        )
        if ticker_col is None:
            continue
        seen: set[str] = set()
        out: list[str] = []
        for raw in df[ticker_col].astype(str):
            if not raw or raw.lower() == "nan":
                continue
            # Strip exchange prefix like "SEHK: 700" → "700", and any .HK suffix
            cleaned = raw.split(":")[-1].strip().replace(".HK", "").strip()
            if not cleaned.isdigit():
                continue
            padded = f"{int(cleaned):04d}.HK"
            if padded in seen:
                continue
            seen.add(padded)
            out.append(padded)
        if len(out) >= 30:  # Hang Seng has ~80 components
            return sorted(out)
    raise RuntimeError("Could not find a Hang Seng ticker column in any Wikipedia table")


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
