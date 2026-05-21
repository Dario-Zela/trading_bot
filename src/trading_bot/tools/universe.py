from __future__ import annotations

import json
import logging
from functools import lru_cache
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


log = logging.getLogger(__name__)


_SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_SP400_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
_SP600_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
_FTSE100_WIKI_URL = "https://en.wikipedia.org/wiki/FTSE_100_Index"
_FTSE250_WIKI_URL = "https://en.wikipedia.org/wiki/FTSE_250_Index"
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

# LSE-listed UCITS ETFs — all stamp-duty exempt (ETFs aren't subject to
# SDRT in the UK), so they carry only the FX fee for non-GBP-denominated
# lines and zero broker-side cost for GBP ones. Hand-curated to cover the
# major asset classes UK retail can access without leaving T212. Adding
# these to the UK-EU candidate pool gives the LLM strategies cheap
# alternatives to UK individual shares (which pay 0.5% stamp duty on
# every purchase) — especially valuable for marginal trades where the
# expected move is under the SDRT hurdle.
_UK_UCITS_ETFS = [
    # Core UK / Europe (GBP)
    "ISF.L",   # iShares Core FTSE 100
    "VUKE.L",  # Vanguard FTSE 100
    "CUKX.L",  # iShares FTSE 100 Acc
    "VMID.L",  # Vanguard FTSE 250
    "VEUR.L",  # Vanguard FTSE Developed Europe ex-UK (EUR exposure)
    "IEUX.L",  # iShares MSCI Europe ex-UK
    # Core US / Global (USD-denominated UCITS — 0.30% FX, no stamp)
    "VUSA.L",  # Vanguard S&P 500
    "IUSA.L",  # iShares S&P 500
    "CSPX.L",  # iShares Core S&P 500 Acc
    "VWRL.L",  # Vanguard FTSE All-World
    "VWRP.L",  # Vanguard FTSE All-World Acc
    "IWDA.L",  # iShares Core MSCI World Acc
    "SWDA.L",  # iShares Core MSCI World (same fund, GBP line)
    "EQQQ.L",  # Invesco NASDAQ 100
    # Asia / Emerging Markets — the bot's stated Asian-exposure pipeline
    "VJPN.L",  # Vanguard FTSE Japan
    "CPJ1.L",  # iShares Core MSCI Japan IMI
    "VAPX.L",  # Vanguard FTSE Developed Asia Pacific ex-Japan
    "VFEM.L",  # Vanguard FTSE Emerging Markets
    "IEEM.L",  # iShares Core MSCI Emerging Markets IMI
    # Commodities (physical-backed ETCs — also exempt from SDRT)
    "SGLN.L",  # iShares Physical Gold (GBP)
    "SSLV.L",  # iShares Physical Silver (GBP)
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


# ---------------------------------------------------------------------------
# T212 instrument file → yfinance universe
# ---------------------------------------------------------------------------

# Map the T212 internal venue-suffix letter (BARC**l**_EQ → LSE) to a
# yfinance ticker suffix. T212 uses a single lowercase character right
# before "_EQ" to encode the listing venue. Two letters share semantics:
# 'n' shows up on Euronext Amsterdam-listed lines (ASMLn_EQ = ASML.AS)
# but 'a' is also Amsterdam — we treat both the same.
_T212_SUFFIX_TO_YF = {
    "l": ".L",     # LSE main + AIM (GBP / GBX)
    "i": ".L",     # International order book (LSE) — GDRs etc
    "a": ".AS",    # Amsterdam (Euronext)
    "n": ".AS",    # Same — Amsterdam tertiary listings
    "p": ".PA",    # Paris (Euronext)
    "d": ".DE",    # Xetra (Deutsche Börse)
    "m": ".MI",    # Milan
    "e": ".MC",    # Madrid
    "s": ".SW",    # Swiss
    "h": ".HE",    # Helsinki
}


def _t212_to_yfinance_ticker(t212_ticker: str, short_name: str | None = None) -> str | None:
    """Convert a T212 instrument (internal ticker + shortName) to a
    yfinance ticker. Returns None if the venue suffix isn't recognised.

    T212's `ticker` field encodes the venue (BARC**l**_EQ = LSE) but
    its base is often a legacy stock-exchange epic that's diverged
    from the modern yfinance symbol (T212 `DCl_EQ` is Currys, but
    yfinance is `CURY.L` — DC was the pre-rebrand Dixons Carphone
    epic). The `shortName` field tracks the current ticker, so we
    prefer it when available.
    """
    if not t212_ticker or not t212_ticker.endswith("_EQ"):
        return None
    if "_US_EQ" in t212_ticker:
        # yfinance uses dash for class shares (BRK-B, BF-B), T212 uses
        # dot in shortName (BRK.B). Normalise to yfinance convention.
        base = (short_name or t212_ticker.replace("_US_EQ", "")).strip().upper()
        return base.replace(".", "-") if base else None
    if len(t212_ticker) < 5:
        return None
    suffix_char = t212_ticker[-4]
    yf_suffix = _T212_SUFFIX_TO_YF.get(suffix_char)
    if yf_suffix is None:
        return None
    base = (short_name or t212_ticker[:-4]).strip().upper()
    if not base or not base.replace(".", "").replace("-", "").isalnum():
        return None
    return f"{base}{yf_suffix}"


# Substrings in instrument names that flag products as non-ISA-eligible
# or otherwise undesirable for the bot (leveraged ETPs, inverse, short).
# T212 ISA explicitly excludes leveraged / inverse products per HMRC's
# "complex instruments" guidance; we filter pre-flight rather than
# discover at order time.
_NON_ISA_NAME_FILTERS = (
    "3X ", "2X ", "DAILY LEVERAGED", "DAILY SHORT", "DAILY INVERSE",
    "LEVERAGED", "INVERSE", "SHORT ETF", "ULTRASHORT", "ULTRA SHORT",
    "BEAR ETF", "BULL ETF",     # Direxion-style branding
)


def _is_isa_eligible(inst: dict) -> bool:
    """Heuristic ISA eligibility filter for a T212 instrument record.

    T212's catalog doesn't carry an explicit ISA flag, but the rules
    are deterministic enough to reproduce:
      - WARRANTs are never ISA-eligible
      - ETFs need to be UCITS-compliant; US-domiciled ETFs (ISIN starts
        with "US") are non-UCITS by definition
      - Leveraged / inverse products are barred regardless of wrapper
    """
    typ = (inst.get("type") or "").upper()
    if typ == "WARRANT":
        return False
    isin = (inst.get("isin") or "").upper()
    if typ == "ETF" and isin.startswith("US"):
        return False
    name = (inst.get("name") or "").upper()
    if any(flag in name for flag in _NON_ISA_NAME_FILTERS):
        return False
    return True


def _t212_instruments_path() -> Path:
    """Locate the cached T212 instrument file. State dir is two levels
    up from this module (src/trading_bot/tools/universe.py)."""
    return Path(__file__).resolve().parents[3] / "state" / "t212_instruments.json"


@lru_cache(maxsize=1)
def _load_t212_universe_raw() -> list[dict]:
    """Read the cached T212 instrument file and return the parsed list.
    Empty list if the file is missing — callers fall back to scraped
    Wikipedia universes."""
    path = _t212_instruments_path()
    if not path.exists():
        log.warning("T212 instrument cache missing at %s — falling back to scraped universes", path)
        return []
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Could not parse %s: %s", path, e)
        return []
    items = raw if isinstance(raw, list) else (raw.get("instruments") or [])
    return [i for i in items if isinstance(i, dict)]


def _t212_venue(t212_ticker: str) -> str:
    """Bucket a T212 ticker into a coarse venue label used by region
    filters. Returns 'US', 'LSE', 'XETR', 'PAR', 'AMS', 'MIL', 'MAD',
    'SWX', 'HEL', or '' if unrecognised."""
    if "_US_EQ" in t212_ticker:
        return "US"
    if not t212_ticker.endswith("_EQ") or len(t212_ticker) < 5:
        return ""
    suffix = t212_ticker[-4]
    return {
        "l": "LSE", "i": "LSE",
        "a": "AMS", "n": "AMS",
        "p": "PAR", "d": "XETR", "m": "MIL",
        "e": "MAD", "s": "SWX", "h": "HEL",
    }.get(suffix, "")


def _t212_isa_eligible_universe(*, venues: set[str] | None = None) -> list[str]:
    """Build a yfinance-ticker universe from the cached T212 instrument
    file, filtered to ISA-eligible instruments and optionally to a set
    of venues (e.g. {'LSE'} for UK-only, {'US'} for US-only).

    Returns sorted list of yfinance tickers. Skips instruments whose
    T212 venue suffix isn't in _T212_SUFFIX_TO_YF (some smaller
    European venues we don't yet handle).

    Result is post-filtered against the local OHLCV cache: tickers that
    yfinance + Stooq couldn't fetch (Xetra structured-product mirrors,
    delisted Q-suffix names, dead small-caps) get dropped so the LLM
    pre-filter doesn't waste tokens ranking instruments the strategies
    can't actually use. Bootstrap-safe: if the cache is sparse (< 1000
    tickers in the last 14 days), the filter is bypassed."""
    items = _load_t212_universe_raw()
    if not items:
        return []
    out: set[str] = set()
    for inst in items:
        if not _is_isa_eligible(inst):
            continue
        t212 = inst.get("ticker") or ""
        if venues is not None and _t212_venue(t212) not in venues:
            continue
        yf = _t212_to_yfinance_ticker(t212, inst.get("shortName"))
        if yf is None:
            continue
        out.add(yf)
    return _filter_against_data_cache(sorted(out))


@lru_cache(maxsize=1)
def _tickers_with_recent_data(window_days: int = 14) -> frozenset[str]:
    """Return the set of yfinance tickers that have at least one bar in
    the local OHLCV cache within the last `window_days`. Empty set if
    the cache is missing / unreadable / not yet populated."""
    try:
        from datetime import date as _date, timedelta as _td
        from trading_bot.tools.ohlcv_store import _conn  # noqa: WPS437
        cutoff = (_date.today() - _td(days=window_days)).isoformat()
        with _conn() as c:
            rows = c.execute(
                "SELECT DISTINCT ticker FROM bars WHERE bar_date >= ?",
                (cutoff,),
            ).fetchall()
        return frozenset(r["ticker"] for r in rows)
    except Exception as e:
        log.warning("could not query OHLCV cache for data-coverage filter: %s", e)
        return frozenset()


def _filter_against_data_cache(
    tickers: list[str],
    *,
    min_cache_size: int = 1000,
) -> list[str]:
    """Drop tickers from `tickers` that have no data in the local
    OHLCV cache. Bootstrap-safe: if the cache has fewer than
    `min_cache_size` tickers with recent bars, return the input
    unchanged — assume the cache is freshly initialised and any
    filtering would be over-restrictive."""
    with_data = _tickers_with_recent_data()
    if len(with_data) < min_cache_size:
        log.info(
            "universe data-coverage filter: cache has only %d tickers with recent data — "
            "bypassing filter (need >= %d to activate)",
            len(with_data), min_cache_size,
        )
        return tickers
    filtered = [t for t in tickers if t in with_data]
    dropped = len(tickers) - len(filtered)
    if dropped:
        log.info(
            "universe data-coverage filter: dropped %d/%d tickers with no recent cache hits",
            dropped, len(tickers),
        )
    return filtered


def get_universe(universe_id: str) -> list[str]:
    """Return a ticker list for the named universe.

    Supported:
    - US equities: 'sp500', 'sp400', 'sp600', 'sp1500' (500+400+600 combined)
    - US ETFs: 'us_etfs_sector', 'us_etfs_bond', 'us_etfs_commodity'
    - UK equities: 'ftse100', 'ftse250', 'ftse350' (100+250 combined)
    - UK ETFs: 'uk_ucits_etfs' (~20 LSE-listed UCITS, SDRT-exempt)
    - EU equities: 'dax40', 'cac40', 'aex25', 'eu_blue_chips' (DAX+CAC+AEX)
    - UK+EU combined: 'uk_eu_blue_chips' (FTSE100+DAX+CAC+AEX),
                       'uk_eu_extended' (FTSE350+DAX+CAC+AEX+UCITS ETFs, ~470 names)
    - T212 ISA-eligible (sourced from cached T212 instrument file, filtered
      to drop warrants + US-domiciled ETFs + leveraged products):
        't212_isa_uk'     — LSE-listed (UK shares + UCITS ETFs + GDRs)
        't212_isa_us'     — US-listed (NYSE/NASDAQ stocks; no US-ISIN ETFs)
        't212_isa_eu'     — XETR + PAR + AMS + MIL + MAD + SWX + HEL
        't212_isa_uk_eu'  — UK + EU combined
        't212_isa_global' — every ISA-eligible instrument T212 lists
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
            | set(_UK_UCITS_ETFS)
        )
        return sorted(combined)
    if universe_id == "uk_ucits_etfs":
        return list(_UK_UCITS_ETFS)
    # T212-sourced ISA-eligible universes (broader than scraped index lists)
    if universe_id == "t212_isa_uk":
        return _t212_isa_eligible_universe(venues={"LSE"})
    if universe_id == "t212_isa_us":
        return _t212_isa_eligible_universe(venues={"US"})
    if universe_id == "t212_isa_eu":
        return _t212_isa_eligible_universe(
            venues={"XETR", "PAR", "AMS", "MIL", "MAD", "SWX", "HEL"}
        )
    if universe_id == "t212_isa_uk_eu":
        return _t212_isa_eligible_universe(
            venues={"LSE", "XETR", "PAR", "AMS", "MIL", "MAD", "SWX", "HEL"}
        )
    if universe_id == "t212_isa_global":
        return _t212_isa_eligible_universe(venues=None)
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
