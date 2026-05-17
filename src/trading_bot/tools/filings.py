"""get_filing_summary — recent SEC EDGAR filings for a US ticker.

EDGAR is free and requires no auth, only a User-Agent that identifies you.
For US tickers we fetch the recent submissions index for the company's CIK
and return metadata + 8-K items list (the most material/timely filings for
trading decisions).

For 8-K filings specifically we optionally include a short excerpt of the
body — those are usually 1-3 pages and the material event is up front.
10-K / 10-Q bodies are too long for inline inclusion in a prompt; we only
return metadata + URL for those.

Companies House (UK) is deferred — most of our universe is US-listed and
the API requires registration.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache

import requests


_USER_AGENT = "trading-bot/0.1 dariozela1@gmail.com"
_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_BODY_PREVIEW_CHARS = 2000

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilingSummary:
    ticker: str
    form_type: str          # "8-K", "10-Q", "10-K", etc.
    filing_date: str        # ISO date the filing was accepted
    accession_number: str
    url: str                # link to the primary document on sec.gov
    items: tuple[str, ...] = field(default_factory=tuple)  # 8-K items (e.g., "5.02", "1.01")
    excerpt: str = ""       # first ~2000 chars of body for 8-K only


def get_filing_summary(
    ticker: str,
    days: int = 30,
    form_types: tuple[str, ...] = ("8-K", "10-Q", "10-K"),
    include_body_for_8k: bool = True,
) -> list[FilingSummary]:
    """Return recent EDGAR filings for a US-listed ticker within the lookback
    window, filtered to the requested form types. Empty list for tickers we
    can't resolve to a CIK (non-US listings, ADRs missing from the map,
    failed network calls)."""
    cik = _ticker_to_cik(ticker)
    if cik is None:
        return []

    try:
        response = requests.get(
            _SUBMISSIONS_URL.format(cik=cik),
            headers={"User-Agent": _USER_AGENT},
            timeout=10,
        )
        response.raise_for_status()
        body = response.json()
    except Exception as e:
        log.debug("EDGAR submissions fetch failed for %s (CIK %s): %s", ticker, cik, e)
        return []

    recent = (body.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    dates = recent.get("filingDate") or []
    accessions = recent.get("accessionNumber") or []
    primary_docs = recent.get("primaryDocument") or []
    items_lists = recent.get("items") or [""] * len(forms)

    cutoff = date.today() - timedelta(days=days)
    cik_int = int(cik)
    out: list[FilingSummary] = []

    for i in range(len(forms)):
        form = forms[i] if i < len(forms) else None
        if form not in form_types:
            continue
        try:
            filed = date.fromisoformat(dates[i])
        except (IndexError, ValueError, TypeError):
            continue
        if filed < cutoff:
            # Submissions are sorted descending; once we're past the cutoff
            # we can stop.
            break

        accession = accessions[i] if i < len(accessions) else ""
        primary = primary_docs[i] if i < len(primary_docs) else ""
        accession_no_dashes = accession.replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
            f"{accession_no_dashes}/{primary}"
        )

        items_raw = items_lists[i] if i < len(items_lists) else ""
        items = tuple(s.strip() for s in items_raw.split(",") if s.strip()) if items_raw else ()

        excerpt = ""
        if form == "8-K" and include_body_for_8k and primary:
            excerpt = _fetch_body_preview(url)

        out.append(
            FilingSummary(
                ticker=ticker,
                form_type=form,
                filing_date=filed.isoformat(),
                accession_number=accession,
                url=url,
                items=items,
                excerpt=excerpt,
            )
        )

    return out


# ---------------------------------------------------------------------------
# Ticker → CIK mapping
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _ticker_map() -> dict[str, str]:
    """Lazy-load + cache the SEC's full ticker → CIK mapping. ~1.5MB JSON,
    rarely changes. Returns {TICKER: 10-digit-zero-padded-CIK}."""
    try:
        response = requests.get(
            _TICKER_MAP_URL, headers={"User-Agent": _USER_AGENT}, timeout=15
        )
        response.raise_for_status()
        raw = response.json()
    except Exception as e:
        log.warning("Could not fetch SEC ticker map: %s", e)
        return {}

    out: dict[str, str] = {}
    # The file is keyed by row index. Each value has {"cik_str": int, "ticker": str, "title": str}
    for entry in raw.values():
        if not isinstance(entry, dict):
            continue
        ticker = entry.get("ticker")
        cik = entry.get("cik_str")
        if ticker and cik is not None:
            out[ticker.upper()] = f"{int(cik):010d}"
    return out


def _ticker_to_cik(ticker: str) -> str | None:
    mapping = _ticker_map()
    return mapping.get(ticker.upper())


# ---------------------------------------------------------------------------
# 8-K body preview
# ---------------------------------------------------------------------------

def _fetch_body_preview(url: str) -> str:
    """Pull the first chunk of an 8-K HTML body and strip to plain-ish text.
    Bounded at ~2000 chars so the resulting prompt stays sane."""
    try:
        response = requests.get(
            url, headers={"User-Agent": _USER_AGENT}, timeout=10
        )
        response.raise_for_status()
        html = response.text
    except Exception as e:
        log.debug("8-K body fetch failed for %s: %s", url, e)
        return ""

    # Strip HTML tags + collapse whitespace
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Skip past the boilerplate header — find the first "Item " heading
    item_match = re.search(r"\bItem\s+\d", text)
    if item_match:
        text = text[item_match.start():]

    return text[:_BODY_PREVIEW_CHARS]
