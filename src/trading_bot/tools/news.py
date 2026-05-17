"""Recent-news tool — Alpaca News for US tickers, yfinance fallback elsewhere.

Alpaca News covers US-listed names well but returns nothing for UK/EU.
For those we fall back to yfinance's `.news` attribute which pulls from
Yahoo Finance and works for any ticker on any exchange.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import requests
import yfinance as yf

from trading_bot.alpaca_slot import load_data_creds


_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NewsItem:
    timestamp: str  # ISO 8601 UTC
    headline: str
    summary: str
    source: str
    url: str
    tickers: tuple[str, ...]


def get_recent_news(
    tickers: str | Iterable[str],
    days: int = 3,
    limit: int = 50,
) -> dict[str, list[NewsItem]]:
    """Fetch recent news for a ticker or list of tickers.

    Returns {ticker: [NewsItem, ...]}. Empty list for tickers with no news in
    the lookback window. Articles tagged against multiple tickers appear in
    each ticker's list.

    Source routing: US-listed tickers (no exchange suffix) go to Alpaca News;
    non-US tickers (.L, .DE, .PA, .AS, etc.) fall back to yfinance.news.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = list(tickers)
    if not tickers:
        return {}

    us_tickers = [t for t in tickers if "." not in t]
    non_us_tickers = [t for t in tickers if "." in t]

    out: dict[str, list[NewsItem]] = {t: [] for t in tickers}
    if us_tickers:
        _fetch_alpaca_news(us_tickers, days=days, limit=limit, out=out)
    for ticker in non_us_tickers:
        items = _fetch_yfinance_news(ticker, days=days)
        out[ticker].extend(items)
    return out


def _fetch_alpaca_news(
    tickers: list[str], *, days: int, limit: int, out: dict[str, list[NewsItem]]
) -> None:
    try:
        creds = load_data_creds()
    except Exception as e:
        log.warning("Alpaca creds unavailable; skipping US news: %s", e)
        return

    start = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    headers = {
        "APCA-API-KEY-ID": creds.api_key,
        "APCA-API-SECRET-KEY": creds.api_secret,
        "accept": "application/json",
    }
    params = {
        "symbols": ",".join(tickers),
        "start": start,
        "limit": min(limit, 50),
        "sort": "desc",
        "include_content": "false",
    }

    next_page = None
    pages_fetched = 0
    while True:
        if next_page:
            params["page_token"] = next_page
        response = requests.get(_NEWS_URL, headers=headers, params=params, timeout=15)
        if not response.ok:
            log.warning("Alpaca News fetch failed: %s %s", response.status_code, response.text[:200])
            return
        body = response.json()
        for article in body.get("news") or []:
            item = NewsItem(
                timestamp=article.get("created_at", ""),
                headline=article.get("headline", ""),
                summary=article.get("summary", ""),
                source=article.get("source", ""),
                url=article.get("url", ""),
                tickers=tuple(article.get("symbols", [])),
            )
            for tk in item.tickers:
                if tk in out:
                    out[tk].append(item)
        next_page = body.get("next_page_token")
        pages_fetched += 1
        if not next_page or pages_fetched >= 4:
            return


def _fetch_yfinance_news(ticker: str, *, days: int) -> list[NewsItem]:
    """Yahoo Finance news for non-US tickers (or as a fallback). yfinance
    returns dicts with 'title', 'publisher', 'providerPublishTime' (epoch),
    'link', and optionally 'relatedTickers'."""
    cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception as e:
        log.debug("yfinance news fetch failed for %s: %s", ticker, e)
        return []

    items: list[NewsItem] = []
    for art in raw[:50]:
        # yfinance's response shape changed across versions — check both top-level
        # and nested 'content' for the fields we need.
        content = art.get("content") if isinstance(art.get("content"), dict) else art
        published = content.get("providerPublishTime") or content.get("pubDate")
        ts = ""
        # Old shape: epoch int
        if isinstance(published, (int, float)) and published > cutoff:
            ts = datetime.fromtimestamp(published, tz=timezone.utc).isoformat()
        elif isinstance(published, str):
            try:
                # ISO date string (newer shape)
                dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                if dt.timestamp() < cutoff:
                    continue
                ts = dt.isoformat()
            except (ValueError, AttributeError):
                continue
        else:
            continue

        headline = content.get("title") or art.get("title") or ""
        summary = content.get("summary") or content.get("description") or ""
        publisher = (content.get("provider") or {}).get("displayName") if isinstance(content.get("provider"), dict) else content.get("publisher", "")
        link = (
            content.get("canonicalUrl", {}).get("url")
            if isinstance(content.get("canonicalUrl"), dict)
            else content.get("link", "") or art.get("link", "")
        )
        related = content.get("relatedTickers") or art.get("relatedTickers") or [ticker]

        if not headline:
            continue
        items.append(NewsItem(
            timestamp=ts,
            headline=headline,
            summary=summary,
            source=publisher or "Yahoo Finance",
            url=link or "",
            tickers=tuple(related) if isinstance(related, (list, tuple)) else (ticker,),
        ))
    return items
