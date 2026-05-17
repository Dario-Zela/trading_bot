"""Recent-news tool — wraps Alpaca's News API.

Alpaca News is bundled with any paper account, no extra signup. The endpoint
returns articles with headline, summary, URL, source, and the tickers it's
tagged against. Free tier limit: 200 req/min — far above anything we need.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import requests

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
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = list(tickers)
    if not tickers:
        return {}

    creds = load_data_creds()
    start = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    headers = {
        "APCA-API-KEY-ID": creds.api_key,
        "APCA-API-SECRET-KEY": creds.api_secret,
        "accept": "application/json",
    }
    params = {
        "symbols": ",".join(tickers),
        "start": start,
        "limit": min(limit, 50),  # API max per page is 50
        "sort": "desc",
        "include_content": "false",
    }

    out: dict[str, list[NewsItem]] = {t: [] for t in tickers}
    next_page = None
    pages_fetched = 0
    while True:
        if next_page:
            params["page_token"] = next_page
        response = requests.get(_NEWS_URL, headers=headers, params=params, timeout=15)
        if not response.ok:
            log.warning("Alpaca News fetch failed: %s %s", response.status_code, response.text[:200])
            break
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
        if not next_page or pages_fetched >= 4:  # cap at 200 articles total
            break
    return out
