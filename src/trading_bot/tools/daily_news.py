"""Read the day's market news brief, if one was generated.

Companion to trading_bot.meta.daily_news (the writer). LLM strategies that
list `get_daily_news_brief` in their tools list get this content injected
into their prompt as today's market-wide context.

Returns empty string if no brief exists for the date — strategies should
gracefully handle absence rather than failing.
"""
from __future__ import annotations

import logging
from datetime import date

from trading_bot.state.paths import STATE_ROOT


log = logging.getLogger(__name__)


def get_daily_news_brief(on_date: date | None = None) -> str:
    """Return the markdown brief for `on_date` (default: today). Empty
    string if no brief was generated."""
    d = on_date or date.today()
    path = STATE_ROOT / "daily_news" / f"{d.isoformat()}.md"
    if not path.exists():
        log.debug("No daily news brief for %s at %s", d.isoformat(), path)
        return ""
    return path.read_text()
