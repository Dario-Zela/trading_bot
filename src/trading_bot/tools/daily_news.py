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
    """Return today's compressed bot-summary brief for strategy prompts.
    Prefers the `.bot.md` companion file (a ~150-word compressed summary
    extracted from the newspaper) and falls back to the full newspaper
    `.md` if no bot version exists (older briefs, or if Claude omitted
    the bot-summary fenced block). Empty string if no brief at all."""
    d = on_date or date.today()
    bot_path = STATE_ROOT / "daily_news" / f"{d.isoformat()}.bot.md"
    if bot_path.exists():
        return bot_path.read_text()
    full_path = STATE_ROOT / "daily_news" / f"{d.isoformat()}.md"
    if full_path.exists():
        log.debug(
            "No bot-summary for %s, falling back to full newspaper at %s",
            d.isoformat(), full_path,
        )
        return full_path.read_text()
    log.debug("No daily news brief for %s", d.isoformat())
    return ""
