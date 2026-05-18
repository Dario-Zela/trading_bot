"""Phase 2F — Bot summary compressor.

Reads the full assembled edition (plan + briefs + articles + floor +
desk's calls) and emits a tight ~150-word `state/daily_news/YYYY-MM-DD.bot.md`
for the trading strategy prompts.

The structure is locked: Risk tone / Themes / Sector lean /
Watchlist / Key data. The strategies have been reading this shape
for weeks — changing the contract is unrelated to the newspaper
restyle and would force re-prompting every strategy.

Compression strategy
====================
We don't send the full rendered HTML to the compressor. We send the
plan + the brief bodies — that's already short enough and contains
everything that matters for the strategies. The full articles add
texture but not new facts.
"""
from __future__ import annotations

import logging
import os
from datetime import date

from trading_bot.llm.claude_code import ClaudeCodeError, run_claude
from trading_bot.meta.news.brief_writer import Brief
from trading_bot.meta.news.desks_calls import DesksCalls
from trading_bot.meta.news.publisher import NewsPlan
from trading_bot.meta.news.trading_floor import FloorBrief

log = logging.getLogger(__name__)

_SUMMARY_TIMEOUT = 240


def compress_bot_summary(
    plan: NewsPlan,
    briefs: dict[str, Brief],
    floor: list[FloorBrief],
    desks: DesksCalls,
    today: date,
) -> str:
    """Returns the compressed bot-summary markdown. Always returns
    *something* — when the LLM is unavailable we synthesise a
    minimal version from the plan headlines so strategies don't get
    an empty brief."""
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        log.error("CLAUDE_CODE_OAUTH_TOKEN not set — using fallback bot summary")
        return _fallback_summary(plan, today)

    prompt = _build_prompt(plan, briefs, floor, desks, today)
    try:
        result = run_claude(prompt, model="haiku", timeout_seconds=_SUMMARY_TIMEOUT)
    except ClaudeCodeError as e:
        log.warning("Bot summary Haiku failed: %s — using fallback", e)
        return _fallback_summary(plan, today)

    text = (result.text or "").strip()
    if not text or "**Risk tone:**" not in text:
        log.warning("Bot summary missing required structure — falling back")
        return _fallback_summary(plan, today)
    return text


def _build_prompt(
    plan: NewsPlan,
    briefs: dict[str, Brief],
    floor: list[FloorBrief],
    desks: DesksCalls,
    today: date,
) -> str:
    edition_lines = []
    for p in plan.pieces:
        b = briefs.get(p.slug)
        body = (b.body_md if b else p.one_line)[:480]
        edition_lines.append(f"### [{p.section}] {p.headline}\n{body}")
    edition_block = "\n\n".join(edition_lines) or "(empty edition)"

    floor_block = ""
    if floor:
        floor_block = "\n## Yesterday's trading floor\n\n" + "\n\n".join(
            f"### {f.headline}\n{f.body_md[:300]}" for f in floor
        )

    calls_block = ""
    if desks.fresh_predictions:
        calls_lines = [
            f"- [{p.horizon} → {p.target_date}, conviction={p.conviction}] {p.claim}"
            for p in desks.fresh_predictions
        ]
        calls_block = "\n## Today's predictions\n\n" + "\n".join(calls_lines)

    question_line = (
        f"\n## Today's question (the publisher's framing — keep it in mind when "
        f"choosing your Risk tone wording; do NOT add a new section for it)\n\n"
        f"> {plan.todays_question}\n"
        if plan.todays_question else ""
    )

    return f"""You are compressing today's edition of The Bot Tribune
into a 150-word strategy brief. {today.isoformat()}.

The strategies that read this need: macro risk tone, the day's
themes, sector lean, a watchlist of tickers, and any data/events
worth flagging. They DO NOT need the news as news — they need it
as signal.
{question_line}

## The edition

{edition_block}
{floor_block}
{calls_block}

## Required output — markdown, EXACTLY this shape

```
**Risk tone:** <risk-on / risk-off / mixed — and a one-line reason rooted in today's facts>

**Themes ({{N}}):**
- _<theme name>_ — <2-3 phrases on what's happening and which sectors it touches>
- _<theme name>_ — ...

**Sector lean:**
- Bullish: <sectors>
- Mildly bullish: <sectors>
- Neutral: <sectors>
- Mildly bearish: <sectors>
- Bearish: <sectors>

**Watchlist (top tickers across regions):** <8-12 tickers, comma-separated>

**Key data / events to watch today:** <one or two lines>
```

## Compression rules

- 150 words ± 20. The output is a *briefing*, not a recap.
- No fluff: cut "as discussed", "notably", "interestingly".
- Tickers: prefer those that actually appeared in the edition; use
  the canonical symbol (NVDA, BP.L for UK).
- Themes: 3-6 max. Each theme should map cleanly to sector lean
  buckets so the strategies can act on it.
- Risk tone: be definite. "Mixed with a downside skew" is fine;
  "uncertain" is not.
- DO NOT add a preamble. Start with **Risk tone:**.
- DO NOT include any code fences in the output — the markdown is
  the final document.

Output only the markdown, no JSON, no preamble.
"""


def _fallback_summary(plan: NewsPlan, today: date) -> str:
    """Used when the LLM is unavailable. Surfaces the headlines so the
    strategies see *something* in the brief shape they expect."""
    headlines = [p.headline for p in plan.pieces[:6]]
    head_block = "\n".join(f"- {h}" for h in headlines) or "- (no headlines available)"
    return (
        f"**Risk tone:** Mixed — LLM unavailable for {today.isoformat()}, "
        "summary is mechanical.\n\n"
        "**Themes (1):**\n"
        f"- _Today's leads_ — fallback summary; see full edition for context.\n"
        f"{head_block}\n\n"
        "**Sector lean:**\n"
        "- Neutral: all\n\n"
        "**Watchlist (top tickers across regions):** SPY, QQQ, IWM, TLT, HSBA.L\n\n"
        "**Key data / events to watch today:** see daily news edition.\n"
    )
