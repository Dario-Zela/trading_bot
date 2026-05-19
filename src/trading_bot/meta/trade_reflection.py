"""Phase 8E — real per-trade LLM reflection.

After each exit, run a single Haiku call per closed trade that takes
in the pre-trade thesis, the day's price action (high/low/close), any
news for the ticker that day, the realised P&L, and the exit reason,
and returns:
- `outcome_notes` — what actually happened, in 1-2 sentences
- `risks_observed` — what the strategy missed or what to watch next time

These overwrite the templated text the executors currently emit. The
evolution agent reads `outcome_notes` / `risks_observed` per trade and
will get much richer "Lessons" content as a result.

Fan-out: max 6 in parallel (matches the existing N-trade-per-day
volume per region; sequential would slow each exit pass by ~30-60s).
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Any

from trading_bot.llm.claude_code import ClaudeCodeError, run_claude_for_json

log = logging.getLogger(__name__)


_MAX_PARALLEL = 6
_TIMEOUT = 180


def reflect_on_exit(
    trade: dict,
    *,
    day_bar: dict[str, float] | None = None,
    recent_news: list[dict] | None = None,
) -> tuple[str, str]:
    """One Haiku call. Returns (outcome_notes, risks_observed).

    `day_bar` is a {open, high, low, close, volume} dict for the exit
    date if available (executors pass it through; shadow has it
    directly from yfinance, paper executors don't always).

    `recent_news` is a list of {timestamp, headline, summary} for the
    same ticker on or just before the exit date.

    On any failure falls back to a single-line templated stand-in.
    """
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return _templated_fallback(trade)
    prompt = _build_prompt(trade, day_bar, recent_news)
    try:
        response = run_claude_for_json(prompt, model="haiku", timeout_seconds=_TIMEOUT)
    except ClaudeCodeError as e:
        log.warning("Reflection LLM call failed for %s: %s", trade.get("ticker"), e)
        return _templated_fallback(trade)
    if not isinstance(response, dict):
        return _templated_fallback(trade)
    outcome = str(response.get("outcome_notes") or "").strip()
    risks = str(response.get("risks_observed") or "").strip()
    if not outcome:
        return _templated_fallback(trade)
    return outcome[:600], risks[:600]


def reflect_batch(
    trades: list[dict],
    *,
    day_bars: dict[str, dict] | None = None,
    news_by_ticker: dict[str, list] | None = None,
) -> dict[str, tuple[str, str]]:
    """Reflect on a batch of exits in parallel. Returns trade_id →
    (outcome, risks). Trades not represented mean their reflection
    call failed completely (caller falls back to templated text)."""
    if not trades:
        return {}
    out: dict[str, tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=_MAX_PARALLEL) as pool:
        futures = {
            pool.submit(
                reflect_on_exit,
                t,
                day_bar=(day_bars or {}).get(t.get("ticker")),
                recent_news=(news_by_ticker or {}).get(t.get("ticker")),
            ): t
            for t in trades
        }
        for fut in as_completed(futures):
            t = futures[fut]
            tid = t.get("trade_id")
            if not tid:
                continue        # nothing to key under
            try:
                out[tid] = fut.result()
            except Exception as e:
                log.warning("Reflection failed for %s: %s", t.get("ticker"), e)
    return out


def _build_prompt(trade: dict, day_bar: dict | None, news: list[dict] | None) -> str:
    ticker = trade.get("ticker", "?")
    thesis = trade.get("thesis") or "(no pre-trade thesis recorded)"
    entry_price = trade.get("entry_price", 0)
    exit_price = trade.get("exit_price", 0)
    pnl_gbp = trade.get("pnl_gbp", 0)
    pnl_pct = trade.get("pnl_pct", 0)
    exit_reason = trade.get("exit_reason", "scheduled")
    fees_gbp = trade.get("fees_gbp", 0) or 0
    strategy_id = trade.get("strategy_id", "?")
    region = trade.get("region", "?")

    bar_block = ""
    if day_bar:
        bar_block = (
            f"- Day bar: open {day_bar.get('open', '—')}, "
            f"high {day_bar.get('high', '—')}, low {day_bar.get('low', '—')}, "
            f"close {day_bar.get('close', '—')}, vol {day_bar.get('volume', '—')}"
        )

    news_block = ""
    if news:
        bullets = []
        for n in news[:5]:
            ts = (n.get("timestamp") or "")[:10] if isinstance(n, dict) else ""
            head = n.get("headline", "") if isinstance(n, dict) else ""
            summ = (n.get("summary") or "")[:150] if isinstance(n, dict) else ""
            line = f"  - [{ts}] {head}"
            if summ:
                line += f" — {summ}"
            bullets.append(line)
        if bullets:
            news_block = "\n## Recent news for " + ticker + "\n" + "\n".join(bullets)

    return f"""You are the trading bot's post-trade reflection agent.
One trade just closed. Read the pre-trade thesis, the realised
outcome, and any news for the ticker today, then return two short
strings: what actually happened, and what risks were exposed for the
strategy.

This is honest self-criticism: surface the thesis-vs-reality gap so
the weekly evolution agent has something concrete to learn from.

## The trade

- **Strategy:** {strategy_id} · region {region}
- **Ticker:** {ticker}
- **Entry → exit:** {entry_price} → {exit_price}
- **P&L (net of fees):** £{pnl_gbp:+,.2f} ({pnl_pct:+.2f}%); fees £{fees_gbp:,.2f}
- **Exit reason:** {exit_reason}
{bar_block}

## Pre-trade thesis (what the strategy said before entering)

> {thesis}
{news_block}

## What we need

Two strings, both brief, both grounded in the data above:

1. **outcome_notes** (≤2 sentences) — describe what actually happened
   intraday. Connect the price action back to the thesis: did it hold,
   did it reverse, was it a coin flip? Cite concrete numbers (high/low
   reach, news catalyst if present, exit reason).

2. **risks_observed** (1-2 sentences) — what does this trade reveal
   about the strategy's risk profile that the evolution agent should
   know? Examples: "stops were too tight given the day's ATR", "thesis
   ignored an earnings catalyst the next morning", "filled at open and
   ran straight against us for 90 minutes", "thesis held but the win
   was eaten by fees on a low-conviction USD trade".

If the trade closed flat with no real signal either way, say so. Do
not manufacture insight where there is none — "exit matched entry
exactly, no intraday movement to read into" is a valid outcome_notes.

## Required output

```json
{{
  "outcome_notes": "<one or two sentences>",
  "risks_observed": "<one or two sentences>"
}}
```
"""


def _templated_fallback(trade: dict) -> tuple[str, str]:
    """When the LLM is unavailable, leave a marker so we can see that
    reflection didn't run on this trade. The evolution agent's prompt
    can de-weight rows with this marker."""
    pnl_pct = float(trade.get("pnl_pct") or 0)
    outcome = (
        f"Closed at {pnl_pct:+.2f}% via {trade.get('exit_reason', 'scheduled')} exit. "
        f"(No LLM reflection — fallback text.)"
    )
    risks = "(reflection agent did not run on this trade)"
    return outcome, risks
