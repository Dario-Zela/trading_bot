"""Prediction grading agent — runs daily, scores open predictions whose
target_date has passed.

For each due prediction, we:
1. Hand the grader WebSearch + WebFetch so it can look up the specific
   data point the claim hinges on — a Treasury yield close, a CPI
   print, an FX cross, an earnings number, whatever. The ETF snapshot
   is still passed as quick context, but for sharply-thresholded
   claims the grader is told to verify with a web search.
2. Ask Claude (Haiku × N parallel) for a verdict: proven / partial /
   falsified / still-open + a one-sentence note citing the source.
3. Mutate the prediction's row with the new status + graded_at + note.

The grader is conservative on "proven": it requires the falsifier to be
clearly *not* satisfied. Anything ambiguous defaults to 'partial' or
'still-open'. Falsified status is the strict reciprocal — the falsifier
must be clearly satisfied. This keeps the "marking the homework" page
honest rather than self-congratulatory.
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, datetime, timezone

from trading_bot.llm.claude_code import ClaudeCodeError, run_claude_for_json
from trading_bot.state.predictions_log import (
    Prediction,
    mark_prediction_graded,
    open_predictions_due_by,
)
from trading_bot.tools import get_history


log = logging.getLogger(__name__)


# Cross-asset baseline used as context for every grader call. Wide enough
# to cover most predictions; specific tickers in the falsifier can be
# fetched lazily by the grader as needed.
_CONTEXT_TICKERS = (
    "SPY", "QQQ", "IWM",         # broad US
    "DIA",                       # Dow proxy
    "TLT", "IEF", "SHY",         # rates
    "HYG", "LQD",                # credit
    "UUP",                       # dollar proxy ETF
    "GLD", "SLV", "USO", "UNG",  # commodities
    "XLE", "XLF", "XLK", "XLV",  # major sectors
    "KRE",                       # regional banks
    "EZU",                       # Europe
    "EWJ",                       # Japan
    "EEM",                       # emerging markets
)

_MAX_PARALLEL = 6
_GRADE_TIMEOUT = 360                                  # web research takes time
_GRADER_TOOLS = ["--allowedTools", "WebSearch,WebFetch"]


def run_daily_grading(today: date) -> dict:
    """Score every open prediction whose target_date has passed.
    Returns a summary dict for logging."""
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        log.error("CLAUDE_CODE_OAUTH_TOKEN not set — skipping prediction grading")
        return {"skipped": True, "reason": "no oauth token"}

    due = open_predictions_due_by(today)
    if not due:
        log.info("No predictions due for grading today (%s)", today.isoformat())
        return {"graded": 0, "skipped_no_due": True}

    log.info("Grading %d predictions due on or before %s", len(due), today.isoformat())

    context_snapshot = _build_context_snapshot(today)

    results = {"proven": 0, "partial": 0, "falsified": 0, "still-open": 0, "errored": 0}

    with ThreadPoolExecutor(max_workers=_MAX_PARALLEL) as pool:
        futures = {
            pool.submit(_grade_one, p, context_snapshot, today): p for p in due
        }
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                verdict, note = fut.result()
            except Exception as e:
                log.warning("Grading failed for %s: %s", p.id, e)
                results["errored"] += 1
                continue
            if verdict in results:
                results[verdict] += 1
            mark_prediction_graded(
                prediction_id=p.id,
                source=p.source,
                new_status=verdict,
                note=note,
            )
            log.info("Graded %s [%s]: %s — %s", p.id, p.source, verdict, note[:80])

    summary = {"graded": sum(v for k, v in results.items() if k != "errored"), **results}
    log.info("Grading complete: %s", summary)
    return summary


def _build_context_snapshot(today: date) -> dict:
    """Compact cross-asset snapshot the LLM uses to score each prediction."""
    snap: dict[str, dict] = {}
    try:
        history = get_history(list(_CONTEXT_TICKERS), lookback_days=22, end_date=today)
    except Exception as e:
        log.warning("Context history fetch failed: %s — grading without snapshot", e)
        return snap
    for ticker, bars in history.items():
        if not bars or len(bars) < 2:
            continue
        latest = bars[-1]
        prev_day = bars[-2] if len(bars) >= 2 else None
        wk_ago = bars[-6] if len(bars) >= 6 else None
        mo_ago = bars[0] if len(bars) >= 20 else None
        snap[ticker] = {
            "as_of": str(latest.bar_date),
            "close": round(latest.close, 4),
            "day_pct": round((latest.close / prev_day.close - 1.0) * 100.0, 3) if prev_day else None,
            "week_pct": round((latest.close / wk_ago.close - 1.0) * 100.0, 3) if wk_ago else None,
            "month_pct": round((latest.close / mo_ago.close - 1.0) * 100.0, 3) if mo_ago else None,
        }
    return snap


def _grade_one(p: Prediction, snapshot: dict, today: date) -> tuple[str, str]:
    """Ask Claude to score one prediction. Returns (status, note).
    The grader runs with WebSearch + WebFetch enabled so it can verify
    sharply-thresholded claims (yields, FX rates, CPI prints, etc.)."""
    prompt = _build_grading_prompt(p, snapshot, today)
    response = run_claude_for_json(
        prompt,
        model="haiku",
        timeout_seconds=_GRADE_TIMEOUT,
        extra_args=_GRADER_TOOLS,
    )
    if not isinstance(response, dict):
        return "still-open", "Grader response was not a JSON object"
    raw_status = (response.get("status") or "").strip().lower()
    note = (response.get("note") or "").strip()[:400]
    if raw_status not in {"proven", "partial", "falsified", "still-open"}:
        return "still-open", f"Grader returned unrecognised status '{raw_status}'"
    return raw_status, note or "(no note)"


def _build_grading_prompt(p: Prediction, snapshot: dict, today: date) -> str:
    snapshot_json = json.dumps(snapshot, indent=2)
    return f"""You are the prediction grading agent for an algorithmic trading
bot. Today is {today.isoformat()}. You are scoring exactly one prediction
that was made earlier and whose target date has now passed.

## The prediction

- **Claim**: {p.claim}
- **Made on**: {p.made_at[:10]}
- **Horizon**: {p.horizon}
- **Target date**: {p.target_date}
- **Falsification criteria**: {p.falsification_criteria}
- **Conviction at the time**: {p.conviction}
- **Source**: {p.source} · section: {p.source_section or '—'}

## ETF snapshot (use as quick proxy only)

```json
{snapshot_json}
```

The ETF snapshot is a baseline — useful when the claim is about an
equity index, sector, or commodity ETF that's in the list. For
*anything else* (Treasury yields, FX crosses, specific stocks not in
the snapshot, economic data prints, individual earnings, regulatory
events) you must look up the real number yourself.

## You have these tools — use them

- **WebSearch** — find the specific data point the falsification
  criteria hinges on. Examples:
  - "10 year Treasury yield close {p.target_date}"
  - "WTI crude close {p.target_date}"
  - "GBP/USD close {p.target_date}"
  - "Nvidia earnings Q1 2026 revenue"
  - "BoE decision {p.target_date}"
- **WebFetch** — pull a specific URL when the search points at the
  right source (FRED, CNBC, Reuters, FT, Yahoo Finance, etc.) and you
  need the precise close / print.

You SHOULD use WebSearch first for any claim with a sharp numeric
threshold. Don't guess from a related ETF; find the actual number.

## Your task

Score the prediction against the falsification criteria using both
the ETF snapshot and (where needed) what you find on the web.

- **proven** — the claim is clearly correct AND the falsifier is clearly NOT met.
- **falsified** — the falsifier is clearly met (the claim is wrong).
- **partial** — the direction was right but the magnitude or timing was off, OR the
  claim was right on one dimension but not another.
- **still-open** — the data genuinely cannot be resolved yet (e.g., a
  print that hasn't been published, settlement that hasn't happened).
  This is for genuine unavailability — NOT for "I didn't look hard
  enough". If you can find the number with a web search, find it.

## Note formatting

Your note MUST cite the specific number(s) you used and the source.
Examples of the right register:

- "10y yield closed at 4.62% on May 19 per CNBC; threshold 4.55% was
  cleared — proven."
- "WTI settled $98.45 on May 23 per Reuters; falsifier requires
  ≥$103 — falsified."
- "Nvidia Q1 revenue $44.06B per company release; threshold $43B
  cleared; stock closed +6.2% — proven."

A note without a specific cited number is incomplete.

## Required output

```json
{{
  "status": "proven" | "partial" | "falsified" | "still-open",
  "note": "<one sentence — cite the specific number(s) and source>"
}}
```

Default to 'partial' or 'still-open' when uncertain. 'Proven' requires
that the claim AND the falsifier both clearly support the verdict.
'Falsified' requires the falsifier to be clearly satisfied. Do not
manufacture wins.
"""


def grade_predictions_cli(on_date: date) -> None:
    """CLI entry point used by `pipeline.py grade-predictions`."""
    summary = run_daily_grading(on_date)
    log.info("Daily grading summary: %s", summary)
