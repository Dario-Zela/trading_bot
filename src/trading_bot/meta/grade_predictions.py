"""Prediction grading agent — runs daily, scores open predictions whose
target_date has passed.

For each due prediction, we:
1. Gather light cross-asset context (yfinance prices for major indices,
   FX, commodities, sectors) so the LLM has something to score against.
2. Ask Claude (Haiku, parallel) for a verdict: proven / partial /
   falsified / still-open + a one-sentence note explaining the call.
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
    """Ask Claude to score one prediction. Returns (status, note)."""
    prompt = _build_grading_prompt(p, snapshot, today)
    response = run_claude_for_json(prompt, model="haiku")
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

## Current market context

Recent prices and percent changes for a baseline cross-asset set:

```json
{snapshot_json}
```

## Your task

Score the prediction against the falsification criteria using the
context above. Be strict and honest.

- **proven** — the claim is clearly correct AND the falsifier is clearly NOT met.
- **falsified** — the falsifier is clearly met (the claim is wrong).
- **partial** — the direction was right but the magnitude or timing was off, OR the
  claim was right on one dimension but not another.
- **still-open** — you cannot resolve from the data above (e.g., a specific data
  release that hasn't been published, a date that hasn't quite passed yet).

Output JSON only:

```json
{{
  "status": "proven" | "partial" | "falsified" | "still-open",
  "note": "<one sentence — what specifically you used to score it>"
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
