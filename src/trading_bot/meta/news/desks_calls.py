"""Phase 2H — The desk's calls (predictions + marking the homework).

Two outputs:

1. **Fresh predictions** — Sonnet writes 4-6 falsifiable calls across
   tomorrow / this-week / this-month. Each prediction has a claim, a
   target_date, a falsification_criteria sentence, and a conviction.
   Each is persisted via `append_prediction()` so the prediction
   grader can score it later.

2. **Marking the homework** — render-only block. Pulls the most
   recently graded news predictions from `read_predictions()` and
   surfaces them with their verdict. No LLM needed at this stage —
   the grader already wrote the grading_note.

This is a heavy single-LLM stage rather than a per-prediction fan-out
because the predictions need to *cohere*. Asking one model to look
across the whole edition and emit 4-6 internally-consistent calls is
better than parallelising and getting six versions of "Fed will cut
in December."
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from trading_bot.llm.claude_code import ClaudeCodeError, run_claude_for_json
from trading_bot.meta.news.publisher import NewsPlan
from trading_bot.meta.news.triage import TriagedCandidate
from trading_bot.state.predictions_log import (
    Prediction,
    append_prediction,
    read_predictions,
)

log = logging.getLogger(__name__)

_DESK_CALLS_TIMEOUT = 240
_MAX_HOMEWORK_ITEMS = 8


@dataclass
class DesksCalls:
    """The desks' calls section content for one edition."""
    fresh_predictions: list[Prediction] = field(default_factory=list)
    homework_items: list[Prediction] = field(default_factory=list)   # graded recently


def build_desks_calls(plan: NewsPlan, triaged: list[TriagedCandidate], today: date) -> DesksCalls:
    """Generate fresh predictions + pull recent graded homework. New
    predictions are persisted to `state/predictions/news.jsonl` so the
    grader will pick them up at their target_date."""
    fresh = _generate_predictions(plan, triaged, today)
    homework = _gather_homework()
    return DesksCalls(fresh_predictions=fresh, homework_items=homework)


def _generate_predictions(plan: NewsPlan, triaged: list[TriagedCandidate], today: date) -> list[Prediction]:
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        log.error("CLAUDE_CODE_OAUTH_TOKEN not set — no fresh predictions written")
        return []
    if not plan.pieces:
        return []

    prompt = _build_prompt(plan, triaged, today)
    try:
        response = run_claude_for_json(prompt, model="sonnet", timeout_seconds=_DESK_CALLS_TIMEOUT)
    except ClaudeCodeError as e:
        log.warning("Desk's calls Sonnet failed: %s — no fresh predictions", e)
        return []

    predictions = _parse_predictions(response, today)
    # Persist each so the grader can later score them
    for p in predictions:
        try:
            append_prediction(p)
        except Exception as e:
            log.warning("Failed to append prediction %s: %s", p.id, e)
    log.info("Desk's calls: %d predictions persisted to news.jsonl", len(predictions))
    return predictions


def _build_prompt(plan: NewsPlan, triaged: list[TriagedCandidate], today: date) -> str:
    edition_block = "\n".join(
        f"  - [{p.section}] {p.headline} — {p.one_line}"
        for p in plan.pieces[:18]
    ) or "  (empty edition)"

    tomorrow = today + timedelta(days=1)
    week_end = today + timedelta(days=7)
    month_end = today + timedelta(days=30)

    return f"""You are the editorial board of The Bot Tribune. Today is
{today.isoformat()}. Write 4-6 falsifiable predictions to run in the
day's "Desk's calls" section.

A prediction is a *bet*, not a hedge. It is concrete enough that a
human grader looking at market data on the target date can say:
proven, partial, or falsified — without ambiguity.

## What's running in today's edition (the context you're working from)

{edition_block}

## What makes a good prediction

- **Concrete:** "10y yield closes below 4.25% on Friday" — yes.
  "Rates will probably drift lower" — no.
- **Time-bounded:** every call has a single target_date. You may
  spread across three horizons (tomorrow / this-week / this-month).
- **Falsifiable:** state EXACTLY what would prove the call wrong.
  "S&P down ≥1% by close" is falsifiable. "Markets pessimistic" is not.
- **Asymmetric:** prefer calls where the contrarian case is the
  default-priced outcome. The fun is in being non-obvious.
- **Honest conviction:** mark conviction = low / medium / high. A
  high-conviction call should be defensible from public information
  you can read today.

## Distribution

Aim for a mix:
- 1-2 **tomorrow** (target_date = {tomorrow.isoformat()})
- 2-3 **this-week** (target_date ≤ {week_end.isoformat()})
- 1-2 **this-month** (target_date ≤ {month_end.isoformat()})

It's fine to lean toward fewer, higher-quality calls. Do not pad.

## Required output

Return JSON only:

```json
{{
  "predictions": [
    {{
      "claim": "<the call in plain English, ≤180 chars>",
      "horizon": "tomorrow" | "this-week" | "this-month",
      "target_date": "YYYY-MM-DD",
      "falsification_criteria": "<one sentence saying exactly what disproves the call>",
      "conviction": "low" | "medium" | "high",
      "rationale": "<1-2 sentence reasoning the article will show — not stored as a prediction field, just for the page>"
    }}
  ]
}}
```

Do not address the reader, do not say "we predict", do not hedge.
A call is a call.
"""


def _parse_predictions(response: dict | list, today: date) -> list[Prediction]:
    if isinstance(response, list):
        response = {"predictions": response}
    if not isinstance(response, dict):
        return []
    raw = response.get("predictions") or []
    if not isinstance(raw, list):
        return []

    out: list[Prediction] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for item in raw:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim", "")).strip()
        if not claim:
            continue
        horizon = str(item.get("horizon", "")).strip()
        if horizon not in {"tomorrow", "this-week", "this-month"}:
            # default the horizon if unrecognised
            horizon = "this-week"
        target = str(item.get("target_date", "")).strip()
        if not _looks_like_iso_date(target):
            # default the target if unrecognised
            target = _default_target_for(horizon, today)
        falsifier = str(item.get("falsification_criteria", "")).strip()
        if not falsifier:
            continue
        conviction = str(item.get("conviction", "medium")).strip().lower()
        if conviction not in {"low", "medium", "high"}:
            conviction = "medium"
        rationale = str(item.get("rationale", "")).strip()[:600]

        pred = Prediction(
            id=str(uuid.uuid4()),
            source="news",
            made_at=now_iso,
            claim=claim[:240],
            horizon=horizon,
            target_date=target,
            falsification_criteria=falsifier[:280],
            conviction=conviction,
            source_section="Desk's calls",
            source_slug=f"desks-calls-{today.isoformat()}",
            status="open",
            grading_note=rationale,   # stash rationale here for now — grader overwrites
        )
        out.append(pred)
    return out


def _gather_homework() -> list[Prediction]:
    """Most recently graded news predictions, newest first."""
    all_news = read_predictions(source="news")
    graded = [p for p in all_news if p.status in {"proven", "partial", "falsified", "still-open"} and p.graded_at]
    graded.sort(key=lambda p: p.graded_at or "", reverse=True)
    return graded[:_MAX_HOMEWORK_ITEMS]


def _looks_like_iso_date(s: str) -> bool:
    try:
        datetime.fromisoformat(s).date()
        return True
    except ValueError:
        return False


def _default_target_for(horizon: str, today: date) -> str:
    deltas = {"tomorrow": 1, "this-week": 7, "this-month": 30}
    return (today + timedelta(days=deltas.get(horizon, 7))).isoformat()
