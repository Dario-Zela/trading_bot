"""Wave 4 — weekly macro agent.

Fires Sunday evening (cron). For each weekly cycle:
1. Gather cross-asset data: yield curve, credit spreads, dollar index, sector
   strength, commodity prices.
2. Read the previous macro view + the predictions it made (predictions.jsonl).
3. Call Claude with everything — receive back a new view markdown + a JSON
   list of new predictions + grades for prior open predictions.
4. Write the new view to `state/macro/views/YYYY-W##.md`.
5. Append the new predictions to `state/macro/predictions.jsonl`.
6. Update grades on prior predictions; failed ones get a line in
   `state/macro/lessons.md`.

Daily strategies (macro-aligned, bond-cycle, commodity-momentum) pick up
the new view automatically via get_macro_view() — they always read the
latest file in macro/views/.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from trading_bot.llm.claude_code import ClaudeCodeError, run_claude
from trading_bot.state.paths import STATE_ROOT
from trading_bot.tools import (
    get_commodity_prices,
    get_credit_spreads,
    get_dollar_index,
    get_sector_strength,
    get_yield_curve,
)


log = logging.getLogger(__name__)


@dataclass
class MacroPrediction:
    """One falsifiable claim made by the macro agent."""

    prediction_id: str
    week: str  # ISO-week like "2026-W20"
    made_at: str  # ISO timestamp
    claim: str
    target_date: str  # ISO date — by when the prediction resolves
    falsification_criteria: str  # specific, measurable
    status: str = "open"  # "open" / "proven" / "falsified"
    graded_at: str | None = None
    grading_note: str = ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_weekly_macro(today: date) -> dict:
    """Run one full weekly cycle. Returns a small summary dict for logging."""
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        log.error("CLAUDE_CODE_OAUTH_TOKEN not set — skipping macro agent")
        return {"skipped": True, "reason": "no oauth token"}

    week_id = _iso_week(today)
    log.info("macro: starting weekly run for %s", week_id)

    prev_view, prev_predictions = _read_prior_state()
    snapshot = _gather_snapshot()

    prompt = _build_prompt(
        today=today,
        week_id=week_id,
        prev_view=prev_view,
        open_predictions=[p for p in prev_predictions if p.status == "open"],
        snapshot=snapshot,
    )

    try:
        result = run_claude(prompt, model="sonnet", retries=2)
    except ClaudeCodeError as e:
        log.error("macro: Claude call failed: %s", e)
        return {"error": str(e)}

    response_text = result.text
    view_md, new_predictions, grades = _parse_response(response_text, week_id)
    if not view_md:
        log.error("macro: response had no view markdown; aborting write")
        return {"error": "no view in response"}

    view_path = _write_view(week_id, view_md)
    n_new = _append_predictions(new_predictions)
    n_graded, n_falsified = _apply_grades(prev_predictions, grades)

    log.info(
        "macro: wrote %s, %d new predictions, %d graded (%d falsified)",
        view_path.name, n_new, n_graded, n_falsified,
    )

    # Phase 3 — also run the v2 multi-desk pipeline. Purely additive;
    # failure here doesn't affect the existing markdown view.
    v2_summary: dict | None = None
    try:
        from trading_bot.meta.macro_v2 import run_macro_v2
        v2_summary = run_macro_v2(today)
    except Exception as e:
        log.warning("Macro v2 render failed (non-fatal): %s", e)

    return {
        "week": week_id,
        "view_path": str(view_path),
        "new_predictions": n_new,
        "graded": n_graded,
        "falsified": n_falsified,
        "total_cost_usd": result.total_cost_usd,
        "v2": v2_summary,
    }


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------

def _gather_snapshot() -> dict:
    """Compact cross-asset summary the prompt can consume."""
    snapshot: dict = {}
    try:
        yc = get_yield_curve()
        snapshot["yield_curve"] = {
            "as_of": yc.as_of,
            "3M": yc.y3m, "5Y": yc.y5y, "10Y": yc.y10y, "30Y": yc.y30y,
            "spread_3m10y": yc.spread_3m10y,
        }
    except Exception as e:
        log.warning("macro: yield curve fetch failed: %s", e)

    try:
        cs = get_credit_spreads()
        snapshot["credit_spreads"] = {
            "as_of": cs.as_of,
            "hyg_5d_return_pct": cs.hyg_5d_return_pct,
            "lqd_5d_return_pct": cs.lqd_5d_return_pct,
            "hy_vs_ig_5d_diff": cs.hy_vs_ig_5d_diff,
        }
    except Exception as e:
        log.warning("macro: credit spreads fetch failed: %s", e)

    try:
        dxy = get_dollar_index()
        snapshot["dollar_index"] = {
            "as_of": dxy.as_of,
            "level": dxy.level,
            "return_5d_pct": dxy.return_5d_pct,
            "return_20d_pct": dxy.return_20d_pct,
        }
    except Exception as e:
        log.warning("macro: DXY fetch failed: %s", e)

    try:
        snapshot["sector_strength"] = [
            {
                "ticker": s.ticker, "label": s.label,
                "5d": s.return_5d_pct, "20d": s.return_20d_pct,
            }
            for s in get_sector_strength()
        ]
    except Exception as e:
        log.warning("macro: sector strength fetch failed: %s", e)

    try:
        snapshot["commodities"] = [
            {
                "name": c.name, "ticker": c.ticker, "close": c.close,
                "5d": c.return_5d_pct, "20d": c.return_20d_pct,
            }
            for c in get_commodity_prices()
        ]
    except Exception as e:
        log.warning("macro: commodities fetch failed: %s", e)

    return snapshot


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _build_prompt(
    *,
    today: date,
    week_id: str,
    prev_view: str,
    open_predictions: list[MacroPrediction],
    snapshot: dict,
) -> str:
    pred_section = ""
    if open_predictions:
        rows = []
        for p in open_predictions:
            rows.append(
                f"- **{p.prediction_id}** (made {p.made_at[:10]}, targets {p.target_date}):\n"
                f"  - Claim: {p.claim}\n"
                f"  - Falsification: {p.falsification_criteria}"
            )
        pred_section = "\n## Prior open predictions (please grade where you can)\n\n" + "\n\n".join(rows)
    else:
        pred_section = "\n## Prior open predictions\n\nNone yet."

    snapshot_json = json.dumps(snapshot, indent=2, default=str)

    return f"""You are the weekly macro agent for the trading bot. Your job: produce a
fresh top-down macro view that the daily trading strategies will consume next week.

Today is {today.isoformat()} (ISO week {week_id}). The previous week's view is
quoted below in full. After it you'll find any still-open predictions and a
JSON cross-asset snapshot from this evening.

## Previous view (for context — feel free to update, contradict, or refine)

{prev_view if prev_view else "(none — this is the first run)"}
{pred_section}

## Cross-asset snapshot (just fetched)

```json
{snapshot_json}
```

## Your output

Produce **three sections in order**:

### 1. New macro view (markdown)

A complete replacement for the previous view. Cover:

- **One-line thesis** — the single most important regime call
- **Cycle stance** — where we are, Fed posture, curve shape, credit
- **Sector view** — table with Tech/Comms/Industrials/Financials/Health/Discretionary/Energy/Staples/Utilities/RealEstate/Materials, each rated Bullish / Mildly Bullish / Neutral / Mildly Bearish / Bearish, with a one-line reason
- **Risks / what would change my mind** — 3–5 things that would invalidate the view

Use proper markdown. Daily strategies will read this verbatim.

### 2. New predictions (JSON)

Output a fenced code block tagged `predictions` with a JSON array. Make 3-5 falsifiable predictions for the coming weeks. Each:

```json
{{
  "claim": "concrete claim with numbers",
  "target_date": "YYYY-MM-DD",
  "falsification_criteria": "specific measurable condition that would prove this wrong"
}}
```

Predictions must be **testable** — a number with a date, not "I think things will go up." If we can't grade it next week, it doesn't belong here.

### 3. Grades for prior open predictions (JSON)

For each prior open prediction listed above, decide one of: **proven**, **falsified**, **still-open**. Use the falsification criteria as the bar. Output a fenced code block tagged `grades` with a JSON array:

```json
{{
  "prediction_id": "<id from above>",
  "status": "proven" | "falsified" | "still-open",
  "note": "1 sentence — what data you used to decide"
}}
```

If there are no prior open predictions, return `[]`.

Be terse, factual, and honest with the grading — falsified is fine and useful. Don't manufacture wins.
"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(text: str, week_id: str) -> tuple[str, list[MacroPrediction], list[dict]]:
    """Extract the three sections from Claude's response."""
    # 1. View markdown: everything before the first ```predictions block
    pred_block_match = re.search(r"```predictions\s*\n", text, re.IGNORECASE)
    view_md = text[: pred_block_match.start()].strip() if pred_block_match else text.strip()
    # Strip a leading "### 1. New macro view" type heading if present
    view_md = re.sub(r"^#{1,3}\s+1\.?\s*New macro view.*?\n", "", view_md, flags=re.IGNORECASE)
    view_md = view_md.strip()

    # 2. Predictions block
    new_predictions: list[MacroPrediction] = []
    pred_match = re.search(r"```(?:predictions)\s*\n(.*?)\n```", text, re.IGNORECASE | re.DOTALL)
    if pred_match:
        try:
            raw_preds = json.loads(pred_match.group(1).strip())
            if isinstance(raw_preds, list):
                now = datetime.now(timezone.utc).isoformat()
                for i, p in enumerate(raw_preds):
                    if not isinstance(p, dict):
                        continue
                    new_predictions.append(
                        MacroPrediction(
                            prediction_id=f"{week_id}-p{i+1:02d}",
                            week=week_id,
                            made_at=now,
                            claim=str(p.get("claim", "")).strip(),
                            target_date=str(p.get("target_date", "")).strip(),
                            falsification_criteria=str(p.get("falsification_criteria", "")).strip(),
                        )
                    )
        except json.JSONDecodeError as e:
            log.warning("macro: failed to parse predictions JSON: %s", e)

    # 3. Grades block
    grades: list[dict] = []
    grade_match = re.search(r"```(?:grades)\s*\n(.*?)\n```", text, re.IGNORECASE | re.DOTALL)
    if grade_match:
        try:
            raw_grades = json.loads(grade_match.group(1).strip())
            if isinstance(raw_grades, list):
                grades = [g for g in raw_grades if isinstance(g, dict)]
        except json.JSONDecodeError as e:
            log.warning("macro: failed to parse grades JSON: %s", e)

    return view_md, new_predictions, grades


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _views_dir() -> Path:
    p = STATE_ROOT / "macro" / "views"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _predictions_path() -> Path:
    p = STATE_ROOT / "macro"
    p.mkdir(parents=True, exist_ok=True)
    return p / "predictions.jsonl"


def _lessons_path() -> Path:
    p = STATE_ROOT / "macro"
    p.mkdir(parents=True, exist_ok=True)
    return p / "lessons.md"


def _iso_week(d: date) -> str:
    year, week, _ = d.isocalendar()
    return f"{year}-W{week:02d}"


def _read_prior_state() -> tuple[str, list[MacroPrediction]]:
    """Return (latest_view_markdown, all_known_predictions)."""
    views_dir = _views_dir()
    view_files = sorted(views_dir.glob("*.md"))
    latest_view = view_files[-1].read_text() if view_files else ""

    predictions: list[MacroPrediction] = []
    p = _predictions_path()
    if p.exists():
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                predictions.append(
                    MacroPrediction(
                        prediction_id=d["prediction_id"],
                        week=d["week"],
                        made_at=d["made_at"],
                        claim=d["claim"],
                        target_date=d["target_date"],
                        falsification_criteria=d["falsification_criteria"],
                        status=d.get("status", "open"),
                        graded_at=d.get("graded_at"),
                        grading_note=d.get("grading_note", ""),
                    )
                )
            except Exception as e:
                log.debug("Skipping malformed prediction row: %s", e)
    return latest_view, predictions


def _write_view(week_id: str, view_md: str) -> Path:
    path = _views_dir() / f"{week_id}.md"
    header = f"# Macro view — {week_id}\n\n*Auto-generated by the weekly macro agent on {date.today().isoformat()}.*\n\n---\n\n"
    path.write_text(header + view_md.strip() + "\n")
    return path


def _append_predictions(predictions: list[MacroPrediction]) -> int:
    if not predictions:
        return 0
    path = _predictions_path()
    with path.open("a") as f:
        for p in predictions:
            f.write(json.dumps(p.__dict__) + "\n")
    return len(predictions)


def _apply_grades(all_predictions: list[MacroPrediction], grades: list[dict]) -> tuple[int, int]:
    """Update statuses + append falsified to lessons.md. Returns (n_graded, n_falsified)."""
    if not grades or not all_predictions:
        return 0, 0

    by_id = {p.prediction_id: p for p in all_predictions}
    n_graded = 0
    falsified_entries: list[MacroPrediction] = []

    for g in grades:
        pid = g.get("prediction_id")
        status = g.get("status", "").strip().lower()
        if not pid or pid not in by_id:
            continue
        if status not in ("proven", "falsified", "still-open"):
            continue
        p = by_id[pid]
        if p.status != "open" and status != "still-open":
            continue
        if status == "still-open":
            continue
        p.status = status
        p.graded_at = datetime.now(timezone.utc).isoformat()
        p.grading_note = str(g.get("note", "")).strip()
        n_graded += 1
        if status == "falsified":
            falsified_entries.append(p)

    # Rewrite the predictions file with the updated statuses
    path = _predictions_path()
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for p in all_predictions:
            f.write(json.dumps(p.__dict__) + "\n")
    tmp.replace(path)

    if falsified_entries:
        lessons = _lessons_path()
        with lessons.open("a") as f:
            for p in falsified_entries:
                f.write(
                    f"\n## {p.prediction_id} — falsified {p.graded_at[:10]}\n\n"
                    f"- **Claim**: {p.claim}\n"
                    f"- **Falsification criteria**: {p.falsification_criteria}\n"
                    f"- **What happened**: {p.grading_note}\n"
                )

    return n_graded, len(falsified_entries)
