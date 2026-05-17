"""Wave 2c — daily reflection agent.

After the exit phase closes today's trades, this module asks Claude to
write per-trade post-mortems that replace the templated `outcome_notes`
and `risks_observed` from ShadowExecutor / AlpacaPaperExecutor with real
analysis. Updates the ledger in place.

Designed to run as the next step after `pipeline exit` (CI workflow wires
this in). Skips silently if CLAUDE_CODE_OAUTH_TOKEN is missing so the rest
of the run still completes.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date

from trading_bot.llm.claude_code import ClaudeCodeError, run_claude_for_json
from trading_bot.state.ledger import _iter_records  # noqa: WPS437 — internal but stable
from trading_bot.state.paths import ledger_path


log = logging.getLogger(__name__)


def reflect_on_day(on_date: date, region: str | None = None) -> int:
    """For each trade exited on `on_date` (optionally filtered by region),
    ask Claude to generate proper outcome_notes + risks_observed. Updates
    the ledger in place and returns the number of trades reflected on.

    Groups by strategy so we make one LLM call per strategy per day rather
    than one per trade — drastically cheaper and lets Claude see the full
    daily basket when reasoning about each pick.
    """
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        log.warning("Reflection skipped: CLAUDE_CODE_OAUTH_TOKEN not set")
        return 0

    target = on_date.isoformat()
    by_strategy: dict[str, list[dict]] = {}
    for record in _iter_records():
        if record.get("exit_date") != target:
            continue
        if region is not None and record.get("region") != region:
            continue
        # Only reflect on trades that actually traded; skip cancelled/cleared
        # placeholders that have no meaningful outcome to analyse.
        if record.get("exit_reason") in ("cancelled", "cleared"):
            continue
        by_strategy.setdefault(record["strategy_id"], []).append(record)

    if not by_strategy:
        log.info("Reflection: no eligible trades on %s", target)
        return 0

    total_updated = 0
    for strategy_id, trades in by_strategy.items():
        try:
            notes = _reflect_strategy(strategy_id, trades, on_date)
        except ClaudeCodeError as e:
            log.error("Reflection failed for %s: %s", strategy_id, e)
            continue

        updated = _apply_reflections(trades, notes)
        log.info("Reflection: %s — updated %d of %d trades", strategy_id, updated, len(trades))
        total_updated += updated

    return total_updated


# ---- internals -----------------------------------------------------------


def _reflect_strategy(strategy_id: str, trades: list[dict], on_date: date) -> dict[str, dict]:
    """Single LLM call summarising the day for one strategy.

    Returns {trade_id: {"outcome_notes": str, "risks_observed": str}}.
    """
    cards = []
    for t in trades:
        entry = float(t["entry_price"])
        exit_price = float(t.get("exit_price") or 0.0)
        pnl_pct = float(t.get("pnl_pct") or 0.0)
        pnl_gbp = float(t.get("pnl_gbp") or 0.0)
        reason = t.get("exit_reason") or "scheduled"
        cards.append(
            f"### {t['trade_id']}\n"
            f"- ticker: {t['ticker']}\n"
            f"- entry: ${entry:.2f}\n"
            f"- exit:  ${exit_price:.2f}\n"
            f"- P&L: £{pnl_gbp:+.2f} ({pnl_pct:+.2f}%)\n"
            f"- exit_reason: {reason}\n"
            f"- entry thesis: {t.get('thesis') or '(none)'}"
        )

    prompt = (
        f"You're writing post-trade reflections for the `{strategy_id}` strategy "
        f"on {on_date.isoformat()}. For each trade below, produce two short paragraphs:\n\n"
        f"1. **outcome_notes** — what actually happened and whether the entry thesis held. "
        f"Be specific about *why* the trade worked or didn't, citing the price action, exit "
        f"reason, and any obvious driver. 1–3 sentences.\n\n"
        f"2. **risks_observed** — what risks materialised (or could have, even if they didn't). "
        f"Concentration, correlation, sector exposure, missing safety nets, hindsight signals. "
        f"1–3 sentences. End-of-day reflection on the basket-level context counts.\n\n"
        f"Tone: factual, terse, useful for the next iteration of this strategy. "
        f"Don't restate the entry thesis or the P&L numbers — those are already shown elsewhere.\n\n"
        f"## Trades to reflect on\n\n" + "\n\n".join(cards) + "\n\n"
        f"## Required output\n\n"
        f"Return a JSON object keyed by trade_id, like:\n\n"
        f"```json\n"
        f"{{\n"
        f"  \"<trade_id>\": {{\n"
        f"    \"outcome_notes\": \"...\",\n"
        f"    \"risks_observed\": \"...\"\n"
        f"  }}\n"
        f"}}\n"
        f"```\n\n"
        f"Only include trades from the list above. Exact `trade_id` strings, please."
    )

    response = run_claude_for_json(prompt, model="sonnet")
    if not isinstance(response, dict):
        raise ClaudeCodeError(
            f"Reflection response was not a JSON object — got {type(response).__name__}"
        )
    return response


def _apply_reflections(trades: list[dict], notes: dict[str, dict]) -> int:
    """Rewrite the ledger to overlay LLM-generated outcome_notes/risks_observed
    on the matching trade rows. Returns number of rows updated."""
    by_id = {t["trade_id"]: t for t in trades}
    path = ledger_path()
    if not path.exists():
        return 0

    rows = list(_iter_records())
    updated = 0
    for row in rows:
        tid = row.get("trade_id")
        if tid not in by_id:
            continue
        reflection = notes.get(tid)
        if not isinstance(reflection, dict):
            continue
        outcome = reflection.get("outcome_notes")
        risks = reflection.get("risks_observed")
        if outcome:
            row["outcome_notes"] = str(outcome).strip()
        if risks:
            row["risks_observed"] = str(risks).strip()
        if outcome or risks:
            updated += 1

    if updated == 0:
        return 0

    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    tmp.replace(path)
    return updated
