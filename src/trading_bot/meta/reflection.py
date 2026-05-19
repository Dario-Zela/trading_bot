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
from trading_bot.state.paths import ledger_path, predictions_path
from trading_bot.tools.history import get_history


log = logging.getLogger(__name__)


def grade_predictions(on_date: date, region: str | None = None) -> int:
    """Fill in actual_return_pct + actual_class on every prediction recorded
    today (open→close return). Cheap — one yfinance batch fetch per region.

    Independent of the LLM-based reflect_on_day; runs even when
    CLAUDE_CODE_OAUTH_TOKEN is missing.
    """
    target = on_date.isoformat()
    path = predictions_path()
    if not path.exists():
        return 0

    rows = []
    tickers_to_fetch: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append(r)
        if r.get("prediction_date") == target and r.get("actual_return_pct") is None:
            if region is None or r.get("region") == region:
                tickers_to_fetch.add(r["ticker"])

    if not tickers_to_fetch:
        return 0

    bars = get_history(list(tickers_to_fetch), lookback_days=1, end_date=on_date)
    actuals: dict[str, float | None] = {}
    for ticker, bar_list in bars.items():
        if not bar_list:
            continue
        b = bar_list[-1]
        if b.open and b.open > 0:
            actuals[ticker] = (b.close / b.open - 1.0) * 100.0

    updated = 0
    for r in rows:
        if r.get("prediction_date") != target:
            continue
        if r.get("actual_return_pct") is not None:
            continue
        if region is not None and r.get("region") != region:
            continue
        actual = actuals.get(r["ticker"])
        if actual is None:
            continue
        r["actual_return_pct"] = round(actual, 2)
        r["actual_class"] = _classify_outcome(actual)
        updated += 1

    if updated == 0:
        return 0

    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)
    return updated


def _classify_outcome(actual_pct: float) -> str:
    if actual_pct >= 4.0:
        return "strong_up"
    if actual_pct >= 1.0:
        return "mild_up"
    if actual_pct <= -4.0:
        return "strong_down"
    if actual_pct <= -1.0:
        return "mild_down"
    return "flat"


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


def reflect_predictions_on_day(on_date: date, region: str | None = None) -> int:
    """Add a one-sentence `reflection` to each *untraded* prediction made
    on `on_date`, explaining why the predicted vs actual class diverged
    (or agreed). Pre-computing this saves the weekly evolution agent
    from re-deriving the same analysis across hundreds of rows.

    Only operates on rows that already have `actual_class` set (i.e.
    grade_predictions has run for the day). Traded predictions get
    their reflection from the corresponding trade's `outcome_notes`
    in the ledger and are intentionally skipped here.

    Returns the number of prediction rows updated.
    """
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        log.warning("Prediction reflection skipped: CLAUDE_CODE_OAUTH_TOKEN not set")
        return 0

    target = on_date.isoformat()
    path = predictions_path()
    if not path.exists():
        return 0

    rows: list[dict] = []
    by_strategy: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append(r)
        if r.get("prediction_date") != target:
            continue
        if region is not None and r.get("region") != region:
            continue
        if r.get("was_traded"):
            continue                         # ledger reflection covers it
        if r.get("actual_class") is None:
            continue                         # not graded yet
        if (r.get("reflection") or "").strip():
            continue                         # already reflected
        by_strategy.setdefault(r["strategy_id"], []).append(r)

    if not by_strategy:
        log.info("Prediction reflection: no eligible rows on %s", target)
        return 0

    notes_by_strategy: dict[str, dict[str, str]] = {}
    for sid, preds in by_strategy.items():
        try:
            notes_by_strategy[sid] = _reflect_predictions_strategy(sid, preds, on_date)
        except ClaudeCodeError as e:
            log.error("Prediction reflection failed for %s: %s", sid, e)
            continue

    if not notes_by_strategy:
        return 0

    # Apply: rows keyed by (strategy_id, ticker) — predictions don't
    # carry a stable trade_id so we match on (sid, ticker) within the
    # day, which is unique by construction.
    apply_index: dict[tuple[str, str], str] = {}
    for sid, by_ticker in notes_by_strategy.items():
        for ticker, refl in by_ticker.items():
            if refl:
                apply_index[(sid, str(ticker).upper())] = refl

    updated = 0
    for r in rows:
        if r.get("prediction_date") != target:
            continue
        key = (r.get("strategy_id"), str(r.get("ticker") or "").upper())
        if key not in apply_index:
            continue
        r["reflection"] = apply_index[key].strip()
        updated += 1

    if updated == 0:
        return 0

    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)
    return updated


def _reflect_predictions_strategy(
    strategy_id: str, predictions: list[dict], on_date: date,
) -> dict[str, str]:
    """One Sonnet call per strategy. Returns {ticker_upper: reflection}.
    The reflection is one short sentence pointing at why the predicted
    class hit or missed — the kind of line the evolution agent would
    otherwise have to derive from raw numbers."""
    cards = []
    for p in predictions:
        pred_class = p.get("predicted_class") or "?"
        actual_class = p.get("actual_class") or "?"
        pred_pct = p.get("predicted_return_pct")
        actual_pct = p.get("actual_return_pct")
        conviction = p.get("conviction")
        rationale = (p.get("rationale") or "").strip()
        pred_s = f"{pred_pct:+.2f}%" if isinstance(pred_pct, (int, float)) else "?"
        actual_s = f"{actual_pct:+.2f}%" if isinstance(actual_pct, (int, float)) else "?"
        conv_s = f"{conviction:.2f}" if isinstance(conviction, (int, float)) else "?"
        cards.append(
            f"### {p.get('ticker')}\n"
            f"- predicted: {pred_class} ({pred_s}), conviction {conv_s}\n"
            f"- actual:    {actual_class} ({actual_s})\n"
            f"- rationale: {rationale or '(none)'}"
        )

    prompt = (
        f"You're annotating the `{strategy_id}` strategy's untraded "
        f"predictions for {on_date.isoformat()}. The strategy scored "
        f"every candidate but only traded the high-conviction subset. "
        f"For each prediction below, write ONE short sentence (≤140 "
        f"chars) connecting the rationale to the realised outcome. "
        f"This pre-computes the analysis the weekly evolution agent "
        f"would otherwise have to derive from scratch — be terse and "
        f"specific, not narrative.\n\n"
        f"Guidance for the sentence:\n"
        f"- If the predicted class matched the actual class: name the "
        f"  signal that worked (e.g. 'RSI 28 mean-reversion played out').\n"
        f"- If they diverged: name what the rationale assumed vs what "
        f"  happened (e.g. 'expected continuation; flatlined on low vol').\n"
        f"- Avoid restating the numbers — they're already shown.\n\n"
        f"## Predictions\n\n" + "\n\n".join(cards) + "\n\n"
        f"## Required output\n\n"
        f"JSON object keyed by ticker symbol (uppercase):\n\n"
        f"```json\n"
        f"{{\n"
        f"  \"TICKER\": \"one-sentence reflection\"\n"
        f"}}\n"
        f"```\n\n"
        f"Use the exact ticker strings from the cards above."
    )

    response = run_claude_for_json(prompt, model="sonnet")
    if not isinstance(response, dict):
        raise ClaudeCodeError(
            f"Prediction reflection response was not a JSON object — "
            f"got {type(response).__name__}"
        )
    # Normalise keys to upper-case + strip non-string values
    out: dict[str, str] = {}
    for k, v in response.items():
        if isinstance(v, str) and v.strip():
            out[str(k).upper()] = v.strip()
    return out


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
