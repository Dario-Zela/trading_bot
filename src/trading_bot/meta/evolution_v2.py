"""Phase 4 — Evolution restyle.

The existing `meta/evolution.py` agent decides actions (promote / demote
/ tune / spawn) and applies them. This module sits *on top*: after the
actions are applied, it generates the editorial layer of the page —
the "this week's read" intro plus per-strategy report cards in the
per-quadrant format from the mockup.

Stages:
1. **Editorial intro** — Sonnet writes a 200-word "this week's read"
   from the snapshot + applied actions.
2. **Per-strategy report** — Haiku × N parallel — for each strategy
   the agent writes a `{what_worked, what_didnt, lessons, going_forward, config_changes}` JSON.
3. **Render** — assembles the editorial + cards + decisions row into
   `docs/evolution.html`.

Pure render layer below the action engine. Safe to fail without
affecting the bot's behaviour.
"""
from __future__ import annotations

import html
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import markdown as md_lib

from trading_bot.llm.claude_code import ClaudeCodeError, run_claude_for_json

log = logging.getLogger(__name__)

_MAX_PARALLEL_REPORTS = 6
_REPORT_TIMEOUT = 240
_EDITORIAL_TIMEOUT = 240


@dataclass
class StrategyReport:
    """One strategy's report card."""
    strategy_id: str
    headline: str                          # the agent's own headline for this strategy this week
    what_worked: list[str] = field(default_factory=list)
    what_didnt: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    going_forward: list[str] = field(default_factory=list)
    config_changes: list[str] = field(default_factory=list)
    failed: bool = False


@dataclass
class EvolutionEdition:
    """The full structured edition for one weekly evolution run."""
    week_end: str
    editorial_md: str
    reports: list[StrategyReport]
    snapshot_rows: list[dict]
    action_log: list[dict]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_and_render_evolution(
    today: date,
    snapshot: list[dict],
    applied_actions: list[dict],
    docs_root: Path,
    shell_fn,
) -> Path:
    """Build the editorial layer and render the evolution page.

    `snapshot` is the per-(strategy, region) snapshot the existing
    evolution agent already produces (`_build_snapshot()` output).
    `applied_actions` is each ActionLog as a dict.
    `shell_fn` is `pages._shell` — accepted as a parameter to avoid
    cycle issues.
    """
    edition = _build_edition(today, snapshot, applied_actions)
    page = _render_page(edition, shell_fn=shell_fn)
    out_path = docs_root / "evolution.html"
    out_path.write_text(page)
    log.info("Rendered evolution edition → %s (%d strategies)", out_path, len(edition.reports))
    return out_path


# ---------------------------------------------------------------------------
# Edition build (LLM stages)
# ---------------------------------------------------------------------------

def _build_edition(today: date, snapshot: list[dict], applied_actions: list[dict]) -> EvolutionEdition:
    """Run the editorial + per-strategy report stages."""
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        log.error("CLAUDE_CODE_OAUTH_TOKEN not set — using fallback edition")
        return _fallback_edition(today, snapshot, applied_actions)

    # Group snapshot rows by strategy
    by_strategy: dict[str, list[dict]] = {}
    for row in snapshot:
        sid = row.get("id") or "(unknown)"
        by_strategy.setdefault(sid, []).append(row)

    actions_by_strategy: dict[str, list[dict]] = {}
    for a in applied_actions:
        sid = a.get("strategy_id") or "(unknown)"
        actions_by_strategy.setdefault(sid, []).append(a)

    # Phase 8G — pairwise daily-P&L correlation across strategies.
    # Flagged pairs go into the editorial intro and into each affected
    # strategy's report card.
    similarity_pairs = _compute_similarity_pairs(today, list(by_strategy.keys()))

    # 1. Editorial intro
    editorial_md = _write_editorial(today, snapshot, applied_actions, similarity_pairs)

    # 2. Per-strategy reports — parallel
    reports: list[StrategyReport] = []
    with ThreadPoolExecutor(max_workers=_MAX_PARALLEL_REPORTS) as pool:
        futures = {
            pool.submit(
                _write_strategy_report, sid, rows,
                actions_by_strategy.get(sid, []), today,
                _similar_pairs_for_strategy(sid, similarity_pairs),
            ): sid
            for sid, rows in by_strategy.items()
        }
        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                reports.append(fut.result())
            except Exception as e:
                log.warning("Strategy report failed for %s: %s — using fallback", sid, e)
                reports.append(_fallback_strategy_report(sid, by_strategy.get(sid, []), actions_by_strategy.get(sid, [])))

    # Stable ordering: by total P&L descending across regions
    def _total_pnl(sid: str) -> float:
        return sum(
            (r.get("metrics") or {}).get("total_pnl_gbp", 0.0) or 0.0
            for r in by_strategy.get(sid, [])
        )
    reports.sort(key=lambda r: _total_pnl(r.strategy_id), reverse=True)

    return EvolutionEdition(
        week_end=today.isoformat(),
        editorial_md=editorial_md,
        reports=reports,
        snapshot_rows=snapshot,
        action_log=applied_actions,
    )


def _write_editorial(today: date, snapshot: list[dict], applied_actions: list[dict],
                     similarity_pairs: list[tuple] | None = None) -> str:
    """Run the editorial-intro Sonnet call."""
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return _fallback_editorial(today, snapshot, applied_actions)
    prompt = _build_editorial_prompt(today, snapshot, applied_actions, similarity_pairs or [])
    try:
        response = run_claude_for_json(prompt, model="sonnet", timeout_seconds=_EDITORIAL_TIMEOUT)
    except ClaudeCodeError as e:
        log.warning("Editorial Sonnet failed: %s — using fallback intro", e)
        return _fallback_editorial(today, snapshot, applied_actions)
    if isinstance(response, dict):
        md = str(response.get("editorial_md") or response.get("body") or "").strip()
        if md:
            return md
    return _fallback_editorial(today, snapshot, applied_actions)


def _build_editorial_prompt(today: date, snapshot: list[dict], applied_actions: list[dict],
                            similarity_pairs: list[tuple]) -> str:
    snapshot_lines = []
    for r in snapshot[:30]:
        m = r.get("metrics") or {}
        snapshot_lines.append(
            f"- {r.get('id')}@{r.get('region')} [{r.get('tier')}]: "
            f"n={m.get('n_trades', 0)}, hit={(m.get('hit_rate') or 0)*100:.0f}%, "
            f"P&L £{(m.get('total_pnl_gbp') or 0):+,.2f}, IC={(m.get('ic') or 0):+.2f}"
            if m else f"- {r.get('id')}@{r.get('region')} [{r.get('tier')}]: (no trades this window)"
        )
    snapshot_block = "\n".join(snapshot_lines) or "(empty)"

    action_lines = []
    for a in applied_actions[:20]:
        scope = f"{a.get('strategy_id')}@{a.get('region')}" if a.get("region") else a.get("strategy_id")
        status = "applied" if a.get("applied") else "skipped"
        action_lines.append(f"- {scope} · {a.get('action')} ({status}) — {a.get('reason', '')[:200]}")
    action_block = "\n".join(action_lines) or "(no actions this week)"

    # Phase 10B — pull cross-cutting signals once for the intro
    from trading_bot.meta.evolution_inputs import (
        divergent_strategies, verdict_rates_by_source,
        halt_history_summary, regime_subtitles,
    )
    divergent = divergent_strategies(snapshot)
    verdict_rates = verdict_rates_by_source()
    halt_summary = halt_history_summary()
    regimes = regime_subtitles(days=7)

    divergence_block = ""
    if divergent:
        lines = [f"- **{d['sid']}**: {d['region_a']} {d['pct_a']:+.2f}% vs {d['region_b']} {d['pct_b']:+.2f}% (Δ {d['delta']:.2f}pp)"
                 for d in divergent[:6]]
        divergence_block = "\n## Strategies diverging across regions (>5pp absolute avg-P&L gap)\n\n" + "\n".join(lines)

    verdict_block = ""
    if verdict_rates:
        lines = []
        for source, c in verdict_rates.items():
            graded = c.get("graded", 0) or 0
            if not graded:
                continue
            proven_pct = c["proven"] / graded * 100
            partial_pct = c["partial"] / graded * 100
            fals_pct = c["falsified"] / graded * 100
            lines.append(f"- **{source}** ({graded} graded): proven {proven_pct:.0f}% · partial {partial_pct:.0f}% · falsified {fals_pct:.0f}%")
        if lines:
            verdict_block = "\n## Falsifiable-call grading (last 4 weeks)\n\n" + "\n".join(lines)

    halt_block = ""
    if halt_summary.get("n_halts", 0) > 0:
        halt_block = (
            f"\n## Kill-switch fired this window\n\n"
            f"{halt_summary['n_halts']} halt(s) over the last 14 days. Most recent reason: "
            f"_{halt_summary['most_recent_reason']}_"
        )

    regime_block = ""
    if regimes:
        lines = [f"- {r['date']}: {r['subtitle']}" for r in regimes]
        regime_block = "\n## Recent regime context (trailing 7 days, daily news subtitles)\n\n" + "\n".join(lines)

    sim_block = ""
    if similarity_pairs:
        lines = []
        for a, b, corr in similarity_pairs[:6]:
            lines.append(f"- **{a}** and **{b}**: daily-P&L correlation {corr:+.2f}")
        sim_block = (
            "\n## Strategies behaving alike (daily-P&L correlation > 0.85 over the window)\n\n"
            + "\n".join(lines)
            + "\n\nLean on this in the editorial: too-similar strategies are "
            "a sign the slate isn't diverse enough. Speculate on the cause "
            "(overlapping universes? same indicator? same news?) and note "
            "whether next week's evolution actions should differentiate."
        )

    return f"""You are the editor of The Bot Tribune writing the "this
week's read" intro for the weekly Evolution page. Week ending
{today.isoformat()}.

The Evolution page reports what each strategy did over the last 14
days and what the auto-evolution agent decided about each. Your
intro sits at the top — 180-260 words, prose, dry. It frames the
week and prepares the reader for the per-strategy report cards
below.

## 14-day snapshot (per strategy × region)

{snapshot_block}

## Auto-actions applied this week

{action_block}
{divergence_block}
{verdict_block}
{halt_block}
{regime_block}
{sim_block}

## Writing rules

- Lead with the *theme of the week*: did regions diverge, did one
  cohort dominate, was the week dull?
- Name 2-3 strategies by name — the winner and the most interesting
  loser, at minimum.
- Reference actions only if they tell a story (e.g., "the agent
  promoted X but demoted Y in the same week").
- No clichés, no hedging, no "we'll see".
- End with a forward-look for next week — what to watch.
- Markdown OK. Use one short H4 quote-pull (`#### "..."`) if it
  earns its keep; otherwise omit.

## Required output

Return JSON only:

```json
{{
  "editorial_md": "<the intro as markdown — no headline, no preamble>"
}}
```
"""


def _write_strategy_report(sid: str, rows: list[dict], actions: list[dict],
                           today: date, similar_pairs: list[tuple] | None = None) -> StrategyReport:
    """Run the per-strategy Haiku call."""
    prompt = _build_strategy_report_prompt(sid, rows, actions, today, similar_pairs or [])
    try:
        response = run_claude_for_json(prompt, model="haiku", timeout_seconds=_REPORT_TIMEOUT)
    except ClaudeCodeError as e:
        log.warning("Strategy report Haiku failed for %s: %s", sid, e)
        return _fallback_strategy_report(sid, rows, actions)
    return _parse_strategy_report(sid, response)


def _build_strategy_report_prompt(sid: str, rows: list[dict], actions: list[dict],
                                  today: date, similar_pairs: list[tuple] = ()) -> str:
    region_lines = []
    for r in rows:
        m = r.get("metrics") or {}
        if m:
            region_lines.append(
                f"- **{r.get('region')}** [{r.get('tier')}]: "
                f"n={m.get('n_trades', 0)}, hit_rate={(m.get('hit_rate') or 0)*100:.0f}%, "
                f"P&L £{(m.get('total_pnl_gbp') or 0):+,.2f}, "
                f"avg_pct={(m.get('avg_pnl_pct') or 0):+.2f}%, "
                f"max_dd={(m.get('max_drawdown_pct') or 0):+.1f}%, "
                f"IC={(m.get('ic') or 0):+.2f}"
            )
        else:
            region_lines.append(f"- **{r.get('region')}** [{r.get('tier')}]: (no trades this window)")
    regions_block = "\n".join(region_lines) or "(no regions configured)"

    # Phase 10C — split applied vs skipped instead of mixing them. The
    # agent was reading both as "what we did this week", but skipped
    # actions are exactly what we DIDN'T do.
    applied_lines: list[str] = []
    skipped_lines: list[str] = []
    for a in actions:
        scope = f"@{a.get('region')}" if a.get("region") else " (strategy-wide)"
        line = f"- `{a.get('action')}`{scope} — {a.get('reason', '')[:200]}"
        if a.get("applied"):
            applied_lines.append(line)
        else:
            skipped_lines.append(line)
    applied_block = "\n".join(applied_lines) or "- _no actions applied this week_"
    skipped_block = "\n".join(skipped_lines) or "- _no actions skipped this week_"
    actions_block = (
        "**Applied:**\n" + applied_block
        + "\n\n**Considered but skipped:**\n" + skipped_block
    )

    # Pull this strategy's misses from the trailing-week missed-movers
    # reports — concrete tickers + catalysts + reason hypotheses that
    # the Lessons quadrant can cite directly.
    missed_block = _strategy_missed_lines(sid)

    # Phase 8E now writes real per-trade outcome_notes + risks_observed
    # via Haiku — these are concrete failure / success modes the agent
    # should cite, not just aggregated metrics.
    notable_trades_block = _notable_trades_lines(sid)

    # Pre-baked reflections on the untraded predictions — the daily
    # reflect cron writes one sentence per prediction connecting the
    # rationale to the realised class. Surfacing the biggest gaps
    # here saves the evolution agent from re-deriving the same
    # rationalisation across 100s of rows.
    prediction_reflections_block = _strategy_prediction_reflections_lines(sid)

    # Phase 10B — derived signals
    from trading_bot.meta.evolution_inputs import (
        fees_pct_of_gross, cost_gate_drop_rate, earnings_gate_hit_rate,
        sector_concentration, trail_activation_rate, parent_deep_analysis,
        ic_noise_floor,
    )
    fee_pct = fees_pct_of_gross(sid)
    cost_drop = cost_gate_drop_rate(sid)
    earnings_hit = earnings_gate_hit_rate(sid)
    sectors = sector_concentration(sid)
    trail_rate = trail_activation_rate(sid)
    deep_md = parent_deep_analysis(sid)
    # Phase 11D / 12 — Monte Carlo noise floor for the strategy's IC.
    # Tells the agent whether the headline IC is "above noise" given
    # sample size, vs an artifact of small-N.
    noise = ic_noise_floor(sid)

    signals_lines = [
        f"- **Fees as share of gross P&L** (last 14d): "
        f"£{fee_pct['fees_gbp']:+,.2f} fees / £{fee_pct['gross_pnl_gbp']:+,.2f} gross "
        f"→ **{fee_pct['fees_pct_of_gross']:.0f}%** of gross eaten by fees ({fee_pct['n_trades']} trades)",
    ]
    if cost_drop["n_picks"]:
        signals_lines.append(
            f"- **Cost-gate drops**: {cost_drop['n_dropped']}/{cost_drop['n_picks']} "
            f"LLM picks dropped by gate ({cost_drop['drop_rate_pct']:.1f}%) — "
            f"the LLM is repeatedly choosing trades the edge can't cover"
        )
    if earnings_hit["n_runs"]:
        signals_lines.append(
            f"- **Earnings-gate hits**: {earnings_hit['candidates_dropped']}/{earnings_hit['candidates_total']} "
            f"candidates dropped pre-earnings over {earnings_hit['n_runs']} runs ({earnings_hit['drop_rate_pct']:.1f}%)"
        )
    if trail_rate["n_exits"]:
        signals_lines.append(
            f"- **Stop / trail rate**: {trail_rate['n_stops']}/{trail_rate['n_exits']} "
            f"exits via stop or trail ({trail_rate['stop_rate_pct']:.0f}%)"
        )
    if sectors:
        sector_str = ", ".join(f"{s['sector']} {s['pct']:.0f}%" for s in sectors[:4])
        signals_lines.append(f"- **Sector concentration** (by traded notional): {sector_str}")
    if noise.get("verdict") not in (None, "too_few"):
        signals_lines.append(
            f"- **IC noise-floor MC**: real IC {noise.get('real_ic', 0):+.3f} vs "
            f"q95 noise {noise.get('noise_floor', 0):+.3f} (n={noise.get('n', 0)}) "
            f"→ **{noise.get('verdict')}**. "
            f"'noise' means the IC could be lucky shuffles, not signal — be sceptical."
        )
    signals_block = "\n".join(signals_lines)

    deep_block = ""
    if deep_md.strip():
        deep_block = (
            "\n### This strategy's `deep_analysis.md` (its current bias)\n\n"
            f"```\n{deep_md}\n```\n"
        )

    sim_lines = []
    for other, corr in similar_pairs:
        sim_lines.append(f"- **{other}** — daily-P&L correlation {corr:+.2f}")
    similar_block = (
        "\n### Strategies this one is moving with\n\n"
        + ("\n".join(sim_lines) or "_(none — this strategy is uncorrelated with its peers)_")
        + "\n\nIf the correlations above are high, the Lessons quadrant "
        "should consider whether this strategy's edge is unique enough "
        "to keep in the slate, or whether it's effectively a copy of "
        "the listed peer(s)."
    )

    return f"""You are the desk reporter for The Bot Tribune's weekly
Evolution page. Week ending {today.isoformat()}. Write the report card
for the strategy below.

## Strategy: {sid}

### 14-day performance by region

{regions_block}

### Actions taken by the evolution agent this week

{actions_block}

### Derived signals (last 14 days)

{signals_block}

### Missed opportunities this week (from daily missed-movers analysis)

{missed_block}

### Notable per-trade reflections (LLM-written outcome + risk notes)

{notable_trades_block}

### Untraded-prediction reflections (one-line analysis per miss)

{prediction_reflections_block}
{similar_block}
{deep_block}

## What we need

A four-quadrant report:

1. **what_worked** — 2-4 short bullet points on what genuinely
   went right. Be specific (a thesis that held up, a name that
   contributed, an edge that compounded).
2. **what_didnt** — 2-4 bullets on what didn't work. Be honest;
   don't pad.
3. **lessons** — 1-3 bullets on what we've *learned* (different
   from "what didn't work" — these are takeaways for the future).
4. **going_forward** — 1-3 bullets on the plan for next week,
   reflecting the actions the agent just took.
5. **config_changes** — 0-3 bullets enumerating any config field
   changes the agent applied (only if `tune` action ran).
6. **headline** — a one-line headline for this strategy's week
   (≤80 chars). Punchy, specific.

## Writing rules

- Short, declarative bullets. No "we", no "the strategy seems to
  have...", no hedging.
- Numbers where they sharpen the point.
- If a quadrant has nothing to say, return an empty array — don't pad.

## Required output

Return JSON only:

```json
{{
  "headline": "<one-line, ≤80 chars>",
  "what_worked":     ["<bullet>", ...],
  "what_didnt":      ["<bullet>", ...],
  "lessons":         ["<bullet>", ...],
  "going_forward":   ["<bullet>", ...],
  "config_changes":  ["<bullet>", ...]
}}
```
"""


_SIMILARITY_THRESHOLD = 0.85
_SIMILARITY_LOOKBACK_DAYS = 14


def _compute_similarity_pairs(today: date, strategy_ids: list[str]) -> list[tuple[str, str, float]]:
    """Pairwise Pearson correlation of daily P&L across strategies
    over the trailing 14 days. Returns only pairs with corr above the
    threshold, sorted by correlation descending."""
    if len(strategy_ids) < 2:
        return []
    from collections import defaultdict
    from datetime import timedelta
    import json
    from trading_bot.state.paths import ledger_path
    p = ledger_path()
    if not p.exists():
        return []
    cutoff = (today - timedelta(days=_SIMILARITY_LOOKBACK_DAYS)).isoformat()
    iso_today = today.isoformat()

    # daily_pnl[sid] = {iso_date: pnl_gbp}
    daily_pnl: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ed = rec.get("exit_date")
            sid = rec.get("strategy_id")
            if not ed or not sid:
                continue
            if ed < cutoff or ed > iso_today:
                continue
            if sid not in strategy_ids:
                continue
            daily_pnl[sid][ed] += float(rec.get("pnl_gbp") or 0)

    if len(daily_pnl) < 2:
        return []

    # Align to a common date axis
    all_dates = sorted({d for series in daily_pnl.values() for d in series.keys()})
    if len(all_dates) < 3:
        return []   # not enough data points to compute correlation

    def _series(sid: str) -> list[float]:
        return [daily_pnl[sid].get(d, 0.0) for d in all_dates]

    pairs: list[tuple[str, str, float]] = []
    sids = sorted(daily_pnl.keys())
    for i, a in enumerate(sids):
        for b in sids[i + 1:]:
            corr = _pearson(_series(a), _series(b))
            if corr is None:
                continue
            if abs(corr) >= _SIMILARITY_THRESHOLD:
                pairs.append((a, b, round(corr, 3)))
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Plain Pearson correlation. Returns None if either series has no
    variance (a constant zero series would otherwise emit NaN)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    den_y = (sum((y - my) ** 2 for y in ys)) ** 0.5
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _similar_pairs_for_strategy(sid: str, pairs: list[tuple[str, str, float]]) -> list[tuple[str, float]]:
    """Filter the global pair list to (other, corr) entries involving
    `sid`. Used to scope the similarity block to each strategy's card."""
    out: list[tuple[str, float]] = []
    for a, b, corr in pairs:
        if a == sid:
            out.append((b, corr))
        elif b == sid:
            out.append((a, corr))
    return out


def _notable_trades_lines(sid: str, *, lookback_days: int = 14, max_trades: int = 8) -> str:
    """Pull this strategy's trades over the last `lookback_days`, sort by
    |pnl_pct| descending, and format the top N with their LLM-written
    outcome_notes + risks_observed. These are concrete failure / success
    modes that the per-strategy report's Lessons + Going Forward
    quadrants can cite directly."""
    import json as _json
    from datetime import timedelta
    from trading_bot.state.paths import ledger_path
    p = ledger_path()
    if not p.exists():
        return "_(ledger not found)_"
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

    rows: list[dict] = []
    try:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if rec.get("strategy_id") != sid:
                    continue
                ed = rec.get("exit_date")
                if not ed or ed < cutoff:
                    continue
                if rec.get("exit_reason") in ("cancelled", "cleared"):
                    continue
                if not (rec.get("outcome_notes") or rec.get("risks_observed")):
                    continue
                rows.append(rec)
    except OSError:
        return "_(could not read ledger)_"
    if not rows:
        return "_(no reflected trades in window)_"

    # Most-extreme P&L first — winners and losers both
    rows.sort(key=lambda r: abs(float(r.get("pnl_pct") or 0)), reverse=True)
    out = []
    for r in rows[:max_trades]:
        pct = float(r.get("pnl_pct") or 0)
        # Clip to sane bounds (long-only can't lose more than 100%)
        pct = max(-100.0, min(500.0, pct))
        outcome = (r.get("outcome_notes") or "").strip()
        risks = (r.get("risks_observed") or "").strip()
        out.append(
            f"- **{r.get('ticker', '?')} {pct:+.2f}%** "
            f"({r.get('exit_date')}, {r.get('exit_reason', '?')})\n"
            f"  - Outcome: {outcome[:280]}\n"
            + (f"  - Risks:   {risks[:280]}" if risks and risks != "(reflection agent did not run on this trade)" else "")
        )
    return "\n".join(out)


def _strategy_prediction_reflections_lines(
    sid: str, *, lookback_days: int = 14, max_rows: int = 10,
) -> str:
    """Surface the most instructive **untraded** prediction reflections
    from the trailing window. These are one-liners pre-computed by
    `reflect_predictions_on_day` — they explain why a given pick's
    rationale agreed or diverged from the realised outcome, without
    the evolution agent having to re-derive that for every miss.

    Picks the rows where the gap between predicted and actual return
    magnitudes is largest (the biggest learn-from cases), capped at
    `max_rows`. Returns "_(no reflected predictions in window)_" when
    the window is empty so the prompt section is clearly empty rather
    than missing."""
    import json as _json
    from datetime import date, timedelta

    from trading_bot.state.paths import predictions_path

    p = predictions_path()
    if not p.exists():
        return "_(no predictions file)_"
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

    rows: list[dict] = []
    try:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if rec.get("strategy_id") != sid:
                    continue
                pd_ = rec.get("prediction_date") or ""
                if not pd_ or pd_ < cutoff:
                    continue
                if not (rec.get("reflection") or "").strip():
                    continue
                rows.append(rec)
    except OSError:
        return "_(could not read predictions)_"

    if not rows:
        return "_(no reflected predictions in window)_"

    def _gap(rec: dict) -> float:
        pr = rec.get("predicted_return_pct")
        ac = rec.get("actual_return_pct")
        try:
            return abs(float(pr or 0) - float(ac or 0))
        except (TypeError, ValueError):
            return 0.0

    rows.sort(key=_gap, reverse=True)
    out: list[str] = []
    for r in rows[:max_rows]:
        pr = r.get("predicted_return_pct")
        ac = r.get("actual_return_pct")
        pr_s = f"{pr:+.2f}%" if isinstance(pr, (int, float)) else "?"
        ac_s = f"{ac:+.2f}%" if isinstance(ac, (int, float)) else "?"
        out.append(
            f"- **{r.get('ticker', '?')}** ({r.get('prediction_date')}, "
            f"{r.get('predicted_class', '?')} pred {pr_s} vs "
            f"{r.get('actual_class', '?')} actual {ac_s}) — "
            f"{(r.get('reflection') or '').strip()[:280]}"
        )
    return "\n".join(out)


def _strategy_missed_lines(sid: str) -> str:
    """Walk the last 7 missed-movers reports and pull the misses where
    this strategy's universe contained the ticker but the strategy
    didn't trade it. Formatted as bullets for the prompt."""
    from trading_bot.meta.missed_movers import load_recent_reports
    reports = load_recent_reports(days=7)
    lines: list[str] = []
    for rep in reports:
        date_str = rep.get("date", "")
        region = rep.get("region", "")
        for m in rep.get("top_movers", []) or []:
            in_universe = sid in (m.get("in_universe_of") or [])
            was_traded = sid in (m.get("was_traded_by") or [])
            if not in_universe or was_traded:
                continue
            ticker = m.get("ticker", "")
            move = m.get("move_pct", 0)
            catalyst = (m.get("catalyst") or "").strip()
            reason = (m.get("miss_reason") or "").strip()
            lines.append(
                f"- [{date_str} {region}] **{ticker} {move:+.2f}%** — "
                f"{catalyst}{' · why missed: ' + reason if reason else ''}"
            )
    if not lines:
        return "_(no missed-movers data — analyser may not have run yet, or this strategy had no misses)_"
    # Cap at 12 to keep the prompt bounded
    if len(lines) > 12:
        lines = lines[:12] + [f"_(... {len(lines) - 12} more)_"]
    return "\n".join(lines)


def _parse_strategy_report(sid: str, response: dict | list) -> StrategyReport:
    if isinstance(response, list) and response:
        response = response[0] if isinstance(response[0], dict) else {}
    if not isinstance(response, dict):
        return _fallback_strategy_report(sid, [], [])

    def _list_field(key: str) -> list[str]:
        raw = response.get(key) or []
        if isinstance(raw, str):
            raw = [raw]
        return [str(x).strip() for x in raw if str(x).strip()][:5]

    return StrategyReport(
        strategy_id=sid,
        headline=str(response.get("headline") or f"{sid} this week").strip()[:120],
        what_worked=_list_field("what_worked"),
        what_didnt=_list_field("what_didnt"),
        lessons=_list_field("lessons"),
        going_forward=_list_field("going_forward"),
        config_changes=_list_field("config_changes"),
        failed=False,
    )


# ---------------------------------------------------------------------------
# Fallback paths (no LLM)
# ---------------------------------------------------------------------------

def _fallback_edition(today: date, snapshot: list[dict], applied_actions: list[dict]) -> EvolutionEdition:
    by_strategy: dict[str, list[dict]] = {}
    for row in snapshot:
        by_strategy.setdefault(row.get("id") or "(unknown)", []).append(row)
    actions_by_strategy: dict[str, list[dict]] = {}
    for a in applied_actions:
        actions_by_strategy.setdefault(a.get("strategy_id") or "(unknown)", []).append(a)

    reports = [
        _fallback_strategy_report(sid, rows, actions_by_strategy.get(sid, []))
        for sid, rows in by_strategy.items()
    ]
    return EvolutionEdition(
        week_end=today.isoformat(),
        editorial_md=_fallback_editorial(today, snapshot, applied_actions),
        reports=reports,
        snapshot_rows=snapshot,
        action_log=applied_actions,
    )


def _fallback_editorial(today: date, snapshot: list[dict], applied_actions: list[dict]) -> str:
    n_actions = sum(1 for a in applied_actions if a.get("applied"))
    return (
        f"Week ending **{today.isoformat()}** — {len(snapshot)} (strategy, region) "
        f"pairs in the snapshot; {n_actions} action{'s' if n_actions != 1 else ''} "
        f"applied. See the report cards below for the per-strategy detail."
    )


def _fallback_strategy_report(sid: str, rows: list[dict], actions: list[dict]) -> StrategyReport:
    total_pnl = sum(
        (r.get("metrics") or {}).get("total_pnl_gbp", 0.0) or 0.0
        for r in rows
    )
    total_n = sum(
        (r.get("metrics") or {}).get("n_trades", 0) or 0
        for r in rows
    )
    headline = (
        f"{sid}: £{total_pnl:+,.2f} across {total_n} trades"
        if total_n
        else f"{sid}: no closed trades this window"
    )
    going_forward = []
    config_changes = []
    for a in actions:
        if a.get("applied"):
            if a.get("action") == "tune":
                config_changes.append(f"{a.get('reason', 'config tuned')[:120]}")
            else:
                going_forward.append(f"{a.get('action')} — {a.get('reason', '')[:120]}")

    return StrategyReport(
        strategy_id=sid,
        headline=headline,
        what_worked=[],
        what_didnt=[],
        lessons=[],
        going_forward=going_forward or ["No action taken; running as configured."],
        config_changes=config_changes,
        failed=True,
    )


# ---------------------------------------------------------------------------
# Render layer
# ---------------------------------------------------------------------------

def _md_to_html(md_text: str) -> str:
    return md_lib.markdown(md_text or "", extensions=["tables", "fenced_code", "sane_lists", "nl2br"])


def _render_page(edition: EvolutionEdition, *, shell_fn) -> str:
    body_html = _render_body(edition)
    return shell_fn(
        title=f"Evolution — week ending {edition.week_end}",
        body_html=body_html,
        current="evolution",
        depth=0,
        page_class="evolution",
    )


def _render_body(edition: EvolutionEdition) -> str:
    parts: list[str] = []
    parts.append('<main class="paper">')

    # Masthead
    parts.append(
        '<header class="masthead">'
        f'  <h1>The Bot Tribune<span class="sub">— Evolution, week ending {html.escape(edition.week_end)}</span></h1>'
        '</header>'
    )

    # Masthead strip
    n_strategies = len({r.strategy_id for r in edition.reports})
    n_actions_applied = sum(1 for a in edition.action_log if a.get("applied"))
    parts.append(
        '<div class="masthead-strip">'
        f'  <span><strong>Evolution</strong></span>'
        f'  <span>{n_strategies} strateg{"ies" if n_strategies != 1 else "y"} reviewed</span>'
        f'  <span>{n_actions_applied} action{"s" if n_actions_applied != 1 else ""} applied</span>'
        '</div>'
    )

    # Editorial intro — body wrapped in .editorial-body so the drop-cap
    # CSS rule targets the body paragraph (not a meta/dek line).
    parts.append('<article class="editorial">')
    parts.append('<div class="editorial-body">')
    parts.append(_md_to_html(edition.editorial_md))
    parts.append('</div>')
    parts.append('</article>')

    # Slate table — one row per (strategy, region) with metrics
    parts.append(_render_slate_table(edition))

    # Per-strategy report cards
    parts.append(
        '<div class="section-label evo">'
        '  <span>The cards</span>'
        f'  <span class="ord">{len(edition.reports)} strateg{"y" if len(edition.reports) == 1 else "ies"} reviewed</span>'
        '</div>'
    )
    for r in edition.reports:
        parts.append(_render_strategy_card(r, edition))

    # Decisions row (chips)
    if edition.action_log:
        parts.append(_render_decisions_row(edition.action_log))

    parts.append(
        '<footer class="colophon">'
        f'Auto-generated by the weekly-evolution agent · {html.escape(_generated_at())}'
        '</footer>'
    )
    parts.append('</main>')
    return "\n".join(parts)


def _render_slate_table(edition: EvolutionEdition) -> str:
    """The snapshot as a single 'slate' table — one row per (strategy, region)."""
    rows = sorted(
        edition.snapshot_rows,
        key=lambda r: ((r.get("metrics") or {}).get("total_pnl_gbp", 0.0) or 0.0),
        reverse=True,
    )
    if not rows:
        return ""

    out = [
        '<div class="section-label evo">'
        '  <span>The slate</span>'
        f'  <span class="ord">{len(rows)} (strategy, region) pair{"s" if len(rows) != 1 else ""}</span>'
        '</div>',
        '<div class="sectors-table">',
        '<h3>14-day rolling performance</h3>',
        '<table><tbody>',
    ]
    for r in rows:
        m = r.get("metrics") or {}
        pnl = (m.get("total_pnl_gbp") or 0.0) or 0.0
        pnl_cls = "up" if pnl > 0 else ("down" if pnl < 0 else "")
        hit = (m.get("hit_rate") or 0.0) * 100
        n = m.get("n_trades", 0)
        ic = m.get("ic", 0.0)
        out.append(
            f'<tr>'
            f'<td class="sector">{html.escape(r.get("id", ""))}</td>'
            f'<td class="lean">{html.escape(r.get("region", "") or "")} · {html.escape(r.get("tier", "") or "")}</td>'
            f'<td class="driver"><span class="{pnl_cls}">£{pnl:+,.2f}</span> '
            f'across {int(n) if n else 0} trades · {hit:.0f}% hit · IC {ic:+.2f}</td>'
            f'</tr>'
        )
    out.append('</tbody></table></div>')
    return "\n".join(out)


def _render_strategy_card(r: StrategyReport, edition: EvolutionEdition) -> str:
    """One per-strategy report card. Four quadrants of bullets."""
    def _bullets(items: list[str]) -> str:
        if not items:
            return '<li><em style="color: var(--ink-muted);">nothing notable</em></li>'
        return "\n".join(f"<li>{html.escape(item)}</li>" for item in items)

    config_changes_html = ""
    if r.config_changes:
        config_changes_html = (
            '<div style="grid-column: 1 / -1; padding-top: 1rem; border-top: 1px solid var(--hairline); margin-top: 0.5rem;">'
            '<div style="font-family: var(--sans); font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: var(--c-action-tune); margin-bottom: 0.4rem;">Config changes</div>'
            f'<ul style="margin: 0; padding-left: 1.2rem; font-size: 0.95rem; line-height: 1.55;">'
            f'{_bullets(r.config_changes)}</ul>'
            '</div>'
        )

    return (
        '<article class="evo-card">'
        f'<p class="meta"><span class="accent" style="color: var(--c-evo);">STRATEGY · {html.escape(r.strategy_id.upper())}</span></p>'
        f'<h3 class="evo-card-headline">{html.escape(r.headline)}</h3>'
        '<div class="evo-card-grid">'
        '<div>'
        '<div style="font-family: var(--sans); font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: var(--c-action-promote); margin-bottom: 0.4rem;">What worked</div>'
        f'<ul style="margin: 0; padding-left: 1.2rem; line-height: 1.55;">{_bullets(r.what_worked)}</ul>'
        '</div>'
        '<div>'
        '<div style="font-family: var(--sans); font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: var(--c-action-demote); margin-bottom: 0.4rem;">What didn\'t</div>'
        f'<ul style="margin: 0; padding-left: 1.2rem; line-height: 1.55;">{_bullets(r.what_didnt)}</ul>'
        '</div>'
        '<div>'
        '<div style="font-family: var(--sans); font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-soft); margin-bottom: 0.4rem;">Lessons</div>'
        f'<ul style="margin: 0; padding-left: 1.2rem; line-height: 1.55;">{_bullets(r.lessons)}</ul>'
        '</div>'
        '<div>'
        '<div style="font-family: var(--sans); font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: var(--c-evo); margin-bottom: 0.4rem;">Going forward</div>'
        f'<ul style="margin: 0; padding-left: 1.2rem; line-height: 1.55;">{_bullets(r.going_forward)}</ul>'
        '</div>'
        f'{config_changes_html}'
        '</div>'
        '</article>'
    )


def _render_decisions_row(action_log: list[dict]) -> str:
    """Decision chips at the bottom — the action summary."""
    chip_styles = {
        "promote":          ("var(--c-action-promote)", "Promoted"),
        "demote":           ("var(--c-action-demote)",  "Demoted"),
        "tune":             ("var(--c-action-tune)",    "Tuned"),
        "spawn-variant":    ("var(--c-action-spawn)",   "Spawned"),
        "request-tier-2":   ("#7e22ce",                 "Tier-2 request"),
        "keep":             ("#6b7280",                 "Kept"),
    }

    chips: list[str] = []
    for a in action_log:
        if not a.get("applied") and a.get("action") != "request-tier-2":
            continue
        color, label = chip_styles.get(a.get("action", ""), ("#6b7280", a.get("action", "")))
        scope = f"{a.get('strategy_id')}" + (f"@{a.get('region')}" if a.get("region") else "")
        chips.append(
            f'<span style="display: inline-flex; align-items: center; gap: 0.5rem; '
            f'padding: 0.4rem 0.85rem; border-radius: 14px; background: rgba(0,0,0,0.03); '
            f'border: 1px solid {color}; color: {color}; font-family: var(--sans); '
            f'font-size: 0.8rem; font-weight: 600; margin: 0.2rem 0.3rem 0.2rem 0;">'
            f'<strong>{html.escape(label)}</strong> {html.escape(scope)}'
            '</span>'
        )

    if not chips:
        return ""

    return (
        '<div class="section-label evo">'
        '  <span>This week\'s decisions</span>'
        f'  <span class="ord">{len(chips)} action{"s" if len(chips) != 1 else ""}</span>'
        '</div>'
        '<div style="margin-top: 0.8rem;">' + "".join(chips) + '</div>'
    )


def _generated_at() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
