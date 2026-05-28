"""Wave 6 — weekly strategy-evolution agent.

Reads each active strategy's rolling metrics (from meta.metrics), gathers
prompts + configs + recent lessons, asks Claude for action recommendations,
then auto-executes anything that stays on Tier 0 / Tier 1:

  - keep             — no change
  - tune             — edit config fields within safety bounds (strategy-wide,
                       all regions share these)
  - promote          — Tier 0 (shadow) → Tier 1 (alpaca-paper) + slot assign,
                       region-specific
  - demote           — Tier 1 (alpaca-paper) → Tier 0 (shadow) + clear_slot,
                       region-specific
  - spawn-variant    — clone an existing strategy with config + prompt diffs
  - request-tier-2   — file GitHub Issue for human approval (never auto)

Tier 2 (live) is fully off-limits to this agent. Anything involving real
money requires explicit human approval per sparknotes design.

Promote/demote are **per-region**: a strategy can be on alpaca-paper in
`us` while still on shadow in `uk-eu`, etc. Tune is strategy-wide because
the bounds (capital, position sizing, stops) are shared across regions.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from trading_bot.llm.claude_code import ClaudeCodeError, run_claude_for_json
from trading_bot.meta.metrics import StrategyMetrics, compute_all_metrics
from trading_bot.state.paths import STATE_ROOT
from trading_bot.strategy.registry import _strategies_dir  # noqa: WPS437 — internal but stable
from trading_bot.t212_slot import T212_PAPER_BUDGET_GBP


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration: thresholds & safety bounds
# ---------------------------------------------------------------------------

MAX_ALPACA_SLOTS = 3  # how many paper accounts we've provisioned
MAX_TOTAL_STRATEGIES = 12  # ceiling on spawned variants
PROMOTION_MIN_TRADES = 10
PROMOTION_MIN_HIT_RATE = 0.50
# Phase 10C — tunable so we can promote net-positive-but-marginal
# strategies in the future without rewriting the gate. Setting it
# higher than 0 means a strategy needs to actually make money to
# promote; setting it to 0 allows break-even promotions.
PROMOTION_MIN_PNL_GBP = 0.0
PROMOTION_MIN_IC = 0.05
# Phase 10C — separate from PROMOTION_MIN_IC so we can relax the
# lower-bound gate independently. Setting this to 0.0 allows promotion
# whenever the lower CI bound is at-or-above "no edge", even if the
# point estimate is well above. Default same as POINT_ESTIMATE so the
# overall gate stays as strict as before.
PROMOTION_MIN_IC_LOWER = 0.05
DEMOTION_MAX_DRAWDOWN_PCT = -10.0
DEMOTION_MIN_HIT_RATE = 0.40

# T212 demo account caps total holdings at £50k. We keep auto-promotion
# under £40k of committed `capital_gbp` so there's £10k of headroom for
# realised losses (the account balance drifts down with each loss, and
# new positions need cash to open). Manual promotions via config edits
# can still push higher — this is the auto-promotion guard only.
T212_PROMOTE_BUDGET_HEADROOM_GBP = 40_000.0

# Phase 9A — A/B confidence intervals. Promotion requires the *lower
# 95% bound* on IC (Fisher z-transform) to clear PROMOTION_MIN_IC_LOWER,
# not just the point estimate. Stops promotions on 14 trades where
# the IC delta is driven by a handful of lucky picks.
PROMOTION_IC_CI_Z = 1.96      # ~95% one-sided

# Fields the agent is allowed to tune on Tier 0/1 strategies (strategy-wide,
# applied at the top level — all regions inherit). Region-specific entries
# in runs_in can override these, but the agent doesn't touch overrides.
TUNABLE_FIELDS = {
    "max_positions": (1, 10),
    "max_position_pct": (5.0, 60.0),
    "min_position_gbp": (10.0, 1000.0),
    "stop_loss_pct": (-15.0, -1.0),
    "take_profit_pct": (1.0, 20.0),
    "capital_gbp": (1000.0, 30000.0),
    # cost_gate_multiplier — minimum predicted-return / round-trip-cost
    # ratio for a pick to survive the gate. 2.0 = "pick must clear 2×
    # fees". Lower lets more marginal trades through (more activity, more
    # vulnerable to flipping on small moves); higher demands stronger
    # conviction. UK-EU strategies tend to need this higher than US
    # because UK Stamp Duty alone is 0.5% on every share purchase.
    "cost_gate_multiplier": (0.5, 5.0),
    # prefilter_top_n — size of the shortlist the LLM pre-filter returns
    # to the strategy. Higher = more diversity for Stage-2 to work with;
    # lower = more concentrated, less LLM token spend at the per-candidate
    # scoring stages. Stage-1 Haiku also caps in its own prompt, so this
    # is the absolute upper bound on what Stage-1 sees.
    "prefilter_top_n": (20, 300),
    # midday_tp_factor — at the midday cron, close positions whose
    # pct_up >= take_profit_pct × midday_tp_factor. Lower = more
    # aggressive locking, more upside left on the table. Tune lower
    # when realised returns consistently fall short of TP despite
    # touching it intraday; tune higher when realised returns
    # consistently exceed TP.
    "midday_tp_factor": (0.3, 1.5),
}

# Enum-style tunable fields. Same `tune` action verb; clamping rule is
# "value must be in the allowed set" instead of "value in numeric range".
TUNABLE_STRING_FIELDS = {
    # prefilter_mode — how the strategy narrows the universe before the
    # expensive yfinance technicals fetch.
    #   "llm":    per-strategy Sonnet call with strategies/<id>/prompts/prefilter.md
    #             — strategy-aware, but ~60-90s per call
    #   "python": fetch all universe technicals, sort by `prefilter_sort_key`
    #   "off":    no pre-filter (universe goes straight to Stage-1; only
    #             safe for small universes or rule-based strategies)
    "prefilter_mode": {"llm", "python", "off"},
    # prefilter_sort_key — when prefilter_mode=python, picks the ranker.
    # Different lenses need different rankers; the legacy default biases
    # every strategy toward biggest movers, which is right for momentum
    # but wrong for macro / sector / mean-reversion strategies.
    #   "abs_return_5d"        — |return_5d_pct| desc (legacy default)
    #   "abs_return_20d"       — |return_20d_pct| desc
    #   "rsi_14_asc"           — rsi_14 asc (oversold first; mean-reverter)
    #   "volume_ratio_desc"    — volume_ratio desc (catalyst flow; news-reactive)
    #   "dollar_volume_desc"   — sma_20 × avg_volume_20 desc (liquidity
    #                            rank — biases toward ETFs/large-caps;
    #                            for macro-aligned, sector-rotator,
    #                            bond-cycle, commodity-momentum)
    "prefilter_sort_key": {
        "abs_return_5d", "abs_return_20d", "rsi_14_asc",
        "volume_ratio_desc", "dollar_volume_desc",
    },
}


@dataclass
class ActionLog:
    strategy_id: str
    region: str | None  # None for strategy-wide actions like tune / spawn
    action: str
    applied: bool
    reason: str
    details: dict


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_weekly_evolution(today: date) -> dict:
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        log.error("CLAUDE_CODE_OAUTH_TOKEN not set — skipping evolution")
        return {"skipped": True, "reason": "no oauth token"}

    log.info("evolution: starting weekly run for %s", today.isoformat())

    metrics = compute_all_metrics(window_days=14, end_date=today)
    configs = _load_all_configs()

    snapshot = _build_snapshot(configs, metrics)
    prompt = _build_prompt(today, snapshot, _read_lessons())

    try:
        recommendations = run_claude_for_json(prompt, model="sonnet", retries=2)
    except ClaudeCodeError as e:
        log.error("evolution: Claude call failed: %s", e)
        return {"error": str(e)}

    actions_raw = (
        recommendations.get("actions") if isinstance(recommendations, dict) else recommendations
    )
    if not isinstance(actions_raw, list):
        log.error("evolution: LLM response wasn't a list of actions")
        return {"error": "bad response shape"}

    applied: list[ActionLog] = []
    pending_tier_2: list[ActionLog] = []
    for raw in actions_raw:
        if not isinstance(raw, dict):
            continue
        try:
            log_entry = _apply_action(raw, metrics=metrics, configs=configs)
        except Exception as e:
            sid = raw.get("strategy_id") or "?"
            action = raw.get("action") or "?"
            log.warning("evolution: action %s for %s threw — recording skip: %s",
                        action, sid, e)
            applied.append(ActionLog(
                sid, raw.get("region"), action, False,
                f"Action raised {type(e).__name__}: {str(e)[:200]}",
                {},
            ))
            continue
        if log_entry is None:
            continue
        if log_entry.action == "request-tier-2":
            pending_tier_2.append(log_entry)
        applied.append(log_entry)

    _append_evolution_log(today, applied)
    issue_url = _maybe_file_issue(today, applied, pending_tier_2)

    # Phase 4 — render the editorial evolution page (per-strategy report cards).
    # This is purely additive: failure here doesn't affect the action engine.
    try:
        from trading_bot.dashboard.pages import _shell, docs_root, pages_url
        from trading_bot.meta.evolution_v2 import build_and_render_evolution
        build_and_render_evolution(
            today=today,
            snapshot=snapshot,
            applied_actions=[asdict_actionlog(a) for a in applied],
            docs_root=docs_root(),
            shell_fn=_shell,
        )
        # Send the weekly evolution email
        try:
            from trading_bot.notify.email import render_evolution_email, send_summary_email
            n_actions_applied_count = sum(1 for a in applied if a.applied)
            n_strategies = len({(a.strategy_id) for a in applied}) or len({s.get("id") for s in snapshot})
            subject, text_body, html_body = render_evolution_email(
                week_end=today.isoformat(),
                n_strategies=n_strategies,
                n_actions_applied=n_actions_applied_count,
                full_brief_url=pages_url("evolution.html"),
            )
            send_summary_email(subject=subject, body_text=text_body, body_html=html_body)
            log.info("Evolution: sent weekly email")
        except Exception as e:
            log.warning("Couldn't send evolution email (non-fatal): %s", e)
    except Exception as e:
        log.warning("Evolution editorial render failed (non-fatal): %s", e)

    summary = {
        "week_end": today.isoformat(),
        "n_snapshot_rows": len(snapshot),
        "n_actions_applied": sum(1 for a in applied if a.applied),
        "n_actions_skipped": sum(1 for a in applied if not a.applied),
        "n_tier_2_requests": len(pending_tier_2),
        "issue_url": issue_url,
    }
    log.info("evolution: %s", summary)
    return summary


def asdict_actionlog(a: ActionLog) -> dict:
    """ActionLog → dict (for handing to the v2 renderer)."""
    return {
        "strategy_id": a.strategy_id,
        "region": a.region,
        "action": a.action,
        "applied": a.applied,
        "reason": a.reason,
        "details": a.details,
    }


# ---------------------------------------------------------------------------
# Snapshot + prompt
# ---------------------------------------------------------------------------

def _build_snapshot(
    configs: dict[str, dict],
    metrics: dict[tuple[str, str], StrategyMetrics],
) -> list[dict]:
    """One row per (strategy_id, region) pair that's actually configured to
    run somewhere. Region-specific tier/slot come from the runs_in entry."""
    rows = []
    for sid, cfg in configs.items():
        for entry in _regions_for_config(cfg):
            region = entry["region"]
            m = metrics.get((sid, region))
            rows.append({
                "id": sid,
                "region": region,
                "tier": entry.get("tier", cfg.get("tier", "shadow")),
                "alpaca_slot": entry.get("alpaca_slot", cfg.get("alpaca_slot")),
                "t212_slot": entry.get("t212_slot", cfg.get("t212_slot")),
                "universe": entry.get("universe", cfg.get("universe")),
                "active": cfg.get("active"),
                # Strategy-wide config fields (shared across regions)
                "max_positions": cfg.get("max_positions"),
                "max_position_pct": cfg.get("max_position_pct"),
                "stop_loss_pct": cfg.get("stop_loss_pct"),
                "take_profit_pct": cfg.get("take_profit_pct"),
                "capital_gbp": cfg.get("capital_gbp"),
                "cost_gate_multiplier": cfg.get("cost_gate_multiplier"),
                "prefilter_mode": cfg.get("prefilter_mode"),
                "prefilter_top_n": cfg.get("prefilter_top_n"),
                "prefilter_sort_key": cfg.get("prefilter_sort_key"),
                "midday_tp_factor": cfg.get("midday_tp_factor"),
                # Tier 2 candidate state — set by a prior evolution run
                # as a self-prediction. Surfacing it here lets the agent
                # grade its own past judgement: did the realised metrics
                # support the prediction?
                "tier2_candidate": bool(cfg.get("tier2_candidate", False)),
                "tier2_marked_at": cfg.get("tier2_marked_at"),
                "tier2_thesis": cfg.get("tier2_thesis"),
                "metrics": _metrics_to_dict(m) if m else None,
                "meets_promotion_criteria": _meets_promotion(m, entry) if m else False,
                "meets_demotion_criteria": _meets_demotion(m, entry) if m else False,
            })
    return rows


def _regions_for_config(cfg: dict) -> list[dict]:
    """Yield each region descriptor for this strategy.

    Multi-region (`runs_in`) configs: yield each entry as-is.
    Single-region configs: yield one synthetic entry combining the
    top-level region / tier / alpaca_slot / universe fields.
    """
    runs_in = cfg.get("runs_in")
    if isinstance(runs_in, list) and runs_in:
        return [dict(e) for e in runs_in if isinstance(e, dict) and e.get("region")]
    return [{
        "region": cfg.get("region", "us"),
        "tier": cfg.get("tier", "shadow"),
        "alpaca_slot": cfg.get("alpaca_slot"),
        "universe": cfg.get("universe"),
    }]


def _metrics_to_dict(m: StrategyMetrics) -> dict:
    return {
        "n_trades": m.n_trades,
        "hit_rate": m.hit_rate,
        "total_pnl_gbp": m.total_pnl_gbp,
        "avg_pnl_pct": m.avg_pnl_pct,
        "max_drawdown_pct": m.max_drawdown_pct,
        "n_predictions_graded": m.n_predictions_graded,
        "ic": m.ic,
        "decile_spread": m.top_minus_bottom_decile_spread,
    }


def _build_tool_attribution_block(snapshot: list[dict]) -> str:
    """Per-strategy tool-attribution summary spliced into the evolution
    prompt. One section per strategy: each tool's IC delta over the
    trailing 60-day window PLUS an IC-by-prefilter-mode comparison
    (lets the agent A/B-test Sonnet pre-filter vs Python heuristic
    vs no filter once at least one strategy has run under both modes).
    Empty for strategies without graded predictions yet."""
    try:
        from trading_bot.meta.tool_attribution import (
            attribution_summary_lines, prefilter_summary_lines,
        )
    except Exception as e:
        log.warning("tool attribution unavailable: %s", e)
        return "_(tool attribution unavailable)_"

    # Strategy IDs appear multiple times in the snapshot (once per
    # region); collapse to a unique set so we don't emit duplicate
    # blocks. Attribution is strategy-wide because the `tools` list
    # is strategy-wide too.
    seen: set[str] = set()
    blocks: list[str] = []
    for row in snapshot:
        sid = row.get("id")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        try:
            tool_lines = attribution_summary_lines(sid)
        except Exception as e:
            log.warning("attribution failed for %s: %s", sid, e)
            tool_lines = ""
        try:
            prefilter_lines = prefilter_summary_lines(sid)
        except Exception as e:
            log.warning("prefilter attribution failed for %s: %s", sid, e)
            prefilter_lines = ""
        if not tool_lines.strip() and not prefilter_lines.strip():
            continue
        parts: list[str] = [f"**{sid}**"]
        if tool_lines.strip():
            parts.append("_Tools:_\n" + tool_lines)
        if prefilter_lines.strip():
            parts.append("_Pre-filter mode comparison:_\n" + prefilter_lines)
        blocks.append("\n".join(parts))
    if not blocks:
        return "_(no graded predictions yet — attribution will populate as data accumulates)_"
    return "\n\n".join(blocks)


def _read_external_research() -> str:
    """Return the latest external research brief as markdown, or a
    placeholder if no brief has been generated yet (e.g. the very
    first weekly run)."""
    try:
        from trading_bot.meta.external_research import latest_brief
    except Exception as e:
        log.warning("could not import external_research module: %s", e)
        return "_(external research module unavailable)_"
    try:
        b = latest_brief()
    except Exception as e:
        log.warning("could not read external research brief: %s", e)
        return "_(external research brief read failed)_"
    if b is None:
        return "_(no external research brief available yet — first scan may not have run)_"
    return f"**Brief from {b.week_iso}** (generated {b.generated_at[:10]})\n\n{b.body_md}"


def _build_prompt(today: date, snapshot: list[dict], lessons: str) -> str:
    free_slots = _free_alpaca_slots(snapshot)
    n_active = sum(1 for s in snapshot if s.get("active"))

    snapshot_json = json.dumps(snapshot, indent=2)

    # Pick up the most recent external research brief (from the
    # scan step that runs earlier in the weekly-evolution workflow).
    # Surfaces what's being researched in quant finance so the agent
    # can ground spawn-variant proposals in current literature
    # rather than re-inventing.
    external_block = _read_external_research()

    # Tool attribution — per-strategy IC delta by tool over the last
    # 60 days. Tells the agent which prompt inputs are pulling weight
    # vs. dead weight that should be tuned out of the `tools` list.
    tool_attribution_block = _build_tool_attribution_block(snapshot)

    return f"""You are the weekly evolution agent for the trading bot. Today is
{today.isoformat()}. Each strategy can run independently across regions
(`us`, `uk-eu`, `asia`), with its own tier and slot per region. Below is a
14-day rolling snapshot of every (strategy, region) pair.

## What you're optimising for — this is a tournament

There is ultimately ONE live (real-money) slot. Every strategy in
shadow and paper tiers is a competitor for that single slot — the
whole slate is a tournament to identify the single best strategy to
run live, not a portfolio of strategies all climbing toward live in
parallel. Your north star each week is: *which one strategy is
proving itself the strongest candidate for the live slot, and what
does it need to get there?*

Concretely, this changes how you treat tier-2 candidacy:
- Tier-2 candidacy is a LEADERBOARD for the one live slot, not a
  checklist of "everything that's mildly positive." When you flag
  candidates, rank them against EACH OTHER — say which is currently
  leading and why it beats the others, not just that each clears some
  bar in isolation.
- Converge over time. If you flagged 2-3 contenders in prior weeks,
  each week should narrow the field as evidence accumulates — drop
  contenders that have been out-competed (`unmark-tier2-candidate`),
  not just ones that turned negative.
- Only escalate `request-tier-2` when one candidate has separated from
  the field on durable, risk-adjusted evidence (IC lower-bound, hit
  rate, net-of-fee P&L over enough trades) — and make the case
  COMPARATIVELY: why this one over the current runner-up. The human
  approves the final live promotion; your job is to hand them a
  clear single winner, not a list.

Tuning, spawning, and demoting all serve this: you're shaping the
field so the best candidate emerges and proves itself, and culling
the ones that won't win the slot.

## Your authority

You can auto-execute these actions on **Tier 0 (shadow)** and **Tier 1
(alpaca-paper)** strategies:
- `keep` — no change
- `tune` — edit specific fields (strategy-wide, no region — affects every region the strategy runs in)
- `promote` — Tier 0 (shadow) → Tier 1 for **one region**. Target tier is region-dependent: US rows promote to `alpaca-paper` (we assign a free numbered Alpaca slot); UK-EU rows promote to `trading212-paper` (single shared slot=1, gated by a £40k capital-budget ceiling across all T212-paper strategies — £10k headroom under the £50k account cap for realised losses).
- `demote` — Tier 1 → Tier 0 (shadow) for **one region**. Works against BOTH paper-broker tiers: `alpaca-paper` (we cancel open orders + free the Alpaca slot) and `trading212-paper` (we free the T212 slot; any open positions exit naturally on the next exit cron).
- `spawn-variant` — clone an existing strategy with prompt + config diffs
- `mark-tier2-candidate` — entry on the leaderboard for the one live
  slot: "this is a live contender, and I'm putting my reputation on
  it." The flag surfaces a gold border on the dashboard and stays set
  until you retract or confirm next week. **Include a `thesis` field**
  that makes the case COMPARATIVELY — why this contender is ahead of
  (or closing on) the others, not just that it cleared a bar in
  isolation. Next week's run reads it back to grade your judgement
  against realised performance. Keep the field small (≤3) and
  ranked — this is the shortlist for a single slot, not a list of
  everything mildly positive.
- `unmark-tier2-candidate` — removes a contender from the leaderboard.
  Use it not only when a candidate turned negative, but when it's been
  OUT-COMPETED by a stronger one — the field should narrow over time
  toward a single winner.

For **Tier 2 (live)** strategies you can ONLY recommend with `request-tier-2`
— that opens a GitHub Issue for the user to approve.

## Safety bounds

Tunable fields (clipped to these ranges if you exceed them; tune is
strategy-wide, applied at the top of config.yaml):
{json.dumps({k: list(v) for k, v in TUNABLE_FIELDS.items()}, indent=2)}

Enum-style tunable fields (value must be one of the allowed strings):
{json.dumps({k: sorted(v) for k, v in TUNABLE_STRING_FIELDS.items()}, indent=2)}

`prefilter_mode` is a structural switch worth special attention:
  - "llm" routes the universe through a Sonnet pre-filter using
    strategies/<id>/prompts/prefilter.md. Strategy-aware (mean-reverter
    sees different names than momentum-trader from the same universe),
    but ~60-90s per call. Recommended default for LLM-driven strategies.
  - "python" fetches all universe technicals and ranks via
    `prefilter_sort_key` (see below). Strategy-aware ONLY IF the
    sort_key matches the strategy's edge.
  - "off" disables the pre-filter entirely. Only safe for rule-based
    strategies (control-rule-based) whose own logic IS the filter.

`prefilter_sort_key` selects which Python ranker runs when
prefilter_mode=python. Picking the wrong key is the second-order
version of the LLM-vs-Python choice — a momentum strategy with
sort_key=rsi_14_asc will get fed oversold names, which is wrong for
its edge. Match the key to the strategy archetype:
  - "abs_return_5d" — biggest absolute 5d movers; momentum-trader,
    control-rule-based (legacy default).
  - "abs_return_20d" — slower momentum; trend strategies.
  - "rsi_14_asc" — most oversold first; mean-reverter.
  - "volume_ratio_desc" — highest volume vs 20d avg; news-reactive
    (catalyst flow).
  - "dollar_volume_desc" — sma_20 × avg_volume_20; biases toward
    ETFs and large-caps; macro-aligned, sector-rotator, bond-cycle,
    commodity-momentum (strategies whose natural vehicles are sector
    ETFs and liquid index names, not microcap discoveries).
If a strategy's picks consistently include microcap noise when the
edge thesis is sector-driven, sort_key=dollar_volume_desc is the
tune. If picks are biased toward big-movers when the edge is mean-
reversion, sort_key=rsi_14_asc is the tune.

Universe / broker reality (do not propose actions that contradict this):
- Everything lives on Trading 212 (the ISA account). US-LISTED ETFs
  (XLE/XLK SPDRs, TLT/IEF, GLD/USO, etc.) are NOT available on T212 —
  no PRIIPs KID for UK retail. The us_etfs_* universes are therefore
  dead for live trading; the sector-rotator / bond-cycle /
  commodity-momentum US sleeves have been deprecated for this reason.
  Do NOT propose re-adding a US-region sleeve for an ETF strategy.
- US/global exposure is still reachable on the UK-EU side via the
  GBP/GBX LSE-listed ETF universes (eu_etfs_sector / eu_etfs_bond /
  eu_etfs_commodity), which now include US-tracking GBP lines (S&P US
  Select Sector, USD-Treasury-in-GBP, etc.). These trade in London
  hours, are stamp-duty exempt, and carry no FX fee. So "we lack US
  exposure" is NOT a valid reason to spawn or re-add anything.
- Individual US/EU stocks ARE tradeable on T212 (the t212_isa_*
  universes), so stock-picking strategies' US sleeves are fine.

Other constraints:
- Free Alpaca slots available: {free_slots}
- Currently active strategies: {n_active} of max {MAX_TOTAL_STRATEGIES}
- Don't spawn a variant if it'd push active total over the cap
- Promote / demote actions require a `region` field naming which region to act on
- Promotion requires `meets_promotion_criteria=true` AND a free slot (US/Alpaca) or budget headroom (UK-EU/T212)
- Demotion requires `meets_demotion_criteria=true`
- Spawned variants start on Tier 0 shadow, must have a different `id`
- Trading212 demo (Tier 1.5, UK-EU only) caps total account balance at £{int(T212_PAPER_BUDGET_GBP):,}.
  If tuning capital_gbp for a UK-EU strategy that's already on `trading212-paper`,
  remember that the sum of capital_gbp across all `trading212-paper` strategies
  shares that single budget. US strategies on `alpaca-paper` are not affected.

## Current snapshot

```json
{snapshot_json}
```

## External research (recent quant finance literature)

{external_block}

Use the external research to inform spawn-variant proposals (anchor
the variant's edge to a cited finding) and tier-2-candidate picks
(if a paper's findings align with a strategy we're already running,
that's a confidence signal worth flagging). Don't propose anything
that requires capability we don't have — options, leverage,
intraday. Don't restate the research; cite it.

## Tool attribution (does each prompt input actually move IC?)

Per-strategy IC contribution by tool, computed over the trailing
60-day prediction window. `with` = days the tool was in the prompt;
`without` = days it wasn't. Δ > +0.05 → keep; Δ < −0.05 → tune
the strategy's `tools` list to drop it. Insufficient = sample too
small to call (one side has <20 rows).

{tool_attribution_block}

When a tool reads as `negative` for a strategy, your `tune` action
can DROP it from the strategy's `tools` list via the standard
`changes` field (`{{"tools": [...]}}` — list the kept tools, omit
the dropped ones). Don't drop tools labelled "insufficient" — wait
for more data before acting. This loop is how the prompt
configuration learns over time.

## Recent lessons (failures / corrections)

{lessons if lessons else "(none yet)"}

## Required output

A JSON object with one key `actions`, a list of action objects. Promote /
demote MUST include `region`; tune / spawn-variant do not:

```json
{{
  "actions": [
    {{ "strategy_id": "<id>", "region": "<region>", "action": "keep", "reason": "1 sentence" }},
    {{ "strategy_id": "<id>", "action": "tune",
       "changes": {{ "max_positions": 4, "stop_loss_pct": -2.5 }},
       "reason": "1-2 sentences citing the metric you're responding to" }},
    {{ "strategy_id": "<id>", "region": "us", "action": "promote", "reason": "..." }},
    {{ "strategy_id": "<id>", "region": "uk-eu", "action": "demote", "reason": "..." }},
    {{ "strategy_id": "<parent_id>", "action": "spawn-variant",
       "variant_id": "<parent_id>-v2-or-some-meaningful-suffix",
       "config_overrides": {{ "max_positions": 3, ... }},
       "deep_analysis_addendum": "Markdown text appended to the parent's deep_analysis.md to express the variant's bias",
       "reason": "1-2 sentences — why this variant has a meaningfully different edge from the parent" }},
    {{ "strategy_id": "<id>", "region": "<region>", "action": "request-tier-2", "reason": "Detailed case for live promotion" }},
    {{ "strategy_id": "<id>", "action": "mark-tier2-candidate",
       "thesis": "One-line prediction (≤300 chars) — what's the edge, what should we see by next week",
       "reason": "Why now (the data point that swung you)" }},
    {{ "strategy_id": "<id>", "action": "unmark-tier2-candidate",
       "reason": "Why we're retracting (prediction missed, conviction faded, etc)" }}
  ]
}}
```

Action thresholds (use the data, don't be sentimental):

- **Keep** is the default. Most rows should be `keep` most weeks.
- **Demote** when a strategy on `alpaca-paper` OR `trading212-paper`
  has *any* of: IC noise-floor verdict = 'noise' AND n ≥ 30, OR
  trailing 14d P&L is negative for the second consecutive week, OR
  trailing hit-rate is under 35% with n ≥ 20. Don't sit on losers
  waiting for them to turn — the slot is more valuable than the
  sunk-cost prompt iteration. Same threshold applies whether the
  strategy is on Alpaca (US) or T212 (UK-EU); the action handler
  picks the right slot to free.
- **Tune** when the live IC / hit-rate / P&L signal points at a
  specific bleeder — cost gate, earnings filter, sizing, or a
  tool the attribution block flags as negative. Target the most
  likely culprit; one tune action can adjust several fields.
- **Spawn-variant** is rare — only when a paper from the external
  research block lines up with a gap in the slate, or when one
  region's metrics are wildly better than another's (suggests a
  prompt that needs regional specialisation).
- **mark-tier2-candidate** for the strongest contender(s) for the one
  live slot — strategies that cleared the IC noise floor with
  conviction AND stack up well against the others on the leaderboard.
  Rank them; say which is leading. **request-tier-2** only when one
  has clearly separated from the field, and argue it comparatively
  against the runner-up.

A strategy performing well in one region but poorly in another is
normal — treat those decisions independently. Don't accumulate
no-edge strategies for breadth; a smaller slate of working
strategies beats a wide slate of mostly-noise ones. Remember the
tournament: the goal is to converge on the single best strategy for
the one live slot, not to keep a wide field of perpetual contenders.
"""


# ---------------------------------------------------------------------------
# Action application
# ---------------------------------------------------------------------------

def _apply_action(
    raw: dict,
    *,
    metrics: dict[tuple[str, str], StrategyMetrics],
    configs: dict[str, dict],
) -> ActionLog | None:
    sid = raw.get("strategy_id")
    action = (raw.get("action") or "").strip().lower()
    region = raw.get("region")
    reason = str(raw.get("reason", "")).strip()
    if not sid or not action:
        return None

    if action == "spawn-variant":
        return _do_spawn(raw, configs)

    if sid not in configs:
        return ActionLog(sid, region, action, False, f"Unknown strategy_id; ignored. ({reason})", {})

    cfg = configs[sid]

    # Strategy-wide actions — no region needed. Dispatch before the
    # region gate below so the LLM can omit `region` on these without
    # tripping the "requires a region" rejection.
    if action == "tune":
        return _do_tune(sid, raw, cfg, reason)
    if action == "mark-tier2-candidate":
        return _do_mark_tier2(sid, cfg, reason, raw)
    if action == "unmark-tier2-candidate":
        return _do_unmark_tier2(sid, cfg, reason)

    # All other actions need a region
    if not region:
        return ActionLog(sid, None, action, False, f"Action '{action}' requires a region", {})

    entry = _find_region_entry(cfg, region)
    if entry is None:
        return ActionLog(sid, region, action, False, f"Strategy {sid} does not run in region {region}", {})

    if entry.get("tier") == "t212-live":
        return ActionLog(sid, region, action, False, "Tier 2 strategy — agent has no authority", {})

    if action == "keep":
        return ActionLog(sid, region, action, True, reason, {})

    if action == "promote":
        return _do_promote(sid, region, cfg, entry, metrics.get((sid, region)), configs, reason)

    if action == "demote":
        return _do_demote(sid, region, cfg, entry, metrics.get((sid, region)), reason)

    if action == "request-tier-2":
        # Recorded only; the issue creator picks these up
        return ActionLog(sid, region, action, False, reason, {"requires_human_approval": True})

    return ActionLog(sid, region, action, False, f"Unknown action; ignored. ({reason})", {})


def _do_tune(sid: str, raw: dict, cfg: dict, reason: str) -> ActionLog:
    changes_raw = raw.get("changes")
    if not isinstance(changes_raw, dict) or not changes_raw:
        return ActionLog(sid, None, "tune", False, "No changes specified", {})

    clamped: dict = {}
    rejected: dict = {}
    for field_name, requested in changes_raw.items():
        # Numeric range-bounded field
        if field_name in TUNABLE_FIELDS:
            lo, hi = TUNABLE_FIELDS[field_name]
            try:
                val = type(lo)(requested)
            except (TypeError, ValueError):
                rejected[field_name] = "wrong type"
                continue
            clamped[field_name] = max(lo, min(hi, val))
            continue
        # String enum field (e.g. prefilter_mode)
        if field_name in TUNABLE_STRING_FIELDS:
            allowed = TUNABLE_STRING_FIELDS[field_name]
            val = str(requested).lower().strip() if requested is not None else ""
            if val not in allowed:
                rejected[field_name] = f"value '{val}' not in allowed set {sorted(allowed)}"
                continue
            clamped[field_name] = val
            continue
        rejected[field_name] = "field not tunable"

    if not clamped:
        return ActionLog(sid, None, "tune", False, f"No applicable changes. Rejected: {rejected}", {})

    cfg.update(clamped)
    # Phase 11B — stamp the tune date so metrics window resets. Avoids
    # mixing pre-tune and post-tune trades when the next evolution run
    # computes IC / hit-rate.
    from datetime import date as _date
    cfg["last_tune_date"] = _date.today().isoformat()
    _write_config(sid, cfg)
    return ActionLog(sid, None, "tune", True, reason, {"applied": clamped, "rejected": rejected})


def _do_mark_tier2(sid: str, cfg: dict, reason: str, raw: dict) -> ActionLog:
    """Flag a strategy as a Tier 2 candidate. The flag is the weekly
    evolution agent's prediction that this strategy is worth elevating;
    the next run scores realised performance against the analysis the
    agent recorded here, so we can measure whether the agent's
    judgement is actually predictive.

    `thesis` (≤300 chars) is the agent's one-line justification — used
    by next week's run as the prediction being graded."""
    from datetime import date as _date
    thesis = str(raw.get("thesis") or "").strip()[:300]
    cfg["tier2_candidate"] = True
    cfg["tier2_marked_at"] = _date.today().isoformat()
    if thesis:
        cfg["tier2_thesis"] = thesis
    _write_config(sid, cfg)
    return ActionLog(
        sid, None, "mark-tier2-candidate", True, reason,
        {"tier2_marked_at": cfg["tier2_marked_at"], "thesis_present": bool(thesis)},
    )


def _do_unmark_tier2(sid: str, cfg: dict, reason: str) -> ActionLog:
    """Clear the Tier 2 candidate flag — the prior week's prediction
    didn't bear out, so the agent retracts it. We keep the
    `tier2_marked_at` + `tier2_thesis` fields nulled so the dashboard
    border drops away cleanly on the next render."""
    cfg["tier2_candidate"] = False
    cfg["tier2_marked_at"] = None
    cfg["tier2_thesis"] = ""
    _write_config(sid, cfg)
    return ActionLog(sid, None, "unmark-tier2-candidate", True, reason, {})


def _do_promote(
    sid: str,
    region: str,
    cfg: dict,
    entry: dict,
    m: StrategyMetrics | None,
    configs: dict[str, dict],
    reason: str,
) -> ActionLog:
    if entry.get("tier") != "shadow":
        return ActionLog(sid, region, "promote", False, f"Not on shadow tier in {region} — nothing to promote from", {})
    if not m or not _meets_promotion(m, entry):
        return ActionLog(sid, region, "promote", False, "Does not meet promotion criteria", {"metrics": _metrics_to_dict(m) if m else None})

    # UK-EU promotions go to T212-paper (single shared slot=1, no
    # slot allocation — the gate is the total capital_gbp budget
    # across all T212-paper strategies).
    if region == "uk-eu":
        candidate_capital = float(cfg.get("capital_gbp") or 0.0)
        committed = _t212_committed_capital(configs)
        if committed + candidate_capital > T212_PROMOTE_BUDGET_HEADROOM_GBP:
            return ActionLog(
                sid, region, "promote", False,
                (f"T212 auto-promote budget exceeded: committed £{committed:,.0f} + "
                 f"this strategy's £{candidate_capital:,.0f} would clear the "
                 f"£{int(T212_PROMOTE_BUDGET_HEADROOM_GBP):,} ceiling"),
                {"committed_gbp": committed, "candidate_gbp": candidate_capital},
            )
        entry["tier"] = "trading212-paper"
        entry["t212_slot"] = 1
        _write_config(sid, cfg)
        return ActionLog(
            sid, region, "promote", True, reason,
            {"target_tier": "trading212-paper", "t212_slot": 1,
             "committed_gbp_after": committed + candidate_capital},
        )

    # US (and any other region) promotions go to alpaca-paper with a
    # numbered slot pool.
    free_slot = _next_free_slot(configs)
    if free_slot is None:
        return ActionLog(sid, region, "promote", False, f"No free Alpaca slot (max={MAX_ALPACA_SLOTS})", {})

    entry["tier"] = "alpaca-paper"
    entry["alpaca_slot"] = free_slot
    _write_config(sid, cfg)
    return ActionLog(sid, region, "promote", True, reason, {"target_tier": "alpaca-paper", "alpaca_slot": free_slot})


def _t212_committed_capital(configs: dict[str, dict]) -> float:
    """Sum of `capital_gbp` across every strategy already on
    `trading212-paper` (across all its runs_in entries). The T212 demo
    account is one shared £50k pool, so this is the budget gate."""
    total = 0.0
    for cfg in configs.values():
        on_t212 = False
        for entry in _regions_for_config(cfg):
            if entry.get("tier") == "trading212-paper":
                on_t212 = True
                break
        if on_t212:
            try:
                total += float(cfg.get("capital_gbp") or 0.0)
            except (TypeError, ValueError):
                continue
    return total


def _do_demote(
    sid: str,
    region: str,
    cfg: dict,
    entry: dict,
    m: StrategyMetrics | None,
    reason: str,
) -> ActionLog:
    tier = entry.get("tier")
    if tier not in ("alpaca-paper", "trading212-paper"):
        return ActionLog(
            sid, region, "demote", False,
            f"Not on a demotable broker tier in {region} (tier={tier}) — nothing to demote from",
            {},
        )
    if not m or not _meets_demotion(m, entry):
        return ActionLog(sid, region, "demote", False, "Does not meet demotion criteria", {"metrics": _metrics_to_dict(m) if m else None})

    details: dict = {"from_tier": tier}
    if tier == "alpaca-paper":
        slot = entry.get("alpaca_slot")
        cleared = _try_clear_slot(slot) if slot else False
        entry["alpaca_slot"] = None
        details.update({"slot_cleared": cleared, "previous_slot": slot, "slot_kind": "alpaca"})
    else:  # trading212-paper
        slot = entry.get("t212_slot")
        # No automatic T212 broker-clear: any open positions on this slot
        # ride out their scheduled exit naturally on the next exit cron.
        # We just free the config so the next entry run no longer routes
        # this strategy to T212 — same semantic as the Alpaca path, just
        # without the active cancellation call.
        entry["t212_slot"] = None
        details.update({"slot_cleared": False, "previous_slot": slot, "slot_kind": "t212"})

    entry["tier"] = "shadow"
    _write_config(sid, cfg)
    return ActionLog(sid, region, "demote", True, reason, details)


_SAFE_VARIANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,40}$")


def _has_t212_live(cfg: dict) -> bool:
    """True if any runs_in entry (or the top-level tier) is t212-live.
    Used to refuse spawn-variant from live-tier strategies."""
    if (cfg.get("tier") or "").lower() == "t212-live":
        return True
    for entry in (cfg.get("runs_in") or []):
        if isinstance(entry, dict) and (entry.get("tier") or "").lower() == "t212-live":
            return True
    return False


def _do_spawn(raw: dict, configs: dict[str, dict]) -> ActionLog:
    parent_id = raw.get("strategy_id")
    variant_id = raw.get("variant_id") or f"{parent_id}-v2"
    reason = str(raw.get("reason", "")).strip()

    if parent_id not in configs:
        return ActionLog(parent_id or "?", None, "spawn-variant", False, "Unknown parent strategy", {})
    if variant_id in configs:
        return ActionLog(parent_id, None, "spawn-variant", False, f"Variant id {variant_id} already exists", {})

    # Validate variant_id is filesystem-safe. The LLM occasionally
    # suggests names with spaces, slashes, or capitals — those would
    # create awkward strategy directories or shell-quoting bugs.
    if not _SAFE_VARIANT_ID_RE.match(variant_id):
        return ActionLog(
            parent_id, None, "spawn-variant", False,
            f"variant_id '{variant_id}' is invalid — must match {_SAFE_VARIANT_ID_RE.pattern}",
            {},
        )

    n_active = sum(1 for c in configs.values() if c.get("active"))
    if n_active >= MAX_TOTAL_STRATEGIES:
        return ActionLog(parent_id, None, "spawn-variant", False, f"Active strategy cap reached ({MAX_TOTAL_STRATEGIES})", {})

    parent_dir = _strategies_dir() / parent_id
    variant_dir = _strategies_dir() / variant_id
    if not parent_dir.exists():
        return ActionLog(parent_id, None, "spawn-variant", False, f"Parent dir missing: {parent_dir}", {})

    # If the parent has any t212-live (Tier 2) configuration, refuse to
    # spawn — variants must start on shadow but we shouldn't be
    # propagating live-tier state through the spawn machinery at all
    # (defense against a bug that lets it slip through).
    parent_cfg_snapshot = configs[parent_id]
    if _has_t212_live(parent_cfg_snapshot):
        return ActionLog(
            parent_id, None, "spawn-variant", False,
            "Parent has tier=t212-live in runs_in — refusing to spawn variants from live strategies",
            {},
        )

    shutil.copytree(parent_dir, variant_dir)

    # Variant config: load parent's, override, force shadow tier across all regions
    variant_cfg = dict(configs[parent_id])
    overrides = raw.get("config_overrides")
    if isinstance(overrides, dict):
        for k, v in overrides.items():
            if k in TUNABLE_FIELDS:
                lo, hi = TUNABLE_FIELDS[k]
                try:
                    v = type(lo)(v)
                    v = max(lo, min(hi, v))
                except (TypeError, ValueError):
                    continue
                variant_cfg[k] = v
    variant_cfg["id"] = variant_id
    variant_cfg["display_name"] = f"{variant_cfg.get('display_name', parent_id)} (variant)"
    variant_cfg["active"] = True
    # Force every region this variant inherits to shadow tier
    if isinstance(variant_cfg.get("runs_in"), list):
        new_runs = []
        for entry in variant_cfg["runs_in"]:
            if not isinstance(entry, dict):
                continue
            ne = dict(entry)
            ne["tier"] = "shadow"
            ne["alpaca_slot"] = None
            new_runs.append(ne)
        variant_cfg["runs_in"] = new_runs
    else:
        variant_cfg["tier"] = "shadow"
        variant_cfg["alpaca_slot"] = None
    variant_cfg["description"] = (
        f"Auto-spawned variant of {parent_id} on {date.today().isoformat()} by the "
        f"weekly evolution agent. Reason: {reason}"
    )
    _write_config(variant_id, variant_cfg)

    # Verify the new variant loads cleanly via the registry before we
    # declare success. Catches malformed configs (missing required
    # fields, broken yaml) before the next pipeline run tries to use
    # the variant and fails opaquely.
    try:
        from trading_bot.strategy.registry import load_strategy_config
        load_strategy_config(variant_id)
    except Exception as e:
        # Roll back — delete the dir + reject the spawn so weekly runs
        # don't accumulate broken variants over time.
        shutil.rmtree(variant_dir, ignore_errors=True)
        return ActionLog(
            parent_id, None, "spawn-variant", False,
            f"Variant {variant_id} failed post-spawn registry load: {e}", {},
        )

    # Append the addendum to the variant's deep_analysis.md (parent stays untouched)
    addendum = raw.get("deep_analysis_addendum")
    if isinstance(addendum, str) and addendum.strip():
        prompts_dir = variant_dir / "prompts"
        deep_path = prompts_dir / "deep_analysis.md"
        if deep_path.exists():
            existing = deep_path.read_text()
            new_text = (
                existing.rstrip()
                + "\n\n## Variant addendum\n\n"
                + addendum.strip()
                + "\n"
            )
            deep_path.write_text(new_text)

    return ActionLog(parent_id, None, "spawn-variant", True, reason, {
        "variant_id": variant_id,
        "addendum_applied": bool(addendum),
    })


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

def _meets_promotion(m: StrategyMetrics | None, entry: dict) -> bool:
    if not m or entry.get("tier") != "shadow":
        return False
    if m.n_trades < PROMOTION_MIN_TRADES:
        return False
    if m.hit_rate < PROMOTION_MIN_HIT_RATE:
        return False
    if m.total_pnl_gbp <= PROMOTION_MIN_PNL_GBP:
        return False
    if m.ic is None or m.ic < PROMOTION_MIN_IC:
        return False
    # Phase 9A + 10C — the lower 95% CI bound on IC (Fisher z-transform)
    # must also clear PROMOTION_MIN_IC_LOWER (separate from point-estimate
    # threshold so the lower-bound gate can be tuned independently).
    ic_lower = _ic_lower_bound(m.ic, m.n_predictions_graded)
    if ic_lower is None or ic_lower < PROMOTION_MIN_IC_LOWER:
        return False
    return True


def _ic_lower_bound(ic: float, n: int) -> float | None:
    """Lower 95% CI for the IC using Fisher's z-transform.

    z = 0.5 * ln((1+r)/(1-r));  SE_z = 1/sqrt(n-3)
    z_lower = z - 1.96 * SE_z
    r_lower = (e^(2z) - 1) / (e^(2z) + 1)

    Returns None when the sample is too small (<5 graded predictions
    we can't credibly compute an interval) or when r is at the
    boundary (|r|=1 means a degenerate sample).
    """
    if n < 5 or ic is None:
        return None
    import math
    if abs(ic) >= 0.999:
        return ic                # boundary case; return point estimate
    try:
        z = 0.5 * math.log((1 + ic) / (1 - ic))
        se = 1.0 / math.sqrt(n - 3)
    except (ValueError, ZeroDivisionError):
        return None
    z_lower = z - PROMOTION_IC_CI_Z * se
    e2z = math.exp(2 * z_lower)
    return (e2z - 1) / (e2z + 1)


def _meets_demotion(m: StrategyMetrics | None, entry: dict) -> bool:
    # Both broker paper tiers are demotable back to shadow when a
    # strategy stops earning its slot. `t212-live` stays out — that
    # path requires human approval.
    if not m or entry.get("tier") not in ("alpaca-paper", "trading212-paper"):
        return False
    if m.max_drawdown_pct <= DEMOTION_MAX_DRAWDOWN_PCT:
        return True
    if m.n_trades >= PROMOTION_MIN_TRADES and m.hit_rate < DEMOTION_MIN_HIT_RATE:
        return True
    return False


# ---------------------------------------------------------------------------
# Slot management + config I/O
# ---------------------------------------------------------------------------

def _find_region_entry(cfg: dict, region: str) -> dict | None:
    """Return a *reference* to the runs_in entry for `region`, or build a
    synthetic top-level wrapper for single-region configs. Mutations to the
    returned dict propagate back into cfg (the caller then writes cfg back
    to disk). For single-region configs we mutate cfg directly, so we wrap
    cfg itself in a thin view to keep the call-site symmetric."""
    runs_in = cfg.get("runs_in")
    if isinstance(runs_in, list):
        for entry in runs_in:
            if isinstance(entry, dict) and entry.get("region") == region:
                return entry
        return None
    # Single-region path — cfg's top-level region must match
    if cfg.get("region", "us") != region:
        return None
    return cfg  # mutating cfg directly is the right thing


def _next_free_slot(configs: dict[str, dict]) -> int | None:
    """Scan every runs_in entry and every top-level config for used slots."""
    used: set[int] = set()
    for cfg in configs.values():
        for entry in _regions_for_config(cfg):
            slot = entry.get("alpaca_slot")
            if slot is not None and entry.get("tier") == "alpaca-paper":
                used.add(int(slot))
    for slot in range(1, MAX_ALPACA_SLOTS + 1):
        if slot not in used:
            return slot
    return None


def _free_alpaca_slots(snapshot: list[dict]) -> list[int]:
    used = {
        int(s["alpaca_slot"])
        for s in snapshot
        if s.get("tier") == "alpaca-paper" and s.get("alpaca_slot") is not None
    }
    return [s for s in range(1, MAX_ALPACA_SLOTS + 1) if s not in used]


def _try_clear_slot(slot: int) -> bool:
    """Best-effort clear of a slot via AlpacaPaperExecutor. Returns True on
    success, False if creds unavailable or call failed (we still proceed
    with the demotion — better to free the config slot than block on a
    transient Alpaca outage)."""
    try:
        from trading_bot.executor.alpaca_paper import AlpacaPaperExecutor
        AlpacaPaperExecutor(slot=slot).clear_slot()
        return True
    except Exception as e:
        log.warning("clear_slot(%d) during demotion failed: %s", slot, e)
        return False


def _load_all_configs() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in _strategies_dir().glob("*/config.yaml"):
        try:
            raw = yaml.safe_load(p.read_text())
            if isinstance(raw, dict) and raw.get("id"):
                out[raw["id"]] = raw
        except yaml.YAMLError as e:
            log.warning("Could not load %s: %s", p, e)
    return out


def _write_config(strategy_id: str, cfg: dict) -> None:
    path = _strategies_dir() / strategy_id / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


# ---------------------------------------------------------------------------
# State file logging
# ---------------------------------------------------------------------------

def _evolution_log_path() -> Path:
    p = STATE_ROOT
    p.mkdir(parents=True, exist_ok=True)
    return p / "evolution.md"


def _lessons_path() -> Path:
    p = STATE_ROOT
    p.mkdir(parents=True, exist_ok=True)
    return p / "lessons.md"


def _read_lessons() -> str:
    path = _lessons_path()
    if not path.exists():
        return ""
    # Cap at last 4000 chars to keep the prompt bounded
    text = path.read_text()
    return text[-4000:] if len(text) > 4000 else text


def _append_evolution_log(today: date, actions: list[ActionLog]) -> None:
    lines = [f"\n## Weekly evolution — {today.isoformat()}\n"]
    if not actions:
        lines.append("_No actions._\n")
    for a in actions:
        status = "✅ applied" if a.applied else "⏭️ skipped"
        scope = f"{a.strategy_id}@{a.region}" if a.region else a.strategy_id
        lines.append(
            f"- **{scope}** · `{a.action}` · {status} — {a.reason}"
        )
        if a.details:
            lines.append(f"  - details: `{json.dumps(a.details)}`")
    with _evolution_log_path().open("a") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# GitHub Issue for tier-2 requests + weekly summary
# ---------------------------------------------------------------------------

def _maybe_file_issue(
    today: date,
    applied: list[ActionLog],
    tier_2_requests: list[ActionLog],
) -> str | None:
    """Open one GitHub Issue per weekly run summarising what happened and
    flagging Tier 2 requests. Uses the `gh` CLI which is pre-installed on
    GH Actions runners (no-op if `gh` isn't available locally)."""
    def _scope(a: ActionLog) -> str:
        return f"{a.strategy_id}@{a.region}" if a.region else a.strategy_id

    body_parts = [f"## Weekly evolution — {today.isoformat()}\n"]
    if tier_2_requests:
        body_parts.append("### Tier 2 promotion requests (your approval needed)\n")
        for a in tier_2_requests:
            body_parts.append(f"- **{_scope(a)}** — {a.reason}")
        body_parts.append("")

    auto_applied = [a for a in applied if a.applied and a.action != "request-tier-2"]
    if auto_applied:
        body_parts.append("### Auto-applied (FYI)\n")
        for a in auto_applied:
            body_parts.append(f"- **{_scope(a)}** · `{a.action}` — {a.reason}")
        body_parts.append("")

    skipped = [a for a in applied if not a.applied and a.action != "request-tier-2"]
    if skipped:
        body_parts.append("### Skipped\n")
        for a in skipped:
            body_parts.append(f"- **{_scope(a)}** · `{a.action}` — {a.reason}")

    body = "\n".join(body_parts).strip() or "_No activity._"
    title = f"Evolution {today.isoformat()} — {len(tier_2_requests)} tier-2 request(s), {len(auto_applied)} auto-applied"

    labels = ["evolution"]
    if tier_2_requests:
        labels.append("promotion")

    try:
        completed = subprocess.run(
            ["gh", "issue", "create", "--title", title, "--body", body, "--label", ",".join(labels)],
            capture_output=True, text=True, timeout=30,
        )
        if completed.returncode != 0:
            log.warning("gh issue create failed: %s", completed.stderr[:200])
            return None
        return completed.stdout.strip()
    except FileNotFoundError:
        log.warning("gh CLI not available; skipping issue creation")
        return None
    except Exception as e:
        log.warning("Issue creation errored: %s", e)
        return None
