"""Wave 6 — weekly strategy-evolution agent.

Reads each active strategy's rolling metrics (from meta.metrics), gathers
prompts + configs + recent lessons, asks Claude for action recommendations,
then auto-executes anything that stays on Tier 0 / Tier 1:

  - keep             — no change
  - tune             — edit config fields within safety bounds
  - promote          — Tier 0 (shadow) → Tier 1 (alpaca-paper) + slot assign
  - demote           — Tier 1 (alpaca-paper) → Tier 0 (shadow) + clear_slot
  - spawn-variant    — clone an existing strategy with config + prompt diffs
  - request-tier-2   — file GitHub Issue for human approval (never auto)

Tier 2 (live) is fully off-limits to this agent. Anything involving real
money requires explicit human approval per sparknotes design.
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


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration: thresholds & safety bounds
# ---------------------------------------------------------------------------

MAX_ALPACA_SLOTS = 3  # how many paper accounts we've provisioned
MAX_TOTAL_STRATEGIES = 12  # ceiling on spawned variants
PROMOTION_MIN_TRADES = 10
PROMOTION_MIN_HIT_RATE = 0.50
PROMOTION_MIN_PNL_GBP = 0.0
PROMOTION_MIN_IC = 0.05
DEMOTION_MAX_DRAWDOWN_PCT = -10.0
DEMOTION_MIN_HIT_RATE = 0.40

# Fields the agent is allowed to tune on Tier 0/1 strategies
TUNABLE_FIELDS = {
    "max_positions": (1, 10),
    "max_position_pct": (5.0, 60.0),
    "min_position_gbp": (10.0, 1000.0),
    "stop_loss_pct": (-15.0, -1.0),
    "take_profit_pct": (1.0, 20.0),
    "capital_gbp": (1000.0, 30000.0),
}


@dataclass
class ActionLog:
    strategy_id: str
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
        recommendations = run_claude_for_json(prompt, model="sonnet")
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
        log_entry = _apply_action(raw, metrics=metrics, configs=configs)
        if log_entry is None:
            continue
        if log_entry.action == "request-tier-2":
            pending_tier_2.append(log_entry)
        applied.append(log_entry)

    _append_evolution_log(today, applied)
    issue_url = _maybe_file_issue(today, applied, pending_tier_2)

    summary = {
        "week_end": today.isoformat(),
        "n_strategies": len(metrics),
        "n_actions_applied": sum(1 for a in applied if a.applied),
        "n_actions_skipped": sum(1 for a in applied if not a.applied),
        "n_tier_2_requests": len(pending_tier_2),
        "issue_url": issue_url,
    }
    log.info("evolution: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Snapshot + prompt
# ---------------------------------------------------------------------------

def _build_snapshot(
    configs: dict[str, dict],
    metrics: dict[str, StrategyMetrics],
) -> list[dict]:
    rows = []
    for sid, cfg in configs.items():
        m = metrics.get(sid)
        rows.append({
            "id": sid,
            "tier": cfg.get("tier"),
            "alpaca_slot": cfg.get("alpaca_slot"),
            "active": cfg.get("active"),
            "max_positions": cfg.get("max_positions"),
            "max_position_pct": cfg.get("max_position_pct"),
            "stop_loss_pct": cfg.get("stop_loss_pct"),
            "take_profit_pct": cfg.get("take_profit_pct"),
            "capital_gbp": cfg.get("capital_gbp"),
            "metrics": _metrics_to_dict(m) if m else None,
            "meets_promotion_criteria": _meets_promotion(m, cfg) if m else False,
            "meets_demotion_criteria": _meets_demotion(m, cfg) if m else False,
        })
    return rows


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


def _build_prompt(today: date, snapshot: list[dict], lessons: str) -> str:
    free_slots = _free_alpaca_slots(snapshot)
    n_active = sum(1 for s in snapshot if s.get("active"))

    snapshot_json = json.dumps(snapshot, indent=2)
    return f"""You are the weekly evolution agent for the trading bot. Today is
{today.isoformat()}. Below is a 14-day rolling snapshot of every strategy:
their config, metrics, and whether they currently meet promotion / demotion
criteria.

## Your authority

You can auto-execute these actions on **Tier 0 (shadow)** and **Tier 1
(alpaca-paper)** strategies:
- `keep` — no change
- `tune` — edit specific fields within bounds
- `promote` — Tier 0 → Tier 1 (we'll assign a free Alpaca slot)
- `demote` — Tier 1 → Tier 0 (we'll clear the slot)
- `spawn-variant` — clone an existing strategy with prompt + config diffs

For **Tier 2 (live)** strategies you can ONLY recommend with `request-tier-2`
— that opens a GitHub Issue for the user to approve.

## Safety bounds

Tunable fields (clipped to these ranges if you exceed them):
{json.dumps({k: list(v) for k, v in TUNABLE_FIELDS.items()}, indent=2)}

Other constraints:
- Free Alpaca slots available: {free_slots}
- Currently active strategies: {n_active} of max {MAX_TOTAL_STRATEGIES}
- Don't spawn a variant if it'd push active total over the cap
- Promotion requires `meets_promotion_criteria=true` AND a free slot
- Demotion requires `meets_demotion_criteria=true`
- Spawned variants start on Tier 0 shadow, must have a different `id`

## Current snapshot

```json
{snapshot_json}
```

## Recent lessons (failures / corrections)

{lessons if lessons else "(none yet)"}

## Required output

A JSON object with one key `actions`, a list of action objects:

```json
{{
  "actions": [
    {{ "strategy_id": "<id>", "action": "keep", "reason": "1 sentence" }},
    {{ "strategy_id": "<id>", "action": "tune",
       "changes": {{ "max_positions": 4, "stop_loss_pct": -2.5 }},
       "reason": "1-2 sentences citing the metric you're responding to" }},
    {{ "strategy_id": "<id>", "action": "promote", "reason": "..." }},
    {{ "strategy_id": "<id>", "action": "demote", "reason": "..." }},
    {{ "strategy_id": "<parent_id>", "action": "spawn-variant",
       "variant_id": "<parent_id>-v2-or-some-meaningful-suffix",
       "config_overrides": {{ "max_positions": 3, ... }},
       "deep_analysis_addendum": "Markdown text appended to the parent's deep_analysis.md to express the variant's bias",
       "reason": "1-2 sentences — why this variant has a meaningfully different edge from the parent" }},
    {{ "strategy_id": "<id>", "action": "request-tier-2", "reason": "Detailed case for live promotion" }}
  ]
}}
```

Be conservative. Most strategies should be `keep` most weeks. Promote only
when the data clearly supports it; demote only when the trend is unmistakable.
Spawning variants should be rare — only when you see a genuinely different
edge that the parent's prompt isn't capturing.
"""


# ---------------------------------------------------------------------------
# Action application
# ---------------------------------------------------------------------------

def _apply_action(
    raw: dict,
    *,
    metrics: dict[str, StrategyMetrics],
    configs: dict[str, dict],
) -> ActionLog | None:
    sid = raw.get("strategy_id")
    action = (raw.get("action") or "").strip().lower()
    reason = str(raw.get("reason", "")).strip()
    if not sid or not action:
        return None

    if action == "spawn-variant":
        return _do_spawn(raw, configs)

    if sid not in configs:
        return ActionLog(sid, action, False, f"Unknown strategy_id; ignored. ({reason})", {})

    cfg = configs[sid]

    if cfg.get("tier") == "t212-live":
        return ActionLog(sid, action, False, "Tier 2 strategy — agent has no authority", {})

    if action == "keep":
        return ActionLog(sid, action, True, reason, {})

    if action == "tune":
        return _do_tune(sid, raw, cfg, reason)

    if action == "promote":
        return _do_promote(sid, cfg, metrics.get(sid), configs, reason)

    if action == "demote":
        return _do_demote(sid, cfg, metrics.get(sid), reason)

    if action == "request-tier-2":
        # Recorded only; the issue creator picks these up
        return ActionLog(sid, action, False, reason, {"requires_human_approval": True})

    return ActionLog(sid, action, False, f"Unknown action; ignored. ({reason})", {})


def _do_tune(sid: str, raw: dict, cfg: dict, reason: str) -> ActionLog:
    changes_raw = raw.get("changes")
    if not isinstance(changes_raw, dict) or not changes_raw:
        return ActionLog(sid, "tune", False, "No changes specified", {})

    clamped: dict = {}
    rejected: dict = {}
    for field_name, requested in changes_raw.items():
        if field_name not in TUNABLE_FIELDS:
            rejected[field_name] = "field not tunable"
            continue
        lo, hi = TUNABLE_FIELDS[field_name]
        try:
            val = type(lo)(requested)
        except (TypeError, ValueError):
            rejected[field_name] = "wrong type"
            continue
        clamped[field_name] = max(lo, min(hi, val))

    if not clamped:
        return ActionLog(sid, "tune", False, f"No applicable changes. Rejected: {rejected}", {})

    cfg.update(clamped)
    _write_config(sid, cfg)
    return ActionLog(sid, "tune", True, reason, {"applied": clamped, "rejected": rejected})


def _do_promote(
    sid: str,
    cfg: dict,
    m: StrategyMetrics | None,
    configs: dict[str, dict],
    reason: str,
) -> ActionLog:
    if cfg.get("tier") != "shadow":
        return ActionLog(sid, "promote", False, "Not on shadow tier — nothing to promote from", {})
    if not m or not _meets_promotion(m, cfg):
        return ActionLog(sid, "promote", False, "Does not meet promotion criteria", {"metrics": _metrics_to_dict(m) if m else None})

    free_slot = _next_free_slot(configs)
    if free_slot is None:
        return ActionLog(sid, "promote", False, f"No free Alpaca slot (max={MAX_ALPACA_SLOTS})", {})

    cfg["tier"] = "alpaca-paper"
    cfg["alpaca_slot"] = free_slot
    _write_config(sid, cfg)
    return ActionLog(sid, "promote", True, reason, {"slot": free_slot})


def _do_demote(sid: str, cfg: dict, m: StrategyMetrics | None, reason: str) -> ActionLog:
    if cfg.get("tier") != "alpaca-paper":
        return ActionLog(sid, "demote", False, "Not on alpaca-paper tier — nothing to demote from", {})
    if not m or not _meets_demotion(m, cfg):
        return ActionLog(sid, "demote", False, "Does not meet demotion criteria", {"metrics": _metrics_to_dict(m) if m else None})

    slot = cfg.get("alpaca_slot")
    cleared = _try_clear_slot(slot) if slot else False

    cfg["tier"] = "shadow"
    cfg["alpaca_slot"] = None
    _write_config(sid, cfg)
    return ActionLog(sid, "demote", True, reason, {"slot_cleared": cleared, "previous_slot": slot})


def _do_spawn(raw: dict, configs: dict[str, dict]) -> ActionLog:
    parent_id = raw.get("strategy_id")
    variant_id = raw.get("variant_id") or f"{parent_id}-v2"
    reason = str(raw.get("reason", "")).strip()

    if parent_id not in configs:
        return ActionLog(parent_id or "?", "spawn-variant", False, "Unknown parent strategy", {})
    if variant_id in configs:
        return ActionLog(parent_id, "spawn-variant", False, f"Variant id {variant_id} already exists", {})

    n_active = sum(1 for c in configs.values() if c.get("active"))
    if n_active >= MAX_TOTAL_STRATEGIES:
        return ActionLog(parent_id, "spawn-variant", False, f"Active strategy cap reached ({MAX_TOTAL_STRATEGIES})", {})

    parent_dir = _strategies_dir() / parent_id
    variant_dir = _strategies_dir() / variant_id
    if not parent_dir.exists():
        return ActionLog(parent_id, "spawn-variant", False, f"Parent dir missing: {parent_dir}", {})

    shutil.copytree(parent_dir, variant_dir)

    # Variant config: load parent's, override, force shadow tier
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
    variant_cfg["tier"] = "shadow"
    variant_cfg["alpaca_slot"] = None
    variant_cfg["active"] = True
    variant_cfg["description"] = (
        f"Auto-spawned variant of {parent_id} on {date.today().isoformat()} by the "
        f"weekly evolution agent. Reason: {reason}"
    )
    _write_config(variant_id, variant_cfg)

    # Append the addendum to the parent's deep_analysis.md (variant inherits parent's prompt
    # but we want the addendum reflected in the variant's copy specifically)
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

    return ActionLog(parent_id, "spawn-variant", True, reason, {
        "variant_id": variant_id,
        "addendum_applied": bool(addendum),
    })


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

def _meets_promotion(m: StrategyMetrics | None, cfg: dict) -> bool:
    if not m or cfg.get("tier") != "shadow":
        return False
    if m.n_trades < PROMOTION_MIN_TRADES:
        return False
    if m.hit_rate < PROMOTION_MIN_HIT_RATE:
        return False
    if m.total_pnl_gbp <= PROMOTION_MIN_PNL_GBP:
        return False
    if m.ic is None or m.ic < PROMOTION_MIN_IC:
        return False
    return True


def _meets_demotion(m: StrategyMetrics | None, cfg: dict) -> bool:
    if not m or cfg.get("tier") != "alpaca-paper":
        return False
    if m.max_drawdown_pct <= DEMOTION_MAX_DRAWDOWN_PCT:
        return True
    if m.n_trades >= PROMOTION_MIN_TRADES and m.hit_rate < DEMOTION_MIN_HIT_RATE:
        return True
    return False


# ---------------------------------------------------------------------------
# Slot management + config I/O
# ---------------------------------------------------------------------------

def _next_free_slot(configs: dict[str, dict]) -> int | None:
    used = {
        c.get("alpaca_slot")
        for c in configs.values()
        if c.get("tier") == "alpaca-paper" and c.get("alpaca_slot") is not None
    }
    for slot in range(1, MAX_ALPACA_SLOTS + 1):
        if slot not in used:
            return slot
    return None


def _free_alpaca_slots(snapshot: list[dict]) -> list[int]:
    used = {
        s.get("alpaca_slot")
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
        lines.append(
            f"- **{a.strategy_id}** · `{a.action}` · {status} — {a.reason}"
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
    body_parts = [f"## Weekly evolution — {today.isoformat()}\n"]
    if tier_2_requests:
        body_parts.append("### Tier 2 promotion requests (your approval needed)\n")
        for a in tier_2_requests:
            body_parts.append(f"- **{a.strategy_id}** — {a.reason}")
        body_parts.append("")

    auto_applied = [a for a in applied if a.applied and a.action != "request-tier-2"]
    if auto_applied:
        body_parts.append("### Auto-applied (FYI)\n")
        for a in auto_applied:
            body_parts.append(f"- **{a.strategy_id}** · `{a.action}` — {a.reason}")
        body_parts.append("")

    skipped = [a for a in applied if not a.applied and a.action != "request-tier-2"]
    if skipped:
        body_parts.append("### Skipped\n")
        for a in skipped:
            body_parts.append(f"- **{a.strategy_id}** · `{a.action}` — {a.reason}")

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
