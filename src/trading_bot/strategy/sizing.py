"""Phase 8A + 8B — post-process LLM picks.

The strategy LLM returns picks with an `allocation_pct` it chose. Two
transforms happen before they become `TradeIntent`s:

1. **Volatility-aware sizing (8A)** — rewrite `allocation_pct` so each
   position carries the same daily-risk budget. High-vol names get
   smaller allocations; low-vol names get larger. Clamped to the
   strategy's `max_position_pct` and `min_position_gbp` so neither
   extreme bites.

2. **FX cost gate (8B)** — drop picks where the LLM's own
   `predicted_return_pct` is below `cost_gate_multiplier × round-trip
   cost`. This is a hard backstop; the LLM already sees the cost line
   per candidate in the prompt and is asked to subtract it, but it
   occasionally tries trades the edge can't cover.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from trading_bot.tools.fees import (
    estimate_round_trip_cost_pct,
    yf_ticker_classify,
)

log = logging.getLogger(__name__)


@dataclass
class PickAdjustment:
    """One pick's path through the post-processing. For visibility in
    the logs — every pick gets one of these so we can audit what
    happened to the LLM's original sizing."""
    ticker: str
    original_alloc_pct: float
    adjusted_alloc_pct: float
    atr_pct: float                          # ATR as % of close
    predicted_return_pct: float | None      # from the predictions list, if available
    round_trip_cost_pct: float
    dropped: bool = False
    drop_reason: str = ""
    sizing_reason: str = ""


def adjust_picks(
    picks_raw: list[dict],
    *,
    candidates: list,                       # list of TechnicalIndicators
    predictions: dict | None,               # ticker → prediction dict (from LLM)
    cfg,                                    # StrategyConfig
) -> tuple[list[dict], list[PickAdjustment]]:
    """Apply vol-aware sizing + FX-cost gate. Returns the filtered +
    rewritten picks alongside an audit log of every adjustment."""
    # Index candidates by ticker for fast lookup
    cand_by_ticker = {c.ticker: c for c in candidates}
    predictions = predictions or {}

    # Phase 10A — load recently trailed-out tickers. Re-picking these
    # within the window pays a fresh round-trip (esp. stamp duty), so
    # we ADD one extra round-trip cost to the gate threshold for
    # them. (Not a multiplier on cost_pct — that would compound with
    # cost_gate_multiplier and over-penalise.)
    try:
        from trading_bot.state.trail_exits import load_recent_trail_exits
        trailed_recently = load_recent_trail_exits(days=3)
    except Exception:
        trailed_recently = {}

    adjustments: list[PickAdjustment] = []
    out: list[dict] = []
    for item in picks_raw:
        if not isinstance(item, dict):
            continue
        ticker = item.get("ticker")
        try:
            orig_alloc = float(item.get("allocation_pct") or 0)
        except (TypeError, ValueError):
            continue
        if not ticker or orig_alloc <= 0:
            continue

        cand = cand_by_ticker.get(ticker)
        if not cand:
            # LLM picked a ticker that wasn't in stage-2 candidates —
            # rare but possible. Pass through unchanged.
            out.append(item)
            continue

        # 8A — vol-aware sizing
        adjusted_alloc, atr_pct, sizing_reason = _vol_adjusted_alloc(
            orig_alloc=orig_alloc, cand=cand, cfg=cfg,
        )

        # 8B — FX cost gate. Drop if predicted return doesn't beat
        # `cost_gate_multiplier × round-trip cost`.
        notional_gbp = cfg.capital_gbp * (adjusted_alloc / 100.0)
        exch, ccy = yf_ticker_classify(ticker)
        cost_est = estimate_round_trip_cost_pct(
            tier=cfg.tier, currency=ccy, exchange=exch,
            instrument_type="share",
            notional_gbp=max(notional_gbp, 1.0),
            quantity=notional_gbp / max(cand.close, 1.0),
        )
        cost_pct = cost_est["total_pct"] * 100.0    # to percentage

        # Phase 10A — re-entry surcharge. Look up case-insensitive
        # because trail_exits stores upper-cased tickers.
        trail_record = trailed_recently.get((ticker or "").upper())

        pred = predictions.get(ticker, {}) if isinstance(predictions, dict) else {}
        pred_return = pred.get("predicted_return_pct")
        try:
            pred_return_f = float(pred_return) if pred_return is not None else None
        except (TypeError, ValueError):
            pred_return_f = None

        adj = PickAdjustment(
            ticker=ticker,
            original_alloc_pct=orig_alloc,
            adjusted_alloc_pct=adjusted_alloc,
            atr_pct=atr_pct,
            predicted_return_pct=pred_return_f,
            round_trip_cost_pct=cost_pct,
            sizing_reason=sizing_reason,
        )

        if pred_return_f is not None:
            # Base threshold: the configured multiplier × round-trip cost.
            # If we're re-entering a recently trailed-out name, we DO pay
            # the round-trip cost a second time within the pair — add ONE
            # extra round-trip cost on top of the configured threshold
            # (NOT a 2× multiplier on cost_pct, which would compound with
            # cost_gate_multiplier to 4× the intended barrier).
            threshold = cfg.cost_gate_multiplier * cost_pct
            if trail_record:
                threshold += cost_pct
            if pred_return_f < threshold:
                adj.dropped = True
                base_threshold = cfg.cost_gate_multiplier * cost_pct
                if trail_record:
                    adj.drop_reason = (
                        f"predicted {pred_return_f:+.2f}% < base {base_threshold:.2f}% "
                        f"+ re-entry surcharge {cost_pct:.2f}% = {threshold:.2f}% threshold"
                    )
                else:
                    adj.drop_reason = (
                        f"predicted {pred_return_f:+.2f}% < "
                        f"{cfg.cost_gate_multiplier:.1f}× round-trip cost "
                        f"{cost_pct:.2f}% (= {threshold:.2f}% threshold)"
                    )
                adjustments.append(adj)
                continue
        else:
            # No prediction available — skip the gate, log it. Defensive
            # against a malformed stage-2 response.
            adj.drop_reason = "no predicted_return_pct available; cost gate skipped"

        adjustments.append(adj)
        item = dict(item)
        item["allocation_pct"] = adjusted_alloc
        out.append(item)

    return out, adjustments


def _vol_adjusted_alloc(*, orig_alloc: float, cand, cfg) -> tuple[float, float, str]:
    """Replace `orig_alloc` with a size that targets `cfg.target_daily_risk_pct`
    of capital per ATR. Clamps to [min_position_gbp/capital, max_position_pct].

    Returns (new_pct, atr_pct, reason_str). The LLM's original alloc is
    used as a *tiebreaker* — among multiple picks we still respect the
    LLM's relative conviction, by scaling all sizes proportionally to
    fit the total LLM-allocated budget (so a 5%/3%/2% split stays
    5:3:2 in relative terms, but adjusted globally for vol)."""
    close = float(cand.close)
    atr = float(cand.atr_14)
    if close <= 0:
        return orig_alloc, 0.0, "fallback: close <= 0"
    atr_pct = (atr / close) * 100.0
    if atr_pct <= 0:
        return orig_alloc, atr_pct, "fallback: zero ATR"

    # Risk budget — daily £ we're willing to lose per position at 1 ATR
    risk_budget_gbp = cfg.capital_gbp * (cfg.target_daily_risk_pct / 100.0)
    # Position size that puts 1 ATR move at exactly risk_budget_gbp
    position_gbp = risk_budget_gbp / (atr_pct / 100.0)
    alloc_pct = (position_gbp / cfg.capital_gbp) * 100.0

    # Clamp — apply MIN first, then MAX last. If min_position_gbp is
    # configured such that min_alloc > max_position_pct (a config
    # mistake), max wins, and the position gets dropped downstream when
    # the LLM-side validation re-checks the bounds.
    min_alloc = (cfg.min_position_gbp / cfg.capital_gbp) * 100.0
    capped = False
    if alloc_pct < min_alloc:
        alloc_pct = min_alloc
        capped = True
    if alloc_pct > cfg.max_position_pct:
        alloc_pct = cfg.max_position_pct
        capped = True

    reason = (
        f"ATR {atr_pct:.2f}% → risk-parity {position_gbp:.0f}£ "
        f"({alloc_pct:.1f}%)"
        + (" [clamped]" if capped else "")
    )
    return round(alloc_pct, 2), round(atr_pct, 3), reason


def persist_adjustments(strategy_id: str, on_date, adjustments: list[PickAdjustment]) -> None:
    """Phase 10B — write adjustments to
    `state/pick_adjustments/{date}.{strategy_id}.jsonl` so the weekly
    evolution agent can see how often the cost gate dropped picks for
    each strategy."""
    if not adjustments:
        return
    from dataclasses import asdict
    from datetime import date as _date
    import json as _json
    from trading_bot.state.paths import STATE_ROOT
    if hasattr(on_date, "isoformat"):
        iso = on_date.isoformat()
    else:
        iso = str(on_date)
    d = STATE_ROOT / "pick_adjustments"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{iso}.{strategy_id}.jsonl"
    try:
        with p.open("a") as f:
            for a in adjustments:
                f.write(_json.dumps(asdict(a)) + "\n")
    except OSError:
        pass


def format_adjustment_log(adjustments: list[PickAdjustment]) -> str:
    """Compact one-line-per-pick log of what post-processing did. Goes
    into the strategy's run log so we can audit drops + resizings."""
    if not adjustments:
        return "(no picks adjusted)"
    lines = []
    for a in adjustments:
        if a.dropped:
            lines.append(f"  · DROPPED {a.ticker}: {a.drop_reason}")
        else:
            note = a.sizing_reason
            if a.original_alloc_pct != a.adjusted_alloc_pct:
                note += f" (was {a.original_alloc_pct:.1f}% → {a.adjusted_alloc_pct:.1f}%)"
            lines.append(f"  · KEPT    {a.ticker}: {note}")
    return "\n".join(lines)
