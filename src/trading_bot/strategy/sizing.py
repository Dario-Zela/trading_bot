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
    infer_instrument_type,
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
    # Phase 12C — record the horizon used at gate time so the audit log
    # can show why a multi-day pick was held to a higher bar.
    hold_days: int = 1


CORRELATION_CLUSTER_THRESHOLD = 0.7
CORRELATION_LOOKBACK_DAYS = 90


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

        # Phase 11A — drop anomalous-price picks before they reach the
        # broker. Catches SNDK-at-$1407 class of yfinance vendor glitches.
        from trading_bot.tools.price_sanity import is_price_anomalous
        bad, reason = is_price_anomalous(close=cand.close, sma_20=cand.sma_20)
        if bad:
            adjustments.append(PickAdjustment(
                ticker=ticker,
                original_alloc_pct=orig_alloc, adjusted_alloc_pct=0.0,
                atr_pct=0.0, predicted_return_pct=None,
                round_trip_cost_pct=0.0,
                dropped=True, drop_reason=f"anomalous price: {reason}",
            ))
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
            instrument_type=infer_instrument_type(ticker),
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

        # Phase 12C — pick up the LLM's requested horizon. The cost gate
        # scales linearly with hold_days so a 5-day pick must beat a 5×
        # higher predicted-return bar than a 1-day pick. Rationale: the
        # round-trip cost is paid once regardless, but the extra days
        # carry extra market-exposure variance — only worth committing
        # capital for if the thesis genuinely has more juice.
        try:
            hold_days = int(item.get("hold_days") or 1)
        except (TypeError, ValueError):
            hold_days = 1
        if hold_days < 1:
            hold_days = 1
        horizon_multiplier = float(hold_days)

        adj = PickAdjustment(
            ticker=ticker,
            original_alloc_pct=orig_alloc,
            adjusted_alloc_pct=adjusted_alloc,
            atr_pct=atr_pct,
            predicted_return_pct=pred_return_f,
            round_trip_cost_pct=cost_pct,
            sizing_reason=sizing_reason,
            hold_days=hold_days,
        )

        if pred_return_f is not None:
            # Base threshold: the configured multiplier × round-trip cost
            # × horizon multiplier (Phase 12C). For hold_days=1 the
            # horizon multiplier is 1.0 and behaviour matches Wave 1.
            # If we're re-entering a recently trailed-out name, we DO pay
            # the round-trip cost a second time within the pair — add ONE
            # extra round-trip cost on top of the configured threshold
            # (NOT a 2× multiplier on cost_pct, which would compound with
            # cost_gate_multiplier to 4× the intended barrier).
            base_threshold = cfg.cost_gate_multiplier * cost_pct * horizon_multiplier
            threshold = base_threshold
            if trail_record:
                threshold += cost_pct
            if pred_return_f < threshold:
                adj.dropped = True
                horizon_tag = (
                    f" × {horizon_multiplier:.0f} (hold {hold_days}d)"
                    if hold_days > 1 else ""
                )
                if trail_record:
                    adj.drop_reason = (
                        f"predicted {pred_return_f:+.2f}% < base {base_threshold:.2f}% "
                        f"{horizon_tag} + re-entry surcharge {cost_pct:.2f}% = "
                        f"{threshold:.2f}% threshold"
                    )
                else:
                    adj.drop_reason = (
                        f"predicted {pred_return_f:+.2f}% < "
                        f"{cfg.cost_gate_multiplier:.1f}× round-trip cost "
                        f"{cost_pct:.2f}%{horizon_tag} (= {threshold:.2f}% threshold)"
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

    # Phase 11E — correlation-aware sizing. Once we know which picks
    # survived the gates above, group them by 90-day return correlation
    # and scale down the allocation of any cluster so the total cluster
    # risk equals the configured per-position risk (rather than summing,
    # which is what naively independent sizing assumes).
    if len(out) >= 2:
        try:
            out, cluster_log = _apply_correlation_discount(out, today=None)
            if cluster_log:
                log.info("Correlation clusters: %s", cluster_log)
        except Exception as e:
            log.debug("Correlation discount skipped: %s", e)

    return out, adjustments


def _apply_correlation_discount(picks: list[dict], *, today=None) -> tuple[list[dict], str]:
    """Cluster picks by 90-day return correlation > threshold and scale
    each cluster's allocations by 1/sqrt(cluster_size). Returns the
    rewritten picks + a short human-readable cluster summary."""
    from datetime import date as _date
    from trading_bot.tools import get_history

    tickers = [p["ticker"] for p in picks if isinstance(p, dict) and p.get("ticker")]
    if len(tickers) < 2:
        return picks, ""

    end = today or _date.today()
    try:
        hist = get_history(tickers, lookback_days=CORRELATION_LOOKBACK_DAYS, end_date=end)
    except Exception:
        return picks, ""

    # Daily returns per ticker, aligned on common dates
    returns_by_ticker: dict[str, dict[str, float]] = {}
    for tkr in tickers:
        bars = hist.get(tkr) or []
        if len(bars) < 10:
            continue
        rets: dict[str, float] = {}
        for i in range(1, len(bars)):
            prev = bars[i - 1].close
            curr = bars[i].close
            if prev > 0:
                rets[str(bars[i].bar_date)] = (curr / prev - 1.0)
        if len(rets) >= 10:
            returns_by_ticker[tkr] = rets
    if len(returns_by_ticker) < 2:
        return picks, ""

    # Pairwise correlation
    def _corr(a: dict[str, float], b: dict[str, float]) -> float | None:
        common = sorted(set(a.keys()) & set(b.keys()))
        if len(common) < 10:
            return None
        xs = [a[d] for d in common]
        ys = [b[d] for d in common]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den_x = (sum((x - mx) ** 2 for x in xs)) ** 0.5
        den_y = (sum((y - my) ** 2 for y in ys)) ** 0.5
        if den_x == 0 or den_y == 0:
            return None
        return num / (den_x * den_y)

    # Union-find over the correlation graph
    parent: dict[str, str] = {t: t for t in returns_by_ticker}

    def find(t: str) -> str:
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    keys = sorted(returns_by_ticker.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            c = _corr(returns_by_ticker[keys[i]], returns_by_ticker[keys[j]])
            if c is not None and c >= CORRELATION_CLUSTER_THRESHOLD:
                union(keys[i], keys[j])

    # Cluster size by representative
    clusters: dict[str, list[str]] = {}
    for t in keys:
        clusters.setdefault(find(t), []).append(t)

    # Apply discount
    summary_parts: list[str] = []
    for rep, members in clusters.items():
        if len(members) < 2:
            continue
        factor = 1.0 / (len(members) ** 0.5)
        for p in picks:
            if p.get("ticker") in members:
                p["allocation_pct"] = round(float(p.get("allocation_pct", 0)) * factor, 2)
        summary_parts.append(f"{{{', '.join(members)}}}×{factor:.2f}")
    return picks, "; ".join(summary_parts)


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
