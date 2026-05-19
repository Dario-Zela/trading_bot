"""Phase 10B — derived inputs for the weekly evolution prompt.

Read-only helpers that pull and aggregate signals scattered across
state/. Each function returns a compact data structure the
evolution_v2 prompt builders can splice in. Designed to be cheap —
no LLM calls, just file reads.

Inputs surfaced:
- `fees_pct_of_gross(sid, region)` — fees ate this share of gross P&L
- `cost_gate_drop_rate(sid)` — fraction of LLM picks the gate dropped
- `verdict_rates_by_source()` — proven/partial/falsified rates per source
- `earnings_gate_hit_rate(sid)` — share of candidates dropped pre-earnings
- `sector_concentration(sid)` — top sectors by notional traded
- `divergent_strategies(snapshot)` — strategies with big region-vs-region gap
- `trail_activation_rate(sid)` — share of exits via stop / trail
- `halt_history_summary()` — count + last reason from halt_history.jsonl
- `regime_subtitles(days)` — trailing daily-news masthead subtitles
- `parent_deep_analysis(sid)` — strategy's own deep_analysis.md content
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from trading_bot.state.paths import STATE_ROOT, ledger_path

log = logging.getLogger(__name__)


_LOOKBACK_DAYS = 14
_VERDICT_LOOKBACK_DAYS = 28
_DIVERGENCE_THRESHOLD_PCT = 5.0           # absolute P&L% delta between regions


# ---------------------------------------------------------------------------
# Ledger-driven helpers
# ---------------------------------------------------------------------------

def _iter_ledger_window(*, sid: str | None = None, days: int = _LOOKBACK_DAYS):
    """Yield ledger rows in the trailing window. Optionally filter by
    strategy_id."""
    p = ledger_path()
    if not p.exists():
        return
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ed = rec.get("exit_date") or ""
            if not ed or ed < cutoff:
                continue
            if sid is not None and rec.get("strategy_id") != sid:
                continue
            if rec.get("exit_reason") in ("cancelled", "cleared"):
                continue
            yield rec


def fees_pct_of_gross(sid: str, region: str | None = None, days: int = _LOOKBACK_DAYS) -> dict:
    """Returns {fees_gbp, gross_pnl_gbp, net_pnl_gbp, fees_pct_of_gross}.

    A strategy that's gross-positive but net-zero because of fees is a
    very different problem from one that's just bad — this surfaces
    that distinction to the evolution agent."""
    fees = 0.0
    net = 0.0
    n = 0
    for rec in _iter_ledger_window(sid=sid, days=days):
        if region is not None and rec.get("region") != region:
            continue
        fees += float(rec.get("fees_gbp") or 0.0)
        net += float(rec.get("pnl_gbp") or 0.0)
        n += 1
    gross = net + fees
    # `fees / gross` flips sign when the strategy is gross-negative,
    # which renders as "−5% of gross eaten by fees" — meaningless. Use
    # abs(gross) for the share calculation; the gross_pnl_gbp number
    # itself still carries the sign for the agent to read.
    pct = (fees / abs(gross) * 100.0) if abs(gross) > 0.01 else 0.0
    return {
        "n_trades": n,
        "fees_gbp": round(fees, 2),
        "gross_pnl_gbp": round(gross, 2),
        "net_pnl_gbp": round(net, 2),
        "fees_pct_of_gross": round(pct, 1),
    }


def sector_concentration(sid: str, *, days: int = _LOOKBACK_DAYS, top_n: int = 5) -> list[dict]:
    """Top sectors by traded notional for this strategy over the window.
    Uses the cached sector lookup so no live yfinance call."""
    from trading_bot.tools.sectors import bulk_lookup
    tickers: set[str] = set()
    notional_by_ticker: dict[str, float] = defaultdict(float)
    for rec in _iter_ledger_window(sid=sid, days=days):
        tkr = rec.get("ticker") or ""
        if not tkr:
            continue
        try:
            n = abs(float(rec.get("entry_price") or 0) * float(rec.get("quantity") or 0))
        except (TypeError, ValueError):
            continue
        notional_by_ticker[tkr] += n
        tickers.add(tkr)
    if not tickers:
        return []
    sector_map = bulk_lookup(sorted(tickers))
    by_sector: dict[str, float] = defaultdict(float)
    for tkr, n in notional_by_ticker.items():
        sector = sector_map.get(tkr) or "Unknown"
        by_sector[sector] += n
    total = sum(by_sector.values()) or 1.0
    rows = [{"sector": s, "pct": round(v / total * 100, 1), "notional": round(v, 0)}
            for s, v in by_sector.items()]
    rows.sort(key=lambda r: r["notional"], reverse=True)
    return rows[:top_n]


def trail_activation_rate(sid: str, *, days: int = _LOOKBACK_DAYS) -> dict:
    """Share of this strategy's exits that fired via stop / trail."""
    total = 0
    stops = 0
    for rec in _iter_ledger_window(sid=sid, days=days):
        total += 1
        reason = (rec.get("exit_reason") or "").lower()
        if reason in {"stop", "trail_stop", "stop_loss"}:
            stops += 1
    return {"n_exits": total, "n_stops": stops,
            "stop_rate_pct": round(stops / total * 100, 1) if total else 0.0}


# ---------------------------------------------------------------------------
# Persistence-driven helpers
# ---------------------------------------------------------------------------

def cost_gate_drop_rate(sid: str, *, days: int = _LOOKBACK_DAYS) -> dict:
    """Aggregate `state/pick_adjustments/<date>.<sid>.jsonl` over the
    window. Returns total picks, dropped count, drop_rate_pct."""
    d = STATE_ROOT / "pick_adjustments"
    if not d.exists():
        return {"n_picks": 0, "n_dropped": 0, "drop_rate_pct": 0.0}
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    total = 0
    dropped = 0
    for p in d.glob(f"*.{sid}.jsonl"):
        date_part = p.stem.split(".", 1)[0]
        if date_part < cutoff:
            continue
        try:
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        adj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    total += 1
                    if adj.get("dropped"):
                        dropped += 1
        except OSError:
            continue
    return {
        "n_picks": total, "n_dropped": dropped,
        "drop_rate_pct": round(dropped / total * 100, 1) if total else 0.0,
    }


def earnings_gate_hit_rate(sid: str, *, days: int = _LOOKBACK_DAYS) -> dict:
    """Per-strategy earnings-gate aggregate. Files are named
    `<date>.<sid>.<region>.json` since f88ac31 added the region
    suffix, so the glob is `<date>.<sid>.*.json`."""
    d = STATE_ROOT / "earnings_gate"
    if not d.exists():
        return {"n_runs": 0, "candidates_total": 0, "candidates_dropped": 0, "drop_rate_pct": 0.0}
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    n_runs = 0
    total = 0
    dropped = 0
    # `*.{sid}.*.json` matches the new region-suffixed format; the
    # double-`*` also tolerates the old (pre-f88ac31) shape `<date>.<sid>.json`.
    for p in list(d.glob(f"*.{sid}.*.json")) + list(d.glob(f"*.{sid}.json")):
        date_part = p.stem.split(".", 1)[0]
        if date_part < cutoff:
            continue
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        n_runs += 1
        total += int(data.get("candidates_total") or 0)
        dropped += int(data.get("candidates_dropped") or 0)
    return {
        "n_runs": n_runs, "candidates_total": total, "candidates_dropped": dropped,
        "drop_rate_pct": round(dropped / total * 100, 1) if total else 0.0,
    }


def verdict_rates_by_source(*, days: int = _VERDICT_LOOKBACK_DAYS) -> dict[str, dict]:
    """Per-source rates of {proven, partial, falsified, still-open}
    over the trailing window. Reads state/predictions/*.jsonl."""
    pred_dir = STATE_ROOT / "predictions"
    if not pred_dir.exists():
        return {}
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    out: dict[str, dict] = {}
    for p in pred_dir.glob("*.jsonl"):
        source = p.stem
        counts = {"proven": 0, "partial": 0, "falsified": 0, "still-open": 0, "graded": 0, "total": 0}
        try:
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Filter by made_at (when the prediction entered the
                    # log). Drop rows with no made_at — they pre-date the
                    # field and aren't safely datable. Also drop anything
                    # older than the window.
                    made_at = (rec.get("made_at") or "")[:10]
                    if not made_at or made_at < cutoff:
                        continue
                    counts["total"] += 1
                    status = rec.get("status") or "open"
                    if status in counts:
                        counts[status] += 1
                    if status in ("proven", "partial", "falsified"):
                        counts["graded"] += 1
        except OSError:
            continue
        if counts["total"]:
            out[source] = counts
    return out


def halt_history_summary(*, days: int = _LOOKBACK_DAYS) -> dict:
    """Aggregate halt events over the window."""
    from trading_bot.state.halt import load_halt_history
    events = load_halt_history(days=days)
    sets = [e for e in events if e.get("type") == "set"]
    return {
        "n_events": len(events),
        "n_halts": len(sets),
        "most_recent_reason": (sets[-1].get("reason") if sets else "") or "",
    }


def regime_subtitles(*, days: int = 7) -> list[dict]:
    """Trailing N days of daily-news masthead subtitles. Gives the
    evolution agent regime context — "Iran tension week, oil spike"
    is a lens for interpreting any single strategy's P&L."""
    state_dir = STATE_ROOT / "daily_news"
    if not state_dir.exists():
        return []
    out: list[dict] = []
    today = date.today()
    for offset in range(1, days + 1):
        d = (today - timedelta(days=offset)).isoformat()
        p = state_dir / f"{d}.pipeline.json"
        if not p.exists():
            continue
        try:
            payload = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        publisher = payload.get("stages", {}).get("publisher", {}) or {}
        subtitle = publisher.get("masthead_subtitle") or ""
        if subtitle:
            out.append({"date": d, "subtitle": subtitle})
    return out


def parent_deep_analysis(sid: str, *, max_chars: int = 2000) -> str:
    """Read the strategy's deep_analysis.md prompt so the agent can
    reason about spawn-variant proposals against the actual bias.
    Uses the registry's absolute path so it works regardless of CWD."""
    from trading_bot.strategy.registry import _strategies_dir
    p = _strategies_dir() / sid / "prompts" / "deep_analysis.md"
    if not p.exists():
        return ""
    try:
        text = p.read_text()
    except OSError:
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + "\n…(truncated)…"
    return text


# ---------------------------------------------------------------------------
# Cross-strategy helpers
# ---------------------------------------------------------------------------

def divergent_strategies(snapshot: list[dict], *, threshold_pct: float = _DIVERGENCE_THRESHOLD_PCT) -> list[dict]:
    """Strategies whose absolute P&L% differs across regions by more
    than `threshold_pct`. Pre-computed for the editorial intro instead
    of letting the LLM scan a flat list of (sid, region) rows."""
    by_sid: dict[str, dict[str, float]] = defaultdict(dict)
    for row in snapshot:
        sid = row.get("id") or "?"
        region = row.get("region") or "?"
        m = row.get("metrics") or {}
        # avg_pnl_pct is ALREADY in percent (mean of ledger pnl_pct);
        # no second multiplication.
        pct = float(m.get("avg_pnl_pct") or 0)
        by_sid[sid][region] = pct
    divergent: list[dict] = []
    for sid, regions in by_sid.items():
        if len(regions) < 2:
            continue
        items = list(regions.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a_region, a_pct = items[i]
                b_region, b_pct = items[j]
                delta = abs(a_pct - b_pct)
                if delta >= threshold_pct:
                    divergent.append({
                        "sid": sid,
                        "region_a": a_region, "pct_a": round(a_pct, 2),
                        "region_b": b_region, "pct_b": round(b_pct, 2),
                        "delta": round(delta, 2),
                    })
    divergent.sort(key=lambda r: r["delta"], reverse=True)
    return divergent
