"""Phase 10A — track positions closed via stop / trail.

When the trailing-stop pass fires (or any bracket-stop hits), the
position closes and we ledger `exit_reason='stop'`. If the strategy
then re-picks the same name tomorrow on a stamp-duty instrument, we
pay the entry tax *again* — which the LLM's cost gate doesn't see.

This module persists every stop-driven exit and exposes:
- `append_trail_exit()` — called from each executor's exit path
- `load_recent_trail_exits(days)` — used by `sizing.adjust_picks()` to
  ADD an extra round-trip cost to the gate threshold for any pick
  whose ticker was trailed out recently (additive, not multiplicative
  — see `sizing.py` for the math). The strategy LLM prompt also
  warns about these tickers explicitly.

The persistence is a single JSONL at `state/trail_exits.jsonl`; one
line per close. Old entries are tolerated forever — the lookback
filter trims them at read time.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from trading_bot.state.paths import STATE_ROOT


log = logging.getLogger(__name__)


@dataclass
class TrailExit:
    """One stop-driven exit, persisted on disk."""
    ticker: str
    region: str
    strategy_id: str
    exit_date: str           # ISO
    exit_reason: str         # "stop" | "trail_stop" | "stop_loss"
    pnl_pct: float
    appended_at: str         # ISO timestamp the row was written


def _path() -> Path:
    p = STATE_ROOT / "trail_exits.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def append_trail_exit(trade: dict) -> None:
    """Append one record. Callers pass a trade dict (the closed-trade
    shape from each executor's exit path); this function pulls the
    fields it needs and is silent if `exit_reason` doesn't look like
    a stop fire."""
    reason = (trade.get("exit_reason") or "").lower()
    if reason not in {"stop", "trail_stop", "stop_loss"}:
        return
    ticker = trade.get("ticker") or ""
    if not ticker:
        return
    rec = TrailExit(
        ticker=ticker.upper(),
        region=trade.get("region", "?"),
        strategy_id=trade.get("strategy_id", "?"),
        exit_date=trade.get("exit_date", ""),
        exit_reason=reason,
        pnl_pct=float(trade.get("pnl_pct") or 0.0),
        appended_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    try:
        with _path().open("a") as f:
            f.write(json.dumps(asdict(rec)) + "\n")
    except OSError as e:
        log.warning("trail-exits: failed to persist %s: %s", ticker, e)


def load_recent_trail_exits(*, days: int = 3) -> dict[str, list[TrailExit]]:
    """Return ticker → list of trail exits inside the last `days` days.
    Used by sizing + strategy prompts to elevate the re-entry cost."""
    p = _path()
    if not p.exists():
        return {}
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    out: dict[str, list[TrailExit]] = {}
    try:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (raw.get("exit_date") or "") < cutoff:
                    continue
                try:
                    rec = TrailExit(**raw)
                except TypeError:
                    continue
                out.setdefault(rec.ticker, []).append(rec)
    except OSError as e:
        log.warning("trail-exits: read failed: %s", e)
        return {}
    return out


def recently_trailed(ticker: str, *, days: int = 3) -> TrailExit | None:
    """Lookup helper. Returns the most recent trail-exit record for
    `ticker` inside the window, or None."""
    rows = load_recent_trail_exits(days=days).get((ticker or "").upper(), [])
    if not rows:
        return None
    rows.sort(key=lambda r: r.exit_date, reverse=True)
    return rows[0]
