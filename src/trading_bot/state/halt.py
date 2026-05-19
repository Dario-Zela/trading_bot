"""Phase 8F — daily-loss kill switch.

A simple file-based gate at `state/halt.json`. When yesterday's
live-tier P&L is materially negative, the entry pass for today writes
this file and refuses to open new positions on any live-tier
strategy. Existing positions still exit on schedule (we don't trap
ourselves in losing trades; we just stop *adding* new ones).

Manual unhalt: delete the file (locally) or via a PR. The dashboard
surfaces halt status so it's not silent.

Threshold is conservative by default (-3% of total live capital) — it
catches news-shock days while leaving normal volatility alone.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from trading_bot.state.paths import STATE_ROOT

log = logging.getLogger(__name__)


# Default threshold: halt if yesterday's live-tier P&L < -3% of capital
DEFAULT_LOSS_THRESHOLD_PCT = -3.0
# Tiers that count as "live" for the kill switch. shadow is excluded —
# it's a simulation, not real money risk.
LIVE_TIERS = {"alpaca-paper", "trading212-paper", "t212-live"}


@dataclass
class HaltRecord:
    halted: bool
    reason: str
    set_at: str
    yesterday_pnl_gbp: float
    yesterday_pnl_pct: float
    capital_gbp: float


def halt_path() -> Path:
    return STATE_ROOT / "halt.json"


def is_halted() -> tuple[bool, HaltRecord | None]:
    """Return (halted, record). Empty/missing file → not halted."""
    p = halt_path()
    if not p.exists():
        return False, None
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        log.warning("halt.json is corrupt — treating as halted to be safe")
        return True, None
    halted = bool(data.get("halted", False))
    rec = None
    try:
        rec = HaltRecord(**{k: data[k] for k in HaltRecord.__annotations__ if k in data})
    except Exception:
        pass
    return halted, rec


def evaluate_and_set_halt(
    today: date,
    *,
    total_live_capital_gbp: float,
    threshold_pct: float = DEFAULT_LOSS_THRESHOLD_PCT,
) -> HaltRecord | None:
    """Compute yesterday's live-tier P&L from the ledger. If it's
    worse than `threshold_pct` of `total_live_capital_gbp`, write the
    halt file and return the record. Otherwise return None.

    Idempotent — if halt.json already exists with halted=True, we
    don't overwrite it (the human needs to clear it manually).
    """
    existing_halted, _ = is_halted()
    if existing_halted:
        log.warning("Kill switch already engaged; entry will skip live-tier strategies")
        return None

    yesterday_pnl = _yesterday_live_pnl(today)
    if total_live_capital_gbp <= 0:
        return None
    pnl_pct = (yesterday_pnl / total_live_capital_gbp) * 100.0

    if pnl_pct <= threshold_pct:
        rec = HaltRecord(
            halted=True,
            reason=(
                f"Yesterday's live-tier P&L of £{yesterday_pnl:+,.2f} "
                f"({pnl_pct:+.2f}%) breached the {threshold_pct:.1f}% loss "
                f"threshold. Entry halted for live-tier strategies. Resolve "
                f"by reviewing the day and deleting state/halt.json."
            ),
            set_at=datetime.now(timezone.utc).isoformat(),
            yesterday_pnl_gbp=round(yesterday_pnl, 2),
            yesterday_pnl_pct=round(pnl_pct, 3),
            capital_gbp=round(total_live_capital_gbp, 2),
        )
        halt_path().write_text(json.dumps(asdict(rec), indent=2))
        log.error("KILL SWITCH ENGAGED: %s", rec.reason)
        return rec

    log.info(
        "Kill-switch check: yesterday %+.2f%% of live capital (threshold %+.2f%%) — OK",
        pnl_pct, threshold_pct,
    )
    return None


def _yesterday_live_pnl(today: date) -> float:
    """Sum pnl_gbp (net of fees) across all live-tier ledger rows that
    exited yesterday or the most recent prior trading day."""
    from trading_bot.state.paths import ledger_path
    p = ledger_path()
    if not p.exists():
        return 0.0
    iso_today = today.isoformat()
    # Find the most recent prior exit_date (yesterday in the
    # calendar-trading-day sense — skips weekends/holidays)
    seen_dates = set()
    rows = []
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
            if not ed or ed >= iso_today:
                continue
            if rec.get("tier") not in LIVE_TIERS:
                continue
            seen_dates.add(ed)
            rows.append(rec)
    if not seen_dates:
        return 0.0
    last_day = max(seen_dates)
    return sum(float(r.get("pnl_gbp") or 0) for r in rows if r.get("exit_date") == last_day)


def clear_halt() -> None:
    """Manual / scripted unhalt. Removes the halt file."""
    p = halt_path()
    if p.exists():
        p.unlink()
        log.info("Kill switch cleared (halt.json removed)")
