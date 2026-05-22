from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Iterator

from trading_bot.state.paths import ledger_path

log = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """One paper trade. Created on entry, updated on exit."""

    trade_id: str
    strategy_id: str
    region: str
    tier: str
    ticker: str
    side: str  # "long" — Wave 1 is long-only
    entry_date: str  # ISO date
    entry_price: float
    quantity: float
    allocation_pct: float
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    thesis: str = ""

    exit_date: str | None = None
    exit_price: float | None = None
    pnl_gbp: float | None = None
    pnl_pct: float | None = None
    exit_reason: str | None = None  # "scheduled" / "stop" / "take_profit"

    # Wave 1.5 — post-trade analysis. Populated when the trade is marked exited.
    # Wave 1 uses templated text from the strategy; Wave 6 replaces with LLM
    # reflection-agent output that's much richer.
    outcome_notes: str | None = None
    risks_observed: str | None = None

    # Broker-side reference. For T212 this is the order id returned by the
    # market-order endpoint. Set when entry is recorded as "pending" because
    # the fill-poll timed out — exit then reconciles via T212 order history.
    broker_order_id: str | None = None

    # Wave 7 — fee accounting. `currency` is the instrument's native currency
    # ("USD" for Alpaca, T212 reports varies); `exchange` is the MIC / common
    # exchange code (LSE / NYSE / NASDAQ / XPAR / ...); `instrument_type` is
    # 'share' / 'etf' / 'aim' / 'bond' / 'gilt' — drives stamp-duty exemption.
    # `fees_gbp` is the aggregate deducted from gross to reach `pnl_gbp` (i.e.
    # pnl_gbp is NET going forward). `fees_breakdown` carries the line items
    # for dashboard transparency. Defaults keep old rows backwards-compatible.
    currency: str = "GBP"
    exchange: str = ""
    instrument_type: str = "share"
    fees_gbp: float = 0.0
    fees_breakdown: dict = field(default_factory=dict)

    # Phase 12A — multi-day positioning. `target_exit_date` is set at
    # entry from `entry_date + hold_days`, skipping weekends. Exit
    # machinery only closes when `today >= target_exit_date`. Legacy
    # rows with no target_exit_date are treated as "exit today" so
    # nothing strands in the ledger after the migration.
    target_exit_date: str | None = None
    hold_days: int = 1

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def append_trade(record: TradeRecord) -> None:
    path = ledger_path()
    with path.open("a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def _iter_records() -> Iterator[dict]:
    path = ledger_path()
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                # A single torn/partial line (e.g. a crash mid-write) must not
                # abort every consumer (exits, reconcile, summary, dashboard).
                log.warning("ledger: skipping unparseable line: %s", e)


def read_open_trades(
    strategy_id: str | None = None,
    region: str | None = None,
    on_date: date | None = None,
    tier: str | None = None,
) -> list[dict]:
    """Return trades that have not yet been exited.

    Optionally filter by strategy, region, entry date, or tier. The default
    returns every open trade in the ledger.

    `tier` matters when a strategy has been promoted/demoted mid-life: the
    executor for the new tier should NOT touch trades belonging to the old
    tier (T212 executor processing a shadow-tier trade caused the
    "T212 position still within hold window" misattribution on 2026-05-21).
    """
    out = []
    for rec in _iter_records():
        if rec.get("exit_date") is not None:
            continue
        if strategy_id is not None and rec.get("strategy_id") != strategy_id:
            continue
        if region is not None and rec.get("region") != region:
            continue
        if on_date is not None and rec.get("entry_date") != on_date.isoformat():
            continue
        if tier is not None and rec.get("tier") != tier:
            continue
        out.append(rec)
    return out


def filter_due_for_exit(open_trades: list[dict], today: date) -> list[dict]:
    """Phase 12A — split open trades into 'due to exit today' vs
    'still held'. A trade is due if its target_exit_date is on-or-
    before today. Legacy rows with no target_exit_date are treated
    as same-day round-trips and exit today — preserves Wave 1
    behaviour for everything written before Phase 12 landed."""
    iso = today.isoformat()
    out: list[dict] = []
    for t in open_trades:
        target = t.get("target_exit_date")
        if not target or target <= iso:
            out.append(t)
    return out


def mark_trade_exited(
    trade_id: str,
    exit_date: date,
    exit_price: float,
    pnl_gbp: float,
    pnl_pct: float,
    exit_reason: str = "scheduled",
    outcome_notes: str | None = None,
    risks_observed: str | None = None,
    fees_gbp: float = 0.0,
    fees_breakdown: dict | None = None,
) -> None:
    """Rewrite the ledger in place, setting exit fields on the matching trade.

    `pnl_gbp` should be NET of `fees_gbp` so downstream aggregations (dashboard
    summaries, evolution metrics) sum to the user's true bottom line without
    needing to subtract fees separately. `fees_breakdown` is for display.

    JSONL append-only files don't support in-place edits cleanly. For Wave 1 the
    ledger is small enough that a rewrite is fine. If this becomes a bottleneck
    we'll move to a per-trade tombstone approach or a sqlite-backed ledger.
    """
    path = ledger_path()
    if not path.exists():
        raise FileNotFoundError(f"Ledger not found at {path}")

    rows = list(_iter_records())
    found = False
    for row in rows:
        if row.get("trade_id") == trade_id:
            row["exit_date"] = exit_date.isoformat()
            row["exit_price"] = exit_price
            row["pnl_gbp"] = pnl_gbp
            row["pnl_pct"] = pnl_pct
            row["exit_reason"] = exit_reason
            row["fees_gbp"] = fees_gbp
            if fees_breakdown is not None:
                row["fees_breakdown"] = fees_breakdown
            if outcome_notes is not None:
                row["outcome_notes"] = outcome_notes
            if risks_observed is not None:
                row["risks_observed"] = risks_observed
            row["updated_at"] = datetime.now(timezone.utc).isoformat()
            found = True
            break

    if not found:
        raise KeyError(f"Trade {trade_id} not found in ledger")

    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    tmp.replace(path)
