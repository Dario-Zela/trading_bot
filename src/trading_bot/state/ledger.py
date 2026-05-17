from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Iterator

from trading_bot.state.paths import ledger_path


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
            if line:
                yield json.loads(line)


def read_open_trades(
    strategy_id: str | None = None,
    region: str | None = None,
    on_date: date | None = None,
) -> list[dict]:
    """Return trades that have not yet been exited.

    Optionally filter by strategy, region, or entry date. The default behaviour
    returns every open trade in the ledger; the exit job uses the date filter
    to find today's positions.
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
        out.append(rec)
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
) -> None:
    """Rewrite the ledger in place, setting exit fields on the matching trade.

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
