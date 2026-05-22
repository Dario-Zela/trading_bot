"""Ledger: append/read round-trip, corrupt-line resilience, exit marking."""
from __future__ import annotations

from datetime import date

from trading_bot.state.ledger import (
    TradeRecord,
    _iter_records,
    append_trade,
    ledger_path,
    mark_trade_exited,
    read_open_trades,
)


def _rec(trade_id: str, **kw) -> TradeRecord:
    base = dict(
        trade_id=trade_id, strategy_id="s", region="us", tier="alpaca-paper",
        ticker="AAPL", side="long", entry_date="2026-05-22", entry_price=100.0,
        quantity=2.0, allocation_pct=10.0,
    )
    base.update(kw)
    return TradeRecord(**base)


def test_append_and_read_open(state_root):
    append_trade(_rec("a"))
    append_trade(_rec("b", region="uk-eu"))
    assert len(read_open_trades()) == 2
    assert {t["trade_id"] for t in read_open_trades(region="us")} == {"a"}


def test_corrupt_line_is_skipped_not_fatal(state_root):
    append_trade(_rec("a"))
    # Simulate a torn write between two good rows.
    with ledger_path().open("a") as f:
        f.write("{ this is not valid json\n")
    append_trade(_rec("b"))
    open_trades = read_open_trades()  # must not raise
    assert {t["trade_id"] for t in open_trades} == {"a", "b"}


def test_mark_trade_exited_persists_fields(state_root):
    append_trade(_rec("a"))
    mark_trade_exited(
        trade_id="a", exit_date=date(2026, 5, 22), exit_price=110.0,
        pnl_gbp=15.0, pnl_pct=10.0, exit_reason="scheduled", fees_gbp=1.5,
    )
    assert read_open_trades() == []  # no longer open
    # The exit fields are actually written back (not just "no longer open").
    row = next(r for r in _iter_records() if r["trade_id"] == "a")
    assert row["exit_date"] == "2026-05-22"
    assert row["exit_price"] == 110.0
    assert row["pnl_gbp"] == 15.0
    assert row["exit_reason"] == "scheduled"
    assert row["fees_gbp"] == 1.5


def test_read_open_trades_tier_filter(state_root):
    # The tier filter exists to stop a new-tier executor touching an old-tier
    # trade (the 2026-05-21 cross-tier misattribution fix).
    append_trade(_rec("a", tier="alpaca-paper"))
    append_trade(_rec("b", tier="shadow"))
    assert {t["trade_id"] for t in read_open_trades(tier="alpaca-paper")} == {"a"}
    assert {t["trade_id"] for t in read_open_trades(tier="shadow")} == {"b"}


def test_entry_fx_rate_defaults_none():
    assert _rec("z").entry_fx_rate is None
