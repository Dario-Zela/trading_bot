"""Regression: the slate-table renderer crashed when a strategy's metrics
had `ic` (or n_trades) set to None — `m.get("ic", 0.0)` only fires the
default for a MISSING key, but compute_all_metrics writes None when there
are no graded predictions, and `None:.+.2f}` raises TypeError. That
killed both the editorial render AND the email send (nested in the same
try)."""
from __future__ import annotations

from trading_bot.meta.evolution_v2 import _render_slate_table, EvolutionEdition


def _mk_edition(snapshot_rows):
    return EvolutionEdition(
        week_end="2026-05-23",
        editorial_md="",
        reports=[],
        snapshot_rows=snapshot_rows,
        action_log=[],
    )


def test_slate_table_handles_none_ic():
    rows = [
        {"id": "x", "region": "us", "tier": "shadow",
         "metrics": {"total_pnl_gbp": None, "hit_rate": None,
                     "n_trades": None, "ic": None}},
    ]
    html = _render_slate_table(_mk_edition(rows))
    # No crash + IC printed as 0.00 (the None-coerced default)
    assert "IC +0.00" in html or "IC -0.00" in html or "IC 0.00" in html


def test_slate_table_renders_populated_row():
    rows = [
        {"id": "macro-aligned", "region": "uk-eu", "tier": "alpaca-paper",
         "metrics": {"total_pnl_gbp": 12.34, "hit_rate": 0.55,
                     "n_trades": 100, "ic": 0.17}},
    ]
    html = _render_slate_table(_mk_edition(rows))
    assert "macro-aligned" in html
    assert "£+12.34" in html
    assert "55% hit" in html
    assert "IC +0.17" in html
    assert "across 100 trades" in html


def test_slate_table_handles_missing_metrics_dict():
    rows = [{"id": "x", "region": "us", "tier": "shadow"}]    # no metrics key
    html = _render_slate_table(_mk_edition(rows))
    assert "x" in html        # didn't crash
