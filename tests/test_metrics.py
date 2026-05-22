"""Metrics: drawdown normalised by the strategy's real capital."""
from __future__ import annotations

import pytest

from trading_bot.meta.metrics import StrategyMetrics, _fill_trade_metrics


def _m():
    return StrategyMetrics(strategy_id="x", region="us", window_days=14,
                           window_start="", window_end="")


@pytest.mark.parametrize("capital,expected", [(5_000, -10.0), (10_000, -5.0), (30_000, -1.67)])
def test_drawdown_scales_with_capital(capital, expected):
    m = _m()
    # A single -£500 trade is a -£500 peak-to-trough drawdown.
    _fill_trade_metrics(m, [{"pnl_gbp": -500, "exit_date": "2026-05-10"}], capital)
    assert m.max_drawdown_pct == pytest.approx(expected, abs=0.01)


def test_drawdown_zero_capital_falls_back_to_10k():
    m = _m()
    _fill_trade_metrics(m, [{"pnl_gbp": -500, "exit_date": "2026-05-10"}], 0.0)
    assert m.max_drawdown_pct == -5.0  # 500/10000


def test_peak_to_trough_not_just_final():
    m = _m()
    # Up 300, down 800 (trough -500 from peak 300), up 200 → max dd = -800.
    trades = [
        {"pnl_gbp": 300, "exit_date": "2026-05-10"},
        {"pnl_gbp": -800, "exit_date": "2026-05-11"},
        {"pnl_gbp": 200, "exit_date": "2026-05-12"},
    ]
    _fill_trade_metrics(m, trades, 10_000)
    assert m.max_drawdown_pct == -8.0  # -800 / 10000
