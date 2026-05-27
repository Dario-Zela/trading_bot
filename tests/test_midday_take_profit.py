"""Midday take-profit pass — threshold math and per-strategy lookups.

We don't test the broker HTTP plumbing here (covered by integration);
this locks down the pure logic: threshold computation from the
strategy's take_profit_pct × midday_tp_factor, fallback behaviour
when the config field is missing, and the per-process config cache.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

from trading_bot.executor.midday_take_profit import (
    DEFAULT_TP_FACTOR, _strategy_thresholds, _STRATEGY_CONFIG_CACHE,
)


@dataclass
class _FakeCfg:
    take_profit_pct: float | None
    midday_tp_factor: float | None = 0.7


def _install(monkey_strategy: str, cfg) -> None:
    """Force a known config into the per-process cache so the lookup
    doesn't try to hit the filesystem."""
    _STRATEGY_CONFIG_CACHE[monkey_strategy] = cfg


def setup_function(_):
    _STRATEGY_CONFIG_CACHE.clear()


def test_threshold_uses_strategy_tp_times_factor():
    _install("test-strategy", _FakeCfg(take_profit_pct=5.0, midday_tp_factor=0.7))
    tp, factor, threshold = _strategy_thresholds("test-strategy", cli_default_factor=DEFAULT_TP_FACTOR)
    assert tp == 5.0
    assert factor == 0.7
    assert threshold == 5.0 * 0.7  # 3.5%


def test_threshold_respects_per_strategy_override():
    """Two strategies with different factors should compute different
    thresholds even with the same TP. The evolution agent uses this to
    tune each strategy independently."""
    _install("aggressive", _FakeCfg(take_profit_pct=5.0, midday_tp_factor=0.5))
    _install("conservative", _FakeCfg(take_profit_pct=5.0, midday_tp_factor=1.0))
    _, _, t_aggr = _strategy_thresholds("aggressive", cli_default_factor=0.7)
    _, _, t_cons = _strategy_thresholds("conservative", cli_default_factor=0.7)
    assert t_aggr == 2.5
    assert t_cons == 5.0


def test_missing_midday_factor_falls_back_to_cli_default():
    _install("legacy", _FakeCfg(take_profit_pct=5.0, midday_tp_factor=None))
    _, factor, threshold = _strategy_thresholds("legacy", cli_default_factor=0.9)
    assert factor == 0.9
    assert threshold == 5.0 * 0.9


def test_no_tp_returns_none_signal():
    """A strategy with no take_profit_pct (e.g. control-rule-based) must
    be skipped entirely — the threshold lookup signals via tp=None."""
    _install("no-tp", _FakeCfg(take_profit_pct=None))
    tp, _, _ = _strategy_thresholds("no-tp", cli_default_factor=0.7)
    assert tp is None


def test_zero_or_negative_tp_treated_as_unset():
    _install("zero-tp", _FakeCfg(take_profit_pct=0.0, midday_tp_factor=0.7))
    tp, _, _ = _strategy_thresholds("zero-tp", cli_default_factor=0.7)
    assert tp is None


# -----------------------------------------------------------------------------
# Evolution-agent integration: midday_tp_factor must be tunable
# -----------------------------------------------------------------------------

def test_midday_tp_factor_is_in_tunable_fields():
    from trading_bot.meta.evolution import TUNABLE_FIELDS
    assert "midday_tp_factor" in TUNABLE_FIELDS
    lo, hi = TUNABLE_FIELDS["midday_tp_factor"]
    assert lo < hi
    assert lo > 0  # factor must be positive
    assert hi <= 2.0  # don't let agent set absurd values
    # 0.7 default must be within the allowed range
    assert lo <= DEFAULT_TP_FACTOR <= hi


# -----------------------------------------------------------------------------
# Config field is wired through registry
# -----------------------------------------------------------------------------

def test_strategy_config_carries_midday_tp_factor_default():
    """A freshly-loaded config without midday_tp_factor in YAML should
    default to 0.7 — preserves backward compat for unedited strategies."""
    from trading_bot.strategy.base import StrategyConfig
    cfg = StrategyConfig(
        id="x", display_name="x", description="", implementation="llm",
        active=True, tier="shadow", region="us", capital_gbp=1000,
        max_positions=1, max_position_pct=10.0, min_position_gbp=10.0,
        use_stops=True, use_take_profits=True,
    )
    assert cfg.midday_tp_factor == 0.7
