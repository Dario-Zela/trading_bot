"""Per-strategy Python prefilter ranker.

The legacy `abs_return_5d` default biases every strategy toward
biggest-movers — fine for momentum, wrong for macro / mean-reversion
lenses. Each archetype gets its own key; the evolution agent can flip
between them with a `tune` action.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from trading_bot.strategy.llm_strategy import _prefilter_rank_fn, _PREFILTER_DESC


@dataclass
class _T:
    """Minimal Technicals stand-in — only the fields the ranker reads."""
    ticker: str
    rsi_14: float | None = 50.0
    return_5d_pct: float | None = 0.0
    return_20d_pct: float | None = 0.0
    volume_ratio: float | None = 1.0
    sma_20: float | None = 100.0
    avg_volume_20: float | None = 1_000_000.0


def _sort(items, key):
    fn = _prefilter_rank_fn(key)
    desc = _PREFILTER_DESC.get(key, True)
    return sorted(items, key=fn, reverse=desc)


def test_abs_return_5d_picks_biggest_movers_symmetrically():
    items = [
        _T("flat", return_5d_pct=0.1),
        _T("big_up", return_5d_pct=15.0),
        _T("big_down", return_5d_pct=-18.0),
        _T("medium", return_5d_pct=5.0),
    ]
    out = _sort(items, "abs_return_5d")
    assert [t.ticker for t in out[:2]] == ["big_down", "big_up"]


def test_rsi_14_asc_promotes_oversold():
    items = [
        _T("neutral", rsi_14=50.0),
        _T("overbought", rsi_14=78.0),
        _T("oversold", rsi_14=22.0),
        _T("deep_oversold", rsi_14=15.0),
    ]
    out = _sort(items, "rsi_14_asc")
    assert [t.ticker for t in out[:2]] == ["deep_oversold", "oversold"]


def test_dollar_volume_desc_promotes_etfs_over_microcaps():
    """The macro-aligned fix. A liquid ETF (high sma_20 × high avg_volume_20)
    has to beat a microcap with a recent huge 5d move — exactly the case
    that produced the CODX-shaped output on 2026-05-26."""
    items = [
        _T("CODX_microcap", sma_20=2.5, avg_volume_20=500_000, return_5d_pct=40.0),
        _T("XLV_sector_etf", sma_20=150.0, avg_volume_20=8_000_000, return_5d_pct=1.5),
        _T("SPY_index", sma_20=540.0, avg_volume_20=45_000_000, return_5d_pct=0.8),
        _T("penny_stock", sma_20=0.20, avg_volume_20=100_000, return_5d_pct=120.0),
    ]
    out = _sort(items, "dollar_volume_desc")
    assert [t.ticker for t in out[:2]] == ["SPY_index", "XLV_sector_etf"]
    # And the microcap should not be near the top
    assert "CODX_microcap" not in [t.ticker for t in out[:2]]


def test_volume_ratio_desc_promotes_catalyst_flow():
    items = [
        _T("normal", volume_ratio=1.0),
        _T("light", volume_ratio=0.4),
        _T("catalyst", volume_ratio=8.5),
        _T("mild", volume_ratio=1.6),
    ]
    out = _sort(items, "volume_ratio_desc")
    assert out[0].ticker == "catalyst"


def test_missing_values_sink_to_bottom_desc():
    """A None field must NEVER beat a real value, regardless of direction."""
    items = [
        _T("missing", sma_20=None, avg_volume_20=None),
        _T("real", sma_20=100.0, avg_volume_20=1_000_000),
    ]
    out = _sort(items, "dollar_volume_desc")
    assert out[0].ticker == "real"
    assert out[-1].ticker == "missing"


def test_missing_values_sink_to_bottom_asc():
    """Same invariant for ascending: missing must sink to bottom, not top."""
    items = [
        _T("missing_rsi", rsi_14=None),
        _T("oversold", rsi_14=20.0),
    ]
    out = _sort(items, "rsi_14_asc")
    assert out[0].ticker == "oversold"
    assert out[-1].ticker == "missing_rsi"


def test_unknown_key_falls_back_to_default():
    items = [
        _T("a", return_5d_pct=5.0),
        _T("b", return_5d_pct=-15.0),
    ]
    out = _sort(items, "not_a_real_key")
    assert out[0].ticker == "b"   # same as abs_return_5d


# -----------------------------------------------------------------------------
# Evolution-agent integration: prefilter_sort_key must be a tunable enum so the
# agent can flip it via a `tune` action.
# -----------------------------------------------------------------------------

def test_prefilter_sort_key_is_in_tunable_string_fields():
    from trading_bot.meta.evolution import TUNABLE_STRING_FIELDS
    assert "prefilter_sort_key" in TUNABLE_STRING_FIELDS
    allowed = TUNABLE_STRING_FIELDS["prefilter_sort_key"]
    # All five expected archetypes must be tunable
    for key in ("abs_return_5d", "abs_return_20d", "rsi_14_asc",
                "volume_ratio_desc", "dollar_volume_desc"):
        assert key in allowed
