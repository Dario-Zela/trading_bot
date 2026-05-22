"""_fmt: optional-numeric formatting for prompt lines must never crash on None.

Guards the regression where an unguarded f-string format on a None technical
(e.g. sma_50 on a <55-bar history) raised TypeError and killed a strategy's
entire pick run (observed live on the 2026-05-22 US entry).
"""
from __future__ import annotations

from trading_bot.strategy.llm_strategy import _fmt


def test_formats_numbers():
    assert _fmt(1.234, ".2f") == "1.23"
    assert _fmt(-0.5, "+.2f") == "-0.50"
    assert _fmt(12.0, "+.3f") == "+12.000"


def test_none_returns_placeholder_not_crash():
    assert _fmt(None, ".2f") == "?"
    assert _fmt(None, "+.3f") == "?"
    assert _fmt(None, ".2f", na="—") == "—"


def test_zero_is_formatted_not_treated_as_missing():
    # 0.0 is falsy but a real value — must not be swallowed as "missing".
    assert _fmt(0.0, ".2f") == "0.00"
