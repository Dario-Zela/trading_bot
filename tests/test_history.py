"""History: MultiIndex flattening + currency-aware pence scaling (hermetic)."""
from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.tools import history
from trading_bot.tools.history import _flatten, _lse_quote_is_pence, _slice_ticker

_IDX = pd.to_datetime(["2026-05-20", "2026-05-21"])


def _flat(close):
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": [1, 2]},
        index=_IDX,
    )


def test_slice_ticker_price_then_ticker_layout():
    df = pd.DataFrame(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], index=_IDX,
        columns=pd.MultiIndex.from_tuples(
            [("Open", "VOD.L"), ("High", "VOD.L"), ("Low", "VOD.L"),
             ("Close", "VOD.L"), ("Volume", "VOD.L")]),
    )
    assert list(_slice_ticker(df, "VOD.L", True).columns) == \
        ["Open", "High", "Low", "Close", "Volume"]


def test_slice_ticker_ticker_then_price_layout():
    df = pd.DataFrame(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], index=_IDX,
        columns=pd.MultiIndex.from_tuples(
            [("VOD.L", "Open"), ("VOD.L", "High"), ("VOD.L", "Low"),
             ("VOD.L", "Close"), ("VOD.L", "Volume")]),
    )
    assert list(_slice_ticker(df, "VOD.L", True).columns) == \
        ["Open", "High", "Low", "Close", "Volume"]


def test_flatten_divides_pence_lines_by_100(monkeypatch):
    monkeypatch.setattr(history, "_LSE_CCY", {"SHEL.L": "GBp"})
    out = _flatten(_flat([3200.0, 3300.0]), ["SHEL.L"], 2)
    assert out["SHEL.L"][-1].close == pytest.approx(33.0)


def test_flatten_leaves_gbp_lines_unscaled(monkeypatch):
    monkeypatch.setattr(history, "_LSE_CCY", {"IGLS.L": "GBP"})
    out = _flatten(_flat([126.0, 127.0]), ["IGLS.L"], 2)
    assert out["IGLS.L"][-1].close == pytest.approx(127.0)


def test_flatten_non_lse_skips_currency_lookup(monkeypatch):
    monkeypatch.setattr(history, "_lse_quote_is_pence",
                        lambda t: pytest.fail("currency lookup should not run for non-.L"))
    out = _flatten(_flat([100.0, 101.5]), ["AAPL"], 2)
    assert out["AAPL"][-1].close == pytest.approx(101.5)


def test_lse_quote_is_pence_reads_cache(monkeypatch):
    monkeypatch.setattr(history, "_LSE_CCY", {"SHEL.L": "GBp", "IGLS.L": "GBP", "IWDA.L": "USD"})
    assert _lse_quote_is_pence("SHEL.L") is True
    assert _lse_quote_is_pence("IGLS.L") is False
    assert _lse_quote_is_pence("IWDA.L") is False
