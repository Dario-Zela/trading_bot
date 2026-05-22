"""Trading212 _to_gbp: returns None (never native units) when it can't convert."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from trading_bot.executor import trading212_demo as t2
from trading_bot.executor.trading212_demo import Trading212DemoExecutor as E


def _executor(instrument, fx_mult, monkeypatch):
    obj = E.__new__(E)  # bypass __init__/creds
    translator = MagicMock()
    translator.get_instrument.return_value = instrument
    obj._get_translator = lambda: translator
    monkeypatch.setattr(t2, "to_gbp_multiplier", lambda ccy: fx_mult)
    return obj


def test_none_when_instrument_missing(monkeypatch):
    obj = _executor(None, 0.79, monkeypatch)
    assert obj._to_gbp("X_EQ", 100.0) is None


def test_none_when_fx_unavailable(monkeypatch):
    obj = _executor({"currencyCode": "USD"}, None, monkeypatch)
    assert obj._to_gbp("X_EQ", 100.0) is None


def test_converts_usd(monkeypatch):
    obj = _executor({"currencyCode": "USD"}, 0.79, monkeypatch)
    assert obj._to_gbp("X_EQ", 100.0) == pytest.approx(79.0)


def test_gbx_pence_divides_by_100(monkeypatch):
    obj = _executor({"currencyCode": "GBX"}, 0.01, monkeypatch)
    assert obj._to_gbp("SHEL_EQ", 3248.0) == pytest.approx(32.48)
