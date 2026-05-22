"""Fee model: GBX-as-GBP, USD-ETF FX, stamp-duty rules, currency table."""
from __future__ import annotations

from trading_bot.tools.fees import (
    _LSE_USD_DENOMINATED,
    TradeContext,
    compute_fees,
    yf_ticker_classify,
)


def _fees(**kw):
    ctx = TradeContext(
        tier="trading212-paper",
        entry_notional_gbp=10_000,
        exit_notional_gbp=10_000,
        quantity=100,
        **kw,
    )
    b = compute_fees(ctx)
    return round(b.stamp_duty_gbp, 2), round(b.fx_fee_entry_gbp + b.fx_fee_exit_gbp, 2)


def test_gbx_share_charges_stamp_and_no_fx():
    # GBX (pence) is sterling — UK stamp duty applies, no FX leg.
    stamp, fx = _fees(currency="GBX", exchange="LSE", instrument_type="share")
    assert stamp > 0
    assert fx == 0.0


def test_gbp_share_matches_gbx():
    assert _fees(currency="GBP", exchange="LSE", instrument_type="share") == _fees(
        currency="GBX", exchange="LSE", instrument_type="share"
    )


def test_usd_gdr_skips_stamp_keeps_fx():
    stamp, fx = _fees(currency="USD", exchange="LSE", instrument_type="share")
    assert stamp == 0.0
    assert fx > 0.0


def test_etf_is_stamp_exempt():
    stamp, _ = _fees(currency="GBX", exchange="LSE", instrument_type="etf")
    assert stamp == 0.0


def test_lse_usd_set_is_only_genuinely_usd_lines():
    # Reconciled against yfinance + T212 currencyCode; GBP/GBX lines removed.
    assert _LSE_USD_DENOMINATED == {"CSPX.L", "IWDA.L", "SSLV.L"}
    assert "VWRL.L" not in _LSE_USD_DENOMINATED  # GBP, not USD
    assert "IUSA.L" not in _LSE_USD_DENOMINATED  # GBX, not USD


def test_classify_corrected_currencies():
    assert yf_ticker_classify("VWRL.L") == ("LSE", "GBP")
    assert yf_ticker_classify("IWDA.L") == ("LSE", "USD")
    assert yf_ticker_classify("AAPL") == ("NYSE", "USD")
