"""Earnings calendar / surprise tool.

yfinance's `.calendar` attribute gives the next earnings date; `.earnings_dates`
gives past earnings + surprise data when available. Both are flaky on some
tickers — we degrade gracefully to None rather than failing the strategy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import yfinance as yf


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EarningsInfo:
    ticker: str
    next_earnings_date: str | None       # ISO date if known
    last_surprise_pct: float | None      # most recent surprise % (actual vs estimate)
    last_eps_actual: float | None
    last_eps_estimate: float | None


def get_earnings_info(ticker: str) -> EarningsInfo:
    next_date: str | None = None
    last_surprise: float | None = None
    last_actual: float | None = None
    last_estimate: float | None = None

    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if isinstance(cal, dict):
            earnings_dates = cal.get("Earnings Date") or []
            if earnings_dates:
                next_date = str(earnings_dates[0])[:10]
        elif cal is not None and hasattr(cal, "loc"):
            try:
                edate = cal.loc["Earnings Date"]
                if hasattr(edate, "__iter__"):
                    next_date = str(list(edate)[0])[:10]
                else:
                    next_date = str(edate)[:10]
            except Exception:
                pass
    except Exception as e:
        log.debug("Calendar fetch failed for %s: %s", ticker, e)

    try:
        ed = yf.Ticker(ticker).earnings_dates
        if ed is not None and not ed.empty:
            # Most recent past row with non-null actual
            past = ed[ed["Reported EPS"].notna()] if "Reported EPS" in ed.columns else None
            if past is not None and not past.empty:
                row = past.iloc[0]
                last_actual = _float_or_none(row.get("Reported EPS"))
                last_estimate = _float_or_none(row.get("EPS Estimate"))
                surprise_pct = row.get("Surprise(%)")
                if surprise_pct is not None:
                    last_surprise = _float_or_none(surprise_pct)
    except Exception as e:
        log.debug("Earnings history fetch failed for %s: %s", ticker, e)

    return EarningsInfo(
        ticker=ticker,
        next_earnings_date=next_date,
        last_surprise_pct=last_surprise,
        last_eps_actual=last_actual,
        last_eps_estimate=last_estimate,
    )


def _float_or_none(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None
