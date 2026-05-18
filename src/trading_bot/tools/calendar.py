"""Market holiday gating: skip entry/exit when the relevant exchange is closed.

Uses pandas_market_calendars which ships with every major exchange's
holiday + early-close schedule. We map each pipeline region to its
canonical exchange:
- us    → NYSE
- uk-eu → LSE (primary; other EU exchanges have similar but not identical
          schedules — we accept the imperfection for now since LSE
          dominates our UK-EU universe)

Weekends are already excluded by cron-job.org's wdays config; this module
exists for things like New Year's Day on a Tuesday, MLK Day, Christmas
Day on a weekday, etc. Half-days where the exchange closes early are
treated as "open" — our daily cron schedule is well-clear of close
either way.
"""
from __future__ import annotations

import logging
from datetime import date
from functools import lru_cache


log = logging.getLogger(__name__)


REGION_TO_EXCHANGE = {
    "us": "NYSE",
    "uk-eu": "LSE",
}


@lru_cache(maxsize=8)
def _calendar(exchange: str):
    """Import lazily — pandas_market_calendars pulls in pytz/pandas at
    import time, which is non-trivial. We only need it on real pipeline
    runs, not at every package import."""
    import pandas_market_calendars as mcal  # noqa: WPS433
    return mcal.get_calendar(exchange)


def is_market_open_on(d: date, region: str) -> bool:
    """True if the region's primary exchange has a trading session on `d`.
    Unknown regions default to True (don't block unexpected pipelines).
    Errors fetching the calendar also default to True — better to attempt
    the run and have a noisy failure than silently skip a real trading day."""
    exchange = REGION_TO_EXCHANGE.get(region)
    if exchange is None:
        return True
    try:
        cal = _calendar(exchange)
        sched = cal.schedule(start_date=d.isoformat(), end_date=d.isoformat())
    except Exception as e:
        log.warning(
            "Calendar lookup failed for region=%s exchange=%s date=%s (%s) — defaulting to OPEN",
            region, exchange, d.isoformat(), e,
        )
        return True
    return not sched.empty
