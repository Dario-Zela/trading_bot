"""Phase 11A — sanity checks on prices fed into the bot.

yfinance occasionally returns anomalous quotes (un-adjusted splits,
delisted tickers stuck at the last good close, vendor errors). The
canonical failure mode is the SNDK-at-$1407 issue we shipped 9
duplicate positions on. This module provides a single small helper
each executor / sizing path can call before committing to a trade.

The check is intentionally cheap: it compares today's price against
the 20-day moving average (or its proxy if we only have a bar list).
A 5× ratio in either direction signals "something is wrong with this
data point", not "a real market event" — single names don't move
that far without a split or vendor glitch.
"""
from __future__ import annotations

import logging
from typing import Sequence

log = logging.getLogger(__name__)


# Anomalous-ratio threshold. A real corporate action / event would move
# a stock by at most ~50% in a day; anything above ~5× the 20-day
# average is almost certainly a data problem.
PRICE_ANOMALY_MAX_RATIO = 5.0
PRICE_ANOMALY_MIN_RATIO = 1.0 / PRICE_ANOMALY_MAX_RATIO


def is_price_anomalous(*, close: float, sma_20: float | None = None,
                       bars: Sequence | None = None) -> tuple[bool, str]:
    """Return (is_anomalous, reason). Accepts either an explicit
    `sma_20` (preferred — LLM strategies already have it from
    technicals) or a list of bars to derive it from.

    Returns (False, "") when the inputs are insufficient to decide —
    callers should treat that as "no anomaly detected" to avoid
    blocking legitimate trades just because the history is short.
    """
    if close <= 0:
        return True, f"close price is non-positive ({close})"

    # Prefer pre-computed sma_20; fall back to mean of bars[-20:].
    if sma_20 is None or sma_20 <= 0:
        if not bars:
            return False, ""
        closes = [getattr(b, "close", None) for b in bars[-20:]
                  if getattr(b, "close", None) is not None]
        if len(closes) < 5:           # not enough history to judge
            return False, ""
        sma_20 = sum(closes) / len(closes)
    if sma_20 <= 0:
        return False, ""

    ratio = close / sma_20
    if ratio > PRICE_ANOMALY_MAX_RATIO:
        return True, (
            f"close {close:.4f} is {ratio:.1f}× the 20-day average "
            f"{sma_20:.4f} — likely an un-adjusted split or vendor glitch"
        )
    if ratio < PRICE_ANOMALY_MIN_RATIO:
        return True, (
            f"close {close:.4f} is {ratio:.2f}× the 20-day average "
            f"{sma_20:.4f} — likely a vendor glitch or stale quote"
        )
    return False, ""
