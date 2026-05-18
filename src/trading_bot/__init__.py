"""trading_bot — self-improving LLM-driven stock trading bot.

Package init also installs a logging filter on yfinance's logger. yfinance
emits ERROR-level logs for every per-ticker 404 ("may be delisted",
"No fundamentals data found", etc.) which floods the pipeline output when
universes contain stale / illiquid tickers — even though our tool wrappers
already catch the underlying exception and return empty results. Filtering
these specific messages keeps the signal/noise ratio sane without hiding
legitimate yfinance failures.
"""
from __future__ import annotations

import logging


__version__ = "0.1.0"


class _YFinance404Filter(logging.Filter):
    """Drop yfinance ERROR records that just report a single ticker missing
    upstream. These are recoverable — our tool layer handles them — and the
    log line is pure noise. Anything else from yfinance still goes through."""

    _NOISY_FRAGMENTS = (
        "HTTP Error 404",
        "may be delisted",
        "No fundamentals data found",
        "No earnings dates found",
        "Quote not found for symbol",
        "no price data found",
        "possibly delisted",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING:
            return True
        msg = record.getMessage()
        for frag in self._NOISY_FRAGMENTS:
            if frag in msg:
                return False
        return True


# Install once at import time, guarded so re-imports don't stack filters.
_yf_logger = logging.getLogger("yfinance")
if not getattr(_yf_logger, "_trading_bot_filter_installed", False):
    _yf_logger.addFilter(_YFinance404Filter())
    _yf_logger._trading_bot_filter_installed = True  # type: ignore[attr-defined]
