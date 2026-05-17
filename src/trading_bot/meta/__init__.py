"""Meta-agents — operate on the bot itself rather than on the market.

Wave 2c: daily reflection (per-trade outcome/risks analysis).
Wave 6 will add: weekly evolution (strategy promotion/demotion), weekly macro.
"""
from trading_bot.meta.reflection import reflect_on_day

__all__ = ["reflect_on_day"]
