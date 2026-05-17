"""Meta-agents — operate on the bot itself rather than on the market.

Wave 2c: daily reflection (per-trade outcome/risks analysis).
Wave 6 will add: weekly evolution (strategy promotion/demotion), weekly macro.
"""
from trading_bot.meta.reflection import grade_predictions, reflect_on_day
from trading_bot.meta.macro import run_weekly_macro
from trading_bot.meta.evolution import run_weekly_evolution
from trading_bot.meta.metrics import StrategyMetrics, compute_all_metrics, compute_metrics
from trading_bot.meta.dst_sync import sync_dst

__all__ = [
    "reflect_on_day",
    "grade_predictions",
    "run_weekly_macro",
    "run_weekly_evolution",
    "sync_dst",
    "StrategyMetrics",
    "compute_metrics",
    "compute_all_metrics",
]
