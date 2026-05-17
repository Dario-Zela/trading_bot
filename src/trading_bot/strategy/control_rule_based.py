from __future__ import annotations

from datetime import date

from trading_bot.executor.base import TradeIntent
from trading_bot.strategy.base import Strategy
from trading_bot.tools import get_history, get_universe


class ControlRuleBased(Strategy):
    """Deterministic baseline: top-N highest previous-day % gainers from the
    configured universe, equal weight, no stops, no take-profits.

    This strategy is the yardstick. Every LLM strategy must beat it consistently
    to justify its existence (and its tokens).
    """

    def select_picks(self, on_date: date) -> list[TradeIntent]:
        cfg = self.config
        tickers = get_universe(cfg.universe)

        # We need yesterday's bar. Pad the lookback to ride over weekends/holidays.
        history = get_history(tickers, lookback_days=2, end_date=on_date)

        ranked: list[tuple[str, float]] = []
        for ticker, bars in history.items():
            if not bars:
                continue
            last = bars[-1]
            # Previous-day return: close vs open of the most recent completed session.
            if last.open <= 0:
                continue
            ret = (last.close / last.open - 1.0) * 100.0
            ranked.append((ticker, ret))

        ranked.sort(key=lambda x: x[1], reverse=True)
        picks = ranked[: cfg.max_positions]
        if not picks:
            return []

        # Equal weight across the slots actually filled
        allocation_each = 100.0 / cfg.max_positions

        intents: list[TradeIntent] = []
        for ticker, ret in picks:
            intents.append(
                TradeIntent(
                    ticker=ticker,
                    allocation_pct=allocation_each,
                    stop_loss_pct=None,
                    take_profit_pct=None,
                    thesis=(
                        f"Top previous-day gainer ({ret:+.2f}% close-to-open). "
                        f"Rule-based control: buying the prior session's strongest movers "
                        f"with no further filter."
                    ),
                )
            )
        return intents
