"""Wave 2a stub for momentum-trader: pure-Python ranking by recent return + RSI band.

The "real" momentum-trader is LLM-driven (Wave 2b). This stub exists so we can
ship Wave 2a — real Alpaca paper fills, bracket orders, multi-tier execution —
without waiting for the LLM pipeline. It also gives us a baseline to compare
the eventual LLM version against (just like control-rule-based is the baseline
across all strategies).

Logic:
  1. Universe: S&P 500
  2. Compute technicals for every name
  3. Filter to those in a healthy uptrend: RSI between 50 and 75 (trending, not
     overbought), above 20-day MA, recent volume not collapsing
  4. Rank by 5-day return
  5. Take top max_positions, equal-weight
  6. Each pick gets the strategy's fixed stop_loss_pct / take_profit_pct
"""
from __future__ import annotations

import logging
from datetime import date

from trading_bot.executor.base import TradeIntent
from trading_bot.strategy.base import Strategy
from trading_bot.tools import get_technicals, get_universe


log = logging.getLogger(__name__)


_RSI_FLOOR = 50.0
_RSI_CEIL = 75.0


class MomentumTraderStub(Strategy):
    def select_picks(self, on_date: date) -> list[TradeIntent]:
        cfg = self.config
        tickers = get_universe(cfg.universe)
        log.info("momentum-stub: scoring %d candidates", len(tickers))

        techs = get_technicals(tickers, end_date=on_date)
        log.info("momentum-stub: technicals computed for %d names", len(techs))

        candidates: list[tuple[str, float, dict]] = []
        for ticker, t in techs.items():
            if t.rsi_14 is None or t.return_5d_pct is None or t.above_sma_20 is None:
                continue
            if not t.above_sma_20:
                continue
            if not (_RSI_FLOOR <= t.rsi_14 <= _RSI_CEIL):
                continue
            # Recent volume should be at least 60% of 20-day average
            if t.volume_ratio is not None and t.volume_ratio < 0.6:
                continue
            candidates.append(
                (ticker, t.return_5d_pct, {"rsi": t.rsi_14, "ret5": t.return_5d_pct})
            )

        candidates.sort(key=lambda x: x[1], reverse=True)
        picks = candidates[: cfg.max_positions]
        if not picks:
            log.info("momentum-stub: no candidates passed the trend filter")
            return []

        allocation_each = min(100.0 / cfg.max_positions, cfg.max_position_pct)

        intents: list[TradeIntent] = []
        for ticker, ret5, meta in picks:
            intents.append(
                TradeIntent(
                    ticker=ticker,
                    allocation_pct=allocation_each,
                    stop_loss_pct=cfg.stop_loss_pct,
                    take_profit_pct=cfg.take_profit_pct,
                    thesis=(
                        f"Momentum stub pick. 5-day return {ret5:+.2f}%, RSI {meta['rsi']:.1f} "
                        f"(in healthy 50–75 band), price above 20-day MA. Bracket: "
                        f"{cfg.stop_loss_pct:+.1f}% stop / {cfg.take_profit_pct:+.1f}% take-profit."
                    ),
                )
            )
        return intents
