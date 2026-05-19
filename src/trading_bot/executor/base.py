from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class TradeIntent:
    """A strategy's intent to enter one position. Sizing is in % of strategy capital."""

    ticker: str
    allocation_pct: float
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    thesis: str = ""
    # Phase 12A — multi-day positioning. Defaults to 1 (same-day round-
    # trip — current Wave 1 behaviour). LLM strategies can return
    # `hold_days ∈ {1, 2, 3, 5, 10}` per pick; rule-based strategies
    # default to 1. Exit machinery checks today >= target_exit_date
    # before closing a position.
    hold_days: int = 1


class Executor(ABC):
    """A broker-tier adapter. Wave 1 has ShadowExecutor only. Later waves add
    AlpacaPaperExecutor (Tier 1) and Trading212ApproveExecutor (Tier 2, deferred).
    """

    @abstractmethod
    def enter(
        self,
        intents: list[TradeIntent],
        *,
        strategy_id: str,
        region: str,
        capital_gbp: float,
        on_date: date,
    ) -> None: ...

    @abstractmethod
    def exit_scheduled(
        self,
        *,
        strategy_id: str,
        region: str,
        on_date: date,
    ) -> list[dict]:
        """Close all positions opened on `on_date` for this strategy. Returns the
        closed trade records (after exit fields are populated)."""
