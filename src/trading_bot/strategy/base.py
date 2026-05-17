from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

from trading_bot.executor.base import TradeIntent


@dataclass
class StrategyConfig:
    id: str
    display_name: str
    description: str
    implementation: str  # "rule_based" | "momentum_stub" | "llm"
    active: bool
    tier: str  # "shadow" | "alpaca-paper" | "t212-live"
    region: str
    capital_gbp: float
    max_positions: int
    max_position_pct: float
    min_position_gbp: float
    use_stops: bool
    use_take_profits: bool
    universe: str = "sp500"
    alpaca_slot: int | None = None      # required for tier=alpaca-paper
    stop_loss_pct: float | None = None  # used by stub implementations; LLM picks per-trade
    take_profit_pct: float | None = None
    tools: list[str] = field(default_factory=list)
    model_assignment: dict = field(default_factory=dict)


class Strategy(ABC):
    """Base class for any tradeable strategy. Rule-based subclass overrides
    select_picks directly; LLM-driven subclasses (Wave 2+) will drive the
    multi-stage pipeline."""

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    @abstractmethod
    def select_picks(self, on_date: date) -> list[TradeIntent]:
        """Decide today's entries. Returns a list of TradeIntent.
        Empty list means 'no trades today'."""
