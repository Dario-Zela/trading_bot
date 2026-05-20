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
    tier: str  # "shadow" | "alpaca-paper" | "trading212-paper" | "t212-live"
    region: str
    capital_gbp: float
    max_positions: int
    max_position_pct: float
    min_position_gbp: float
    use_stops: bool
    use_take_profits: bool
    universe: str = "sp500"
    alpaca_slot: int | None = None      # required for tier=alpaca-paper
    t212_slot: int | None = None        # required for tier=trading212-paper
    stop_loss_pct: float | None = None  # used by stub implementations; LLM picks per-trade
    take_profit_pct: float | None = None
    tools: list[str] = field(default_factory=list)
    model_assignment: dict = field(default_factory=dict)

    # Phase 8A — volatility-aware position sizing. Target fraction of
    # `capital_gbp` we're willing to lose on a 1-ATR adverse move per
    # position. The strategy post-processes the LLM's `allocation_pct`
    # so high-vol names get smaller sizes and low-vol names get larger
    # ones, with `max_position_pct` as the hard cap. Default 1% gives a
    # ~3-5x size range across the typical RSI / momentum candidate set.
    target_daily_risk_pct: float = 1.0
    # Phase 8B — pre-trade FX cost gate. Drop picks where the LLM's
    # predicted_return_pct is less than this multiplier × the round-trip
    # cost (FX + stamp duty + FTT). 2x default means we need ≥2:1
    # signal-to-cost odds.
    cost_gate_multiplier: float = 2.0
    # Phase 8C — earnings gating. Skip candidates with earnings inside
    # this many days. 0 = disabled; 1 = avoid binary events tomorrow.
    skip_if_earnings_in_days: int = 0
    # Phase 11B — set by the evolution agent on each `tune` action.
    # `compute_metrics` clips its window to this date so post-tune
    # IC / hit-rate isn't diluted by pre-tune trades.
    last_tune_date: str | None = None

    # Tier 2 candidate flag — set by the weekly evolution agent on
    # strategies it thinks are worth elevating to a graduated tier.
    # The flag is the agent's prediction; next week's run grades the
    # candidate's realised performance to validate the analysis. Two
    # ledger-side fields support that self-check:
    #
    #   tier2_candidate         True if the agent has flagged it this
    #                           cycle; the dashboard renders a gold
    #                           border for these strategies.
    #   tier2_marked_at         ISO date the flag was set, so the
    #                           next evolution run knows how much
    #                           realised performance to score against.
    #   tier2_thesis            One-line explanation from the agent
    #                           of why this strategy deserves the
    #                           candidate flag; the next run reads
    #                           this back when grading.
    tier2_candidate: bool = False
    tier2_marked_at: str | None = None
    tier2_thesis: str = ""


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
