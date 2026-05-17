from __future__ import annotations

import os
from pathlib import Path

import yaml

from trading_bot.strategy.base import Strategy, StrategyConfig


def _strategies_dir() -> Path:
    override = os.environ.get("TRADING_BOT_STRATEGIES_DIR")
    if override:
        return Path(override)
    # src/trading_bot/strategy/registry.py → repo root is 4 parents up
    return Path(__file__).resolve().parents[3] / "strategies"


def load_strategy_config(strategy_id: str) -> StrategyConfig:
    config_path = _strategies_dir() / strategy_id / "config.yaml"
    raw = yaml.safe_load(config_path.read_text())
    return _to_config(raw)


def _to_config(raw: dict) -> StrategyConfig:
    alpaca_slot = raw.get("alpaca_slot")
    return StrategyConfig(
        id=raw["id"],
        display_name=raw["display_name"],
        description=raw.get("description", ""),
        implementation=raw["implementation"],
        active=raw.get("active", False),
        tier=raw.get("tier", "shadow"),
        region=raw.get("region", "us"),
        capital_gbp=float(raw.get("capital_gbp", 10000)),
        max_positions=int(raw.get("max_positions", 5)),
        max_position_pct=float(raw.get("max_position_pct", 30)),
        min_position_gbp=float(raw.get("min_position_gbp", 50)),
        use_stops=bool(raw.get("use_stops", False)),
        use_take_profits=bool(raw.get("use_take_profits", False)),
        universe=raw.get("universe", "sp500"),
        alpaca_slot=int(alpaca_slot) if alpaca_slot is not None else None,
        stop_loss_pct=float(raw["stop_loss_pct"]) if raw.get("stop_loss_pct") is not None else None,
        take_profit_pct=float(raw["take_profit_pct"]) if raw.get("take_profit_pct") is not None else None,
        tools=list(raw.get("tools", [])),
        model_assignment=dict(raw.get("model_assignment", {})),
    )


def load_active_strategies(region: str | None = None) -> list[Strategy]:
    """Discover and instantiate every active strategy for the given region."""
    from trading_bot.strategy.control_rule_based import ControlRuleBased
    from trading_bot.strategy.momentum_stub import MomentumTraderStub

    out: list[Strategy] = []
    for config_path in _strategies_dir().glob("*/config.yaml"):
        raw = yaml.safe_load(config_path.read_text())
        config = _to_config(raw)
        if not config.active:
            continue
        if region is not None and config.region != region:
            continue
        if config.implementation == "rule_based":
            out.append(ControlRuleBased(config))
        elif config.implementation == "momentum_stub":
            out.append(MomentumTraderStub(config))
        elif config.implementation == "llm":
            # Wave 2b will activate these
            continue
        else:
            raise ValueError(f"Unknown strategy implementation: {config.implementation}")
    return out
