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
    t212_slot = raw.get("t212_slot")
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
        t212_slot=int(t212_slot) if t212_slot is not None else None,
        stop_loss_pct=float(raw["stop_loss_pct"]) if raw.get("stop_loss_pct") is not None else None,
        take_profit_pct=float(raw["take_profit_pct"]) if raw.get("take_profit_pct") is not None else None,
        tools=list(raw.get("tools", [])),
        model_assignment=dict(raw.get("model_assignment", {})),
    )


def load_active_strategies(region: str | None = None) -> list[Strategy]:
    """Discover and instantiate every active strategy for the given region.

    Supports two config shapes:
    - Single-region: `region: us` top-level field.
    - Multi-region: a `runs_in` list of {region, universe, tier, alpaca_slot}
      entries. The loader expands one Strategy instance per entry whose
      region matches the filter.
    """
    from trading_bot.strategy.control_rule_based import ControlRuleBased
    from trading_bot.strategy.llm_strategy import LLMStrategy
    from trading_bot.strategy.momentum_stub import MomentumTraderStub

    def _instantiate(cfg: StrategyConfig) -> Strategy:
        if cfg.implementation == "rule_based":
            return ControlRuleBased(cfg)
        if cfg.implementation == "momentum_stub":
            return MomentumTraderStub(cfg)
        if cfg.implementation == "llm":
            return LLMStrategy(cfg)
        raise ValueError(f"Unknown strategy implementation: {cfg.implementation}")

    out: list[Strategy] = []
    for config_path in _strategies_dir().glob("*/config.yaml"):
        raw = yaml.safe_load(config_path.read_text())
        if not raw.get("active", False):
            continue
        for derived in _expand_regions(raw):
            if region is not None and derived.get("region") != region:
                continue
            out.append(_instantiate(_to_config(derived)))
    return out


def _expand_regions(raw: dict) -> list[dict]:
    """Yield one fully-merged config dict per region this strategy runs in.

    Single-region configs (with top-level `region` and no `runs_in`) yield
    a single entry unchanged. Multi-region configs (with `runs_in: [...]`)
    yield one merged dict per region entry — top-level fields are the
    defaults, region entries override.
    """
    runs_in = raw.get("runs_in")
    if not runs_in:
        return [raw]

    out = []
    for entry in runs_in:
        if not isinstance(entry, dict) or not entry.get("region"):
            continue
        merged = {k: v for k, v in raw.items() if k != "runs_in"}
        merged.update(entry)
        # If region-specific entry doesn't override tier/universe etc, the
        # top-level defaults still apply via the merge above.
        out.append(merged)
    return out
