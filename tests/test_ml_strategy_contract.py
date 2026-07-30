"""Strategy-contract test: select_picks returns TradeIntents, every
logged row constructs a valid PredictionRecord, and the registry
dispatches `implementation: ml_challenger`. Hermetic — the registry is
pointed at a fixture config via TRADING_BOT_STRATEGIES_DIR, the
universe and OHLCV store are synthetic, no network."""
from __future__ import annotations

import json
import random
from datetime import date, timedelta

import pytest

from trading_bot.executor.base import TradeIntent
from trading_bot.ml import features as F
from trading_bot.ml.labels import compute_labels
from trading_bot.ml.train import DEFAULT_MIDPOINTS, _lgbm_params, train_lgbm
from trading_bot.state.predictions import PredictionRecord

N_TICKERS = 10
N_DAYS = 100


def _synthetic_bars(seed: int = 21):
    from trading_bot.tools.ohlcv_store import StoredBar
    rng = random.Random(seed)
    start = date(2026, 1, 5)  # a Monday
    days: list[date] = []
    d = start
    while len(days) < N_DAYS:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    bars = []
    for i in range(N_TICKERS):
        price = 40.0 + 5 * i
        for day in days:
            drift = rng.gauss(0, 0.02)
            open_ = price * (1 + rng.gauss(0, 0.005))
            close = price * (1 + drift)
            bars.append(StoredBar(
                ticker=f"TK{i}", bar_date=day,
                open=open_, high=max(open_, close) * 1.004,
                low=min(open_, close) * 0.996, close=close,
                volume=int(1e6 * (1 + rng.random())),
            ))
            price = close
    return bars, days


@pytest.fixture
def ml_strategy_env(tmp_path, monkeypatch):
    """Fixture strategies dir + synthetic runtime OHLCV store + a tiny
    trained model whose manifest matches the live feature spec."""
    # State root — predictions.jsonl lands here; ohlcv_store binds
    # STATE_ROOT at import so patch its module copy too.
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr("trading_bot.state.paths.STATE_ROOT", state)
    monkeypatch.setattr("trading_bot.tools.ohlcv_store.STATE_ROOT", state)

    from trading_bot.tools.ohlcv_store import write_bars
    bars, days = _synthetic_bars()
    write_bars(bars)

    # Train a tiny real model on the synthetic panel so the full
    # load-model → assert-manifest → predict path runs for real.
    panel = F.panel_from_bars({b.ticker: [x for x in bars if x.ticker == b.ticker]
                               for b in bars})
    feats = F.compute_feature_panel(panel)
    labels = compute_labels(panel)
    df = feats.merge(labels, on=["ticker", "date"], how="inner")
    y = df["actual_class"].map(F.CLASSES.index).to_numpy()
    X = df[F.FEATURE_COLUMNS].to_numpy()
    params = _lgbm_params(0.1, 4, seed=42)
    params["min_data_in_leaf"] = 5
    booster = train_lgbm(X, y, X, y, params)

    sdir = tmp_path / "strategies" / "ml-challenger"
    model_dir = sdir / "model"
    model_dir.mkdir(parents=True)
    booster.save_model(str(model_dir / "model.txt"))
    F.write_manifest(model_dir / "feature_manifest.json", class_midpoints=DEFAULT_MIDPOINTS)
    (sdir / "config.yaml").write_text(
        "id: ml-challenger\n"
        "display_name: ML challenger\n"
        "implementation: ml_challenger\n"
        "active: true\n"
        "capital_gbp: 10000\n"
        "max_positions: 3\n"
        "max_position_pct: 30\n"
        "use_stops: false\n"
        "use_take_profits: false\n"
        "prefilter_mode: 'off'\n"
        "skip_if_earnings_in_days: 0\n"   # hermetic — no yfinance earnings lookups
        "evolution_frozen: true\n"
        "runs_in:\n"
        "  - {region: us, tier: shadow, universe: t212_isa_us}\n"
    )
    monkeypatch.setenv("TRADING_BOT_STRATEGIES_DIR", str(tmp_path / "strategies"))

    tickers = sorted({b.ticker for b in bars})
    monkeypatch.setattr("trading_bot.strategy.ml_challenger.get_universe", lambda _u: tickers)
    # Trade "today" = the session after the last synthetic bar.
    on_date = days[-1] + timedelta(days=3 if days[-1].weekday() == 4 else 1)
    return {"state": state, "on_date": on_date, "tickers": tickers}


def test_registry_dispatches_ml_challenger(ml_strategy_env):
    from trading_bot.strategy.ml_challenger import MLChallengerStrategy
    from trading_bot.strategy.registry import load_active_strategies
    active = load_active_strategies(region="us")
    assert len(active) == 1
    assert isinstance(active[0], MLChallengerStrategy)
    assert active[0].config.tier == "shadow"


def test_select_picks_contract(ml_strategy_env, monkeypatch):
    from trading_bot.strategy import ml_challenger as mc
    from trading_bot.strategy.registry import load_active_strategies

    # Floor to −1 so the synthetic model always yields tradeable edge —
    # the contract under test is shape, not signal.
    monkeypatch.setattr(mc, "CONFIDENCE_FLOOR", -1.0)

    strategy = load_active_strategies(region="us")[0]
    on_date = ml_strategy_env["on_date"]
    intents = strategy.select_picks(on_date)

    assert intents, "expected picks with the floor disabled"
    assert len(intents) <= strategy.config.max_positions
    for i in intents:
        assert isinstance(i, TradeIntent)
        assert i.ticker in ml_strategy_env["tickers"]
        assert i.hold_days == 1
        assert i.allocation_pct == pytest.approx(100.0 / strategy.config.max_positions)
        assert "ML challenger" in i.thesis

    # Every scored candidate got logged; each row round-trips through
    # the PredictionRecord dataclass (the schema contract).
    pred_path = ml_strategy_env["state"] / "predictions.jsonl"
    rows = [json.loads(line) for line in pred_path.read_text().splitlines() if line.strip()]
    assert len(rows) == len(ml_strategy_env["tickers"])
    picked = {i.ticker for i in intents}
    for r in rows:
        rec = PredictionRecord(**r)
        assert rec.strategy_id == "ml-challenger"
        assert rec.region == "us"
        assert rec.prediction_date == on_date.isoformat()
        assert rec.predicted_class in F.CLASSES
        assert 0.0 <= rec.conviction <= 1.0
        assert rec.actual_return_pct is None          # graded later, for free
        assert rec.was_traded == (rec.ticker in picked)
        assert rec.prefilter_mode == "off"
        assert rec.rationale                          # top-3 feature attribution


def test_manifest_drift_refuses_to_predict(ml_strategy_env, monkeypatch):
    """A model trained against a different feature spec must never be
    served — the load-time assert is the train/serve-skew fuse."""
    import json as _json

    from trading_bot.ml.features import ManifestMismatchError
    from trading_bot.strategy.registry import _strategies_dir, load_active_strategies

    manifest_path = _strategies_dir() / "ml-challenger" / "model" / "feature_manifest.json"
    raw = _json.loads(manifest_path.read_text())
    raw["spec_hash"] = "0000000000000000"
    manifest_path.write_text(_json.dumps(raw))

    strategy = load_active_strategies(region="us")[0]
    with pytest.raises(ManifestMismatchError):
        strategy.select_picks(ml_strategy_env["on_date"])


def test_evolution_frozen_guard_blocks_mutations():
    from trading_bot.meta.evolution import _apply_action
    cfg = {"active": True, "evolution_frozen": True,
           "runs_in": [{"region": "us", "tier": "shadow"}]}
    for action, extra in [
        ("tune", {"changes": {"max_positions": 2}}),
        ("deactivate", {}),
        ("demote", {"region": "us"}),
        ("promote", {"region": "us"}),
    ]:
        result = _apply_action(
            {"strategy_id": "ml-challenger", "action": action, "reason": "test", **extra},
            metrics={}, configs={"ml-challenger": dict(cfg)},
        )
        assert result is not None
        assert result.applied is False
        assert "frozen" in result.reason.lower()
