"""ML challenger — the bot's first *learned* strategy.

In place of the LLM prompt stages: load the committed LightGBM model,
build point-in-time features for the previous completed session, emit
5-class probabilities for every scorable name in the universe, and log
a PredictionRecord for each — same records, same vocabulary, same
grader as every LLM strategy, so `metrics.py` and weekly evolution
judge ML vs LLM head-to-head with zero new machinery.

Deployed at shadow tier; `prefilter_mode: off` — the model ranks the
full cross-section (a movers-biased shortlist would bias the ranking).
Model hyperparameters live in strategies/ml-challenger/model/, outside
config.yaml, so the evolution agent's `tune` action can't touch them;
the config additionally carries `evolution_frozen: true` (see
meta.evolution._apply_action) to keep the experiment clean.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np

from trading_bot.executor.base import TradeIntent
from trading_bot.ml.features import build_features, load_manifest, ManifestMismatchError
from trading_bot.state.predictions import PredictionRecord, append_prediction
from trading_bot.strategy.base import Strategy
from trading_bot.tools import get_universe


log = logging.getLogger(__name__)

# Minimum long-side edge — P(up classes) − P(down classes) — for a name
# to be tradeable. Deliberately a module constant, not config: the
# evolution agent may not tune the experiment's decision rule.
CONFIDENCE_FLOOR = 0.10

# If the freshest feature date is older than this many calendar days
# before the expected previous session, the runtime OHLCV cache is
# stale — refuse to trade on rotten features.
MAX_FEATURE_STALENESS_DAYS = 4


class MLChallengerStrategy(Strategy):
    """5-class LightGBM over point-in-time OHLCV features, shadow tier."""

    def select_picks(self, on_date: date) -> list[TradeIntent]:
        import lightgbm as lgb
        cfg = self.config

        model_dir = self._model_dir()
        manifest = load_manifest(model_dir / "feature_manifest.json")
        booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
        if booster.feature_name() != manifest["columns"]:
            raise ManifestMismatchError(
                f"{cfg.id}: model.txt feature names differ from feature_manifest.json — "
                f"the committed artifacts are out of sync"
            )

        tickers = get_universe(cfg.universe)
        as_of = _previous_trading_day(on_date, cfg.region)

        feats = build_features(tickers, as_of)
        if feats.empty:
            log.warning("%s: no feature rows for %s — OHLCV cache too thin, no picks",
                        cfg.id, as_of.isoformat())
            return []
        feature_date = feats["date"].iloc[0]
        if (as_of - feature_date).days > MAX_FEATURE_STALENESS_DAYS:
            log.warning("%s: freshest features are %s but previous session is %s — "
                        "stale cache, refusing to predict", cfg.id, feature_date, as_of)
            return []

        X = feats[manifest["columns"]].to_numpy()
        probs = booster.predict(X)
        classes: list[str] = manifest["classes"]
        midpoints = np.array([manifest["class_midpoints"][c] for c in classes])
        up = [i for i, c in enumerate(classes) if c.endswith("_up")]
        down = [i for i, c in enumerate(classes) if c.endswith("_down")]

        scores = probs @ midpoints                       # predicted_return_pct
        edge = probs[:, up].sum(axis=1) - probs[:, down].sum(axis=1)
        pred_idx = probs.argmax(axis=1)
        conviction = probs.max(axis=1)
        rationales = _rationales(booster, X, pred_idx, manifest["columns"])

        candidates = feats[["ticker"]].copy()
        candidates["score"] = scores
        candidates["edge"] = edge
        candidates["pred_class"] = [classes[i] for i in pred_idx]
        candidates["conviction"] = conviction
        candidates["rationale"] = rationales
        log.info("%s: scored %d candidates for %s (features from %s)",
                 cfg.id, len(candidates), on_date.isoformat(), feature_date)

        # Picks: top-N by long edge above the floor, equal weight like
        # control-rule-based, hold_days=1. The earnings gate only prunes
        # the tradeable shortlist — every scored name is still logged
        # below, because grading is free and more graded rows = tighter IC.
        shortlist = candidates[candidates["edge"] >= CONFIDENCE_FLOOR].nlargest(
            max(cfg.max_positions * 2, cfg.max_positions), "edge",
        )
        if cfg.skip_if_earnings_in_days > 0 and not shortlist.empty:
            keep = [t for t in shortlist["ticker"]
                    if not _earnings_within(t, on_date, cfg.skip_if_earnings_in_days)]
            shortlist = shortlist[shortlist["ticker"].isin(keep)]
        picks = shortlist.nlargest(cfg.max_positions, "edge")

        intents: list[TradeIntent] = []
        allocation_each = 100.0 / cfg.max_positions
        for _, row in picks.iterrows():
            intents.append(TradeIntent(
                ticker=row["ticker"],
                allocation_pct=allocation_each,
                stop_loss_pct=None,
                take_profit_pct=None,
                thesis=(
                    f"ML challenger: P(up)−P(down)={row['edge']:+.2f}, "
                    f"predicted {row['pred_class']} ({row['score']:+.2f}%). "
                    f"Top features: {row['rationale']}"
                ),
                hold_days=1,
            ))

        self._log_predictions(candidates, picked={i.ticker for i in intents}, on_date=on_date)
        return intents

    # -- internals ---------------------------------------------------------

    def _model_dir(self):
        from trading_bot.strategy.registry import _strategies_dir
        return _strategies_dir() / self.config.id / "model"

    def _log_predictions(self, candidates, *, picked: set[str], on_date: date) -> None:
        """One PredictionRecord per scored candidate — prediction logging
        lives inside the strategy (see LLMStrategy._log_predictions), not
        the pipeline. The exit-step grader fills actuals for free."""
        cfg = self.config
        for _, row in candidates.iterrows():
            append_prediction(PredictionRecord(
                strategy_id=cfg.id,
                region=cfg.region,
                prediction_date=on_date.isoformat(),
                ticker=row["ticker"],
                predicted_class=row["pred_class"],
                predicted_return_pct=round(float(row["score"]), 2),
                conviction=round(float(row["conviction"]), 2),
                rationale=row["rationale"],
                was_traded=row["ticker"] in picked,
                tools_used=[],
                prefilter_mode=(cfg.prefilter_mode or "").lower().strip(),
            ))


def _rationales(booster, X: np.ndarray, pred_idx: np.ndarray, columns: list[str]) -> list[str]:
    """Top-3 features by |contribution| toward the argmax class, with
    their raw values — gives the dashboard and the reflection pass
    something legible ('ret_21d=+0.031, vol_10d=0.42, …')."""
    try:
        contrib = booster.predict(X, pred_contrib=True)
    except Exception as e:
        log.warning("pred_contrib failed (%s) — falling back to blank rationales", e)
        return ["(no attribution)"] * len(X)
    n_feat = len(columns)
    out = []
    for i in range(len(X)):
        block = contrib[i, pred_idx[i] * (n_feat + 1) : pred_idx[i] * (n_feat + 1) + n_feat]
        top = np.argsort(-np.abs(block))[:3]
        out.append(", ".join(f"{columns[j]}={X[i, j]:+.3g}" for j in top))
    return out


def _previous_trading_day(on_date: date, region: str) -> date:
    """The last completed session before `on_date` — the pipeline fires
    pre-open, so bar t−1 is the freshest completed bar."""
    try:
        from trading_bot.tools.calendar import add_trading_days
        return add_trading_days(on_date, -1, region)
    except Exception as e:
        log.warning("trading-calendar lookup failed (%s) — falling back to weekday walk", e)
        d = on_date - timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return d


def _earnings_within(ticker: str, on_date: date, window_days: int) -> bool:
    """Best-effort, matching LLMStrategy._filter_pre_earnings: if the
    lookup fails or returns nothing, keep the candidate."""
    try:
        from trading_bot.tools.earnings import get_earnings_info
        info = get_earnings_info(ticker)
        if not info.next_earnings_date:
            return False
        next_date = date.fromisoformat(info.next_earnings_date)
    except Exception:
        return False
    return on_date <= next_date <= on_date + timedelta(days=window_days)
