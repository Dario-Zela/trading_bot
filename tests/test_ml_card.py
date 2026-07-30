"""Smoke test: evaluate_oos → render_card renders a complete markdown
card from a synthetic OOS frame — a format regression must fail here,
not at the end of a 40-minute training run."""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from trading_bot.ml.features import CLASSES, FEATURE_COLUMNS
from trading_bot.ml.train import evaluate_oos, render_card


def _synthetic_oos(n_days: int = 30, n_tickers: int = 40, seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    d = date(2026, 1, 5)
    for _ in range(n_days):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        for t in range(n_tickers):
            probs = rng.dirichlet(np.ones(len(CLASSES)))
            actual = float(rng.normal(0, 2))
            rows.append({
                "date": d, "ticker": f"TK{t}", "fold": 0,
                "actual_return_pct": actual,
                "y": int(np.clip((actual > 1) * 3 + (actual > 4) + 2 - (actual < -1) - (actual < -4), 0, 4)),
                **{f"p_{c}": probs[i] for i, c in enumerate(CLASSES)},
                "score": float(probs @ np.array([-5.5, -1.9, 0.0, 1.9, 5.5])),
                "pred_class_idx": int(probs.argmax()),
            })
        d += timedelta(days=1)
    return pd.DataFrame(rows)


def test_render_card_complete():
    oos = _synthetic_oos()
    challenger = evaluate_oos(oos, "LightGBM challenger")
    momentum = evaluate_oos(oos.assign(score=-oos["score"]), "momentum", with_probs=False)
    report = {
        "generated_at": "2026-07-30T12:00:00+00:00",
        "seed": 42,
        "db": "state/ml/train.db",
        "snapshot_hash": "abcd1234abcd1234",
        "audit": {"total_rows": 500000, "n_covered": 1400, "date_min": "2025-01-27",
                  "date_max": "2026-07-29", "n_universe": 6900, "median_depth": 379,
                  "p10_depth": 300, "n_depth_ge_150": 1400},
        "n_rows": len(oos), "n_tickers": 40, "n_dates": 30,
        "date_range": ["2025-05-01", "2026-07-29"],
        "class_support": {c: 100 + i for i, c in enumerate(CLASSES)},
        "midpoints": {"strong_down": -5.5, "mild_down": -1.9, "flat": 0.0,
                      "mild_up": 1.9, "strong_up": 5.5},
        "best": {"learning_rate": 0.05, "max_depth": 6, "median_best_iter": 210,
                 "grid": [{"combo": [0.05, 6], "logloss": 1.4, "pooled_ic": 0.03}]},
        "folds": [{"fold": 0, "train_through": "2026-01-30", "val_start": "2026-02-09",
                   "val_end": "2026-03-09", "n_val": 1200, "logloss": 1.42,
                   "pooled_ic": 0.031, "mean_daily_ic": 0.025, "best_iter": 200}],
        "challenger": challenger,
        "noise_floor": 0.012,
        "fisher_z_lower_bound": 0.019,
        "ic_decay": [{"horizon_days": h, "pooled_ic": 0.02 / h, "n": 1000} for h in (1, 2, 3)],
        "baselines": {
            "uniform": {"label": "Uniform prior", "n": challenger["n"],
                        "logloss": round(math.log(5), 4), "pooled_ic": None,
                        "daily_ic": {"mean": None}, "decile_spread": None},
            "momentum": momentum,
            "logistic": evaluate_oos(oos, "logistic"),
            "control_rule_based": {"n": 800, "pooled_ic": 0.01,
                                   "decile_spread": 0.5, "hit_rate": 0.49},
        },
        "portfolio": {"n_days": 30, "mean_daily_net_pct": 0.05, "hit_rate": 0.53,
                      "cumulative_net_pct": 1.5, "worst_day_pct": -2.1,
                      "cost_pct_per_trade": 0.3},
        "cost_note": "fallback 2×0.15% FX legs",
        "importances": [[c, 100.0] for c in FEATURE_COLUMNS[:20]],
        # Stretch sections
        "region": "us",
        "cpcv": {"n_paths": 15, "mean": 0.02, "std": 0.01, "min": -0.01, "max": 0.04,
                 "share_positive": 0.8},
        "regime_ic": [
            {"regime": "calm (VIX ≤ 14.0)", "n": 400, "n_days": 10, "pooled_ic": 0.01,
             "mean_daily_ic": 0.005},
            {"regime": "stressed (VIX > 20.0)", "n": 400, "n_days": 10, "pooled_ic": 0.03,
             "mean_daily_ic": 0.02},
        ],
        "horizons": {
            1: {"n": 1200, "pooled_ic": 0.02, "mean_daily_ic": 0.01, "logloss": 1.5,
                "class_support": {c: 100 for c in CLASSES}},
            3: {"n": 1100, "pooled_ic": 0.03, "mean_daily_ic": 0.015, "logloss": 1.55,
                "class_support": {c: 100 for c in CLASSES}},
        },
        "shap": [{"feature": c, "mean_abs_shap": 0.01, "mean_signed_strong_up": 0.001}
                 for c in FEATURE_COLUMNS],
    }
    report["baselines"]["mlp"] = evaluate_oos(oos, "MLP (PyTorch, v2 preview)")
    card = render_card(report)
    for heading in ("# Model card", "## Data", "## Label", "## Features",
                    "## Validation", "## Results vs baselines", "## Significance",
                    "## CPCV", "## Regime-conditional IC", "## Multi-horizon heads",
                    "## Calibration", "## Toy top-5", "## Feature attribution",
                    "## Limitations", "## Reproduce"):
        assert heading in card
    for feature in FEATURE_COLUMNS:
        assert f"`{feature}`" in card
    assert "Fisher-z" in card and "noise floor" in card.lower()
    assert "MLP (PyTorch, v2 preview)" in card
    # The SHAP table must not contain a broken header (pipes inside cells)
    assert "mean abs SHAP" in card
