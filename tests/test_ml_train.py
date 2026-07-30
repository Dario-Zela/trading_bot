"""Trainer tests: determinism (fixed seed → identical model), frozen
midpoints, and the metric helpers the card quotes."""
from __future__ import annotations

import hashlib
import math
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from trading_bot.ml.splits import purged_walk_forward
from trading_bot.ml.train import (
    CLASSES,
    _lgbm_params,
    fisher_z_lower_bound,
    fit_class_midpoints,
    multiclass_logloss,
    permutation_noise_floor,
    pooled_ic,
    score_from_probs,
    train_lgbm,
)
from trading_bot.ml import features as F


def _tiny_matrix(n: int = 600, seed: int = 5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(F.FEATURE_COLUMNS)))
    y = rng.integers(0, len(CLASSES), size=n)
    return X, y


def test_fixed_seed_identical_model_hash():
    X, y = _tiny_matrix()
    params = _lgbm_params(0.1, 4, seed=42)
    params["min_data_in_leaf"] = 5  # tiny synthetic set
    hashes = []
    for _ in range(2):
        booster = train_lgbm(X[:500], y[:500], X[500:], y[500:], params)
        hashes.append(hashlib.sha256(booster.model_to_string().encode()).hexdigest())
    assert hashes[0] == hashes[1]


def test_different_seed_different_model():
    X, y = _tiny_matrix()
    boosters = []
    for seed in (42, 43):
        params = _lgbm_params(0.1, 4, seed=seed)
        params["min_data_in_leaf"] = 5
        boosters.append(train_lgbm(X[:500], y[:500], X[500:], y[500:], params))
    assert boosters[0].model_to_string() != boosters[1].model_to_string()


def test_midpoints_frozen_from_first_training_block_only():
    """Rows outside the first fold's training window — including every
    validation window — must not move the midpoints."""
    days = []
    d = date(2025, 1, 6)
    while len(days) < 160:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)

    rng = random.Random(9)
    rows = []
    for day in days:
        for t in range(12):
            r = rng.gauss(0, 2.5)
            cls = ("strong_up" if r >= 4 else "mild_up" if r >= 1 else
                   "strong_down" if r <= -4 else "mild_down" if r <= -1 else "flat")
            rows.append({"date": day, "ticker": f"T{t}", "actual_return_pct": r, "actual_class": cls})
    df = pd.DataFrame(rows)

    folds = purged_walk_forward(days, burn_in=100, val_size=21, purge=1, embargo=5, step=21)
    mids = fit_class_midpoints(df, folds[0])

    # Corrupt everything OUTSIDE the first training block; midpoints
    # must be byte-identical.
    train_set = set(folds[0].train_dates)
    poisoned = df.copy()
    poisoned.loc[~poisoned["date"].isin(train_set), "actual_return_pct"] = 99.0
    assert fit_class_midpoints(poisoned, folds[0]) == mids

    # Ordering sanity: up midpoints above flat above down.
    assert mids["strong_up"] > mids["mild_up"] > mids["flat"] > mids["mild_down"] > mids["strong_down"]


def test_score_from_probs_is_probability_weighted_midpoint():
    mids = {"strong_down": -5.0, "mild_down": -2.0, "flat": 0.0, "mild_up": 2.0, "strong_up": 5.0}
    probs = np.array([
        [0.0, 0.0, 1.0, 0.0, 0.0],   # all flat → 0
        [0.0, 0.0, 0.0, 0.0, 1.0],   # all strong_up → +5
        [0.5, 0.0, 0.0, 0.0, 0.5],   # symmetric extremes → 0
    ])
    scores = score_from_probs(probs, mids)
    assert scores == pytest.approx([0.0, 5.0, 0.0])


def test_fisher_z_lower_bound_matches_closed_form():
    ic, n = 0.05, 5000
    expected = math.tanh(math.atanh(ic) - 1.96 / math.sqrt(n - 3))
    assert fisher_z_lower_bound(ic, n) == pytest.approx(expected, abs=1e-4)
    assert fisher_z_lower_bound(None, 100) is None
    assert fisher_z_lower_bound(0.5, 3) is None


def test_noise_floor_beats_null_and_flags_signal():
    """A genuinely predictive score must clear the permutation floor; a
    random score must not (with overwhelming probability at n=5000)."""
    rng = np.random.default_rng(3)
    actual = rng.normal(size=5000)
    signal = actual + rng.normal(scale=3.0, size=5000)     # IC ≈ 0.3
    noise = rng.normal(size=5000)
    df_sig = pd.DataFrame({"score": signal, "actual_return_pct": actual})
    df_noise = pd.DataFrame({"score": noise, "actual_return_pct": actual})

    floor_sig = permutation_noise_floor(df_sig, n_iter=200)
    assert pooled_ic(df_sig) > floor_sig
    floor_noise = permutation_noise_floor(df_noise, n_iter=200)
    assert pooled_ic(df_noise) < floor_noise + 0.05


def test_multiclass_logloss_perfect_and_uniform():
    y = np.array([0, 1, 2, 3, 4])
    perfect = np.eye(5)[y]
    assert multiclass_logloss(y, perfect) == pytest.approx(0.0, abs=1e-6)
    uniform = np.full((5, 5), 0.2)
    assert multiclass_logloss(y, uniform) == pytest.approx(math.log(5), abs=1e-3)
