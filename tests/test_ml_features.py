"""ML challenger feature-pipeline tests — the leakage tests exist
BEFORE the first real training run, per the design doc.

The no-lookahead test is the one that matters: features at date t
computed from the full dataset must be byte-identical to features at
date t computed from data truncated at t. Everything in
compute_feature_panel is backward-looking per ticker or cross-sectional
within a single date, so truncation must be a no-op for surviving rows.
"""
from __future__ import annotations

import math
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from trading_bot.ml.features import (
    CLASSES,
    FEATURE_COLUMNS,
    MIN_HISTORY_SESSIONS,
    ManifestMismatchError,
    compute_feature_panel,
    load_manifest,
    spec_hash,
    write_manifest,
)


def _synthetic_panel(n_tickers: int = 8, n_days: int = 120, seed: int = 7) -> pd.DataFrame:
    """Random-walk OHLCV bars on weekdays only."""
    rng = random.Random(seed)
    start = date(2025, 1, 6)  # a Monday
    days: list[date] = []
    d = start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)

    rows = []
    for i in range(n_tickers):
        ticker = f"TK{i}"
        price = 50.0 + 10.0 * i
        for day in days:
            drift = rng.gauss(0, 0.02)
            open_ = price * (1 + rng.gauss(0, 0.005))
            close = price * (1 + drift)
            high = max(open_, close) * (1 + abs(rng.gauss(0, 0.004)))
            low = min(open_, close) * (1 - abs(rng.gauss(0, 0.004)))
            volume = int(1e6 * (1 + rng.random()))
            rows.append((ticker, day, open_, high, low, close, volume))
            price = close
    return pd.DataFrame(
        rows, columns=["ticker", "date", "open", "high", "low", "close", "volume"],
    )


def test_no_lookahead_truncation_identical():
    """Features at date t from the full panel == features at date t from
    the panel truncated at t, over 20 random (ticker, date) pairs."""
    panel = _synthetic_panel()
    full = compute_feature_panel(panel).set_index(["date", "ticker"]).sort_index()

    eligible = full.index.tolist()
    rng = random.Random(11)
    pairs = rng.sample(eligible, 20)

    for (d, ticker) in pairs:
        truncated_panel = panel[panel["date"] <= d]
        truncated = compute_feature_panel(truncated_panel).set_index(["date", "ticker"])
        row_full = full.loc[(d, ticker)]
        row_trunc = truncated.loc[(d, ticker)]
        pd.testing.assert_series_equal(row_full, row_trunc, check_names=False)


def test_min_history_rows_dropped():
    panel = _synthetic_panel(n_tickers=3, n_days=90)
    feats = compute_feature_panel(panel)
    # Every ticker has 90 bars; the first valid row is bar #MIN_HISTORY_SESSIONS.
    per_ticker = feats.groupby("ticker").size()
    assert (per_ticker == 90 - MIN_HISTORY_SESSIONS + 1).all()


def test_late_listing_ticker_dropped_until_enough_history():
    panel = _synthetic_panel(n_tickers=4, n_days=120)
    # TK3 "lists" 80 sessions in — only 40 bars of history, < 63.
    dates = sorted(panel["date"].unique())
    cutoff = dates[80]
    panel = panel[~((panel["ticker"] == "TK3") & (panel["date"] < cutoff))]
    feats = compute_feature_panel(panel)
    assert "TK3" not in set(feats["ticker"])


def test_feature_columns_complete_and_finite():
    feats = compute_feature_panel(_synthetic_panel())
    assert list(feats.columns) == ["date", "ticker", *FEATURE_COLUMNS]
    values = feats[FEATURE_COLUMNS].to_numpy()
    assert np.isfinite(values).all()


def test_winsorisation_is_per_date_cross_section():
    """An extreme outlier on one date must be clipped to that date's
    cross-sectional 1%/99% band, and other dates must be unaffected."""
    panel = _synthetic_panel(n_tickers=12, n_days=80)
    dates = sorted(panel["date"].unique())
    target = dates[-1]
    baseline = compute_feature_panel(panel)

    # 30% single-day crash for TK0 on the last date only.
    mask = (panel["ticker"] == "TK0") & (panel["date"] == target)
    panel.loc[mask, "close"] = panel.loc[mask, "close"] * 0.7
    shocked = compute_feature_panel(panel)

    b = baseline.set_index(["date", "ticker"])
    s = shocked.set_index(["date", "ticker"])
    other = [(d, t) for d, t in b.index if d != target]
    pd.testing.assert_frame_equal(b.loc[other], s.loc[other])

    # The shocked ret_1d is clipped to the date's cross-sectional band —
    # within (or at) the min of the others, never the raw −30%.
    row = s.loc[(target, "TK0")]
    others_ret = s.loc[[(target, f"TK{i}") for i in range(1, 12)], "ret_1d"]
    assert row["ret_1d"] >= others_ret.min() + math.log(0.7)  # clipped well above raw
    assert row["ret_1d"] <= others_ret.max()


def test_day_of_week_one_hot():
    feats = compute_feature_panel(_synthetic_panel())
    dow_cols = ["dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri"]
    assert (feats[dow_cols].sum(axis=1) == 1.0).all()
    mondays = feats[pd.to_datetime(feats["date"]).dt.weekday == 0]
    assert (mondays["dow_mon"] == 1.0).all()


def test_manifest_round_trip_and_drift_detection(tmp_path):
    p = tmp_path / "feature_manifest.json"
    midpoints = {c: 0.0 for c in CLASSES}
    write_manifest(p, class_midpoints=midpoints)
    m = load_manifest(p)
    assert m["columns"] == FEATURE_COLUMNS
    assert m["spec_hash"] == spec_hash()

    # Tamper with the stored hash — load must refuse.
    import json
    raw = json.loads(p.read_text())
    raw["spec_hash"] = "deadbeefdeadbeef"
    p.write_text(json.dumps(raw))
    with pytest.raises(ManifestMismatchError):
        load_manifest(p)
