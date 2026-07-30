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


def test_interleaved_venue_holidays_do_not_drop_dates():
    """Spec-v2 regression: a two-venue universe with interleaved
    holidays (venue A closed while B trades, and vice versa) must not
    lose whole dates — the grid-based rolling windows used to NaN out
    every name's vol features after any venue holiday, cascading into
    complete cross-section drops (the uk-eu May-2025/2026 holes)."""
    panel = _synthetic_panel(n_tickers=8, n_days=120, seed=13)
    dates = sorted(panel["date"].unique())
    venue_a = {f"TK{i}" for i in range(4)}          # e.g. LSE names
    holiday_a = {dates[70]}                          # A closed, B trades
    holiday_b = {dates[75], dates[76]}               # B closed, A trades
    mask_a = panel["ticker"].isin(venue_a) & panel["date"].isin(holiday_a)
    mask_b = ~panel["ticker"].isin(venue_a) & panel["date"].isin(holiday_b)
    panel = panel[~(mask_a | mask_b)]

    feats = compute_feature_panel(panel)
    feat_dates = set(feats["date"])
    # Every date from the burn-in point onward survives.
    for d in dates[MIN_HISTORY_SESSIONS + 5 :]:
        assert d in feat_dates, f"date {d} was dropped by the holiday cascade"
    # And every feature is populated (no NaN row survived).
    assert feats[FEATURE_COLUMNS].notna().all().all()
    # On venue A's holiday only venue B names appear — no fabricated bars.
    on_holiday = set(feats.loc[feats["date"].isin(holiday_a), "ticker"])
    assert on_holiday and on_holiday.isdisjoint(venue_a)


def test_rolling_windows_span_a_tickers_own_sessions():
    """After a name's own holiday, its 1-day return is measured from its
    last *traded* session — not NaN'd against a grid date."""
    # Wide cross-section, and the gap ticker gets a deterministic
    # low-drift price path: a skipped session makes a name's return a
    # two-session move, which for a normal-vol name is the day's extreme
    # and gets winsorised — a tiny-drift name stays mid-pack, so the
    # window mechanics are observable through the clip.
    panel = _synthetic_panel(n_tickers=30, n_days=100, seed=15)
    dates = sorted(panel["date"].unique())
    flat_price = {d: 100.0 * (1.0001 ** i) for i, d in enumerate(dates)}
    tk0 = panel["ticker"] == "TK0"
    panel.loc[tk0, "close"] = panel.loc[tk0, "date"].map(flat_price)
    panel.loc[tk0, "open"] = panel.loc[tk0, "close"] * 0.9999
    panel.loc[tk0, "high"] = panel.loc[tk0, "close"] * 1.0002
    panel.loc[tk0, "low"] = panel.loc[tk0, "close"] * 0.9998

    gap_day = dates[80]
    with_gap = panel[~(tk0 & (panel["date"] == gap_day))]
    feats = compute_feature_panel(with_gap).set_index(["date", "ticker"])

    after = dates[81]
    row = feats.loc[(after, "TK0")]
    expected = math.log(flat_price[after] / flat_price[dates[79]])   # skips the gap day
    assert row["ret_1d"] == pytest.approx(expected, abs=1e-9)
    # The old grid version left vol_10d NaN (then median-imputed) for 10
    # sessions after the gap; own-session windows keep it defined and
    # tiny for the flat name.
    assert row["vol_10d"] < 0.05
