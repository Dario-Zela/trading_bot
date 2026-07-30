"""Point-in-time feature pipeline for the ML challenger.

The point-in-time contract: the entry pipeline fires at 09:05 ET, 25
minutes before the open — the last completed bar at prediction time is
t−1. `build_features(tickers, as_of)` therefore uses only bars with
date ≤ as_of, enforced structurally (the loader reads
`read_bars_bulk(tickers, as_of − 400d, as_of)`, so later bars never
enter the frame) and verified by test (tests/test_ml_features.py).

The same `compute_feature_panel` serves training (all dates at once)
and inference (yesterday only) — train/serve skew is eliminated by
construction. Every op is either backward-looking per ticker (rolling
windows, lags) or cross-sectional within a single date (z-scores,
ranks, winsorisation, imputation). No global statistics exist, so
nothing can leak across time by construction.

Feature set v1 — all from OHLCV, deliberately no external data.
Sector one-hots are deliberately dropped: the tools.sectors cache maps
each ticker's *current* sector onto all history (mild look-ahead), so
excluding it is the cleaner v1 call (documented in the model card).
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Bump when the feature definitions below change in any way that would
# make an old model's inputs incompatible. Part of the spec hash.
# v2: rolling windows compute over each ticker's OWN sessions rather
# than a union-of-venues date grid — on mixed-calendar universes
# (uk-eu) the grid approach NaN-poisoned every name's windows after
# any interleaved venue holiday, cascading into whole-date drops.
FEATURE_SPEC_VERSION = 2

# Calendar-day read window handed to the bar loader. 400 calendar days
# comfortably covers the 63-session SMA burn-in plus weekends/holidays.
LOOKBACK_CALENDAR_DAYS = 400

# Rows with fewer than this many bars of history are dropped — the
# 63-day SMA (the deepest window) is undefined before this.
MIN_HISTORY_SESSIONS = 63

# Canonical class order — index positions are the LightGBM class ids.
# Vocabulary matches meta.reflection._classify_outcome exactly.
CLASSES = ["strong_down", "mild_down", "flat", "mild_up", "strong_up"]

_DOW_COLUMNS = ["dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri"]

# Continuous features, winsorised at the 1st/99th percentile per date
# cross-section and median-imputed per date cross-section.
_CONTINUOUS_COLUMNS = [
    # Returns: ln(C/C_lag) for lags 1/5/10/21; overnight gap into the bar
    "ret_1d", "ret_5d", "ret_10d", "ret_21d", "overnight_gap",
    # Trend: C/SMA_k − 1; 21-day return z-scored cross-sectionally per date
    "close_vs_sma10", "close_vs_sma21", "close_vs_sma63", "ret_21d_z",
    # Volatility: 10/21-day std of daily log returns (annualised);
    # 10-day Parkinson; vol-of-vol = 21-day std of the rolling 10-day vol
    "vol_10d", "vol_21d", "parkinson_10d", "vol_of_vol",
    # Volume: ln(mean(V,5)/mean(V,21)); dollar-volume percentile within
    # that date's cross-section (C × mean(V,21))
    "volume_ratio", "dollar_volume_pctile",
    # Cross-sectional context: previous-day return rank within universe
    "ret_1d_rank",
]

FEATURE_COLUMNS = _CONTINUOUS_COLUMNS + _DOW_COLUMNS

# Human-readable definitions, rendered into the model card's feature table.
FEATURE_DEFINITIONS = {
    "ret_1d": "ln(C_t / C_{t−1})",
    "ret_5d": "ln(C_t / C_{t−5})",
    "ret_10d": "ln(C_t / C_{t−10})",
    "ret_21d": "ln(C_t / C_{t−21})",
    "overnight_gap": "ln(O_t / C_{t−1}) — the gap into the feature bar",
    "close_vs_sma10": "C / SMA₁₀ − 1",
    "close_vs_sma21": "C / SMA₂₁ − 1",
    "close_vs_sma63": "C / SMA₆₃ − 1",
    "ret_21d_z": "21-day return z-scored cross-sectionally per date",
    "vol_10d": "10-day std of daily log returns, annualised (×√252)",
    "vol_21d": "21-day std of daily log returns, annualised (×√252)",
    "parkinson_10d": "√(mean₁₀(ln(H/L)²) / (4 ln 2))",
    "vol_of_vol": "21-day std of the rolling 10-day vol",
    "volume_ratio": "ln(mean(V,5) / mean(V,21))",
    "dollar_volume_pctile": "percentile of C × mean(V,21) within that date's cross-section",
    "ret_1d_rank": "previous-day return percentile rank within the universe that day",
    "dow_mon": "day-of-week one-hot (Monday)",
    "dow_tue": "day-of-week one-hot (Tuesday)",
    "dow_wed": "day-of-week one-hot (Wednesday)",
    "dow_thu": "day-of-week one-hot (Thursday)",
    "dow_fri": "day-of-week one-hot (Friday)",
}


def spec_hash() -> str:
    """Hash of the feature spec — column order, parameters, version.
    Asserted against the model's manifest at load time so a silently
    drifted feature pipeline can never feed a stale model."""
    spec = {
        "version": FEATURE_SPEC_VERSION,
        "columns": FEATURE_COLUMNS,
        "min_history_sessions": MIN_HISTORY_SESSIONS,
        "winsorise_pct": [0.01, 0.99],
        "classes": CLASSES,
    }
    canonical = json.dumps(spec, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def panel_from_bars(bars_by_ticker: dict[str, list]) -> pd.DataFrame:
    """Long frame [ticker, date, open, high, low, close, volume] from the
    {ticker: [StoredBar]} shape both ohlcv_store.read_bars_bulk and the
    training snapshot reader return."""
    rows = []
    for ticker, bars in bars_by_ticker.items():
        for b in bars:
            rows.append((ticker, b.bar_date, b.open, b.high, b.low, b.close, b.volume))
    return pd.DataFrame(
        rows, columns=["ticker", "date", "open", "high", "low", "close", "volume"],
    )


def compute_feature_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Features for every (date, ticker) in the panel with enough history.

    `panel` is the long bar frame from `panel_from_bars`. Returns a long
    frame [date, ticker] + FEATURE_COLUMNS. Rows with < MIN_HISTORY_SESSIONS
    bars of history are dropped; remaining gaps are winsorised then
    median-imputed within each date's cross-section.

    Rolling windows run over each ticker's OWN sessions ("the last 10
    sessions this name actually traded") — not a union-of-venues date
    grid. On mixed-calendar universes the grid version NaN-poisoned
    every window that overlapped an interleaved venue holiday, which
    cascaded into whole trading dates being dropped (spec v2 fix).
    """
    if panel.empty:
        return pd.DataFrame(columns=["date", "ticker", *FEATURE_COLUMNS])

    df = panel.drop_duplicates(subset=["ticker", "date"], keep="last")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["close"] = df["close"].where(df["close"] > 0)
    df["open"] = df["open"].where(df["open"] > 0)
    df["logc"] = np.log(df["close"])

    g = df.groupby("ticker", sort=False)

    def roll(col: str, window: int, fn: str) -> pd.Series:
        r = g[col].rolling(window, min_periods=window)
        return getattr(r, fn)().reset_index(level=0, drop=True)

    df["ret_1d"] = g["logc"].diff(1)
    df["ret_5d"] = g["logc"].diff(5)
    df["ret_10d"] = g["logc"].diff(10)
    df["ret_21d"] = g["logc"].diff(21)
    df["overnight_gap"] = np.log(df["open"]) - g["logc"].shift(1)

    for k in (10, 21, 63):
        df[f"close_vs_sma{k}"] = df["close"] / roll("close", k, "mean") - 1.0

    df["vol_10d"] = roll("ret_1d", 10, "std") * math.sqrt(252)
    df["vol_21d"] = roll("ret_1d", 21, "std") * math.sqrt(252)
    df["hl2"] = np.log(df["high"].where(df["high"] > 0) / df["low"].where(df["low"] > 0)) ** 2
    df["parkinson_10d"] = np.sqrt(roll("hl2", 10, "mean") / (4 * math.log(2)))
    # vol_of_vol nests a rolling on a rolled column — recompute the group
    # handle so the new vol_10d column is visible to it.
    g = df.groupby("ticker", sort=False)
    df["vol_of_vol"] = roll("vol_10d", 21, "std")

    v5 = roll("volume", 5, "mean")
    v21 = roll("volume", 21, "mean")
    df["volume_ratio"] = np.log(v5.where(v5 > 0) / v21.where(v21 > 0))
    df["dollar_vol"] = df["close"] * v21

    # Cross-sectional context — within each date, over the names that
    # traded that date.
    by_date = df.groupby("date")
    df["dollar_volume_pctile"] = by_date["dollar_vol"].rank(pct=True)
    df["ret_1d_rank"] = by_date["ret_1d"].rank(pct=True)
    r21_mean = by_date["ret_21d"].transform("mean")
    r21_std = by_date["ret_21d"].transform("std")
    df["ret_21d_z"] = (df["ret_21d"] - r21_mean) / r21_std

    # Validity: enough history (bars up to and including this row, per
    # ticker) and a real close on the date.
    df["history_count"] = df.groupby("ticker", sort=False).cumcount() + 1
    valid = df["close"].notna() & (df["history_count"] >= MIN_HISTORY_SESSIONS)
    df = df[valid]

    # Winsorise at 1st/99th pct *per date cross-section* over surviving
    # rows, then impute remaining gaps with that date's cross-sectional
    # median.
    df = df.replace([np.inf, -np.inf], np.nan)
    by_date = df.groupby("date")
    lo = by_date[_CONTINUOUS_COLUMNS].transform(lambda s: s.quantile(0.01))
    hi = by_date[_CONTINUOUS_COLUMNS].transform(lambda s: s.quantile(0.99))
    clipped = df[_CONTINUOUS_COLUMNS].clip(lower=lo, upper=hi, axis=0)
    med = clipped.groupby(df["date"]).transform("median")
    df = df.assign(**{c: clipped[c].fillna(med[c]) for c in _CONTINUOUS_COLUMNS})

    long = df.dropna(subset=_CONTINUOUS_COLUMNS).copy()

    # Calendar: day-of-week one-hots (Mon/Fri effects — cheap, honest).
    dow = pd.to_datetime(long["date"]).dt.weekday
    for i, col in enumerate(_DOW_COLUMNS):
        long[col] = (dow == i).astype(float)

    return long.sort_values(["date", "ticker"])[["date", "ticker", *FEATURE_COLUMNS]].reset_index(
        drop=True
    )


def build_features(
    tickers: list[str],
    as_of: date,
    *,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Features for the most recent completed session ≤ `as_of`.

    Serves inference: the strategy calls this with as_of = the previous
    trading day. Truncation is structural — the loader only reads bars
    in [as_of − LOOKBACK_CALENDAR_DAYS, as_of], so no future bar can
    influence any feature. Returns the feature rows for the latest bar
    date in that window (with its `date` column, so the caller can
    check staleness).
    """
    start = as_of - timedelta(days=LOOKBACK_CALENDAR_DAYS)
    if db_path is not None:
        from trading_bot.ml.data import read_snapshot_bars
        bars = read_snapshot_bars(db_path, tickers, start, as_of)
    else:
        from trading_bot.tools.ohlcv_store import read_bars_bulk
        bars = read_bars_bulk(tickers, start, as_of)

    panel = panel_from_bars(bars)
    feats = compute_feature_panel(panel)
    if feats.empty:
        return feats
    last_date = feats["date"].max()
    return feats[feats["date"] == last_date].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Manifest — ordered column list + spec hash, asserted at model load
# ---------------------------------------------------------------------------

class ManifestMismatchError(RuntimeError):
    """The model on disk was trained against a different feature spec."""


def write_manifest(path: Path, *, class_midpoints: dict[str, float], extra: dict | None = None) -> None:
    manifest = {
        "columns": FEATURE_COLUMNS,
        "spec_hash": spec_hash(),
        "spec_version": FEATURE_SPEC_VERSION,
        "classes": CLASSES,
        "class_midpoints": class_midpoints,
        "label": "next-session open→close simple return %, classes per meta.reflection._classify_outcome",
    }
    if extra:
        manifest.update(extra)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def load_manifest(path: Path) -> dict:
    """Load + validate the manifest against the live feature spec.
    Raises ManifestMismatchError on any drift — a silently drifted
    pipeline must never feed a stale model."""
    manifest = json.loads(path.read_text())
    if manifest.get("spec_hash") != spec_hash():
        raise ManifestMismatchError(
            f"feature_manifest spec_hash {manifest.get('spec_hash')} != live spec {spec_hash()} "
            f"— retrain the model or check out the matching code version"
        )
    if manifest.get("columns") != FEATURE_COLUMNS:
        raise ManifestMismatchError("feature_manifest column list differs from live FEATURE_COLUMNS")
    return manifest
