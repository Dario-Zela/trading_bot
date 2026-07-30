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
FEATURE_SPEC_VERSION = 1

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
    """
    if panel.empty:
        return pd.DataFrame(columns=["date", "ticker", *FEATURE_COLUMNS])

    def pivot(field: str) -> pd.DataFrame:
        p = panel.pivot_table(index="date", columns="ticker", values=field, aggfunc="last")
        return p.sort_index()

    open_, high, low = pivot("open"), pivot("high"), pivot("low")
    close, volume = pivot("close"), pivot("volume")
    close = close.where(close > 0)
    open_ = open_.where(open_ > 0)

    logc = np.log(close)
    ret_1d = logc.diff(1)

    feats: dict[str, pd.DataFrame] = {}
    feats["ret_1d"] = ret_1d
    feats["ret_5d"] = logc.diff(5)
    feats["ret_10d"] = logc.diff(10)
    feats["ret_21d"] = logc.diff(21)
    feats["overnight_gap"] = np.log(open_) - logc.shift(1)

    for k in (10, 21, 63):
        feats[f"close_vs_sma{k}"] = close / close.rolling(k, min_periods=k).mean() - 1.0

    r21 = feats["ret_21d"]
    feats["ret_21d_z"] = r21.sub(r21.mean(axis=1), axis=0).div(r21.std(axis=1), axis=0)

    feats["vol_10d"] = ret_1d.rolling(10, min_periods=10).std() * math.sqrt(252)
    feats["vol_21d"] = ret_1d.rolling(21, min_periods=21).std() * math.sqrt(252)
    hl = np.log((high.where(high > 0)) / (low.where(low > 0))) ** 2
    feats["parkinson_10d"] = np.sqrt(hl.rolling(10, min_periods=10).mean() / (4 * math.log(2)))
    feats["vol_of_vol"] = feats["vol_10d"].rolling(21, min_periods=21).std()

    v5 = volume.rolling(5, min_periods=5).mean()
    v21 = volume.rolling(21, min_periods=21).mean()
    feats["volume_ratio"] = np.log(v5.where(v5 > 0) / v21.where(v21 > 0))
    dollar_vol = close * v21
    feats["dollar_volume_pctile"] = dollar_vol.rank(axis=1, pct=True)

    feats["ret_1d_rank"] = ret_1d.rank(axis=1, pct=True)

    # Validity: enough history AND a bar on this date. History counts
    # bars up to and including the row's date, per ticker.
    history_count = close.notna().cumsum()
    valid = close.notna() & (history_count >= MIN_HISTORY_SESSIONS)

    # Winsorise at 1st/99th pct *per date cross-section*, over valid
    # cells only, then impute missing with that date's cross-sectional
    # median. Invalid rows never surface (masked out at melt time).
    for name in _CONTINUOUS_COLUMNS:
        f = feats[name].where(valid)
        f = f.replace([np.inf, -np.inf], np.nan)
        lo = f.quantile(0.01, axis=1)
        hi = f.quantile(0.99, axis=1)
        f = f.clip(lower=lo, upper=hi, axis=0)
        med = f.median(axis=1)
        f = f.T.fillna(med).T
        feats[name] = f

    long = pd.concat(
        {name: feats[name].where(valid).stack() for name in _CONTINUOUS_COLUMNS},
        axis=1,
    )
    long = long.dropna(how="any")
    long.index.names = ["date", "ticker"]
    long = long.reset_index()

    # Calendar: day-of-week one-hots (Mon/Fri effects — cheap, honest).
    dow = pd.to_datetime(long["date"]).dt.weekday
    for i, col in enumerate(_DOW_COLUMNS):
        long[col] = (dow == i).astype(float)

    return long[["date", "ticker", *FEATURE_COLUMNS]]


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
