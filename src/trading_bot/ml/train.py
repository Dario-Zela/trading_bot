"""Walk-forward trainer for the ML challenger.

LightGBM multiclass (5) on the point-in-time feature panel, validated
with purged + embargoed expanding walk-forward, benchmarked against
four mandatory baselines, reported as an auto-generated model card that
speaks the evolution machinery's language (pooled Spearman IC,
permutation noise floor, Fisher-z 95% lower bound).

Hyper-search is *not* the project: a small fixed grid (3 learning rates
× 2 depths), selected on pooled validation log-loss.

Run:
  python -m trading_bot.ml.train [--db state/ml/train.db] \
      [--out strategies/ml-challenger/model] [--seed 42]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot.ml import features as F
from trading_bot.ml.data import depth_report, read_snapshot_bars, snapshot_path, snapshot_tickers
from trading_bot.ml.labels import compute_labels
from trading_bot.ml.splits import Fold, purged_walk_forward
from trading_bot.state.paths import STATE_ROOT

log = logging.getLogger(__name__)

SEED = 42
CLASSES = F.CLASSES
N_CLASSES = len(CLASSES)
_FLAT = CLASSES.index("flat")
_UP = [CLASSES.index("mild_up"), CLASSES.index("strong_up")]
_DOWN = [CLASSES.index("mild_down"), CLASSES.index("strong_down")]

# Small fixed grid — 3 learning rates × 2 depths.
GRID = [(lr, depth) for lr in (0.03, 0.05, 0.10) for depth in (4, 6)]

MAX_ROUNDS = 1500
EARLY_STOPPING_ROUNDS = 50

# Multi-horizon stretch: h=1 is the graded head (grid-searched); the
# longer heads reuse h=1's winning params and drive TradeIntent.hold_days.
HORIZONS = [1, 2, 3, 5]

# With ~3 years of snapshot the naive fold count balloons to ~28;
# cap at the most recent MAX_FOLDS windows (~1 year of OOS) so early
# history only ever trains — standard expanding-window practice.
MAX_FOLDS = 12

# CPCV (stretch): 6 blocks choose 2 test blocks = 15 OOS paths.
CPCV_GROUPS = 6
CPCV_TEST_GROUPS = 2

# Fallback class midpoints if a class has < MIN_MIDPOINT_SUPPORT rows in
# the first training block (§5 of the design doc).
DEFAULT_MIDPOINTS = {
    "strong_down": -5.5, "mild_down": -1.9, "flat": 0.0,
    "mild_up": 1.9, "strong_up": 5.5,
}
MIN_MIDPOINT_SUPPORT = 50

# Toy-portfolio round-trip cost fallback: 2 × 0.15% T212 FX legs. Used
# when the live fee model can't run (e.g. no FX cache in a cold clone).
FALLBACK_ROUND_TRIP_COST_PCT = 0.30
TOP_N_PORTFOLIO = 5


def _lgbm_params(learning_rate: float, max_depth: int, seed: int) -> dict:
    return {
        "objective": "multiclass",
        "num_class": N_CLASSES,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 4,
        "verbosity": -1,
    }


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def load_region_panel(db_path: Path, *, region: str = "us", end_date: date | None = None) -> pd.DataFrame:
    """Long bar frame for one region's trainable universe. Index tickers
    (^VIX etc.) are excluded here and read separately for regime slicing.

    Bars dated today or later are dropped — a same-day bar fetched
    intraday is partial and would poison both features and labels.
    """
    from trading_bot.ml.data import region_tickers
    end = end_date or date.today()
    snap = [t for t in snapshot_tickers(db_path) if not t.startswith("^")]
    if not snap:
        raise SystemExit(f"No tickers in snapshot {db_path} — run `python -m trading_bot.ml.data backfill`")
    wanted = sorted(set(snap) & set(region_tickers(region)))
    if not wanted:
        raise SystemExit(f"No {region} tickers in snapshot — run "
                         f"`python -m trading_bot.ml.data backfill --region {region}`")
    bars = read_snapshot_bars(db_path, wanted, date(1970, 1, 1), end)
    panel = F.panel_from_bars(bars)
    return panel[panel["date"] < date.today()]


def build_frame(panel: pd.DataFrame, *, horizon: int = 1) -> pd.DataFrame:
    """Features ⋈ labels for one horizon: one row per (ticker, feature
    date) with FEATURE_COLUMNS, actual_return_pct, actual_class, y, and
    prev_intraday_ret (the momentum baseline's score — not a feature)."""
    feats = F.compute_feature_panel(panel)
    labels = compute_labels(panel, horizon=horizon)
    df = feats.merge(labels, on=["ticker", "date"], how="inner")

    # Baseline score: the feature date's own open→close intraday return
    # ("yesterday's sign" relative to the label day).
    intraday = panel[["ticker", "date", "open", "close"]].copy()
    intraday = intraday[intraday["open"] > 0]
    intraday["prev_intraday_ret"] = (intraday["close"] / intraday["open"] - 1.0) * 100.0
    df = df.merge(intraday[["ticker", "date", "prev_intraday_ret"]], on=["ticker", "date"], how="left")

    df["y"] = df["actual_class"].map(CLASSES.index)
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def load_training_frame(db_path: Path, *, region: str = "us", end_date: date | None = None) -> pd.DataFrame:
    """Back-compat convenience: h=1 frame for one region."""
    return build_frame(load_region_panel(db_path, region=region, end_date=end_date), horizon=1)


def auto_burn_in(n_dates: int, *, step: int = 21, val_size: int = 21, embargo: int = 5) -> int:
    """Cap the fold count at MAX_FOLDS by growing the burn-in: with ~3y
    of history the most recent ~MAX_FOLDS×21 sessions validate and
    everything earlier only ever trains."""
    return max(100, n_dates - (MAX_FOLDS * step + embargo + val_size))


def fit_class_midpoints(df: pd.DataFrame, first_fold: Fold) -> dict[str, float]:
    """Within-class mean returns fitted on the FIRST training block only
    and frozen — nothing from any validation window touches them (§5)."""
    block = df[df["date"].isin(set(first_fold.train_dates))]
    out: dict[str, float] = {}
    for cls in CLASSES:
        rows = block.loc[block["actual_class"] == cls, "actual_return_pct"]
        if len(rows) >= MIN_MIDPOINT_SUPPORT:
            out[cls] = round(float(rows.mean()), 3)
        else:
            out[cls] = DEFAULT_MIDPOINTS[cls]
    return out


def score_from_probs(probs: np.ndarray, midpoints: dict[str, float]) -> np.ndarray:
    """predicted_return_pct = Σ_c p_c · m_c — the continuous score that
    metrics.py computes IC on at runtime."""
    m = np.array([midpoints[c] for c in CLASSES])
    return probs @ m


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    counts[counts == 0] = 1.0
    per_class = len(y) / (N_CLASSES * counts)
    return per_class[y]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    from scipy.stats import spearmanr
    if len(a) < 4:
        return None
    rho = spearmanr(a, b).statistic
    return None if np.isnan(rho) else float(rho)


def per_date_ic(oos: pd.DataFrame, score_col: str = "score") -> dict:
    """For each validation date d, Spearman across tickers between score
    and realised return; mean, std, t-stat = mean/std × √N_days."""
    ics = []
    for _, day in oos.groupby("date"):
        rho = spearman(day[score_col].to_numpy(), day["actual_return_pct"].to_numpy())
        if rho is not None:
            ics.append(rho)
    if not ics:
        return {"n_days": 0, "mean": None, "std": None, "t_stat": None}
    arr = np.array(ics)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    t = float(arr.mean() / std * math.sqrt(len(arr))) if std > 0 else None
    return {"n_days": len(arr), "mean": round(float(arr.mean()), 4),
            "std": round(std, 4), "t_stat": round(t, 2) if t is not None else None}


def pooled_ic(oos: pd.DataFrame, score_col: str = "score") -> float | None:
    """All graded rows in the window → one Spearman — what metrics.py
    computes at runtime, so the card's offline number is directly
    comparable to what weekly evolution will print."""
    rho = spearman(oos[score_col].to_numpy(), oos["actual_return_pct"].to_numpy())
    return round(rho, 4) if rho is not None else None


def permutation_noise_floor(
    oos: pd.DataFrame, *, score_col: str = "score",
    n_iter: int = 1000, quantile: float = 0.95, seed: int = SEED,
) -> float | None:
    """Exactly as scripts/ic_noise_floor.py does for live strategies:
    shuffle realised returns n_iter times, 95th percentile = floor."""
    scores = oos[score_col].to_numpy()
    actuals = oos["actual_return_pct"].to_numpy().copy()
    if len(scores) < 10:
        return None
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_iter):
        rng.shuffle(actuals)
        rho = spearman(scores, actuals)
        if rho is not None:
            nulls.append(rho)
    return round(float(np.quantile(nulls, quantile)), 4) if nulls else None


def fisher_z_lower_bound(ic: float | None, n: int, z: float = 1.96) -> float | None:
    """Evolution's promotion gate speaks Fisher-z: the 95% lower bound
    on the pooled IC (PROMOTION_IC_CI_Z = 1.96 in meta.evolution)."""
    if ic is None or n < 4 or abs(ic) >= 1:
        return None
    return round(math.tanh(math.atanh(ic) - z / math.sqrt(n - 3)), 4)


def multiclass_logloss(y: np.ndarray, probs: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0)
    return round(float(-np.mean(np.log(p))), 4)


def per_class_report(y: np.ndarray, pred: np.ndarray) -> list[dict]:
    out = []
    for i, cls in enumerate(CLASSES):
        tp = int(((pred == i) & (y == i)).sum())
        fp = int(((pred == i) & (y != i)).sum())
        fn = int(((pred != i) & (y == i)).sum())
        out.append({
            "class": cls,
            "support": int((y == i).sum()),
            "predicted": int((pred == i).sum()),
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        })
    return out


def directional_hit_rate(oos: pd.DataFrame) -> dict:
    """Hit rate on non-flat predictions: predicted up-class and the
    realised return was positive, or predicted down and negative."""
    nonflat = oos[oos["pred_class_idx"] != _FLAT]
    if nonflat.empty:
        return {"n": 0, "hit_rate": None}
    up = nonflat["pred_class_idx"].isin(_UP)
    hits = np.where(up, nonflat["actual_return_pct"] > 0, nonflat["actual_return_pct"] < 0)
    return {"n": len(nonflat), "hit_rate": round(float(hits.mean()), 3)}


def decile_spread(oos: pd.DataFrame, score_col: str = "score") -> float | None:
    if len(oos) < 20:
        return None
    s = oos.sort_values(score_col, ascending=False)
    decile = max(1, len(s) // 10)
    return round(float(s.head(decile)["actual_return_pct"].mean()
                       - s.tail(decile)["actual_return_pct"].mean()), 3)


def calibration_report(y: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> dict:
    """Per-class reliability curves + ECE + multiclass Brier. The LLM
    strategies' conviction values are famously uncalibrated; a
    demonstrably calibrated challenger is a headline result."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    brier = round(float(np.mean(((probs - onehot) ** 2).sum(axis=1))), 4)

    # Argmax-confidence ECE
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.sum():
            ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())

    per_class = []
    for i, cls in enumerate(CLASSES):
        rows = []
        cbins = np.clip((probs[:, i] * n_bins).astype(int), 0, n_bins - 1)
        for b in range(n_bins):
            mask = cbins == b
            if mask.sum() >= 50:
                rows.append({
                    "bin": f"{b / n_bins:.1f}–{(b + 1) / n_bins:.1f}",
                    "n": int(mask.sum()),
                    "mean_predicted": round(float(probs[mask, i].mean()), 3),
                    "empirical_freq": round(float((y[mask] == i).mean()), 3),
                })
        per_class.append({"class": cls, "bins": rows})
    return {"brier": brier, "ece": round(float(ece), 4), "per_class": per_class}


def _round_trip_cost_pct() -> tuple[float, str]:
    """Round-trip cost for a representative US shadow trade, from the
    live fee model; falls back to 2 × FX legs if it can't run."""
    try:
        from trading_bot.tools.fees import estimate_round_trip_cost_pct
        est = estimate_round_trip_cost_pct(
            tier="shadow", currency="USD", exchange="NASDAQ",
            instrument_type="stock", notional_gbp=2000.0,
        )
        return round(float(est["total_pct"]) * 100.0, 4), est.get("note", "tools/fees.py estimate")
    except Exception as e:  # FX cache absent, etc.
        return FALLBACK_ROUND_TRIP_COST_PCT, f"fallback 2×0.15% FX legs ({e})"


def toy_portfolio(oos: pd.DataFrame, cost_pct: float, top_n: int = TOP_N_PORTFOLIO) -> dict:
    """Equal-weight long the top-N scores per validation date, net of
    round-trip costs. A sanity harness, not a backtest."""
    daily = []
    for _, day in oos.groupby("date"):
        picks = day.nlargest(top_n, "score")
        if len(picks) >= top_n:
            daily.append(float(picks["actual_return_pct"].mean()) - cost_pct)
    if not daily:
        return {"n_days": 0}
    arr = np.array(daily)
    return {
        "n_days": len(arr),
        "mean_daily_net_pct": round(float(arr.mean()), 4),
        "hit_rate": round(float((arr > 0).mean()), 3),
        "cumulative_net_pct": round(float(arr.sum()), 2),
        "worst_day_pct": round(float(arr.min()), 2),
        "cost_pct_per_trade": cost_pct,
    }


# ---------------------------------------------------------------------------
# Model fits
# ---------------------------------------------------------------------------

def train_lgbm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    params: dict,
):
    """One fold: balanced sample weights, early stopping on val logloss."""
    import lightgbm as lgb
    dtrain = lgb.Dataset(X_train, label=y_train, weight=_balanced_weights(y_train),
                         feature_name=F.FEATURE_COLUMNS)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    booster = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return booster


def run_walk_forward(
    df: pd.DataFrame, folds: list[Fold], midpoints: dict[str, float], seed: int,
    *, grid: list[tuple[float, int]] | None = None, train_window_sessions: int = 0,
) -> tuple[dict, pd.DataFrame, list[dict]]:
    """Grid × folds. Returns (best_combo, oos_frame_for_best, fold_rows).
    Selection: pooled OOS log-loss (noted in the card — 6-combo grid,
    selection pressure minimal). Pass a single-combo `grid` to reuse
    already-selected params (the h>1 horizon heads do this).

    `train_window_sessions` > 0 switches from expanding-window training
    to a rolling most-recent-N-sessions window per fold — the
    depth-vs-recency ablation for non-stationary markets. Validation
    windows are identical either way, so the comparison is clean."""
    Xcols = F.FEATURE_COLUMNS
    by_combo: dict[tuple[float, int], list[pd.DataFrame]] = {}
    iters_by_combo: dict[tuple[float, int], list[int]] = {}

    for lr, depth in (grid or GRID):
        params = _lgbm_params(lr, depth, seed)
        oos_parts: list[pd.DataFrame] = []
        for k, fold in enumerate(folds):
            tr_dates = (fold.train_dates[-train_window_sessions:]
                        if train_window_sessions else fold.train_dates)
            tr = df[df["date"].isin(set(tr_dates))]
            va = df[df["date"].isin(set(fold.val_dates))]
            booster = train_lgbm(
                tr[Xcols].to_numpy(), tr["y"].to_numpy(),
                va[Xcols].to_numpy(), va["y"].to_numpy(), params,
            )
            probs = booster.predict(va[Xcols].to_numpy(), num_iteration=booster.best_iteration)
            part = va[["date", "ticker", "actual_return_pct", "y", "prev_intraday_ret"]].copy()
            part["fold"] = k
            for i, cls in enumerate(CLASSES):
                part[f"p_{cls}"] = probs[:, i]
            part["score"] = score_from_probs(probs, midpoints)
            part["pred_class_idx"] = probs.argmax(axis=1)
            oos_parts.append(part)
            iters_by_combo.setdefault((lr, depth), []).append(booster.best_iteration or MAX_ROUNDS)
            log.info("combo lr=%s depth=%s fold %d/%d: best_iter=%s val_n=%d",
                     lr, depth, k + 1, len(folds), booster.best_iteration, len(va))
        by_combo[(lr, depth)] = oos_parts

    results = []
    for combo, parts in by_combo.items():
        oos = pd.concat(parts, ignore_index=True)
        probs = oos[[f"p_{c}" for c in CLASSES]].to_numpy()
        ll = multiclass_logloss(oos["y"].to_numpy(), probs)
        results.append({"combo": combo, "logloss": ll, "pooled_ic": pooled_ic(oos)})
        log.info("combo lr=%s depth=%s pooled OOS logloss=%.4f pooled IC=%s",
                 combo[0], combo[1], ll, pooled_ic(oos))

    best = min(results, key=lambda r: r["logloss"])
    best_combo = best["combo"]
    best_oos = pd.concat(by_combo[best_combo], ignore_index=True)

    fold_rows = []
    for k, part in enumerate(by_combo[best_combo]):
        probs = part[[f"p_{c}" for c in CLASSES]].to_numpy()
        fold_rows.append({
            "fold": k,
            "train_through": max(folds[k].train_dates).isoformat(),
            "val_start": min(folds[k].val_dates).isoformat(),
            "val_end": max(folds[k].val_dates).isoformat(),
            "n_val": len(part),
            "logloss": multiclass_logloss(part["y"].to_numpy(), probs),
            "pooled_ic": pooled_ic(part),
            "mean_daily_ic": per_date_ic(part)["mean"],
            "best_iter": iters_by_combo[best_combo][k],
        })
    return (
        {"learning_rate": best_combo[0], "max_depth": best_combo[1],
         "grid": [{"combo": list(r["combo"]), "logloss": r["logloss"], "pooled_ic": r["pooled_ic"]}
                  for r in sorted(results, key=lambda r: r["logloss"])],
         "median_best_iter": int(np.median(iters_by_combo[best_combo]))},
        best_oos,
        fold_rows,
    )


def logistic_baseline(
    df: pd.DataFrame, folds: list[Fold], midpoints: dict[str, float], seed: int,
    *, train_window_sessions: int = 0,
) -> pd.DataFrame:
    """Multinomial logistic on the identical feature matrix — does the
    GBM earn its nonlinearity? Standardisation is fit on each fold's
    training window only (no global statistics). The training window
    mode (expanding/rolling) follows the challenger's, so the
    comparison stays apples-to-apples."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xcols = F.FEATURE_COLUMNS
    parts = []
    for k, fold in enumerate(folds):
        tr_dates = (fold.train_dates[-train_window_sessions:]
                    if train_window_sessions else fold.train_dates)
        tr = df[df["date"].isin(set(tr_dates))]
        va = df[df["date"].isin(set(fold.val_dates))]
        scaler = StandardScaler().fit(tr[Xcols].to_numpy())
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0, random_state=seed,
        ).fit(scaler.transform(tr[Xcols].to_numpy()), tr["y"].to_numpy())
        probs_raw = clf.predict_proba(scaler.transform(va[Xcols].to_numpy()))
        # Column-align to canonical class order (sklearn sorts classes_)
        probs = np.zeros((len(va), N_CLASSES))
        for j, cls_idx in enumerate(clf.classes_):
            probs[:, int(cls_idx)] = probs_raw[:, j]
        part = va[["date", "ticker", "actual_return_pct", "y"]].copy()
        for i, cls in enumerate(CLASSES):
            part[f"p_{cls}"] = probs[:, i]
        part["score"] = score_from_probs(probs, midpoints)
        part["pred_class_idx"] = probs.argmax(axis=1)
        parts.append(part)
        log.info("logistic baseline fold %d/%d done", k + 1, len(folds))
    return pd.concat(parts, ignore_index=True)


def mlp_baseline(
    df: pd.DataFrame, folds: list[Fold], midpoints: dict[str, float], seed: int,
    *, train_window_sessions: int = 0,
) -> dict | None:
    """Small PyTorch net on the identical features and folds — the v2
    preview that quantifies the data-volume conversation against a
    strong GBM. Returns None (card row says 'not run') without torch;
    the CI retrain never installs it.

    Runs in an ISOLATED subprocess (trading_bot.ml.mlp_worker): LightGBM
    (Homebrew libomp) and torch (bundled libomp) deadlock when they
    share one process on macOS, and KMP_DUPLICATE_LIB_OK does not
    reliably clear it. The worker never imports lightgbm."""
    import importlib.util
    import pickle
    import subprocess
    import sys
    import tempfile

    if importlib.util.find_spec("torch") is None:
        log.warning("torch not installed — MLP v2 baseline skipped (pip install -e '.[ml-v2]')")
        return None

    payload = {
        "df": df,
        # Same training-window mode as the challenger — fair comparison.
        "folds": [
            (list(f.train_dates[-train_window_sessions:] if train_window_sessions
                  else f.train_dates), list(f.val_dates))
            for f in folds
        ],
        "midpoints": midpoints,
        "seed": seed,
    }
    with tempfile.TemporaryDirectory(prefix="mlp_worker_") as tmp:
        in_path = f"{tmp}/in.pkl"
        out_path = f"{tmp}/out.pkl"
        with open(in_path, "wb") as f:
            pickle.dump(payload, f)
        result = subprocess.run(
            [sys.executable, "-m", "trading_bot.ml.mlp_worker", in_path, out_path],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            log.warning("mlp worker failed (rc=%d) — baseline skipped:\n%s",
                        result.returncode, result.stderr[-2000:])
            return None
        for line in result.stderr.splitlines():
            if line.strip():
                log.info("%s", line.strip())
        with open(out_path, "rb") as f:
            oos = pickle.load(f)
    return evaluate_oos(oos, "MLP (PyTorch, v2 preview)")


def control_rule_based_record() -> dict:
    """The deactivated control-rule-based strategy's frozen live record.

    It predates full prediction logging, so its history is *trade-level*
    (state/ledger.jsonl): hit rate and avg P&L over its shadow trades —
    no IC is computable. If prediction rows ever exist they're preferred;
    otherwise the ledger fallback is the yardstick, flagged with
    `source: ledger` so the card explains the comparison basis."""
    path = STATE_ROOT / "predictions.jsonl"
    preds, actuals = [], []
    n_hit = n_nonflat = 0
    if path.exists():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("strategy_id") != "control-rule-based" or r.get("region") != "us":
                    continue
                p, a = r.get("predicted_return_pct"), r.get("actual_return_pct")
                if p is None or a is None:
                    continue
                preds.append(float(p))
                actuals.append(float(a))
                if r.get("predicted_class") != "flat":
                    n_nonflat += 1
                    up = r.get("predicted_class") in ("mild_up", "strong_up")
                    if (up and float(a) > 0) or (not up and float(a) < 0):
                        n_hit += 1
    if len(preds) < 4:
        return _control_ledger_record()
    frame = pd.DataFrame({"score": preds, "actual_return_pct": actuals})
    return {
        "n": len(preds),
        "pooled_ic": pooled_ic(frame),
        "decile_spread": decile_spread(frame),
        "hit_rate": round(n_hit / n_nonflat, 3) if n_nonflat else None,
    }


def _control_ledger_record() -> dict:
    """Trade-level fallback: control-rule-based's shadow trades from
    state/ledger.jsonl (n, hit rate, avg pnl_pct)."""
    path = STATE_ROOT / "ledger.jsonl"
    n = wins = 0
    pnl_pcts: list[float] = []
    if path.exists():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("strategy_id") != "control-rule-based":
                    continue
                if (r.get("region") or "us") != "us" or not r.get("exit_date"):
                    continue
                if r.get("exit_reason") in ("cancelled", "cleared"):
                    continue
                n += 1
                if float(r.get("pnl_gbp") or 0) > 0:
                    wins += 1
                pnl_pcts.append(float(r.get("pnl_pct") or 0))
    return {
        "n": n, "source": "ledger", "pooled_ic": None, "decile_spread": None,
        "hit_rate": round(wins / n, 3) if n else None,
        "avg_pnl_pct": round(sum(pnl_pcts) / n, 3) if n else None,
    }


def evaluate_oos(oos: pd.DataFrame, label: str, *, with_probs: bool = True) -> dict:
    probs = oos[[f"p_{c}" for c in CLASSES]].to_numpy() if with_probs else None
    y = oos["y"].to_numpy()
    out = {
        "label": label,
        "n": len(oos),
        "pooled_ic": pooled_ic(oos),
        "daily_ic": per_date_ic(oos),
        "decile_spread": decile_spread(oos),
    }
    if with_probs:
        out["logloss"] = multiclass_logloss(y, probs)
        out["hit_rate_nonflat"] = directional_hit_rate(oos)
        out["per_class"] = per_class_report(y, oos["pred_class_idx"].to_numpy())
        out["calibration"] = calibration_report(y, probs)
    return out


def ic_decay(oos: pd.DataFrame, panel: pd.DataFrame) -> list[dict]:
    """Spearman of the day-t score against open(t+1)→close(t+h) returns,
    h ∈ {1,2,3} — does the signal die within a day (it should)?"""
    close = panel.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()
    open_ = panel.pivot_table(index="date", columns="ticker", values="open", aggfunc="last").sort_index()
    out = []
    for h in (1, 2, 3):
        ret_h = (close.shift(-h) / open_.shift(-1) - 1.0) * 100.0
        long = ret_h.stack().rename("fwd").reset_index()
        long.columns = ["date", "ticker", "fwd"]
        merged = oos.merge(long, on=["date", "ticker"], how="inner").dropna(subset=["fwd"])
        rho = spearman(merged["score"].to_numpy(), merged["fwd"].to_numpy())
        out.append({"horizon_days": h, "pooled_ic": round(rho, 4) if rho is not None else None,
                    "n": len(merged)})
    return out


# ---------------------------------------------------------------------------
# Stretch analyses — SHAP, regime IC, CPCV
# ---------------------------------------------------------------------------

def shap_summary(booster, df: pd.DataFrame, *, sample: int = 20000, seed: int = SEED) -> list[dict]:
    """TreeSHAP feature attributions (LightGBM's pred_contrib IS SHAP for
    trees) on a row sample: mean |contribution| across all classes, plus
    the mean signed push toward strong_up — consistent attributions the
    gain table can't give."""
    rows = df.sample(n=min(sample, len(df)), random_state=seed)
    X = rows[F.FEATURE_COLUMNS].to_numpy()
    contrib = booster.predict(X, pred_contrib=True)
    n_feat = len(F.FEATURE_COLUMNS)
    # contrib layout: per class, n_feat contributions + 1 bias term.
    per_class = contrib.reshape(len(X), N_CLASSES, n_feat + 1)[:, :, :n_feat]
    mean_abs = np.abs(per_class).mean(axis=(0, 1))
    strong_up_signed = per_class[:, CLASSES.index("strong_up"), :].mean(axis=0)
    order = np.argsort(-mean_abs)
    return [
        {"feature": F.FEATURE_COLUMNS[i],
         "mean_abs_shap": round(float(mean_abs[i]), 5),
         "mean_signed_strong_up": round(float(strong_up_signed[i]), 5)}
        for i in order
    ]


def vix_regime_ic(oos: pd.DataFrame, db_path: Path) -> list[dict]:
    """Regime-conditional IC: slice the OOS window by terciles of the
    VIX close on the *feature* date (known at prediction time). Answers
    'when does it work', not just 'does it'. Repo precedent:
    momentum-trader-vix-gated."""
    bars = read_snapshot_bars(db_path, ["^VIX"], date(1970, 1, 1), date.today())
    vix = {b.bar_date: b.close for b in bars.get("^VIX", []) if b.close > 0}
    if not vix:
        return []
    with_vix = oos.assign(vix=oos["date"].map(vix)).dropna(subset=["vix"])
    if len(with_vix) < 100:
        return []
    lo, hi = with_vix["vix"].quantile([1 / 3, 2 / 3])
    out = []
    for name, sel in [
        (f"calm (VIX ≤ {lo:.1f})", with_vix["vix"] <= lo),
        (f"mid ({lo:.1f} – {hi:.1f})", (with_vix["vix"] > lo) & (with_vix["vix"] <= hi)),
        (f"stressed (VIX > {hi:.1f})", with_vix["vix"] > hi),
    ]:
        part = with_vix[sel]
        daily = per_date_ic(part)
        out.append({
            "regime": name, "n": int(len(part)), "n_days": daily["n_days"],
            "pooled_ic": pooled_ic(part), "mean_daily_ic": daily["mean"],
        })
    return out


def cpcv_report(
    df: pd.DataFrame, best: dict, midpoints: dict[str, float], seed: int,
) -> dict:
    """CPCV with the winning params only (grid re-selection inside CPCV
    would be circular and 6× the cost): pooled IC per path → the
    distribution simple walk-forward can't give."""
    from trading_bot.ml.splits import cpcv
    dates = sorted(df["date"].unique())
    folds = cpcv(dates, n_groups=CPCV_GROUPS, n_test_groups=CPCV_TEST_GROUPS)
    if not folds:
        return {"n_paths": 0}
    params = _lgbm_params(best["learning_rate"], best["max_depth"], seed)
    ics = []
    for i, fold in enumerate(folds):
        tr = df[df["date"].isin(set(fold.train_dates))]
        va = df[df["date"].isin(set(fold.val_dates))]
        booster = train_lgbm(
            tr[F.FEATURE_COLUMNS].to_numpy(), tr["y"].to_numpy(),
            va[F.FEATURE_COLUMNS].to_numpy(), va["y"].to_numpy(), params,
        )
        probs = booster.predict(va[F.FEATURE_COLUMNS].to_numpy(),
                                num_iteration=booster.best_iteration)
        part = va[["date", "actual_return_pct"]].copy()
        part["score"] = score_from_probs(probs, midpoints)
        ic = pooled_ic(part)
        if ic is not None:
            ics.append(ic)
        log.info("cpcv path %d/%d: pooled IC %s", i + 1, len(folds), ic)
    if not ics:
        return {"n_paths": 0}
    arr = np.array(ics)
    return {
        "n_paths": len(arr),
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std(ddof=1)), 4) if len(arr) > 1 else None,
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "share_positive": round(float((arr > 0).mean()), 3),
    }


def _region_round_trip_cost_pct(region: str) -> tuple[float, str]:
    """Round-trip cost for a representative shadow trade in the region.
    UK-EU is stamp-duty-heavy (0.5% on LSE purchases) — the reason it
    waited for v1.1 and gets a cost gate at pick time."""
    try:
        from trading_bot.tools.fees import estimate_round_trip_cost_pct
        if region == "uk-eu":
            est = estimate_round_trip_cost_pct(
                tier="shadow", currency="GBP", exchange="LSE",
                instrument_type="stock", notional_gbp=2000.0,
            )
        else:
            est = estimate_round_trip_cost_pct(
                tier="shadow", currency="USD", exchange="NASDAQ",
                instrument_type="stock", notional_gbp=2000.0,
            )
        return round(float(est["total_pct"]) * 100.0, 4), est.get("note", "tools/fees.py estimate")
    except Exception as e:
        fallback = 0.55 if region == "uk-eu" else FALLBACK_ROUND_TRIP_COST_PCT
        return fallback, f"fallback flat estimate ({e})"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _refit_final(df: pd.DataFrame, best: dict, seed: int, *, train_window_sessions: int = 0):
    """Refit with the winning combo on all dates (expanding) or on the
    most recent N sessions (rolling); boosting rounds = median of the
    folds' early-stopped best iterations."""
    import lightgbm as lgb
    if train_window_sessions:
        recent = sorted(df["date"].unique())[-train_window_sessions:]
        df = df[df["date"].isin(set(recent))]
    params = _lgbm_params(best["learning_rate"], best["max_depth"], seed)
    dall = lgb.Dataset(
        df[F.FEATURE_COLUMNS].to_numpy(), label=df["y"].to_numpy(),
        weight=_balanced_weights(df["y"].to_numpy()), feature_name=F.FEATURE_COLUMNS,
    )
    return lgb.train(params, dall, num_boost_round=best["median_best_iter"])


def train_and_evaluate(
    db_path: Path | None = None,
    out_dir: Path | None = None,
    *,
    region: str = "us",
    seed: int = SEED,
    end_date: date | None = None,
    with_mlp: bool = False,
    train_window_sessions: int = 0,
) -> dict:
    db = db_path or _default_db()
    out = out_dir or (_repo_root() / "strategies" / "ml-challenger" / "model" / region)
    out.mkdir(parents=True, exist_ok=True)

    log.info("loading %s panel from %s", region, db)
    panel = load_region_panel(db, region=region, end_date=end_date)
    df = build_frame(panel, horizon=1)
    dates = sorted(df["date"].unique())
    log.info("h=1 frame: %d rows, %d tickers, %d feature dates (%s → %s)",
             len(df), df["ticker"].nunique(), len(dates), dates[0], dates[-1])

    burn_in = auto_burn_in(len(dates))
    folds = purged_walk_forward(dates, burn_in=burn_in)
    if len(folds) < 3:
        raise SystemExit(f"Only {len(folds)} walk-forward folds — snapshot too shallow to train honestly")
    midpoints = fit_class_midpoints(df, folds[0])
    log.info("class midpoints (frozen from first training block): %s", midpoints)

    best, oos, fold_rows = run_walk_forward(
        df, folds, midpoints, seed, train_window_sessions=train_window_sessions,
    )
    challenger = evaluate_oos(oos, "LightGBM challenger")

    noise_floor = permutation_noise_floor(oos, seed=seed)
    fisher_lb = fisher_z_lower_bound(challenger["pooled_ic"], challenger["n"])

    # Baselines (mandatory, §6): uniform prior · yesterday's-sign
    # momentum · logistic on the identical matrix · control-rule-based.
    log.info("running baselines")
    uniform = {
        "label": "Uniform prior", "n": challenger["n"],
        "logloss": round(math.log(N_CLASSES), 4),
        "pooled_ic": None, "daily_ic": {"mean": None}, "decile_spread": None,
        "note": "equal probability on every class — no ranking, log-loss anchor only",
    }
    momentum_oos = oos[["date", "ticker", "actual_return_pct", "y"]].copy()
    momentum_oos["score"] = oos["prev_intraday_ret"]
    momentum = evaluate_oos(momentum_oos.dropna(subset=["score"]), "Yesterday's-sign momentum",
                            with_probs=False)
    logistic_oos = logistic_baseline(df, folds, midpoints, seed,
                                     train_window_sessions=train_window_sessions)
    logistic = evaluate_oos(logistic_oos, "Logistic (same features)")
    control = control_rule_based_record()
    mlp = (mlp_baseline(df, folds, midpoints, seed,
                        train_window_sessions=train_window_sessions)
           if with_mlp else None)

    # Cost-aware toy portfolio — region-specific fee model
    cost_pct, cost_note = _region_round_trip_cost_pct(region)
    portfolio = toy_portfolio(oos, cost_pct)

    decay = ic_decay(oos, panel)

    # Stretch analyses on the h=1 head
    regime = vix_regime_ic(oos, db)
    log.info("running CPCV (%d choose %d)", CPCV_GROUPS, CPCV_TEST_GROUPS)
    cpcv_dist = cpcv_report(df, best, midpoints, seed)

    # Multi-horizon heads: reuse h=1's winning params; purge scales with
    # the horizon (h-day labels overlap h days across rows).
    horizons_report: dict[int, dict] = {}
    finals: dict[int, object] = {}
    midpoints_by_horizon: dict[int, dict[str, float]] = {1: midpoints}
    finals[1] = _refit_final(df, best, seed, train_window_sessions=train_window_sessions)
    for h in HORIZONS:
        if h == 1:
            probs_cols = [f"p_{c}" for c in CLASSES]
            horizons_report[1] = {
                "n": challenger["n"], "pooled_ic": challenger["pooled_ic"],
                "mean_daily_ic": challenger["daily_ic"]["mean"],
                "logloss": challenger["logloss"],
                "class_support": {c: int((df["actual_class"] == c).sum()) for c in CLASSES},
            }
            continue
        df_h = build_frame(panel, horizon=h)
        dates_h = sorted(df_h["date"].unique())
        folds_h = purged_walk_forward(dates_h, burn_in=auto_burn_in(len(dates_h)), purge=h)
        if len(folds_h) < 3:
            log.warning("h=%d: only %d folds — skipping horizon", h, len(folds_h))
            continue
        mids_h = fit_class_midpoints(df_h, folds_h[0])
        midpoints_by_horizon[h] = mids_h
        best_h, oos_h, _ = run_walk_forward(
            df_h, folds_h, mids_h, seed,
            grid=[(best["learning_rate"], best["max_depth"])],
            train_window_sessions=train_window_sessions,
        )
        probs_h = oos_h[[f"p_{c}" for c in CLASSES]].to_numpy()
        horizons_report[h] = {
            "n": int(len(oos_h)), "pooled_ic": pooled_ic(oos_h),
            "mean_daily_ic": per_date_ic(oos_h)["mean"],
            "logloss": multiclass_logloss(oos_h["y"].to_numpy(), probs_h),
            "class_support": {c: int((df_h["actual_class"] == c).sum()) for c in CLASSES},
        }
        finals[h] = _refit_final(df_h, best_h, seed, train_window_sessions=train_window_sessions)
        log.info("h=%d head: pooled IC %s over %d rows", h,
                 horizons_report[h]["pooled_ic"], horizons_report[h]["n"])

    final = finals[1]
    importances = sorted(
        zip(F.FEATURE_COLUMNS, final.feature_importance(importance_type="gain")),
        key=lambda x: -x[1],
    )[:20]
    shap = shap_summary(final, df, seed=seed)

    snapshot_hash = hashlib.sha256(db.read_bytes()).hexdigest()[:16]
    audit_row = depth_report(db, sorted(df["ticker"].unique()))
    try:
        db_display = str(db.resolve().relative_to(_repo_root()))
    except ValueError:
        db_display = str(db)

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "region": region,
        "db": db_display,
        "snapshot_hash": snapshot_hash,
        "audit": audit_row,
        "n_rows": len(df),
        "n_tickers": int(df["ticker"].nunique()),
        "n_dates": len(dates),
        "burn_in": burn_in,
        "train_window_sessions": train_window_sessions,
        "date_range": [dates[0].isoformat(), dates[-1].isoformat()],
        "class_support": {c: int((df["actual_class"] == c).sum()) for c in CLASSES},
        "midpoints": midpoints,
        "best": best,
        "folds": fold_rows,
        "challenger": challenger,
        "noise_floor": noise_floor,
        "fisher_z_lower_bound": fisher_lb,
        "ic_decay": decay,
        "regime_ic": regime,
        "cpcv": cpcv_dist,
        "horizons": horizons_report,
        "shap": shap,
        "baselines": {"uniform": uniform, "momentum": momentum,
                      "logistic": logistic, "control_rule_based": control,
                      "mlp": mlp},
        "portfolio": portfolio,
        "cost_note": cost_note,
        "importances": [[n, round(float(v), 1)] for n, v in importances],
    }

    # Artifacts: model_h*.txt (LightGBM native text), manifest, card,
    # machine-readable metrics for the retrain workflow's promotion gate.
    for h, booster in finals.items():
        booster.save_model(str(out / f"model_h{h}.txt"))
    F.write_manifest(
        out / "feature_manifest.json",
        class_midpoints=midpoints,
        extra={
            "region": region,
            "horizons": sorted(finals.keys()),
            "midpoints_by_horizon": {str(h): m for h, m in midpoints_by_horizon.items()
                                     if h in finals},
            "trained_at": report["generated_at"],
            "seed": seed,
            "snapshot_hash": snapshot_hash,
            "train_window_sessions": train_window_sessions,
            "train_date_range": report["date_range"],
            "best_params": {"learning_rate": best["learning_rate"], "max_depth": best["max_depth"],
                            "num_boost_round": best["median_best_iter"]},
        },
    )
    (out / "metrics.json").write_text(json.dumps({
        "region": region,
        "pooled_oos_ic": challenger["pooled_ic"],
        "fisher_z_lower_bound": fisher_lb,
        "noise_floor": noise_floor,
        "n_oos": challenger["n"],
        "logloss": challenger["logloss"],
        "cpcv_mean_ic": cpcv_dist.get("mean"),
        "horizon_pooled_ic": {str(h): r["pooled_ic"] for h, r in horizons_report.items()},
        "generated_at": report["generated_at"],
    }, indent=2) + "\n")
    (out / "card.md").write_text(render_card(report))
    log.info("wrote %s → %s", ", ".join(f"model_h{h}.txt" for h in sorted(finals)), out)
    return report


def _default_db() -> Path:
    """Prefer the training snapshot; fall back to the runtime store
    (the retrain workflow restores it from actions/cache)."""
    snap = snapshot_path()
    if snap.exists():
        return snap
    runtime = STATE_ROOT / "ohlcv.db"
    if runtime.exists():
        return runtime
    raise SystemExit("No bars db found — run `python -m trading_bot.ml.data backfill` first")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------

def _fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return str(v)


def render_card(r: dict) -> str:
    ch = r["challenger"]
    base = r["baselines"]
    lines: list[str] = []
    w = lines.append

    region = r.get("region", "us")
    w(f"# Model card — ml-challenger · {region} (LightGBM, 5-class)")
    w("")
    w(f"*Generated {r['generated_at']} by `python -m trading_bot.ml.train --region {region}` · "
      f"seed {r['seed']} · data snapshot `{r['snapshot_hash']}`*")
    w("")
    w("Gradient-boosted trees predicting the same 5-class next-session")
    w("open→close vocabulary the LLM strategies are graded on, trained on")
    w("point-in-time OHLCV features with purged walk-forward validation.")
    w("Deployed at shadow tier so the existing daily grading and weekly")
    w("evolution machinery judges ML vs LLM head-to-head, forward, out of")
    w("sample.")
    w("")

    w("## Data")
    w("")
    a = r["audit"]
    w(f"- Snapshot: `{r['db']}` — {a['total_rows']:,} bars, {a['n_covered']:,} tickers, "
      f"{a['date_min']} → {a['date_max']}")
    w(f"- Training frame: **{r['n_rows']:,} rows** × {r['n_tickers']:,} tickers × "
      f"{r['n_dates']} feature dates ({r['date_range'][0]} → {r['date_range'][1]})")
    w("- Universe: S&P 1500 ∩ t212_isa_us (~1.5k liquid US names the bot can trade)")
    w("- Prices are unadjusted (auto_adjust=False), matching what tools.history writes")
    w("  into the runtime store the model is served from")
    w("")

    w("## Label")
    w("")
    w("Next-session **open→close simple return** `(C_t/O_t − 1) × 100`, bucketed by")
    w("`meta.reflection._classify_outcome` itself (±1% / ±4% fixed thresholds — the")
    w("grader defines the label). Features end at the t−1 close; the overnight gap")
    w("sits between feature time and label start and is excluded from the target.")
    w("Quantile thresholds would be statistically prettier but would corrupt every")
    w("downstream hit-rate comparison against the graded `actual_class` vocabulary.")
    w("")
    w("| class | support | share |")
    w("|---|---|---|")
    total = sum(r["class_support"].values())
    for cls in CLASSES:
        n = r["class_support"][cls]
        w(f"| {cls} | {n:,} | {n / total:.1%} |")
    w("")
    w("Class imbalance is why training uses balanced class weights and this card")
    w("never quotes raw accuracy — predict-always-flat scores deceptively well.")
    w("")
    w("Class midpoints (within-class mean returns, fitted on the first training")
    w("block only and frozen): "
      + ", ".join(f"{c}={_fmt(v, 3)}" for c, v in r["midpoints"].items()))
    w("`predicted_return_pct = Σ_c p_c · m_c` — the continuous score `metrics.py`")
    w("computes IC on; `predicted_class` = argmax; `conviction` = max probability.")
    w("")

    w("## Features")
    w("")
    w("| feature | definition |")
    w("|---|---|")
    for col in F.FEATURE_COLUMNS:
        w(f"| `{col}` | {F.FEATURE_DEFINITIONS[col]} |")
    w("")
    w("All rolling windows are backward-looking; winsorisation (1st/99th pct) and")
    w("median imputation are per-date cross-sectional — no global statistics exist,")
    w("so nothing can leak across time by construction. Rows with <63 sessions of")
    w("history are dropped. Sector one-hots are deliberately omitted: the sectors")
    w("cache maps each ticker's *current* sector onto all history (look-ahead).")
    w("")

    w("## Validation — purged walk-forward")
    w("")
    w("```")
    w("train ──────────────────┤purge 1│ embargo 5 │ validate 21 │→ roll 21, expand")
    w("```")
    w("")
    w("Grid: 3 learning rates × 2 depths, selected on pooled OOS log-loss")
    w(f"(winner: lr={r['best']['learning_rate']}, depth={r['best']['max_depth']}, "
      f"{r['best']['median_best_iter']} rounds). The label is intraday so labels never")
    w("overlap across days; purge+embargo are kept anyway — 21/63-day rolling")
    w("features decay slowly, so adjacent-day rows are heavily correlated.")
    tws = r.get("train_window_sessions") or 0
    if tws:
        w("")
        w(f"**Training window: rolling, most recent {tws} sessions per fold** — the")
        w("depth-vs-recency ablation showed recent-only training beats expanding over")
        w("the full snapshot on identical validation folds (non-stationarity wins).")
        w("The full 3-year history still serves validation depth and the regime")
        w("sections; CPCV remains expanding (rolling windows are ill-defined over")
        w("unordered block combinations).")
    w("")
    w("| fold | train through | validation | n | log-loss | pooled IC | mean daily IC |")
    w("|---|---|---|---|---|---|---|")
    for f in r["folds"]:
        w(f"| {f['fold']} | {f['train_through']} | {f['val_start']} → {f['val_end']} | "
          f"{f['n_val']:,} | {_fmt(f['logloss'])} | {_fmt(f['pooled_ic'])} | {_fmt(f['mean_daily_ic'])} |")
    w("")

    w("## Results vs baselines (pooled out-of-sample)")
    w("")
    w("| model | n | log-loss | pooled IC | mean daily IC (t-stat) | decile spread | non-flat hit rate |")
    w("|---|---|---|---|---|---|---|")
    mom, lgst = base["momentum"], base["logistic"]
    rows = [
        ("**LightGBM challenger**", ch["n"], ch["logloss"], ch["pooled_ic"],
         f"{_fmt(ch['daily_ic']['mean'])} ({_fmt(ch['daily_ic']['t_stat'], 2)})",
         ch["decile_spread"], ch["hit_rate_nonflat"]["hit_rate"]),
        ("Logistic (same features)", lgst["n"], lgst["logloss"], lgst["pooled_ic"],
         f"{_fmt(lgst['daily_ic']['mean'])} ({_fmt(lgst['daily_ic']['t_stat'], 2)})",
         lgst["decile_spread"], lgst["hit_rate_nonflat"]["hit_rate"]),
        ("Yesterday's-sign momentum", mom["n"], None, mom["pooled_ic"],
         f"{_fmt(mom['daily_ic']['mean'])} ({_fmt(mom['daily_ic']['t_stat'], 2)})",
         mom["decile_spread"], None),
        ("Uniform prior", base["uniform"]["n"], base["uniform"]["logloss"], None, "—", None, None),
        ("control-rule-based (live record)", base["control_rule_based"]["n"], None,
         base["control_rule_based"]["pooled_ic"], "—",
         base["control_rule_based"].get("decile_spread"), base["control_rule_based"]["hit_rate"]),
    ]
    mlp = base.get("mlp")
    if mlp:
        rows.insert(2, (
            "MLP (PyTorch, v2 preview)", mlp["n"], mlp["logloss"], mlp["pooled_ic"],
            f"{_fmt(mlp['daily_ic']['mean'])} ({_fmt(mlp['daily_ic']['t_stat'], 2)})",
            mlp["decile_spread"], mlp["hit_rate_nonflat"]["hit_rate"],
        ))
    for label, n, ll, pic, dic, ds, hr in rows:
        w(f"| {label} | {n:,} | {_fmt(ll)} | {_fmt(pic)} | {dic} | {_fmt(ds, 3)} | {_fmt(hr, 3)} |")
    w("")
    ctl = base["control_rule_based"]
    if ctl.get("source") == "ledger":
        w(f"control-rule-based is the deactivated (2026-06-07) rule baseline. It predates")
        w(f"prediction logging, so its frozen record is *trade-level* (ledger): "
          f"{ctl['n']} shadow trades, hit rate {_fmt(ctl['hit_rate'], 3)}, "
          f"avg P&L {_fmt(ctl.get('avg_pnl_pct'), 3)}%/trade — no IC is computable,")
        w("so compare it on hit rate and the toy portfolio, not on ranking metrics.")
    else:
        w("control-rule-based is the deactivated (2026-06-07) rule baseline's own live")
        w("graded record — a frozen yardstick on a different sample (~100 movers/day,")
        w("its own forward window), so compare direction not decimals.")
    w("")

    w("## Significance — the evolution gate's own vocabulary")
    w("")
    w(f"- Pooled OOS rank IC: **{_fmt(ch['pooled_ic'])}** over {ch['n']:,} graded rows")
    w(f"- Permutation noise floor (1000 shuffles, 95th pct, as `scripts/ic_noise_floor.py`): "
      f"**{_fmt(r['noise_floor'])}**")
    w(f"- Fisher-z 95% lower bound (`PROMOTION_IC_CI_Z = 1.96`): **{_fmt(r['fisher_z_lower_bound'])}**")
    verdict = ("clears" if (ch["pooled_ic"] or 0) > (r["noise_floor"] or 0) else "does NOT clear")
    w(f"- The pooled IC {verdict} the noise floor.")
    w("")
    w("| horizon | pooled IC | n |")
    w("|---|---|---|")
    for d in r["ic_decay"]:
        w(f"| open(t+1)→close(t+{d['horizon_days']}) | {_fmt(d['pooled_ic'])} | {d['n']:,} |")
    w("")

    cp = r.get("cpcv") or {}
    if cp.get("n_paths"):
        w("## CPCV — combinatorial purged cross-validation")
        w("")
        w(f"{CPCV_GROUPS} blocks choose {CPCV_TEST_GROUPS} = {cp['n_paths']} OOS paths with the")
        w("winning params (grid re-selection inside CPCV would be circular). Where simple")
        w("walk-forward gives one pooled IC, CPCV gives a distribution:")
        w("")
        w(f"mean **{_fmt(cp['mean'])}** · std {_fmt(cp['std'])} · range "
          f"[{_fmt(cp['min'])}, {_fmt(cp['max'])}] · {_fmt(cp['share_positive'], 3)} of paths positive.")
        w("")
        w("Caveat: CPCV trains on data that postdates some validation blocks — acceptable")
        w("for a stationary-ish feature spec, and the reason promotion still keys on the")
        w("walk-forward number (the only one runtime evolution can observe).")
        w("")

    if r.get("regime_ic"):
        w("## Regime-conditional IC (VIX terciles)")
        w("")
        w("Sliced by the VIX close on the *feature* date (known at prediction time).")
        w("Repo precedent: momentum-trader-vix-gated.")
        w("")
        w("| regime | n | days | pooled IC | mean daily IC |")
        w("|---|---|---|---|---|")
        for row in r["regime_ic"]:
            w(f"| {row['regime']} | {row['n']:,} | {row['n_days']} | "
              f"{_fmt(row['pooled_ic'])} | {_fmt(row['mean_daily_ic'])} |")
        w("")

    hz = r.get("horizons") or {}
    if len(hz) > 1:
        w("## Multi-horizon heads")
        w("")
        w("Separate models per holding horizon, sharing h=1's winning params; purge")
        w("scales with the horizon (h-day labels overlap h days). Only h=1 is graded at")
        w("runtime — the longer heads drive `TradeIntent.hold_days` on picked names,")
        w("exercising the Phase 12A multi-day machinery. One class vocabulary at every")
        w("horizon (±1%/±4%); h-day moves are ~√h larger so class balance shifts, which")
        w("balanced weights absorb.")
        w("")
        w("| horizon (sessions) | n OOS | pooled IC | mean daily IC | log-loss |")
        w("|---|---|---|---|---|")
        for h in sorted(hz):
            row = hz[h]
            w(f"| {h} | {row['n']:,} | {_fmt(row['pooled_ic'])} | "
              f"{_fmt(row['mean_daily_ic'])} | {_fmt(row['logloss'])} |")
        w("")

    w("## Calibration")
    w("")
    cal = ch["calibration"]
    w(f"Multiclass Brier **{_fmt(cal['brier'])}** · argmax-confidence ECE **{_fmt(cal['ece'])}**.")
    w("LLM conviction values are famously uncalibrated; a demonstrably calibrated")
    w("challenger is a headline result even where IC ties.")
    w("")
    for pc in cal["per_class"]:
        if not pc["bins"]:
            continue
        w(f"<details><summary>Reliability — {pc['class']}</summary>")
        w("")
        w("| predicted prob | n | mean predicted | empirical freq |")
        w("|---|---|---|---|")
        for b in pc["bins"]:
            w(f"| {b['bin']} | {b['n']:,} | {_fmt(b['mean_predicted'], 3)} | {_fmt(b['empirical_freq'], 3)} |")
        w("")
        w("</details>")
    w("")

    w("## Per-class precision / recall (challenger)")
    w("")
    w("| class | support | predicted | precision | recall |")
    w("|---|---|---|---|---|")
    for row in ch["per_class"]:
        w(f"| {row['class']} | {row['support']:,} | {row['predicted']:,} | "
          f"{_fmt(row['precision'], 3)} | {_fmt(row['recall'], 3)} |")
    w("")

    w("## Toy top-5 long portfolio (net of costs)")
    w("")
    p = r["portfolio"]
    if p.get("n_days"):
        w(f"Equal-weight long the top-5 scores each validation day, net of "
          f"{_fmt(p['cost_pct_per_trade'], 3)}% round-trip ({r['cost_note']}):")
        w(f"mean daily net {_fmt(p['mean_daily_net_pct'])}% · hit rate {_fmt(p['hit_rate'], 3)} · "
          f"cumulative {_fmt(p['cumulative_net_pct'], 2)}% over {p['n_days']} days · "
          f"worst day {_fmt(p['worst_day_pct'], 2)}%.")
        w("A sanity harness, not a backtest — no slippage, no capacity, fills at the")
        w("label's own open. UK-EU waits for v1.1 because 0.5% stamp duty moves this")
        w("from marginal to negative.")
    else:
        w("(not enough validation days)")
    w("")

    w("## Feature attribution — gain vs TreeSHAP")
    w("")
    w("Gain importances (split quality) next to TreeSHAP mean-|contribution| on a")
    w("20k-row sample (consistent attributions — LightGBM's `pred_contrib` is exact")
    w("TreeSHAP for trees). The signed column is the mean push toward `strong_up`.")
    w("")
    w("| feature | gain | mean abs SHAP | signed → strong_up |")
    w("|---|---|---|---|")
    shap_by_feat = {row["feature"]: row for row in r.get("shap", [])}
    for name, gain in r["importances"]:
        s = shap_by_feat.get(name, {})
        w(f"| `{name}` | {gain:,.0f} | {_fmt(s.get('mean_abs_shap'), 5)} | "
          f"{_fmt(s.get('mean_signed_strong_up'), 5)} |")
    w("")

    w("## Limitations")
    w("")
    daily_mean = ch["daily_ic"]["mean"]
    if (ch["pooled_ic"] or 0) > (daily_mean or 0) + 0.003:
        w(f"- **Pooled IC ({_fmt(ch['pooled_ic'])}) exceeds the mean per-date IC "
          f"({_fmt(daily_mean)}).** Part of the pooled ranking edge comes from")
        w("  cross-date level effects rather than within-day stock selection. The")
        w("  pooled number is quoted because it is what `metrics.py` computes at")
        w("  runtime; the per-date IC and the toy portfolio are the sober view.")
    w("- **Survivorship:** today's index membership applied historically omits")
    w("  delisted names and flatters the backtest slightly. The forward shadow run")
    w("  is immune — which is the argument for shadow deployment.")
    w(f"- **Depth:** the snapshot spans {r['date_range'][0]} → {r['date_range'][1]} "
      f"(~3 years — the deep-history stretch). Broader than v1's single year, still")
    w("  far from a full cycle; the VIX-tercile table above is the regime lens.")
    w("  Deeper history also worsens survivorship slightly (more delistings missing).")
    w("- **Grid selected on the same OOS folds it reports.** 6 combos — selection")
    w("  pressure is minimal, but the pooled numbers carry that footnote.")
    w("- **No text/news features.** By design: if an LLM strategy beats this, the")
    w("  interesting ablation is exactly the features the GBM can't see.")
    w("")

    w("## Reproduce")
    w("")
    w("```bash")
    w(f"python -m trading_bot.ml.data backfill --region {region}   # rebuild state/ml/train.db")
    w(f"python -m trading_bot.ml.train --region {region} --seed {r['seed']}  # card + model_h*.txt + manifest")
    w("```")
    w("")
    w(f"Data snapshot sha256[:16] `{r['snapshot_hash']}` · feature spec `{F.spec_hash()}` "
      f"· LightGBM params: lr={r['best']['learning_rate']}, depth={r['best']['max_depth']}, "
      f"leaves=31, min_leaf=100, feature/bagging 0.8, balanced weights.")
    w("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=None, help="bars db (default: snapshot, then runtime store)")
    p.add_argument("--out", type=Path, default=None, help="model output dir")
    p.add_argument("--region", default="us", choices=["us", "uk-eu"])
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--with-mlp", action="store_true",
                   help="also run the PyTorch MLP v2 baseline (needs the ml-v2 extra)")
    p.add_argument("--train-window-sessions", type=int, default=0,
                   help="rolling most-recent-N-sessions training window per fold "
                        "(0 = expanding over all history)")
    args = p.parse_args(argv)
    report = train_and_evaluate(args.db, args.out, region=args.region, seed=args.seed,
                                with_mlp=args.with_mlp,
                                train_window_sessions=args.train_window_sessions)
    ch = report["challenger"]
    print(json.dumps({
        "pooled_oos_ic": ch["pooled_ic"],
        "noise_floor": report["noise_floor"],
        "fisher_z_lower_bound": report["fisher_z_lower_bound"],
        "logloss": ch["logloss"],
        "n_oos": ch["n"],
        "best": {k: report["best"][k] for k in ("learning_rate", "max_depth", "median_best_iter")},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
