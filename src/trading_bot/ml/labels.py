"""Labels for the ML challenger — the grader defines the label.

`meta.reflection.grade_predictions` scores every prediction with the
same-day open→close simple return, `(C_t/O_t − 1) × 100` rounded to
2 dp, and buckets `actual_class` with fixed ±1%/±4% thresholds in
`_classify_outcome`. We import that function rather than copying its
constants, so our labels can never drift from what the runtime grader
will print.

Alignment: features end at the t−1 close; the label for those features
is day t's open→close return. The overnight gap (C_{t−1} → O_t) sits
between feature time and label start and is *excluded from the target*
— predictions are graded on the intraday move only.
"""
from __future__ import annotations

import pandas as pd

from trading_bot.meta.reflection import _classify_outcome

# A feature row's label must open at the ticker's *next* bar, and that
# bar must be within this many calendar days — rides over weekends and
# holidays but skips long halts/delistings, where "next bar" isn't the
# session the live pipeline would have traded.
MAX_LABEL_GAP_DAYS = 5

# Multi-horizon stretch: per extra holding day, allow ~2 more calendar
# days of gap (weekend-heavy stretches) before declaring the sequence
# too holey to represent a real h-day hold.
MAX_GAP_PER_EXTRA_DAY = 2


def classify_outcome(actual_pct: float) -> str:
    """The grader's bucketing, re-exported for the ml package."""
    return _classify_outcome(actual_pct)


def compute_labels(panel: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Per (ticker, feature date): the open(t+1)→close(t+horizon) label.

    horizon=1 (the graded default) is the next session's open→close —
    exactly what `reflection.grade_predictions` scores. Higher horizons
    (the Phase-12A multi-day stretch) open at the same next-session bar
    but close `horizon` sessions out; only h=1 rows are ever graded at
    runtime, so h>1 models drive `TradeIntent.hold_days`, not the
    prediction log. All horizons share the grader's ±1%/±4% class
    thresholds — one vocabulary, documented in the card (h-day moves
    are √h-ish larger, so class balance shifts with horizon; balanced
    class weights absorb it).

    `panel` is the long bar frame [ticker, date, open, high, low, close,
    volume]. Returns [ticker, date, label_date, actual_return_pct,
    actual_class] where `date` is the feature date (bar t−1) and
    `label_date` is the bar the return finishes on (bar t+horizon−1
    relative to the label's opening bar).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be ≥ 1, got {horizon}")
    if panel.empty:
        return pd.DataFrame(
            columns=["ticker", "date", "label_date", "actual_return_pct", "actual_class"]
        )

    df = panel.sort_values(["ticker", "date"]).copy()
    grouped = df.groupby("ticker", sort=False)
    df["label_date"] = grouped["date"].shift(-horizon)
    df["label_open"] = grouped["open"].shift(-1)
    df["label_close"] = grouped["close"].shift(-horizon)

    df = df.dropna(subset=["label_date"])
    gap = (pd.to_datetime(df["label_date"]) - pd.to_datetime(df["date"])).dt.days
    max_gap = MAX_LABEL_GAP_DAYS + (horizon - 1) * (1 + MAX_GAP_PER_EXTRA_DAY)
    df = df[gap <= max_gap]
    df = df[(df["label_open"] > 0) & (df["label_close"] > 0)]

    # Match the grader exactly: reflection.grade_predictions stores the
    # return rounded to 2 dp but classifies the *unrounded* value — so
    # do we, in the same order.
    raw = (df["label_close"] / df["label_open"] - 1.0) * 100.0
    df["actual_return_pct"] = raw.round(2)
    df["actual_class"] = raw.map(_classify_outcome)

    return df[["ticker", "date", "label_date", "actual_return_pct", "actual_class"]].reset_index(
        drop=True
    )
