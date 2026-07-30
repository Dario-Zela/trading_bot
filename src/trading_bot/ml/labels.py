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

# A feature row's label must come from the ticker's *next* bar, and that
# bar must be within this many calendar days — rides over weekends and
# holidays but skips long halts/delistings, where "next bar" isn't the
# session the live pipeline would have traded.
MAX_LABEL_GAP_DAYS = 5


def classify_outcome(actual_pct: float) -> str:
    """The grader's bucketing, re-exported for the ml package."""
    return _classify_outcome(actual_pct)


def compute_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """Per (ticker, feature date): the next session's open→close label.

    `panel` is the long bar frame [ticker, date, open, high, low, close,
    volume]. Returns [ticker, date, label_date, actual_return_pct,
    actual_class] where `date` is the feature date (bar t−1) and
    `label_date` is the bar the return realises on (bar t).
    """
    if panel.empty:
        return pd.DataFrame(
            columns=["ticker", "date", "label_date", "actual_return_pct", "actual_class"]
        )

    df = panel.sort_values(["ticker", "date"]).copy()
    grouped = df.groupby("ticker", sort=False)
    df["label_date"] = grouped["date"].shift(-1)
    df["label_open"] = grouped["open"].shift(-1)
    df["label_close"] = grouped["close"].shift(-1)

    df = df.dropna(subset=["label_date"])
    gap = (pd.to_datetime(df["label_date"]) - pd.to_datetime(df["date"])).dt.days
    df = df[gap <= MAX_LABEL_GAP_DAYS]
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
