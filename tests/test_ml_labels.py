"""Label-alignment tests — the grader defines the label, so these pin
equality to meta.reflection._classify_outcome itself, not to a copy of
its constants."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from trading_bot.meta.reflection import _classify_outcome
from trading_bot.ml.labels import MAX_LABEL_GAP_DAYS, compute_labels


def _panel(rows):
    return pd.DataFrame(
        rows, columns=["ticker", "date", "open", "high", "low", "close", "volume"],
    )


def test_label_is_next_session_open_to_close():
    panel = _panel([
        # Feature date Mon; label = Tue's open→close = 100 → 103 = +3.00%
        ("AAA", date(2026, 1, 5), 95.0, 99.0, 94.0, 98.0, 1000),
        ("AAA", date(2026, 1, 6), 100.0, 104.0, 99.0, 103.0, 1000),
    ])
    labels = compute_labels(panel)
    assert len(labels) == 1
    row = labels.iloc[0]
    assert row["date"] == date(2026, 1, 5)
    assert row["label_date"] == date(2026, 1, 6)
    assert row["actual_return_pct"] == pytest.approx(3.00)
    assert row["actual_class"] == "mild_up"


def test_overnight_gap_excluded_from_target():
    """A +10% overnight gap followed by a flat intraday session must be
    labelled flat — the gap sits between feature time and label start."""
    panel = _panel([
        ("AAA", date(2026, 1, 5), 100.0, 101.0, 99.0, 100.0, 1000),
        # Opens at 110 (gap +10%), closes at 110 → open→close = 0%.
        ("AAA", date(2026, 1, 6), 110.0, 111.0, 109.0, 110.0, 1000),
    ])
    labels = compute_labels(panel)
    assert labels.iloc[0]["actual_return_pct"] == pytest.approx(0.0)
    assert labels.iloc[0]["actual_class"] == "flat"


@pytest.mark.parametrize("pct", [1.0, -1.0, 4.0, -4.0, 0.999, -0.999, 3.999, -3.999, 0.0])
def test_bucketing_equals_grader_at_boundaries(pct):
    """Our bucketing must equal reflection._classify_outcome at the
    boundary values — pinned by constructing a bar with exactly that
    open→close return and comparing against the grader's own function."""
    open_ = 100.0
    close = open_ * (1 + pct / 100.0)
    panel = _panel([
        ("AAA", date(2026, 1, 5), 100.0, 101.0, 99.0, 100.0, 1000),
        ("AAA", date(2026, 1, 6), open_, max(open_, close) + 1, min(open_, close) - 1, close, 1000),
    ])
    labels = compute_labels(panel)
    assert labels.iloc[0]["actual_class"] == _classify_outcome(pct)


def test_classifies_unrounded_value_like_grader():
    """grade_predictions stores round(x, 2) but classifies the raw
    value. 0.996% rounds to 1.00 (display) yet must classify as flat."""
    open_ = 100.0
    close = 100.996
    panel = _panel([
        ("AAA", date(2026, 1, 5), 100.0, 101.0, 99.0, 100.0, 1000),
        ("AAA", date(2026, 1, 6), open_, 102.0, 99.0, close, 1000),
    ])
    labels = compute_labels(panel)
    row = labels.iloc[0]
    assert row["actual_return_pct"] == pytest.approx(1.00)      # stored rounded
    assert row["actual_class"] == _classify_outcome(0.996)      # == "flat"
    assert row["actual_class"] == "flat"


def test_long_gap_rows_excluded():
    """A 'next bar' more than MAX_LABEL_GAP_DAYS away (halt/delisting)
    is not the session the live pipeline would have traded — drop it."""
    panel = _panel([
        ("AAA", date(2026, 1, 5), 100.0, 101.0, 99.0, 100.0, 1000),
        ("AAA", date(2026, 1, 5 + MAX_LABEL_GAP_DAYS + 3), 100.0, 101.0, 99.0, 100.5, 1000),
    ])
    labels = compute_labels(panel)
    assert labels.empty


def test_last_bar_has_no_label():
    panel = _panel([
        ("AAA", date(2026, 1, 5), 100.0, 101.0, 99.0, 100.0, 1000),
        ("AAA", date(2026, 1, 6), 100.0, 101.0, 99.0, 102.0, 1000),
    ])
    labels = compute_labels(panel)
    # Only the first bar gets a label; the last bar's future is unknown.
    assert list(labels["date"]) == [date(2026, 1, 5)]


def test_multi_horizon_label_opens_next_session_closes_h_out():
    """h=3: opens at bar t+1's open, closes at bar t+3's close."""
    rows = []
    opens = [100.0, 102.0, 104.0, 106.0, 108.0]
    closes = [101.0, 103.0, 105.0, 107.0, 110.0]
    for i in range(5):
        d = date(2026, 1, 5 + i)  # Mon..Fri
        rows.append(("AAA", d, opens[i], closes[i] + 1, opens[i] - 1, closes[i], 1000))
    labels = compute_labels(_panel(rows), horizon=3)
    first = labels.iloc[0]
    # Feature date Mon: open Tue @102 → close Thu @107 = +4.902%
    assert first["date"] == date(2026, 1, 5)
    assert first["label_date"] == date(2026, 1, 8)
    assert first["actual_return_pct"] == pytest.approx((107.0 / 102.0 - 1) * 100, abs=0.01)
    assert first["actual_class"] == "strong_up"
    # Only Mon and Tue have 3 future bars
    assert len(labels) == 2


def test_horizon_one_unchanged_by_generalisation():
    rows = [
        ("AAA", date(2026, 1, 5), 95.0, 99.0, 94.0, 98.0, 1000),
        ("AAA", date(2026, 1, 6), 100.0, 104.0, 99.0, 103.0, 1000),
    ]
    default = compute_labels(_panel(rows))
    explicit = compute_labels(_panel(rows), horizon=1)
    pd.testing.assert_frame_equal(default, explicit)
