"""Purged walk-forward splitter tests, including the synthetic-leak
test: a signal that only 'works' through train/validation boundary
contamination must die under purge + embargo."""
from __future__ import annotations

import random
from datetime import date, timedelta

import numpy as np

from trading_bot.ml.splits import purged_walk_forward


def _weekdays(n: int, start: date = date(2025, 1, 6)) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_fold_mechanics_worked_example():
    """Purge drops the last training date, embargo skips the next 5,
    validation is the following 21, folds expand by 21."""
    dates = _weekdays(250)
    folds = purged_walk_forward(dates, burn_in=100, val_size=21, purge=1, embargo=5, step=21)
    assert len(folds) >= 5

    for k, f in enumerate(folds):
        train_end_idx = 100 + k * 21
        assert list(f.train_dates) == dates[: train_end_idx - 1]        # purge=1
        assert list(f.val_dates) == dates[train_end_idx + 5 : train_end_idx + 5 + 21]
        # No date in both; embargo gap of exactly 5 sessions + 1 purged.
        assert not (set(f.train_dates) & set(f.val_dates))
        gap_sessions = dates.index(f.val_dates[0]) - dates.index(f.train_dates[-1]) - 1
        assert gap_sessions == 6  # 1 purged + 5 embargoed


def test_expanding_windows():
    dates = _weekdays(250)
    folds = purged_walk_forward(dates)
    for earlier, later in zip(folds, folds[1:]):
        assert set(earlier.train_dates) < set(later.train_dates)
        assert later.val_dates[0] > earlier.val_dates[0]


def test_short_final_validation_window_dropped():
    dates = _weekdays(130)
    folds = purged_walk_forward(dates, burn_in=100, val_size=21, purge=1, embargo=5,
                                step=21, min_val_size=21)
    # 130 sessions: fold 0 val = dates[105:126] fits; fold 1 would need
    # dates[126:147] — only 4 left, dropped.
    assert len(folds) == 1


def test_synthetic_leak_blocked_by_purge_embargo():
    """Labels are constant within 6-session blocks and a feature leaks
    the block value (the label copied into future-dated features within
    the block). A naive train/next-day-validate split scores way above
    chance on boundary days via that contamination; with purge=1 +
    embargo=5 (≥ block span) the validation window never shares a block
    with training, and performance collapses to ~chance."""
    rng = random.Random(3)
    n_days = 400
    block = 6
    dates = _weekdays(n_days)
    # Per block: a random feature value and an INDEPENDENT random label
    # bit shared by every row in the block. The feature has no global
    # relationship to the label — it can only 'predict' via exact-match
    # memorization of a block seen in training. That is precisely the
    # boundary-contamination channel purge+embargo must close.
    X: dict[date, float] = {}
    y: dict[date, int] = {}
    for i in range(0, n_days, block):
        value = rng.random()
        label = rng.randint(0, 1)
        for j in range(i, min(i + block, n_days)):
            X[dates[j]] = value
            y[dates[j]] = label

    def nearest_neighbour_accuracy(train_dates, val_dates) -> float:
        """1-NN on the single feature — the strongest possible
        exploiter of the leak, no model-fitting noise."""
        train_x = np.array([X[d] for d in train_dates])
        train_y = np.array([y[d] for d in train_dates])
        hits = 0
        for d in val_dates:
            pred = train_y[np.abs(train_x - X[d]).argmin()]
            hits += int(pred == y[d])
        return hits / len(val_dates)

    # Naive splitter: validate on the sessions immediately after training.
    naive_acc = []
    purged_acc = []
    folds = purged_walk_forward(dates, burn_in=100, val_size=21, purge=1, embargo=5, step=21)
    for f in folds:
        train_end_idx = dates.index(f.train_dates[-1]) + 2  # undo purge
        naive_val = dates[train_end_idx : train_end_idx + 6]  # the leaked boundary days
        naive_acc.append(nearest_neighbour_accuracy(dates[:train_end_idx], naive_val))
        purged_acc.append(nearest_neighbour_accuracy(f.train_dates, f.val_dates))

    # Boundary contamination: near-perfect. Purged+embargoed: ~chance.
    assert np.mean(naive_acc) > 0.8
    assert abs(np.mean(purged_acc) - 0.5) < 0.15


def test_no_validation_date_ever_in_training_across_folds():
    dates = _weekdays(300)
    folds = purged_walk_forward(dates)
    for f in folds:
        # Every training date strictly precedes every validation date.
        assert max(f.train_dates) < min(f.val_dates)
