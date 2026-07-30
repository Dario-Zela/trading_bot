"""Purged + embargoed walk-forward splitter.

Expanding-window folds over the sorted unique trading dates: train on
[t0, tk], purge the final label-horizon of training (1 day), embargo 5
further trading days, validate on the next ~21 trading days; roll by 21.

Because the label is intraday (open→close within one day), labels never
overlap across days — the purge requirement is milder than the textbook
overlapping-horizon case. Purge + embargo are kept anyway: rolling-window
features (21/63-day vols) decay slowly, so adjacent-day rows are heavily
correlated, and the mechanics must be defensible independent of the
label choice.

Worked example (matches the design doc): fold k trains through
2026-01-30. Purge drops 2026-01-30's labels. Embargo skips
2026-02-02 → 2026-02-06. Validation = 2026-02-09 → ~2026-03-09
(21 sessions). The next fold's training window extends to 2026-03-09.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Fold:
    train_dates: tuple[date, ...]
    val_dates: tuple[date, ...]


def purged_walk_forward(
    dates: list[date],
    *,
    burn_in: int = 100,
    val_size: int = 21,
    purge: int = 1,
    embargo: int = 5,
    step: int = 21,
    min_val_size: int = 5,
) -> list[Fold]:
    """Expanding-window folds over sorted unique `dates`.

    Fold k's training window ends at index `burn_in + k*step − 1`
    (inclusive); the last `purge` dates of that window are dropped from
    training, the next `embargo` dates are skipped entirely, and the
    following `val_size` dates form the validation window. Folds whose
    validation window would be shorter than `min_val_size` are dropped.
    """
    uniq = sorted(set(dates))
    folds: list[Fold] = []
    k = 0
    while True:
        train_end = burn_in + k * step  # exclusive index of train window end
        if train_end > len(uniq):
            break
        val_start = train_end + embargo
        val_dates = uniq[val_start : val_start + val_size]
        if len(val_dates) < min_val_size:
            break
        train_dates = uniq[: train_end - purge]
        folds.append(Fold(train_dates=tuple(train_dates), val_dates=tuple(val_dates)))
        k += 1
    return folds
