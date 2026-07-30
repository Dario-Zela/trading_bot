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
from itertools import combinations


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


def cpcv(
    dates: list[date],
    *,
    n_groups: int = 6,
    n_test_groups: int = 2,
    purge: int = 1,
    embargo: int = 5,
) -> list[Fold]:
    """Combinatorial purged cross-validation (López de Prado) — the
    stretch-goal variance tightener over simple walk-forward.

    The sorted unique dates are cut into `n_groups` contiguous blocks;
    every combination of `n_test_groups` blocks becomes one fold's
    validation set (C(6,2) = 15 paths by default), training on the
    remaining blocks with `purge` sessions dropped from each training
    edge that touches a test block and `embargo` further sessions
    dropped from training AFTER each test block (features look
    backwards, so contamination flows forward: train rows just after a
    test block carry the test period in their rolling windows).

    Unlike walk-forward, training data may lie after validation data —
    CPCV trades that (acceptable for a stationary-ish feature spec) for
    many more OOS paths, giving a distribution of pooled ICs instead of
    a single number. The card reports both; promotion still keys on the
    walk-forward number, which is what runtime evolution can observe.
    """
    uniq = sorted(set(dates))
    n = len(uniq)
    if n < n_groups * 2:
        return []
    bounds = [round(i * n / n_groups) for i in range(n_groups + 1)]
    blocks = [range(bounds[i], bounds[i + 1]) for i in range(n_groups)]

    folds: list[Fold] = []
    for test_ids in combinations(range(n_groups), n_test_groups):
        test_idx: set[int] = set()
        for b in test_ids:
            test_idx.update(blocks[b])
        banned = set(test_idx)
        for b in test_ids:
            first, last = blocks[b][0], blocks[b][-1]
            banned.update(range(max(0, first - purge), first))            # purge before
            banned.update(range(last + 1, min(n, last + 1 + purge + embargo)))  # purge+embargo after
        train_dates = tuple(uniq[i] for i in range(n) if i not in banned)
        val_dates = tuple(uniq[i] for i in sorted(test_idx))
        if train_dates and val_dates:
            folds.append(Fold(train_dates=train_dates, val_dates=val_dates))
    return folds
