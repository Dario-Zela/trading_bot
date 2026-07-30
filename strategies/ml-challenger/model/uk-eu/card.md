# Model card — ml-challenger · uk-eu (LightGBM, 5-class)

*Generated 2026-07-30T16:36:18+00:00 by `python -m trading_bot.ml.train --region uk-eu` · seed 42 · data snapshot `b9d68a5efc4126fa`*

Gradient-boosted trees predicting the same 5-class next-session
open→close vocabulary the LLM strategies are graded on, trained on
point-in-time OHLCV features with purged walk-forward validation.
Deployed at shadow tier so the existing daily grading and weekly
evolution machinery judges ML vs LLM head-to-head, forward, out of
sample.

## Data

- Snapshot: `state/ml/train.db` — 1,541,954 bars, 466 tickers, 2023-06-06 → 2026-07-30
- Training frame: **258,641 rows** × 466 tickers × 561 feature dates (2023-08-31 → 2026-05-01)
- Universe: S&P 1500 ∩ t212_isa_us (~1.5k liquid US names the bot can trade)
- Prices are unadjusted (auto_adjust=False), matching what tools.history writes
  into the runtime store the model is served from

## Label

Next-session **open→close simple return** `(C_t/O_t − 1) × 100`, bucketed by
`meta.reflection._classify_outcome` itself (±1% / ±4% fixed thresholds — the
grader defines the label). Features end at the t−1 close; the overnight gap
sits between feature time and label start and is excluded from the target.
Quantile thresholds would be statistically prettier but would corrupt every
downstream hit-rate comparison against the graded `actual_class` vocabulary.

| class | support | share |
|---|---|---|
| strong_down | 5,391 | 2.1% |
| mild_down | 50,619 | 19.6% |
| flat | 146,859 | 56.8% |
| mild_up | 50,848 | 19.7% |
| strong_up | 4,924 | 1.9% |

Class imbalance is why training uses balanced class weights and this card
never quotes raw accuracy — predict-always-flat scores deceptively well.

Class midpoints (within-class mean returns, fitted on the first training
block only and frozen): strong_down=-5.521, mild_down=-1.899, flat=0.018, mild_up=1.871, strong_up=5.744
`predicted_return_pct = Σ_c p_c · m_c` — the continuous score `metrics.py`
computes IC on; `predicted_class` = argmax; `conviction` = max probability.

## Features

| feature | definition |
|---|---|
| `ret_1d` | ln(C_t / C_{t−1}) |
| `ret_5d` | ln(C_t / C_{t−5}) |
| `ret_10d` | ln(C_t / C_{t−10}) |
| `ret_21d` | ln(C_t / C_{t−21}) |
| `overnight_gap` | ln(O_t / C_{t−1}) — the gap into the feature bar |
| `close_vs_sma10` | C / SMA₁₀ − 1 |
| `close_vs_sma21` | C / SMA₂₁ − 1 |
| `close_vs_sma63` | C / SMA₆₃ − 1 |
| `ret_21d_z` | 21-day return z-scored cross-sectionally per date |
| `vol_10d` | 10-day std of daily log returns, annualised (×√252) |
| `vol_21d` | 21-day std of daily log returns, annualised (×√252) |
| `parkinson_10d` | √(mean₁₀(ln(H/L)²) / (4 ln 2)) |
| `vol_of_vol` | 21-day std of the rolling 10-day vol |
| `volume_ratio` | ln(mean(V,5) / mean(V,21)) |
| `dollar_volume_pctile` | percentile of C × mean(V,21) within that date's cross-section |
| `ret_1d_rank` | previous-day return percentile rank within the universe that day |
| `dow_mon` | day-of-week one-hot (Monday) |
| `dow_tue` | day-of-week one-hot (Tuesday) |
| `dow_wed` | day-of-week one-hot (Wednesday) |
| `dow_thu` | day-of-week one-hot (Thursday) |
| `dow_fri` | day-of-week one-hot (Friday) |

All rolling windows are backward-looking; winsorisation (1st/99th pct) and
median imputation are per-date cross-sectional — no global statistics exist,
so nothing can leak across time by construction. Rows with <63 sessions of
history are dropped. Sector one-hots are deliberately omitted: the sectors
cache maps each ticker's *current* sector onto all history (look-ahead).

## Validation — purged walk-forward

```
train ──────────────────┤purge 1│ embargo 5 │ validate 21 │→ roll 21, expand
```

Grid: 3 learning rates × 2 depths, selected on pooled OOS log-loss
(winner: lr=0.1, depth=6, 1012 rounds). The label is intraday so labels never
overlap across days; purge+embargo are kept anyway — 21/63-day rolling
features decay slowly, so adjacent-day rows are heavily correlated.

**Training window: rolling, most recent 250 sessions per fold** — the
depth-vs-recency ablation showed recent-only training beats expanding over
the full snapshot on identical validation folds (non-stationarity wins).
The full 3-year history still serves validation depth and the regime
sections; CPCV remains expanding (rolling windows are ill-defined over
unordered block combinations).

| fold | train through | validation | n | log-loss | pooled IC | mean daily IC |
|---|---|---|---|---|---|---|
| 0 | 2024-12-31 | 2025-01-10 → 2025-02-07 | 9,755 | 1.2054 | -0.017 | -0.0068 |
| 1 | 2025-01-30 | 2025-02-10 → 2025-03-10 | 9,764 | 1.2358 | 0.0533 | 0.021 |
| 2 | 2025-02-28 | 2025-03-11 → 2025-04-08 | 9,764 | 1.5607 | 0.0218 | 0.004 |
| 3 | 2025-03-31 | 2025-04-09 → 2025-08-04 | 9,662 | 1.6064 | 0.0132 | -0.0049 |
| 4 | 2025-05-01 | 2025-08-05 → 2025-09-02 | 9,403 | 1.1389 | 0.0485 | 0.0279 |
| 5 | 2025-08-25 | 2025-09-03 → 2025-10-01 | 9,765 | 1.1376 | 0.0767 | 0.0196 |
| 6 | 2025-09-23 | 2025-10-02 → 2025-10-30 | 9,747 | 1.1595 | -0.0123 | -0.0003 |
| 7 | 2025-10-22 | 2025-10-31 → 2025-11-28 | 9,786 | 1.2253 | 0.0013 | 0.0107 |
| 8 | 2025-11-20 | 2025-12-01 → 2025-12-31 | 9,669 | 1.1148 | 0.0246 | 0.0282 |
| 9 | 2025-12-19 | 2026-01-02 → 2026-01-30 | 9,786 | 1.1831 | 0.0109 | 0.0128 |
| 10 | 2026-01-22 | 2026-02-02 → 2026-03-02 | 9,785 | 1.34 | -0.0262 | 0.0019 |
| 11 | 2026-02-20 | 2026-03-03 → 2026-03-31 | 9,772 | 1.6112 | 0.0557 | 0.0173 |
| 12 | 2026-03-23 | 2026-04-01 → 2026-05-01 | 9,682 | 1.3403 | 0.0097 | 0.0154 |

## Results vs baselines (pooled out-of-sample)

| model | n | log-loss | pooled IC | mean daily IC (t-stat) | decile spread | non-flat hit rate |
|---|---|---|---|---|---|---|
| **LightGBM challenger** | 126,340 | 1.2972 | 0.0192 | 0.0113 (2.87) | 0.075 | 0.492 |
| Logistic (same features) | 126,340 | 1.5194 | 0.017 | 0.0104 (1.67) | 0.123 | 0.492 |
| MLP (PyTorch, v2 preview) | 126,340 | 1.4977 | 0.0289 | 0.0028 (0.48) | 0.263 | 0.498 |
| Yesterday's-sign momentum | 126,340 | — | 0.0233 | 0.0231 (4.13) | 0.094 | — |
| Uniform prior | 126,340 | 1.6094 | — | — | — | — |
| control-rule-based (live record) | 111 | — | — | — | — | 0.081 |

control-rule-based is the deactivated (2026-06-07) rule baseline. It predates
prediction logging, so its frozen record is *trade-level* (ledger): 111 shadow trades, hit rate 0.081, avg P&L 0.319%/trade — no IC is computable,
so compare it on hit rate and the toy portfolio, not on ranking metrics.

## Significance — the evolution gate's own vocabulary

- Pooled OOS rank IC: **0.0192** over 126,340 graded rows
- Permutation noise floor (1000 shuffles, 95th pct, as `scripts/ic_noise_floor.py`): **0.0048**
- Fisher-z 95% lower bound (`PROMOTION_IC_CI_Z = 1.96`): **0.0137**
- The pooled IC clears the noise floor.

| horizon | pooled IC | n |
|---|---|---|
| open(t+1)→close(t+1) | 0.0184 | 124,970 |
| open(t+1)→close(t+2) | 0.0146 | 123,561 |
| open(t+1)→close(t+3) | 0.0096 | 123,600 |

## CPCV — combinatorial purged cross-validation

6 blocks choose 2 = 15 OOS paths with the
winning params (grid re-selection inside CPCV would be circular). Where simple
walk-forward gives one pooled IC, CPCV gives a distribution:

mean **0.0138** · std 0.008 · range [-0.0032, 0.0261] · 0.933 of paths positive.

Caveat: CPCV trains on data that postdates some validation blocks — acceptable
for a stationary-ish feature spec, and the reason promotion still keys on the
walk-forward number (the only one runtime evolution can observe).

## Regime-conditional IC (VIX terciles)

Sliced by the VIX close on the *feature* date (known at prediction time).
Repo precedent: momentum-trader-vix-gated.

| regime | n | days | pooled IC | mean daily IC |
|---|---|---|---|---|
| calm (VIX ≤ 16.4) | 41,842 | 91 | 0.0241 | 0.0163 |
| mid (16.4 – 19.6) | 40,863 | 88 | 0.0026 | 0.0094 |
| stressed (VIX > 19.6) | 40,843 | 88 | 0.0267 | 0.0091 |

## Multi-horizon heads

Separate models per holding horizon, sharing h=1's winning params; purge
scales with the horizon (h-day labels overlap h days). Only h=1 is graded at
runtime — the longer heads drive `TradeIntent.hold_days` on picked names,
exercising the Phase 12A multi-day machinery. One class vocabulary at every
horizon (±1%/±4%); h-day moves are ~√h larger so class balance shifts, which
balanced weights absorb.

| horizon (sessions) | n OOS | pooled IC | mean daily IC | log-loss |
|---|---|---|---|---|
| 1 | 126,340 | 0.0192 | 0.0113 | 1.2972 |
| 2 | 126,379 | 0.0127 | -0.0003 | 1.4746 |
| 3 | 126,379 | 0.0173 | -0.0029 | 1.5357 |
| 5 | 126,379 | 0.0346 | 0.001 | 1.5632 |

## Calibration

Multiclass Brier **0.6855** · argmax-confidence ECE **0.0299**.
LLM conviction values are famously uncalibrated; a demonstrably calibrated
challenger is a headline result even where IC ties.

<details><summary>Reliability — strong_down</summary>

| predicted prob | n | mean predicted | empirical freq |
|---|---|---|---|
| 0.0–0.1 | 88,003 | 0.018 | 0.016 |
| 0.1–0.2 | 21,823 | 0.167 | 0.03 |
| 0.2–0.3 | 14,049 | 0.222 | 0.054 |
| 0.3–0.4 | 1,651 | 0.336 | 0.088 |
| 0.4–0.5 | 390 | 0.444 | 0.095 |
| 0.5–0.6 | 222 | 0.542 | 0.086 |
| 0.6–0.7 | 119 | 0.641 | 0.092 |
| 0.7–0.8 | 60 | 0.748 | 0.1 |

</details>
<details><summary>Reliability — mild_down</summary>

| predicted prob | n | mean predicted | empirical freq |
|---|---|---|---|
| 0.0–0.1 | 7,474 | 0.059 | 0.109 |
| 0.1–0.2 | 33,722 | 0.167 | 0.193 |
| 0.2–0.3 | 47,930 | 0.236 | 0.204 |
| 0.3–0.4 | 22,302 | 0.345 | 0.219 |
| 0.4–0.5 | 10,069 | 0.442 | 0.245 |
| 0.5–0.6 | 3,490 | 0.541 | 0.246 |
| 0.6–0.7 | 1,041 | 0.639 | 0.246 |
| 0.7–0.8 | 263 | 0.739 | 0.224 |

</details>
<details><summary>Reliability — flat</summary>

| predicted prob | n | mean predicted | empirical freq |
|---|---|---|---|
| 0.0–0.1 | 4,261 | 0.072 | 0.316 |
| 0.1–0.2 | 33,620 | 0.168 | 0.414 |
| 0.2–0.3 | 27,355 | 0.242 | 0.515 |
| 0.3–0.4 | 20,137 | 0.348 | 0.566 |
| 0.4–0.5 | 16,220 | 0.448 | 0.625 |
| 0.5–0.6 | 11,034 | 0.546 | 0.688 |
| 0.6–0.7 | 6,211 | 0.646 | 0.741 |
| 0.7–0.8 | 3,598 | 0.745 | 0.807 |
| 0.8–0.9 | 2,176 | 0.847 | 0.874 |
| 0.9–1.0 | 1,728 | 0.944 | 0.927 |

</details>
<details><summary>Reliability — mild_up</summary>

| predicted prob | n | mean predicted | empirical freq |
|---|---|---|---|
| 0.0–0.1 | 6,641 | 0.06 | 0.1 |
| 0.1–0.2 | 32,586 | 0.168 | 0.202 |
| 0.2–0.3 | 49,439 | 0.238 | 0.201 |
| 0.3–0.4 | 22,875 | 0.345 | 0.214 |
| 0.4–0.5 | 10,017 | 0.442 | 0.228 |
| 0.5–0.6 | 3,411 | 0.541 | 0.235 |
| 0.6–0.7 | 1,049 | 0.64 | 0.256 |
| 0.7–0.8 | 277 | 0.74 | 0.238 |

</details>
<details><summary>Reliability — strong_up</summary>

| predicted prob | n | mean predicted | empirical freq |
|---|---|---|---|
| 0.0–0.1 | 88,996 | 0.016 | 0.015 |
| 0.1–0.2 | 20,685 | 0.166 | 0.021 |
| 0.2–0.3 | 14,181 | 0.224 | 0.048 |
| 0.3–0.4 | 1,719 | 0.335 | 0.073 |
| 0.4–0.5 | 398 | 0.444 | 0.065 |
| 0.5–0.6 | 202 | 0.546 | 0.084 |
| 0.6–0.7 | 92 | 0.645 | 0.087 |

</details>

## Per-class precision / recall (challenger)

| class | support | predicted | precision | recall |
|---|---|---|---|---|
| strong_down | 3,053 | 7,301 | 0.068 | 0.162 |
| mild_down | 25,657 | 27,668 | 0.243 | 0.262 |
| flat | 69,501 | 53,923 | 0.677 | 0.525 |
| mild_up | 25,497 | 28,006 | 0.23 | 0.253 |
| strong_up | 2,632 | 9,442 | 0.053 | 0.191 |

## Toy top-5 long portfolio (net of costs)

Equal-weight long the top-5 scores each validation day, net of 0.5% round-trip (~0.50% round-trip cost on £2,000 (stamp 0.5%)):
mean daily net -0.6232% · hit rate 0.344 · cumulative -170.13% over 273 days · worst day -5.95%.
A sanity harness, not a backtest — no slippage, no capacity, fills at the
label's own open. UK-EU waits for v1.1 because 0.5% stamp duty moves this
from marginal to negative.

## Feature attribution — gain vs TreeSHAP

Gain importances (split quality) next to TreeSHAP mean-|contribution| on a
20k-row sample (consistent attributions — LightGBM's `pred_contrib` is exact
TreeSHAP for trees). The signed column is the mean push toward `strong_up`.

| feature | gain | mean abs SHAP | signed → strong_up |
|---|---|---|---|
| `parkinson_10d` | 168,856 | 0.61527 | -0.19245 |
| `overnight_gap` | 89,529 | 0.10689 | 0.011 |
| `ret_1d` | 81,418 | 0.14983 | -0.02229 |
| `dollar_volume_pctile` | 76,964 | 0.13264 | 0.00423 |
| `volume_ratio` | 76,113 | 0.10708 | -0.00771 |
| `ret_1d_rank` | 75,368 | 0.13484 | 0.03022 |
| `close_vs_sma63` | 73,786 | 0.10991 | -0.00239 |
| `vol_of_vol` | 72,283 | 0.13498 | 0.04355 |
| `ret_21d_z` | 70,972 | 0.12409 | -0.01471 |
| `ret_5d` | 69,753 | 0.09868 | 0.01806 |
| `vol_21d` | 69,604 | 0.21043 | -0.12076 |
| `ret_10d` | 68,339 | 0.11545 | -0.00463 |
| `vol_10d` | 68,118 | 0.15801 | 0.03015 |
| `ret_21d` | 65,269 | 0.11999 | 0.01915 |
| `close_vs_sma10` | 61,886 | 0.1103 | 0.01914 |
| `close_vs_sma21` | 55,196 | 0.09108 | 0.00487 |
| `dow_thu` | 13,152 | 0.03548 | 0.00129 |
| `dow_wed` | 10,051 | 0.02641 | 0.00033 |
| `dow_mon` | 8,320 | 0.02723 | 0.00006 |
| `dow_tue` | 7,918 | 0.04537 | 0.00036 |

## Limitations

- **Pooled IC (0.0192) exceeds the mean per-date IC (0.0113).** Part of the pooled ranking edge comes from
  cross-date level effects rather than within-day stock selection. The
  pooled number is quoted because it is what `metrics.py` computes at
  runtime; the per-date IC and the toy portfolio are the sober view.
- **Survivorship:** today's index membership applied historically omits
  delisted names and flatters the backtest slightly. The forward shadow run
  is immune — which is the argument for shadow deployment.
- **Depth:** the snapshot spans 2023-08-31 → 2026-05-01 (~3 years — the deep-history stretch). Broader than v1's single year, still
  far from a full cycle; the VIX-tercile table above is the regime lens.
  Deeper history also worsens survivorship slightly (more delistings missing).
- **Grid selected on the same OOS folds it reports.** 6 combos — selection
  pressure is minimal, but the pooled numbers carry that footnote.
- **No text/news features.** By design: if an LLM strategy beats this, the
  interesting ablation is exactly the features the GBM can't see.

## Reproduce

```bash
python -m trading_bot.ml.data backfill --region uk-eu   # rebuild state/ml/train.db
python -m trading_bot.ml.train --region uk-eu --seed 42  # card + model_h*.txt + manifest
```

Data snapshot sha256[:16] `b9d68a5efc4126fa` · feature spec `98e07f46d8aeaeef` · LightGBM params: lr=0.1, depth=6, leaves=31, min_leaf=100, feature/bagging 0.8, balanced weights.
