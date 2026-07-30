# Model card — ml-challenger v1 (LightGBM, 5-class)

*Generated 2026-07-30T14:42:41+00:00 by `python -m trading_bot.ml.train` · seed 42 · data snapshot `048c793f26a86909`*

Gradient-boosted trees predicting the same 5-class next-session
open→close vocabulary the LLM strategies are graded on, trained on
point-in-time OHLCV features with purged walk-forward validation.
Deployed at shadow tier so the existing daily grading and weekly
evolution machinery judges ML vs LLM head-to-head, forward, out of
sample.

## Data

- Snapshot: `state/ml/train.db` — 564,063 bars, 1,491 tickers, 2025-01-27 → 2026-07-30
- Training frame: **468,639 rows** × 1,491 tickers × 315 feature dates (2025-04-25 → 2026-07-28)
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
| strong_down | 13,498 | 2.9% |
| mild_down | 105,080 | 22.4% |
| flat | 223,735 | 47.7% |
| mild_up | 111,426 | 23.8% |
| strong_up | 14,900 | 3.2% |

Class imbalance is why training uses balanced class weights and this card
never quotes raw accuracy — predict-always-flat scores deceptively well.

Class midpoints (within-class mean returns, fitted on the first training
block only and frozen): strong_down=-5.88, mild_down=-1.873, flat=0.013, mild_up=1.901, strong_up=5.843
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
(winner: lr=0.1, depth=6, 13 rounds). The label is intraday so labels never
overlap across days; purge+embargo are kept anyway — 21/63-day rolling
features decay slowly, so adjacent-day rows are heavily correlated.

| fold | train through | validation | n | log-loss | pooled IC | mean daily IC |
|---|---|---|---|---|---|---|
| 0 | 2025-09-16 | 2025-09-25 → 2025-10-23 | 31,227 | 1.3606 | 0.0125 | -0.0121 |
| 1 | 2025-10-15 | 2025-10-24 → 2025-11-21 | 31,226 | 1.6016 | -0.0632 | -0.0215 |
| 2 | 2025-11-13 | 2025-11-24 → 2025-12-23 | 31,227 | 1.35 | 0.0125 | -0.0038 |
| 3 | 2025-12-15 | 2025-12-24 → 2026-01-26 | 31,232 | 1.3519 | 0.0186 | 0.0089 |
| 4 | 2026-01-15 | 2026-01-27 → 2026-02-25 | 31,269 | 1.6079 | 0.0464 | 0.0163 |
| 5 | 2026-02-17 | 2026-02-26 → 2026-03-26 | 31,277 | 1.6099 | 0.1052 | 0.0304 |
| 6 | 2026-03-18 | 2026-03-27 → 2026-04-27 | 31,290 | 1.4364 | 0.0245 | 0.0125 |
| 7 | 2026-04-17 | 2026-04-28 → 2026-05-27 | 31,290 | 1.5869 | -0.0251 | -0.0262 |
| 8 | 2026-05-18 | 2026-05-28 → 2026-06-26 | 31,291 | 1.5655 | 0.0258 | -0.0157 |
| 9 | 2026-06-17 | 2026-06-29 → 2026-07-28 | 31,311 | 1.5577 | 0.0219 | 0.0021 |

## Results vs baselines (pooled out-of-sample)

| model | n | log-loss | pooled IC | mean daily IC (t-stat) | decile spread | non-flat hit rate |
|---|---|---|---|---|---|---|
| **LightGBM challenger** | 312,640 | 1.5029 | 0.007 | -0.0009 (-0.14) | -0.006 | 0.505 |
| Logistic (same features) | 312,640 | 1.6136 | -0.0166 | 0.0045 (0.49) | 0.047 | 0.496 |
| Yesterday's-sign momentum | 312,640 | — | 0.0038 | 0.0081 (0.89) | 0.057 | — |
| Uniform prior | 312,640 | 1.6094 | — | — | — | — |
| control-rule-based (live record) | 111 | — | — | — | — | 0.081 |

control-rule-based is the deactivated (2026-06-07) rule baseline. It predates
prediction logging, so its frozen record is *trade-level* (ledger): 111 shadow trades, hit rate 0.081, avg P&L 0.319%/trade — no IC is computable,
so compare it on hit rate and the toy portfolio, not on ranking metrics.

## Significance — the evolution gate's own vocabulary

- Pooled OOS rank IC: **0.007** over 312,640 graded rows
- Permutation noise floor (1000 shuffles, 95th pct, as `scripts/ic_noise_floor.py`): **0.003**
- Fisher-z 95% lower bound (`PROMOTION_IC_CI_Z = 1.96`): **0.0035**
- The pooled IC clears the noise floor.

| horizon | pooled IC | n |
|---|---|---|
| open(t+1)→close(t+1) | 0.007 | 312,639 |
| open(t+1)→close(t+2) | 0.0007 | 311,147 |
| open(t+1)→close(t+3) | 0.0023 | 309,656 |

## Calibration

Multiclass Brier **0.7671** · argmax-confidence ECE **0.0329**.
LLM conviction values are famously uncalibrated; a demonstrably calibrated
challenger is a headline result even where IC ties.

<details><summary>Reliability — strong_down</summary>

| predicted prob | n | mean predicted | empirical freq |
|---|---|---|---|
| 0.0–0.1 | 98,630 | 0.032 | 0.019 |
| 0.1–0.2 | 112,179 | 0.166 | 0.021 |
| 0.2–0.3 | 91,899 | 0.227 | 0.059 |
| 0.3–0.4 | 7,626 | 0.333 | 0.116 |
| 0.4–0.5 | 1,319 | 0.443 | 0.089 |
| 0.5–0.6 | 553 | 0.545 | 0.083 |
| 0.6–0.7 | 283 | 0.642 | 0.124 |
| 0.7–0.8 | 111 | 0.741 | 0.063 |

</details>
<details><summary>Reliability — mild_down</summary>

| predicted prob | n | mean predicted | empirical freq |
|---|---|---|---|
| 0.0–0.1 | 5,913 | 0.073 | 0.191 |
| 0.1–0.2 | 113,722 | 0.177 | 0.239 |
| 0.2–0.3 | 142,757 | 0.228 | 0.232 |
| 0.3–0.4 | 32,477 | 0.344 | 0.216 |
| 0.4–0.5 | 12,587 | 0.441 | 0.218 |
| 0.5–0.6 | 3,970 | 0.54 | 0.212 |
| 0.6–0.7 | 973 | 0.639 | 0.224 |
| 0.7–0.8 | 214 | 0.736 | 0.252 |

</details>
<details><summary>Reliability — flat</summary>

| predicted prob | n | mean predicted | empirical freq |
|---|---|---|---|
| 0.0–0.1 | 13,609 | 0.076 | 0.275 |
| 0.1–0.2 | 127,931 | 0.169 | 0.366 |
| 0.2–0.3 | 95,942 | 0.236 | 0.506 |
| 0.3–0.4 | 38,852 | 0.344 | 0.55 |
| 0.4–0.5 | 21,407 | 0.445 | 0.582 |
| 0.5–0.6 | 10,195 | 0.543 | 0.603 |
| 0.6–0.7 | 3,491 | 0.639 | 0.635 |
| 0.7–0.8 | 989 | 0.74 | 0.672 |
| 0.8–0.9 | 204 | 0.838 | 0.696 |

</details>
<details><summary>Reliability — mild_up</summary>

| predicted prob | n | mean predicted | empirical freq |
|---|---|---|---|
| 0.0–0.1 | 6,307 | 0.076 | 0.231 |
| 0.1–0.2 | 123,863 | 0.176 | 0.254 |
| 0.2–0.3 | 143,774 | 0.226 | 0.233 |
| 0.3–0.4 | 24,474 | 0.343 | 0.24 |
| 0.4–0.5 | 9,542 | 0.442 | 0.251 |
| 0.5–0.6 | 3,329 | 0.542 | 0.263 |
| 0.6–0.7 | 1,049 | 0.641 | 0.316 |
| 0.7–0.8 | 254 | 0.738 | 0.311 |

</details>
<details><summary>Reliability — strong_up</summary>

| predicted prob | n | mean predicted | empirical freq |
|---|---|---|---|
| 0.0–0.1 | 100,925 | 0.031 | 0.019 |
| 0.1–0.2 | 110,266 | 0.164 | 0.021 |
| 0.2–0.3 | 85,520 | 0.228 | 0.063 |
| 0.3–0.4 | 11,964 | 0.338 | 0.105 |
| 0.4–0.5 | 2,379 | 0.438 | 0.117 |
| 0.5–0.6 | 848 | 0.545 | 0.113 |
| 0.6–0.7 | 439 | 0.645 | 0.139 |
| 0.7–0.8 | 207 | 0.744 | 0.126 |
| 0.8–0.9 | 71 | 0.836 | 0.099 |

</details>

## Per-class precision / recall (challenger)

| class | support | predicted | precision | recall |
|---|---|---|---|---|
| strong_down | 10,764 | 49,660 | 0.061 | 0.283 |
| mild_down | 72,419 | 62,363 | 0.242 | 0.208 |
| flat | 142,054 | 94,810 | 0.572 | 0.381 |
| mild_up | 76,114 | 50,675 | 0.264 | 0.176 |
| strong_up | 11,289 | 55,132 | 0.074 | 0.361 |

## Toy top-5 long portfolio (net of costs)

Equal-weight long the top-5 scores each validation day, net of 0.303% round-trip (~0.30% round-trip cost on £2,000 (FX 0.30% rt)):
mean daily net -0.4178% · hit rate 0.371 · cumulative -87.73% over 210 days · worst day -7.35%.
A sanity harness, not a backtest — no slippage, no capacity, fills at the
label's own open. UK-EU waits for v1.1 because 0.5% stamp duty moves this
from marginal to negative.

## Top-20 gain importances

| feature | gain |
|---|---|
| `parkinson_10d` | 339,191 |
| `vol_21d` | 47,018 |
| `overnight_gap` | 11,944 |
| `ret_1d` | 11,369 |
| `vol_10d` | 10,115 |
| `ret_1d_rank` | 8,737 |
| `dow_mon` | 6,039 |
| `volume_ratio` | 5,026 |
| `ret_21d` | 4,684 |
| `vol_of_vol` | 4,554 |
| `dollar_volume_pctile` | 4,395 |
| `dow_tue` | 3,882 |
| `close_vs_sma10` | 3,687 |
| `ret_5d` | 3,530 |
| `dow_wed` | 3,373 |
| `dow_fri` | 3,156 |
| `close_vs_sma63` | 2,843 |
| `ret_21d_z` | 2,379 |
| `close_vs_sma21` | 1,882 |
| `ret_10d` | 1,517 |

## Limitations

- **Pooled IC (0.007) exceeds the mean per-date IC (-0.0009).** Part of the pooled ranking edge comes from
  cross-date level effects rather than within-day stock selection. The
  pooled number is quoted because it is what `metrics.py` computes at
  runtime; the per-date IC and the toy portfolio are the sober view.
- **Survivorship:** today's index membership applied historically omits
  delisted names and flatters the backtest slightly. The forward shadow run
  is immune — which is the argument for shadow deployment.
- **~1 year of depth ≈ one regime.** The snapshot spans 2025-04-25 → 2026-07-28; nothing here says how the model
  behaves in a regime it hasn't seen. Stooq deep-history backfill is the v1.1 fix.
- **Single region.** US only; UK-EU needs a separate model and a
  stamp-duty-aware cost gate.
- **Grid selected on the same OOS folds it reports.** 6 combos — selection
  pressure is minimal, but the pooled numbers carry that footnote.
- **No text/news features.** By design: if an LLM strategy beats this, the
  interesting ablation is exactly the features the GBM can't see.

## Reproduce

```bash
python -m trading_bot.ml.data backfill        # rebuild state/ml/train.db
python -m trading_bot.ml.train --seed 42   # this card, model.txt, manifest
```

Data snapshot sha256[:16] `048c793f26a86909` · feature spec `98e07f46d8aeaeef` · LightGBM params: lr=0.1, depth=6, leaves=31, min_leaf=100, feature/bagging 0.8, balanced weights.
