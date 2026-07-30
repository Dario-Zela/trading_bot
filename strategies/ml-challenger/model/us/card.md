# Model card — ml-challenger · us (LightGBM, 5-class)

*Generated 2026-07-30T19:13:20+00:00 by `python -m trading_bot.ml.train --region us` · seed 42 · data snapshot `b9d68a5efc4126fa`*

Gradient-boosted trees predicting the same 5-class next-session
open→close vocabulary the LLM strategies are graded on, trained on
point-in-time OHLCV features with purged walk-forward validation.
Deployed at shadow tier so the existing daily grading and weekly
evolution machinery judges ML vs LLM head-to-head, forward, out of
sample.

## Data

- Snapshot: `state/ml/train.db` — this region's slice: 1,168,366 bars, 1,491 tickers, 2023-06-06 → 2026-07-29
- Training frame: **1,074,433 rows** × 1,491 tickers × 726 feature dates (2023-09-05 → 2026-07-28)
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
| strong_down | 28,573 | 2.7% |
| mild_down | 239,175 | 22.3% |
| flat | 534,568 | 49.8% |
| mild_up | 242,923 | 22.6% |
| strong_up | 29,194 | 2.7% |

Class imbalance is why training uses balanced class weights and this card
never quotes raw accuracy — predict-always-flat scores deceptively well.

Class midpoints (within-class mean returns, fitted on the first training
block only and frozen): strong_down=-5.832, mild_down=-1.914, flat=0.009, mild_up=1.892, strong_up=6.425
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
(winner: lr=0.1, depth=6, 63 rounds). The label is intraday so labels never
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
| 0 | 2025-06-16 | 2025-06-26 → 2025-07-25 | 31,206 | 1.4017 | -0.0589 | -0.0348 |
| 1 | 2025-07-17 | 2025-07-28 → 2025-08-25 | 31,206 | 1.372 | 0.0614 | 0.0055 |
| 2 | 2025-08-15 | 2025-08-26 → 2025-09-24 | 31,208 | 1.2924 | -0.036 | -0.0303 |
| 3 | 2025-09-16 | 2025-09-25 → 2025-10-23 | 31,227 | 1.376 | 0.0507 | 0.0027 |
| 4 | 2025-10-15 | 2025-10-24 → 2025-11-21 | 31,226 | 1.5461 | -0.0019 | 0.0021 |
| 5 | 2025-11-13 | 2025-11-24 → 2025-12-23 | 31,227 | 1.4497 | -0.0178 | 0.0273 |
| 6 | 2025-12-15 | 2025-12-24 → 2026-01-26 | 31,232 | 1.3461 | -0.0107 | -0.0022 |
| 7 | 2026-01-15 | 2026-01-27 → 2026-02-25 | 31,269 | 1.5108 | 0.0282 | 0.0329 |
| 8 | 2026-02-17 | 2026-02-26 → 2026-03-26 | 31,277 | 1.5972 | 0.0759 | 0.0429 |
| 9 | 2026-03-18 | 2026-03-27 → 2026-04-27 | 31,290 | 1.4244 | 0.0192 | 0.0277 |
| 10 | 2026-04-17 | 2026-04-28 → 2026-05-27 | 31,290 | 1.5829 | -0.0188 | -0.0212 |
| 11 | 2026-05-18 | 2026-05-28 → 2026-06-26 | 31,291 | 1.568 | 0.0147 | -0.0321 |
| 12 | 2026-06-17 | 2026-06-29 → 2026-07-28 | 31,311 | 1.555 | 0.0202 | 0.0033 |

## Results vs baselines (pooled out-of-sample)

| model | n | log-loss | pooled IC | mean daily IC (t-stat) | decile spread | non-flat hit rate |
|---|---|---|---|---|---|---|
| **LightGBM challenger** | 406,260 | 1.4633 | 0.0066 | 0.0018 (0.31) | 0.056 | 0.501 |
| Logistic (same features) | 406,260 | 1.5215 | -0.0029 | -0.0038 (-0.49) | 0.01 | 0.503 |
| MLP (PyTorch, v2 preview) | 406,260 | 1.5065 | 0.0043 | 0.0025 (0.31) | 0.126 | 0.504 |
| Yesterday's-sign momentum | 406,260 | — | -0.0019 | 0.0028 (0.35) | 0.05 | — |
| Uniform prior | 406,260 | 1.6094 | — | — | — | — |
| control-rule-based (live record) | 111 | — | — | — | — | 0.081 |

control-rule-based is the deactivated (2026-06-07) rule baseline. It predates
prediction logging, so its frozen record is *trade-level* (ledger): 111 shadow trades, hit rate 0.081, avg P&L 0.319%/trade — no IC is computable,
so compare it on hit rate and the toy portfolio, not on ranking metrics.

## Ship gate — beat every baseline, pooled out-of-sample

| baseline | baseline pooled IC | challenger pooled IC | verdict |
|---|---|---|---|
| Yesterday's-sign momentum | -0.0019 | 0.0066 | beats |
| Logistic (same features) | -0.0029 | 0.0066 | beats |
| MLP (PyTorch, v2 preview) | 0.0043 | 0.0066 | beats |
| Uniform prior (log-loss basis) | — | 0.0066 | beats |

**Verdict: passes** — the challenger beats every comparable baseline pooled
out-of-sample (control-rule-based's trade-level record has no IC basis and
is compared in the toy-portfolio section instead).

## Significance — the evolution gate's own vocabulary

- Pooled OOS rank IC: **0.0066** over 406,260 graded rows
- Permutation noise floor (1000 shuffles, 95th pct, as `scripts/ic_noise_floor.py`): **0.0026**
- Fisher-z 95% lower bound (`PROMOTION_IC_CI_Z = 1.96`): **0.0035**
- The pooled IC clears the noise floor.

| horizon | pooled IC | n |
|---|---|---|
| open(t+1)→close(t+1) | 0.0066 | 406,259 |
| open(t+1)→close(t+2) | 0.0055 | 404,767 |
| open(t+1)→close(t+3) | 0.0102 | 403,276 |

## CPCV — combinatorial purged cross-validation

6 blocks choose 2 = 15 OOS paths with the
winning params (grid re-selection inside CPCV would be circular). Where simple
walk-forward gives one pooled IC, CPCV gives a distribution:

mean **-0.0001** · std 0.0098 · range [-0.0221, 0.0143] · 0.467 of paths positive.

Caveat: CPCV trains on data that postdates some validation blocks — acceptable
for a stationary-ish feature spec, and the reason promotion still keys on the
walk-forward number (the only one runtime evolution can observe).

## Regime-conditional IC (VIX terciles)

Sliced by the VIX close on the *feature* date (known at prediction time).
Repo precedent: momentum-trader-vix-gated.

| regime | n | days | pooled IC | mean daily IC |
|---|---|---|---|---|
| calm (VIX ≤ 16.4) | 136,823 | 92 | -0.0243 | -0.0197 |
| mid (16.4 – 18.2) | 135,421 | 91 | -0.0026 | -0.0006 |
| stressed (VIX > 18.2) | 134,016 | 90 | 0.028 | 0.0263 |

## Multi-horizon heads

Separate models per holding horizon, sharing h=1's winning params; purge
scales with the horizon (h-day labels overlap h days). Only h=1 is graded at
runtime — the longer heads drive `TradeIntent.hold_days` on picked names,
exercising the Phase 12A multi-day machinery. One class vocabulary at every
horizon (±1%/±4%); h-day moves are ~√h larger so class balance shifts, which
balanced weights absorb.

| horizon (sessions) | n OOS | pooled IC | mean daily IC | log-loss |
|---|---|---|---|---|
| 1 | 406,260 | 0.0066 | 0.0018 | 1.4633 |
| 2 | 406,255 | 0.001 | 0.0057 | 1.5513 |
| 3 | 406,250 | -0.0012 | 0.0047 | 1.5634 |
| 5 | 406,240 | 0.0237 | 0.019 | 1.5722 |

## Calibration

Multiclass Brier **0.7504** · argmax-confidence ECE **0.016**.
LLM conviction values are famously uncalibrated; a demonstrably calibrated
challenger is a headline result even where IC ties.

<details><summary>Reliability — strong_down</summary>

| predicted prob | n | mean predicted | empirical freq |
|---|---|---|---|
| 0.0–0.1 | 150,466 | 0.046 | 0.015 |
| 0.1–0.2 | 143,649 | 0.151 | 0.023 |
| 0.2–0.3 | 88,877 | 0.24 | 0.055 |
| 0.3–0.4 | 17,271 | 0.336 | 0.093 |
| 0.4–0.5 | 3,820 | 0.441 | 0.099 |
| 0.5–0.6 | 1,326 | 0.542 | 0.104 |
| 0.6–0.7 | 559 | 0.645 | 0.109 |
| 0.7–0.8 | 213 | 0.739 | 0.155 |
| 0.8–0.9 | 72 | 0.843 | 0.139 |

</details>
<details><summary>Reliability — mild_down</summary>

| predicted prob | n | mean predicted | empirical freq |
|---|---|---|---|
| 0.0–0.1 | 6,640 | 0.076 | 0.211 |
| 0.1–0.2 | 113,479 | 0.169 | 0.24 |
| 0.2–0.3 | 223,342 | 0.24 | 0.227 |
| 0.3–0.4 | 48,137 | 0.338 | 0.224 |
| 0.4–0.5 | 11,276 | 0.439 | 0.242 |
| 0.5–0.6 | 2,673 | 0.539 | 0.235 |
| 0.6–0.7 | 596 | 0.637 | 0.237 |
| 0.7–0.8 | 101 | 0.727 | 0.178 |

</details>
<details><summary>Reliability — flat</summary>

| predicted prob | n | mean predicted | empirical freq |
|---|---|---|---|
| 0.0–0.1 | 25,286 | 0.072 | 0.271 |
| 0.1–0.2 | 134,107 | 0.155 | 0.369 |
| 0.2–0.3 | 117,592 | 0.246 | 0.49 |
| 0.3–0.4 | 72,803 | 0.345 | 0.561 |
| 0.4–0.5 | 38,435 | 0.443 | 0.61 |
| 0.5–0.6 | 13,563 | 0.54 | 0.64 |
| 0.6–0.7 | 3,587 | 0.639 | 0.667 |
| 0.7–0.8 | 786 | 0.736 | 0.701 |
| 0.8–0.9 | 95 | 0.836 | 0.821 |

</details>
<details><summary>Reliability — mild_up</summary>

| predicted prob | n | mean predicted | empirical freq |
|---|---|---|---|
| 0.0–0.1 | 8,528 | 0.076 | 0.239 |
| 0.1–0.2 | 131,701 | 0.168 | 0.246 |
| 0.2–0.3 | 215,933 | 0.236 | 0.23 |
| 0.3–0.4 | 38,772 | 0.338 | 0.239 |
| 0.4–0.5 | 8,694 | 0.438 | 0.26 |
| 0.5–0.6 | 2,064 | 0.538 | 0.25 |
| 0.6–0.7 | 459 | 0.638 | 0.275 |
| 0.7–0.8 | 94 | 0.735 | 0.287 |

</details>
<details><summary>Reliability — strong_up</summary>

| predicted prob | n | mean predicted | empirical freq |
|---|---|---|---|
| 0.0–0.1 | 171,151 | 0.041 | 0.016 |
| 0.1–0.2 | 134,784 | 0.15 | 0.025 |
| 0.2–0.3 | 73,453 | 0.241 | 0.064 |
| 0.3–0.4 | 19,843 | 0.338 | 0.103 |
| 0.4–0.5 | 4,590 | 0.439 | 0.116 |
| 0.5–0.6 | 1,438 | 0.542 | 0.121 |
| 0.6–0.7 | 617 | 0.643 | 0.112 |
| 0.7–0.8 | 267 | 0.744 | 0.18 |
| 0.8–0.9 | 96 | 0.838 | 0.167 |

</details>

## Per-class precision / recall (challenger)

| class | support | predicted | precision | recall |
|---|---|---|---|---|
| strong_down | 12,546 | 51,802 | 0.063 | 0.26 |
| mild_down | 93,576 | 87,207 | 0.248 | 0.231 |
| flat | 190,013 | 144,378 | 0.582 | 0.442 |
| mild_up | 96,416 | 68,968 | 0.258 | 0.184 |
| strong_up | 13,709 | 53,905 | 0.081 | 0.318 |

## Toy top-5 long portfolio (net of costs)

Equal-weight long the top-5 scores each validation day, net of 0.303% round-trip (~0.30% round-trip cost on £2,000 (FX 0.30% rt)):
mean daily net -0.3764% · hit rate 0.443 · cumulative -102.76% over 273 days · worst day -11.66%.
A sanity harness, not a backtest — no slippage, no capacity, fills at the
label's own open. UK-EU runs as a separate sleeve with a stamp-duty-aware
cost gate — see its own card.

## Feature attribution — gain vs TreeSHAP

Gain importances (split quality) next to TreeSHAP mean-|contribution| on a
20k-row sample (consistent attributions — LightGBM's `pred_contrib` is exact
TreeSHAP for trees). The signed column is the mean push toward `strong_up`.

| feature | gain | mean abs SHAP | signed → strong_up |
|---|---|---|---|
| `parkinson_10d` | 284,660 | 0.30263 | -0.12237 |
| `vol_21d` | 66,569 | 0.12637 | -0.0583 |
| `overnight_gap` | 32,606 | 0.04173 | 0.0071 |
| `ret_1d` | 29,769 | 0.05043 | -0.00546 |
| `ret_1d_rank` | 20,896 | 0.04073 | 0.00388 |
| `vol_10d` | 18,630 | 0.04315 | 0.00422 |
| `volume_ratio` | 17,047 | 0.02894 | 0.00433 |
| `ret_21d` | 15,286 | 0.03804 | -0.00272 |
| `dollar_volume_pctile` | 14,144 | 0.02781 | 0.00051 |
| `ret_5d` | 13,812 | 0.02021 | 0.00463 |
| `close_vs_sma63` | 13,085 | 0.01852 | 0.00241 |
| `ret_21d_z` | 11,912 | 0.02471 | 0.00423 |
| `vol_of_vol` | 11,154 | 0.02357 | 0.01048 |
| `ret_10d` | 11,062 | 0.02273 | 0.00514 |
| `close_vs_sma10` | 10,288 | 0.01942 | -0.00466 |
| `close_vs_sma21` | 10,082 | 0.02135 | -0.00019 |
| `dow_mon` | 9,012 | 0.02836 | -0.00045 |
| `dow_wed` | 8,220 | 0.02369 | 0.00001 |
| `dow_thu` | 6,358 | 0.00989 | -0.00056 |
| `dow_fri` | 5,921 | 0.02693 | -0.00152 |

## Limitations

- **Pooled IC (0.0066) exceeds the mean per-date IC (0.0018).** Part of the pooled ranking edge comes from
  cross-date level effects rather than within-day stock selection. The
  pooled number is quoted because it is what `metrics.py` computes at
  runtime; the per-date IC and the toy portfolio are the sober view.
- **Survivorship:** today's index membership applied historically omits
  delisted names and flatters the backtest slightly. The forward shadow run
  is immune — which is the argument for shadow deployment.
- **Depth:** the snapshot spans 2023-09-05 → 2026-07-28 (~3 years — the deep-history stretch). Broader than v1's single year, still
  far from a full cycle; the VIX-tercile table above is the regime lens.
  Deeper history also worsens survivorship slightly (more delistings missing).
- **Grid selected on the same OOS folds it reports.** 6 combos — selection
  pressure is minimal, but the pooled numbers carry that footnote.
- **No text/news features.** By design: if an LLM strategy beats this, the
  interesting ablation is exactly the features the GBM can't see.

## Reproduce

```bash
python -m trading_bot.ml.data backfill --region us   # rebuild state/ml/train.db
python -m trading_bot.ml.train --region us --seed 42  # card + model_h*.txt + manifest
```

Data snapshot sha256[:16] `b9d68a5efc4126fa` · feature spec `58bd0e3b705e8d26` · LightGBM params: lr=0.1, depth=6, leaves=31, min_leaf=100, feature/bagging 0.8, balanced weights.
