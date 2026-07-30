# Model card — ml-challenger · uk-eu (LightGBM, 5-class)

*Generated 2026-07-30T19:58:13+00:00 by `python -m trading_bot.ml.train --region uk-eu` · seed 42 · data snapshot `b9d68a5efc4126fa`*

Gradient-boosted trees predicting the same 5-class next-session
open→close vocabulary the LLM strategies are graded on, trained on
point-in-time OHLCV features with purged walk-forward validation.
Deployed at shadow tier so the existing daily grading and weekly
evolution machinery judges ML vs LLM head-to-head, forward, out of
sample.

## Data

- Snapshot: `state/ml/train.db` — this region's slice: 370,840 bars, 466 tickers, 2023-06-06 → 2026-07-29
- Training frame: **341,443 rows** × 466 tickers × 744 feature dates (2023-08-31 → 2026-07-28)
- Universe: (FTSE 350 + DAX + CAC + AEX + UCITS ETFs) ∩ t212_isa_uk_eu (~470 UK/EU names)
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
| strong_down | 6,859 | 2.0% |
| mild_down | 66,674 | 19.5% |
| flat | 193,444 | 56.7% |
| mild_up | 67,751 | 19.8% |
| strong_up | 6,715 | 2.0% |

Class imbalance is why training uses balanced class weights and this card
never quotes raw accuracy — predict-always-flat scores deceptively well.

Class midpoints (within-class mean returns, fitted on the first training
block only and frozen): strong_down=-5.6, mild_down=-1.903, flat=0.017, mild_up=1.879, strong_up=5.724
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
(winner: lr=0.1, depth=6, 1258 rounds). The label is intraday so labels never
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
| 0 | 2025-06-25 | 2025-07-04 → 2025-08-01 | 9,764 | 1.1335 | 0.0122 | 0.0099 |
| 1 | 2025-07-24 | 2025-08-04 → 2025-09-01 | 9,403 | 1.1413 | 0.0387 | 0.0174 |
| 2 | 2025-08-22 | 2025-09-02 → 2025-09-30 | 9,765 | 1.1422 | 0.0166 | 0.0077 |
| 3 | 2025-09-22 | 2025-10-01 → 2025-10-29 | 9,746 | 1.1493 | 0.0061 | 0.0129 |
| 4 | 2025-10-21 | 2025-10-30 → 2025-11-27 | 9,786 | 1.2382 | -0.0326 | 0.0082 |
| 5 | 2025-11-19 | 2025-11-28 → 2025-12-30 | 9,708 | 1.1149 | 0.0295 | 0.0205 |
| 6 | 2025-12-18 | 2025-12-31 → 2026-01-29 | 9,747 | 1.1654 | 0.0147 | 0.014 |
| 7 | 2026-01-21 | 2026-01-30 → 2026-02-27 | 9,785 | 1.3075 | 0.0019 | 0.0291 |
| 8 | 2026-02-19 | 2026-03-02 → 2026-03-30 | 9,772 | 1.6083 | 0.0525 | 0.0055 |
| 9 | 2026-03-20 | 2026-03-31 → 2026-04-30 | 9,785 | 1.3637 | -0.0369 | -0.0232 |
| 10 | 2026-04-22 | 2026-05-01 → 2026-05-29 | 8,957 | 1.3283 | 0.0041 | -0.0082 |
| 11 | 2026-05-21 | 2026-06-01 → 2026-06-29 | 9,786 | 1.2872 | -0.0033 | -0.0201 |
| 12 | 2026-06-19 | 2026-06-30 → 2026-07-28 | 9,785 | 1.2784 | -0.0154 | -0.0174 |

## Results vs baselines (pooled out-of-sample)

| model | n | log-loss | pooled IC | mean daily IC (t-stat) | decile spread | non-flat hit rate |
|---|---|---|---|---|---|---|
| **LightGBM challenger** | 125,789 | 1.2506 | 0.0031 | 0.0043 (1.19) | 0 | 0.489 |
| Logistic (same features) | 125,789 | 1.486 | 0.0062 | -0.0001 (-0.01) | 0.182 | 0.492 |
| MLP (PyTorch, v2 preview) | 125,789 | 1.4566 | 0.0151 | 0.0035 (0.6) | 0.277 | 0.494 |
| Yesterday's-sign momentum | 125,789 | — | 0.0157 | 0.0246 (4.08) | 0.134 | — |
| Uniform prior | 125,789 | 1.6094 | — | — | — | — |
| control-rule-based (live record) | 111 | — | — | — | — | 0.081 |

control-rule-based is the deactivated (2026-06-07) rule baseline. It predates
prediction logging, so its frozen record is *trade-level* (ledger): 111 shadow trades, hit rate 0.081, avg P&L 0.319%/trade — no IC is computable,
so compare it on hit rate and the toy portfolio, not on ranking metrics.

## Ship gate — beat every baseline, pooled out-of-sample

| baseline | baseline pooled IC | challenger pooled IC | verdict |
|---|---|---|---|
| Yesterday's-sign momentum | 0.0157 | 0.0031 | **LOSES** |
| Logistic (same features) | 0.0062 | 0.0031 | **LOSES** |
| MLP (PyTorch, v2 preview) | 0.0151 | 0.0031 | **LOSES** |
| Uniform prior (log-loss basis) | — | 0.0031 | beats |

**Verdict: FAILS the design gate** — the challenger loses to Yesterday's-sign momentum, Logistic (same features), MLP (PyTorch, v2 preview) on pooled OOS IC. It ships at shadow tier anyway,
stated here rather than buried, because: (1) shadow risks nothing and the
forward run on live grading — not this backtest — is the decisive test;
(2) the challenger's advantages are on other axes (log-loss, calibration,
per-day IC consistency) that the pooled-IC gate doesn't see; (3) a simple
baseline being hard to beat is itself a finding worth publishing. If the
forward record confirms the loss, the sleeve gets pulled.

## Significance — the evolution gate's own vocabulary

- Pooled OOS rank IC: **0.0031** over 125,789 graded rows
- Permutation noise floor (1000 shuffles, 95th pct, as `scripts/ic_noise_floor.py`): **0.005**
- Fisher-z 95% lower bound (`PROMOTION_IC_CI_Z = 1.96`): **-0.0024**
- The pooled IC does NOT clear the noise floor.

| horizon | pooled IC | n |
|---|---|---|
| open(t+1)→close(t+1) | 0.0024 | 124,523 |
| open(t+1)→close(t+2) | 0 | 122,752 |
| open(t+1)→close(t+3) | -0.0063 | 122,325 |

## CPCV — combinatorial purged cross-validation

6 blocks choose 2 = 15 OOS paths with the
winning params (grid re-selection inside CPCV would be circular). Where simple
walk-forward gives one pooled IC, CPCV gives a distribution:

mean **0.0173** · std 0.004 · range [0.0099, 0.0222] · 1 of paths positive.

Caveat: CPCV trains on data that postdates some validation blocks — acceptable
for a stationary-ish feature spec, and the reason promotion still keys on the
walk-forward number (the only one runtime evolution can observe).

## Regime-conditional IC (VIX terciles)

Sliced by the VIX close on the *feature* date (known at prediction time).
Repo precedent: momentum-trader-vix-gated.

| regime | n | days | pooled IC | mean daily IC |
|---|---|---|---|---|
| calm (VIX ≤ 16.4) | 41,396 | 90 | 0.0049 | 0.0134 |
| mid (16.4 – 18.2) | 40,512 | 88 | 0.008 | -0.0025 |
| stressed (VIX > 18.2) | 40,621 | 88 | -0.0122 | -0.0007 |

## Multi-horizon heads

Separate models per holding horizon, sharing h=1's winning params; purge
scales with the horizon (h-day labels overlap h days). Only h=1 is graded at
runtime — the longer heads drive `TradeIntent.hold_days` on picked names,
exercising the Phase 12A multi-day machinery. One class vocabulary at every
horizon (±1%/±4%); h-day moves are ~√h larger so class balance shifts, which
balanced weights absorb.

| horizon (sessions) | n OOS | pooled IC | mean daily IC | log-loss |
|---|---|---|---|---|
| 1 | 125,789 | 0.0031 | 0.0043 | 1.2506 |
| 2 | 125,827 | -0.0064 | -0.0034 | 1.4504 |
| 3 | 125,826 | -0.0083 | -0.0108 | 1.5196 |
| 5 | 125,824 | -0.0082 | -0.0048 | 1.5462 |

## Calibration

Multiclass Brier **0.6701** · argmax-confidence ECE **0.0549**.
LLM conviction values are famously uncalibrated; a demonstrably calibrated
challenger is a headline result even where IC ties.

<details><summary>Reliability — strong_down</summary>

| predicted prob | n | mean predicted | empirical freq |
|---|---|---|---|
| 0.0–0.1 | 104,388 | 0.017 | 0.016 |
| 0.1–0.2 | 10,892 | 0.159 | 0.039 |
| 0.2–0.3 | 8,223 | 0.219 | 0.058 |
| 0.3–0.4 | 1,074 | 0.343 | 0.064 |
| 0.4–0.5 | 566 | 0.445 | 0.065 |
| 0.5–0.6 | 308 | 0.546 | 0.088 |
| 0.6–0.7 | 179 | 0.647 | 0.061 |
| 0.7–0.8 | 91 | 0.74 | 0.077 |
| 0.8–0.9 | 51 | 0.839 | 0.157 |

</details>
<details><summary>Reliability — mild_down</summary>

| predicted prob | n | mean predicted | empirical freq |
|---|---|---|---|
| 0.0–0.1 | 9,474 | 0.054 | 0.098 |
| 0.1–0.2 | 27,801 | 0.162 | 0.195 |
| 0.2–0.3 | 41,115 | 0.244 | 0.213 |
| 0.3–0.4 | 27,564 | 0.346 | 0.22 |
| 0.4–0.5 | 13,251 | 0.442 | 0.225 |
| 0.5–0.6 | 4,641 | 0.541 | 0.237 |
| 0.6–0.7 | 1,484 | 0.641 | 0.237 |
| 0.7–0.8 | 385 | 0.741 | 0.244 |
| 0.8–0.9 | 71 | 0.831 | 0.282 |

</details>
<details><summary>Reliability — flat</summary>

| predicted prob | n | mean predicted | empirical freq |
|---|---|---|---|
| 0.0–0.1 | 5,406 | 0.069 | 0.31 |
| 0.1–0.2 | 25,513 | 0.165 | 0.4 |
| 0.2–0.3 | 27,464 | 0.246 | 0.481 |
| 0.3–0.4 | 21,910 | 0.349 | 0.555 |
| 0.4–0.5 | 17,220 | 0.447 | 0.616 |
| 0.5–0.6 | 11,225 | 0.546 | 0.675 |
| 0.6–0.7 | 6,754 | 0.646 | 0.738 |
| 0.7–0.8 | 4,251 | 0.746 | 0.807 |
| 0.8–0.9 | 2,835 | 0.848 | 0.864 |
| 0.9–1.0 | 3,211 | 0.955 | 0.943 |

</details>
<details><summary>Reliability — mild_up</summary>

| predicted prob | n | mean predicted | empirical freq |
|---|---|---|---|
| 0.0–0.1 | 10,002 | 0.053 | 0.1 |
| 0.1–0.2 | 27,959 | 0.163 | 0.189 |
| 0.2–0.3 | 42,605 | 0.244 | 0.207 |
| 0.3–0.4 | 27,424 | 0.345 | 0.221 |
| 0.4–0.5 | 12,105 | 0.442 | 0.231 |
| 0.5–0.6 | 4,107 | 0.54 | 0.235 |
| 0.6–0.7 | 1,171 | 0.64 | 0.254 |
| 0.7–0.8 | 346 | 0.737 | 0.254 |
| 0.8–0.9 | 65 | 0.831 | 0.169 |

</details>
<details><summary>Reliability — strong_up</summary>

| predicted prob | n | mean predicted | empirical freq |
|---|---|---|---|
| 0.0–0.1 | 105,084 | 0.015 | 0.017 |
| 0.1–0.2 | 12,327 | 0.164 | 0.033 |
| 0.2–0.3 | 6,374 | 0.223 | 0.057 |
| 0.3–0.4 | 1,047 | 0.343 | 0.071 |
| 0.4–0.5 | 510 | 0.442 | 0.076 |
| 0.5–0.6 | 231 | 0.544 | 0.095 |
| 0.6–0.7 | 126 | 0.646 | 0.119 |
| 0.7–0.8 | 64 | 0.744 | 0.172 |

</details>

## Per-class precision / recall (challenger)

| class | support | predicted | precision | recall |
|---|---|---|---|---|
| strong_down | 2,749 | 4,083 | 0.072 | 0.107 |
| mild_down | 25,724 | 33,865 | 0.238 | 0.313 |
| flat | 69,340 | 54,440 | 0.678 | 0.532 |
| mild_up | 25,301 | 28,870 | 0.237 | 0.27 |
| strong_up | 2,675 | 4,531 | 0.062 | 0.104 |

## Toy top-5 long portfolio (net of costs)

Equal-weight long the top-5 scores each validation day, net of 0.5% round-trip (~0.50% round-trip cost on £2,000 (stamp 0.5%)):
mean daily net -0.6215% · hit rate 0.359 · cumulative -169.66% over 273 days · worst day -5.06%.
A sanity harness, not a backtest — no slippage, no capacity, fills at the
label's own open. The 0.5% LSE stamp duty dominates this region's cost
estimate — which is why the live strategy also applies a pick-time cost
gate (predicted move ≥ cost_gate_multiplier × estimated round-trip).

## Feature attribution — gain vs TreeSHAP

Gain importances (split quality) next to TreeSHAP mean-|contribution| on a
20k-row sample (consistent attributions — LightGBM's `pred_contrib` is exact
TreeSHAP for trees). The signed column is the mean push toward `strong_up`.

| feature | gain | mean abs SHAP | signed → strong_up |
|---|---|---|---|
| `parkinson_10d` | 199,723 | 0.69577 | -0.22243 |
| `overnight_gap` | 92,644 | 0.10654 | 0.00894 |
| `dollar_volume_pctile` | 88,987 | 0.15519 | 0.00483 |
| `vol_21d` | 83,299 | 0.29794 | -0.11981 |
| `ret_1d` | 80,863 | 0.12459 | -0.00403 |
| `volume_ratio` | 80,746 | 0.12709 | 0.0005 |
| `vol_of_vol` | 77,999 | 0.16311 | 0.04768 |
| `vol_10d` | 77,837 | 0.17962 | 0.07922 |
| `ret_1d_rank` | 77,347 | 0.12694 | 0.00447 |
| `close_vs_sma63` | 76,433 | 0.12372 | 0.08168 |
| `ret_5d` | 75,548 | 0.13368 | 0.00605 |
| `ret_21d_z` | 73,227 | 0.15675 | -0.00693 |
| `ret_10d` | 71,022 | 0.13101 | 0.02955 |
| `ret_21d` | 66,583 | 0.15537 | -0.05519 |
| `close_vs_sma10` | 64,700 | 0.11572 | 0.00273 |
| `close_vs_sma21` | 61,608 | 0.14063 | -0.03513 |
| `dow_tue` | 8,766 | 0.05339 | -0.0008 |
| `dow_fri` | 8,418 | 0.05368 | 0.00032 |
| `dow_thu` | 7,916 | 0.05586 | -0.00134 |
| `dow_mon` | 6,882 | 0.03495 | 0.00112 |

## Limitations

- **Loses to Yesterday's-sign momentum, Logistic (same features), MLP (PyTorch, v2 preview) on pooled OOS IC** — the design's
  ship gate fails for this sleeve; see the Ship gate section for the
  numbers and the explicit case for running it forward anyway.
- **Survivorship:** today's index membership applied historically omits
  delisted names and flatters the backtest slightly. The forward shadow run
  is immune — which is the argument for shadow deployment.
- **Depth:** the snapshot spans 2023-08-31 → 2026-07-28 (~3 years — the deep-history stretch). Broader than v1's single year, still
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

Data snapshot sha256[:16] `b9d68a5efc4126fa` · feature spec `58bd0e3b705e8d26` · LightGBM params: lr=0.1, depth=6, leaves=31, min_leaf=100, feature/bagging 0.8, balanced weights.
