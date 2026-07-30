# trading_bot

Self-improving stock trading bot. Runs unattended on GitHub Actions, places paper-money orders on real brokers, evolves its own strategies week-over-week. Most strategies are LLM-driven (Claude); one — [`ml-challenger`](#ml-challenger-the-learned-strategy) — is a trained LightGBM model running against them as a controlled experiment. Live (real-money) graduation is gated behind explicit human approval.

**Live dashboard:** https://dario-zela.github.io/trading_bot/ · design rationale in [`sparknotes.md`](./sparknotes.md).

## What it does

Every weekday morning, for each region (`us`, `uk-eu`), the entry cron:
1. Pulls a candidate universe (T212-ISA-tradeable US / UK-EU names).
2. Runs each active strategy: the LLM ones rank and select picks from prices, news, filings, macro view, and the strategy's prompt; the ML one scores the full cross-section with a committed model.
3. Records 5-class predictions (`strong_up` / `mild_up` / `flat` / `mild_down` / `strong_down`) on every candidate so strategies are scored statistically, beyond just the trades they took.
4. Submits orders via the strategy's configured executor.

Every weekday evening, the exit cron closes all positions opened today (or reads filled bracket children if Alpaca's stop / take-profit fired intraday), computes P&L from broker-reported fill prices, runs LLM reflection on each trade, rebuilds the dashboard, and emails a summary.

Once a week the macro agent refreshes the cross-asset regime view; the evolution agent reads each strategy's rolling 14-day metrics (hit rate, IC, drawdown) and auto-promotes / demotes / tunes / spawns variants within bounds. Tier-2 (real-money) promotions are recommended only — they require a human-approved GitHub Issue.

## Tiers

| Tier | Executor | Money | Who picks strategies |
|---|---|---|---|
| 0 — shadow | `ShadowExecutor` (yfinance prices, no orders) | None | Bot |
| 1 — Alpaca paper (US only) | `AlpacaPaperExecutor` (bracket orders) | Paper | Bot, auto-promotion from tier 0 |
| 1.5 — T212 demo (UK-EU only) | `Trading212DemoExecutor` (market orders) | Paper (£50k cap) | Bot, manual promotion |
| 2 — T212 live | _deferred_ | Real | Human approval only |

Recorded fill prices and P&L come exclusively from broker order endpoints — yfinance is only used as a sizing seed for tier 1 / 1.5 and as the actual fill source on tier 0 (shadow).

## Strategies

The roster is **evolution-managed** — the weekly agent promotes, demotes, retires and spawns variants, so the set below is a snapshot (2026-07-30; the authoritative list is `strategies/*/config.yaml` with `active: true`, rendered live on the dashboard). Ten active:

| ID | Style | Tier (US) | Tier (UK-EU) |
|---|---|---|---|
| `momentum-trader-vix-gated` | LLM trend-following, VIX-gated | Alpaca slot 1 | shadow |
| `mean-reverter` | LLM counter-trend | Alpaca slot 2 | T212 slot 1 |
| `news-reactive-disclosure` | LLM event-driven (filings) | Alpaca slot 3 | shadow |
| `news-reactive` / `-buzz-fade` / `-sentiment-gate` | LLM event-driven family | shadow | shadow |
| `macro-aligned` / `-hmm` / `-macro-attn` | LLM top-down family | shadow | shadow |
| `ml-challenger` | **Learned (LightGBM)** — see below | shadow | shadow |

Each strategy has a `config.yaml` with `runs_in:` per-region entries (region, tier, slot, universe); LLM strategies add prompts (`prefilter.md`, `deep_analysis.md`, `final_select.md`). Retired strategies (`momentum-trader`, `control-rule-based`, `sector-rotator`, …) keep their configs and history under `strategies/` as frozen yardsticks. The evolution agent edits configs and prompts within safety bounds; human approval is needed only for tier-2 transitions.

## ml-challenger: the learned strategy

The one strategy that predicts with **gradient-boosted trees instead of an
LLM**: per-region LightGBM 5-class models over point-in-time OHLCV features,
running at shadow tier in both regions as a frozen experiment control
(`evolution_frozen: true` — the evolution agent observes but can't touch it).
It emits the same `PredictionRecord`s in the grader's exact class
vocabulary, so grading, metrics, dashboards and evolution needed zero new
machinery — the ML-vs-LLM comparison falls out of the existing pipeline.

The methodology is the point: structurally point-in-time features (leakage
CI-tested), labels defined by the grader itself, purged walk-forward + CPCV
validation, permutation noise floor and Fisher-z bounds, five baselines
including a PyTorch MLP, TreeSHAP attributions, VIX-regime slices, a
depth-vs-recency training-window ablation, multi-horizon heads driving
`hold_days`, and a stamp-duty-aware cost gate for the UK-EU sleeve.

- **Read the results**: the [ML lab dashboard page](https://dario-zela.github.io/trading_bot/ml/)
  or the auto-generated model cards in the repo
  ([us](strategies/ml-challenger/model/us/card.md) ·
  [uk-eu](strategies/ml-challenger/model/uk-eu/card.md)).
- **Retraining**: monthly via `retrain-challenger.yml`, gated per region on
  the pooled OOS rank-IC Fisher-z lower bound; rejected retrains open an
  issue with the metric diff instead of committing.
- **Head-to-head tracking**: the `mart_llm_vs_ml` dbt mart rolls up daily
  per-strategy rank IC, non-flat hit rate and conviction-vs-realised
  calibration gap.
- **Reproduce**: `python -m trading_bot.ml.data backfill --region us` then
  `python -m trading_bot.ml.train --region us` (training snapshot is
  separate from the runtime OHLCV cache, which is never widened for
  training).

## Daily cycle

GitHub Actions' built-in cron is unreliable (we observed silent dropped triggers during high-load windows). Scheduling is handled by **cron-job.org**, an independent service that calls each workflow via GitHub's REST API at the right local time. Each schedule runs in the market's local timezone, so DST is handled automatically.

| Workflow | When (market-local) | Region | Action |
|---|---|---|---|
| `grade-predictions.yml` | Daily 05:00 UTC | — | score predictions whose target date has passed |
| `daily-news-brief.yml` | Daily ~07:15 UK | — | generate the news brief (dashboard "Bot Tribune") |
| `pipeline-uk-eu.yml` | Mon–Fri 08:35 UK | UK-EU | entry |
| `pipeline-us.yml` | Mon–Fri 09:35 ET | US | entry |
| `midday-trail-uk-eu.yml` | Mon–Fri ~12:00 UK | UK-EU | ratchet trailing stops mid-session |
| `midday-trail-us.yml` | Mon–Fri ~12:30 ET | US | ratchet trailing stops mid-session |
| `pipeline-uk-eu.yml` | Mon–Fri 16:00 UK | UK-EU | exit + reflect + dashboard + email |
| `pipeline-us.yml` | Mon–Fri 15:30 ET | US | exit + reflect + dashboard + email |
| `weekly-evolution.yml` | Sat 09:00 UTC | — | strategy promote / demote / tune / spawn |
| `weekly-macro.yml` | Sun 17:00 UTC | — | macro view refresh |

(Plus maintenance crons: `health-check`, `archive-trim`, and the `ohlcv-*` cache jobs — see `scripts/setup_cron_jobs.py` for the authoritative schedule.)

> **Trailing stops are GBP-only on T212.** T212's API rejects stop orders on non-base-currency instruments (`POST /equity/orders/stop` → 400 "Invalid payload"), so the UK-EU midday trail only protects GBP/GBX (UK) positions. EU/US instruments in the `t212_isa_uk_eu` universe are bought fine (market orders auto-FX) but get **no intraday trailing stop** — they're closed by the scheduled EOD exit instead. Alpaca (US tier) has no such restriction.

Provision the cron-job.org schedules from `scripts/setup_cron_jobs.py` (one-shot). All workflows also accept `workflow_dispatch` for manual runs from the Actions tab.

## Dashboard + email

GitHub Pages-hosted dashboard at https://dario-zela.github.io/trading_bot/: per-strategy + per-region equity curves, recent trades, prediction calibration, the daily Bot Tribune news brief, the weekly evolution log, and the [ML lab](https://dario-zela.github.io/trading_bot/ml/) (the ml-challenger model cards rendered from the committed artifacts). End-of-day email summary (Brevo) groups by strategy with a table of contents.

## Analytics: DuckDB + dbt + Metabase

The append-only JSONL state files (`ledger`, `predictions`, `decision_log`, `trail_exits`) double as the raw layer of a dbt project under `dbt/`:

- **Staging** (`stg_*`, views): 1:1 typed reads of each JSONL source via DuckDB's `read_json_auto`, with the OHLCV SQLite store attached as a schema.
- **Marts** (`mart_*`, tables): the business facts consumed by the dashboard, evolution agent, and LLM prompts — recent + weekly strategy performance, exit attribution, fee drag — each registered in `schema.yml` with column tests.

```bash
# Rebuild the analytics store (wraps `dbt run` with the right env)
python -c "from trading_bot.analytics.dbt_runner import build; build()"
```

Consumers go through the typed accessors in `src/trading_bot/analytics/reader.py`, which open `dbt/analytics.duckdb` read-only.

For interactive dashboards, `docker compose up -d` serves **Metabase** over the same DuckDB store — mounted read-only so dashboard tiles never contend with `dbt run` for DuckDB's exclusive write lock. Zero-install alternative: the repo's devcontainer boots the full stack in GitHub Codespaces (deps, DuckDB driver, `dbt run`, Metabase on port 3000). Details in [`dbt/README.md`](./dbt/README.md) and [`metabase/README.md`](./metabase/README.md).

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Set at minimum ALPACA_API_KEY__1 / ALPACA_API_SECRET__1 + CLAUDE_CODE_OAUTH_TOKEN
# Optional: T212_API_KEY__1 / T212_API_SECRET__1 if testing UK-EU tier 1.5
```

## Running manually

```bash
# Morning entry (decide picks, place orders, record predictions)
python -m trading_bot.pipeline entry --region us
python -m trading_bot.pipeline entry --region uk-eu

# Evening exit (close positions, compute P&L, reflect, email)
python -m trading_bot.pipeline exit --region us --email
python -m trading_bot.pipeline exit --region uk-eu --email

# Weekly meta-jobs
python -m trading_bot.pipeline weekly-macro
python -m trading_bot.pipeline weekly-evolution

# Slot management (Alpaca only — wipes the slot for re-assignment)
python -m trading_bot.pipeline clear-slot --slot 1
```

## Required secrets

Repo secrets (GitHub Actions environment):

| Secret | Used by |
|---|---|
| `CLAUDE_CODE_OAUTH_TOKEN` | Every LLM call (Claude Max plan token from `claude setup-token`) |
| `ALPACA_API_KEY__N` / `ALPACA_API_SECRET__N` | Tier-1 US strategies on slot N (1–3) |
| `T212_API_KEY__1` / `T212_API_SECRET__1` | Tier-1.5 UK-EU strategies |
| `BREVO_API_KEY` | Daily email summary |
| `NOTIFY_EMAIL_FROM` / `NOTIFY_EMAIL_TO` | Email addresses |

## Project layout

```
.
├── sparknotes.md             # design discussion
├── pyproject.toml
├── .github/workflows/        # cron-scheduled CI workflows
├── src/trading_bot/
│   ├── pipeline.py           # CLI entry point (entry / exit / reflect / summary / weekly-* / grade-predictions / daily-news-brief / ohlcv-* / clear-slot)
│   ├── tools/                # universes, history, news, filings, macro view, T212 instruments
│   ├── strategy/             # base classes + registry + per-implementation strategies
│   ├── ml/                   # ml-challenger: features, labels, splits, trainer, data snapshot
│   ├── executor/             # ShadowExecutor, AlpacaPaperExecutor, Trading212DemoExecutor
│   ├── state/                # ledger / predictions / paths
│   ├── meta/                 # metrics, reflection, macro, evolution, grade_predictions, backtest
│   ├── analytics/            # dbt runner + typed read-only accessors over the marts
│   ├── notify/               # email rendering + send
│   └── dashboard/            # static HTML build
├── strategies/               # per-strategy config.yaml + prompts (LLM-evolvable)
├── state/                    # runtime ledger, predictions, evolution log (committed by CI)
├── dbt/                      # analytics layer: staging views + tested marts over state/*.jsonl
├── metabase/                 # Metabase dashboards over dbt/analytics.duckdb (see docker-compose.yml)
└── docs/                     # GitHub Pages dashboard output (committed by CI)
```
