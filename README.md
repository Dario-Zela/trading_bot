# trading_bot

Self-improving LLM-driven stock trading bot. Runs unattended on GitHub Actions, places paper-money orders on real brokers, evolves its own strategies week-over-week. Live (real-money) graduation is gated behind explicit human approval.

See [`sparknotes.md`](./sparknotes.md) for the full design rationale.

## What it does

Every weekday morning, for each region (`us`, `uk-eu`), the entry cron:
1. Pulls a candidate universe (S&P 1500 for US, FTSE 350 + DAX + CAC + AEX for UK-EU).
2. Runs each active strategy: LLM (Claude) ranks and selects today's picks based on prices, news, filings, macro view, and the strategy's prompt.
3. Records 5-class predictions (`strong_up` / `mild_up` / `flat` / `mild_down` / `strong_down`) on every candidate so we can score the LLM statistically beyond just the trades it took.
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

Eight active strategies under `strategies/`:

| ID | Style | Tier (US) | Tier (UK-EU) |
|---|---|---|---|
| `momentum-trader` | LLM trend-following | Alpaca slot 1 | T212 slot 1 |
| `mean-reverter` | LLM counter-trend | Alpaca slot 2 | T212 slot 1 |
| `news-reactive` | LLM event-driven | Alpaca slot 3 | shadow |
| `macro-aligned` | LLM top-down (sector + macro view) | shadow | shadow |
| `control-rule-based` | Deterministic baseline (top-N prior-day gainers) | shadow | shadow |
| `commodity-momentum` | LLM commodity-ETF | shadow | shadow |
| `sector-rotator` | LLM sector ETF rotation | shadow | shadow |
| `bond-cycle` | LLM rates / duration | shadow | shadow |

Each strategy has a `config.yaml` with `runs_in:` per-region entries (region, tier, slot, universe), plus prompts (`wide_scoring.md`, `deep_analysis.md`, `final_select.md`) for the LLM stages. The evolution agent can edit configs and prompts within safety bounds; human approval is needed only for tier-2 transitions.

## Daily cycle

GitHub Actions' built-in cron is unreliable (we observed silent dropped triggers during high-load windows). Scheduling is handled by **cron-job.org**, an independent service that calls each workflow via GitHub's REST API at the right local time. Each schedule runs in the market's local timezone, so DST is handled automatically.

| Workflow | When (market-local) | Region | Action |
|---|---|---|---|
| `pipeline-us.yml` | Mon–Fri 09:35 ET | US | entry |
| `pipeline-us.yml` | Mon–Fri 15:30 ET | US | exit + reflect + dashboard + email |
| `pipeline-uk-eu.yml` | Mon–Fri 08:35 UK | UK-EU | entry |
| `pipeline-uk-eu.yml` | Mon–Fri 16:00 UK | UK-EU | exit + reflect + dashboard + email |
| `weekly-macro.yml` | Sun 17:00 UTC | — | macro view refresh |
| `weekly-evolution.yml` | Sat 09:00 UTC | — | strategy promote / demote / tune / spawn |

Provision the cron-job.org schedules from `scripts/setup_cron_jobs.py` (one-shot). All workflows also accept `workflow_dispatch` for manual runs from the Actions tab.

## Dashboard + email

GitHub Pages-hosted dashboard at the repo's Pages URL: per-strategy + per-region equity curves, recent trades table, prediction calibration. End-of-day email summary (Brevo) groups by strategy with a table of contents.

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
python -m trading_bot.pipeline dst-sync

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
│   ├── pipeline.py           # CLI entry point (entry / exit / reflect / summary / weekly-*)
│   ├── tools/                # universes, history, news, filings, macro view, T212 instruments
│   ├── strategy/             # base classes + registry + per-implementation strategies
│   ├── executor/             # ShadowExecutor, AlpacaPaperExecutor, Trading212DemoExecutor
│   ├── state/                # ledger / predictions / paths
│   ├── meta/                 # metrics, reflection, macro, evolution, dst_sync
│   ├── notify/               # email rendering + send
│   └── dashboard/            # static HTML build
├── strategies/               # per-strategy config.yaml + prompts (LLM-evolvable)
├── state/                    # runtime ledger, predictions, evolution log (committed by CI)
└── docs/                     # GitHub Pages dashboard output (committed by CI)
```
