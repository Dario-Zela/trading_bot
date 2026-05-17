# Trading Bot — Sparknotes

Working notes from the initial design discussion. Not a finished plan — a scratchpad we iterate on before writing code.

## Goal

A personal trading bot that, on weekday mornings, identifies short-term stock opportunities and exits them before the day ends. Same-day round trip, no overnight holds. Runs unattended on GitHub Actions (free, no infra). LLM-driven strategy that improves itself over time on paper accounts before any live capital is committed.

## Tiered execution

The bot runs many strategies in parallel across two active tiers. A third tier (live) is deferred — **no real money trades until a strategy has demonstrably converged and is explicitly chosen for graduation.** Until then everything is paper.

| Tier | Executor | Used by | Realism | Parallelism |
|---|---|---|---|---|
| 0 — Shadow | `ShadowExecutor` (records intent + reads real prices, no orders placed anywhere) | Every new / unpromoted strategy. Default. | No fills, no slippage modelling | Unlimited — cost is effectively zero |
| 1 — Paper-fills | `AlpacaPaperExecutor` | Strategies promoted from Tier 0 by the meta-agent (with user approval) | Real fill mechanics, slippage, partial fills | Limited — Alpaca supports multiple paper accounts under one user, so we can run 2–3 in parallel |
| 2 — Live (deferred) | `Trading212ApproveExecutor` (Option B, one-tap approval) | The single strategy chosen for live graduation. Activated only once. | Real money | 1 strategy at a time, starting £100, scaling up on track record |

### Why this shape

- Alpaca is US-equities-only, so UK/EU strategies can't reach Tier 1 — they stay in shadow until/unless they graduate to live
- Tier 0 lets us run essentially unlimited parallel strategies because there's no broker friction or account-balance constraint
- Tier 1 catches anything that only breaks at the realistic-fill layer (slippage-sensitive strategies, low-liquidity picks)
- Tier 2 is the destination, not a parallel mode. We don't need to design its mechanics until we're close to using it. Option B + the approve-channel discussion is deferred indefinitely until then.

## Position sizing & risk

- Each strategy starts with a **paper capital allocation** (default £10k, configurable per strategy). Realistic relative to the £20k/year ISA cap, big enough that fractional-share math doesn't dominate behaviour.
- The LLM's final-selection step outputs per pick:
  - `ticker`
  - `allocation_pct` — fraction of the strategy's capital
  - `stop_loss_pct` — optional, e.g., -3.0
  - `take_profit_pct` — optional, e.g., +5.0
  - `rationale`
- Hard constraints in the strategy config:
  - `max_position_pct` (e.g., 30%)
  - `min_position_size` (e.g., £50 at paper scale)
  - `max_positions`
  - allocations sum to ≤ 100% (strategy may hold cash if nothing looks compelling)
- Sizing and stop/take-profit style are part of each strategy's identity. The meta-agent evolves them the same way it evolves other config and prompts.
- Stops and take-profits are executed as **bracket orders** server-side (Alpaca `OrderClass.BRACKET`; T212 stop orders) — no mid-day cron monitoring needed. The Shadow executor simulates them at end-of-day by checking the intra-day high/low against the trigger prices.
- Predictions are graded against the EOD price regardless of whether a stop or take-profit fired mid-day. Stop/TP affect *trade P&L*; the *prediction-accuracy* metric is unaffected.
- **Daily strategy-level kill-switch is deferred** — on paper we want to see what happens when a strategy goes wrong rather than mask it. The meta-agent can recommend demotion at the weekly evolution review.

## Two layers — strategies and tools

The bot's intelligence is split across two distinct layers with different evolution models.

### Strategies (top-level) — LLM-evolvable

A strategy is a "trader persona" — a complete pipeline config plus prompts at every stage. Different strategies encode different *biases* (momentum, mean-reversion, macro-aligned, news-reactive, etc). The meta-agent evolves strategies (Species A — prompts and configs only, never Python).

### Tools (mid-level) — human-controlled

A tool is a pure analysis function exposed via MCP — `get_technicals(ticker)`, `summarize_recent_news(ticker, days)`, `get_filing_summary(ticker, type)`, etc. Tools are **Python code**, written once, shared across all strategies, **never modified by the LLM**. They handle the "what is true"; strategies handle the "what to do about it." If an LLM strategy wants a tool that doesn't exist, it files a `change-request` issue and the human implements it.

### Why the split matters

- Without tools: every strategy re-derives technicals, news summaries, etc. from raw data in its prompt. Massive token waste, slow, inconsistent across strategies.
- With tools: deep-analysis prompts become recipes ("for each candidate, call `get_technicals` + `get_recent_news`, weight per your bias, output a structured score"). Tokens spent on judgment, not computation. Consistent across strategies.
- Tool library compounds across every strategy that ever runs — single most valuable shared asset of the system.

### Seed strategies (8 total — 1 control + 4 equity LLM + 3 cross-asset LLM)

Deliberately diverse — different lenses, different asset focuses, different signal-to-noise profiles. The meta-agent will later spawn variants within each school of thought.

**Equity / single-stock**

| Strategy | Bias / persona |
|---|---|
| `control-rule-based` | Deterministic baseline: top-5 highest-volume previous-day gainers, equal weight, no stops, no LLM. The yardstick every other strategy must beat. |
| `momentum-trader` | Buy recent winners; price action + volume; trend-following |
| `mean-reverter` | Buy oversold bounces; RSI + distance from MA; counter-trend |
| `news-reactive` | Earnings surprises, M&A, guidance changes; catalyst-driven |
| `macro-aligned` | Reads weekly macro view; buys bullish sectors, avoids bearish; regime-aware |

**Cross-asset / ETF**

| Strategy | Bias / persona |
|---|---|
| `sector-rotator` | Trades SPDR sector ETFs (XLF, XLE, XLK, XLV, XLY, XLP, XLI, XLU, XLB, XLRE, XLC); relative momentum across sectors |
| `bond-cycle` | Trades bond ETFs (TLT, IEF, SHY, HYG, LQD); yield-curve and credit-spread signals |
| `commodity-momentum` | Trades commodity ETFs (GLD, SLV, USO, DBA, DBB); macro and dollar-index signals |

### Seed tool library

| Tool | Returns |
|---|---|
| `get_universe(name)` | Tickers for a named universe — supports `sp500`, `us_etfs_sector`, `us_etfs_bond`, `us_etfs_commodity` |
| `get_history(tickers, lookback)` | OHLCV history |
| `get_technicals(ticker)` | RSI, MACD, ATR, moving averages, volume profile |
| `get_recent_news(ticker, days)` | Summarized headlines with sentiment tagging |
| `get_filing_summary(ticker, type)` | Key takeaways from latest 10-K / 10-Q / 8-K (SEC EDGAR) or UK equivalent |
| `get_earnings_info(ticker)` | Next earnings date, last surprise %, analyst consensus |
| `get_insider_trades(ticker, days)` | Form 4 summary |
| `get_sector_strength()` | Ranked SPDR sector ETF performance (1d / 5d / 20d) |
| `get_etf_relative_strength(category)` | Rank a category's ETFs by N-day return |
| `get_yield_curve()` | US Treasury yields by tenor; curve-shape snapshots |
| `get_credit_spreads()` | HYG/LQD spread + direction |
| `get_dollar_index()` | DXY level and trend |
| `get_commodity_prices()` | Spot / ETF proxies for gold, silver, oil, gas, ags |
| `get_macro_view()` | Latest weekly macro view as structured text |

New tools are added via `change-request` issues. Library grows monotonically.

## The brain — pipeline overview

The strategy layer is a multi-stage Claude pipeline, not fixed rules. The `Strategy` interface is pluggable so we can run multiple variants in parallel.

### Daily pipeline (six stages, one per region)

0. **Macro context** — read the latest `macro/views/YYYY-WW.md` produced by the weekly macro agent (see below). This is the regime/sector lens through which today's news is interpreted.
1. **News & macro digest** — geopolitical headlines, forward economic predictions, sector moves, earnings calendar, economic releases. Output: a "what's happening today" thesis + themes / sectors / keywords.
2. **Universe filter** — narrow the broker's full tradable catalog (Alpaca for Tier 1, T212 for Tier 0/2) to candidates matching today's themes. Yields a few hundred tickers.
3. **Wide scoring** — score every candidate (cheap model, batched). Each ticker gets a predicted return and a class (strong-up / mild-up / flat / mild-down / strong-down).
4. **Deep analysis** — top ~15 candidates by absolute conviction get a deeper pass: news, technicals (RSI / MACD / ATR / volume), upcoming catalysts, company filings (SEC EDGAR / Companies House) if relevant. Returns `(upside_estimate, downside_estimate, conviction, rationale)`.
5. **Final selection & sizing** — Claude picks the final N trades with thesis, entry zone, optional stop, optional take-profit, and suggested allocation %. Hard rules apply: liquidity floor, max position size, ticker blacklist, ISA-eligibility filter (Tier 2 only).

### Weekly macro agent

Fires Sunday evening (~18:00 UK, before Asia Monday open). Long-horizon counterpart to the daily pipeline.

- **Cross-asset digest**: yields (2Y, 10Y, curve), DXY, oil, gold, VIX, credit spreads, sector ETF performance, week's economic releases, central-bank speakers, geopolitics
- **Regime read + sector view**: where are we in the cycle; per-sector bullish/neutral/bearish + why
- **Macro predictions with falsification criteria**: dated, specific, measurable (e.g., "10Y yield exceeds 5% by 2026-Q4 — falsified if it closes below 4.5% at any point before then")
- **Self-grading**: review last week's open predictions; mark proven / falsified / still-open; failed predictions → `macro/lessons.md`. Track record influences future predictions.

Output artifacts (append-only, version-controlled):
- `macro/views/YYYY-WW.md` — fresh snapshot each week, read by the daily pipeline as Stage 0
- `macro/predictions.jsonl` — every macro prediction with status
- `macro/lessons.md` — what the macro agent got wrong and why

This gives the system **two time-scale learning loops**: daily ticker predictions and weekly macro predictions. Both are evaluable artifacts; both have prediction-tracking and lessons; both feed each other.

### Company filings as a data source

- SEC EDGAR (US, free, no auth) and Companies House (UK, free) accessed via Claude Code's `WebFetch`
- Daily pipeline (Stage 4 deep analysis) can pull recent 10-K / 10-Q / 8-K / Form 4 when relevant
- Weekly macro agent can sample filings during earnings seasons to spot management-tone themes
- If filing access becomes a hot path, we promote it to a thin MCP server (`get_latest_filing(ticker, type)`); v1 uses raw `WebFetch`

### Prediction vs trade decoupling

## Prediction vs trade decoupling

- We **execute** only the top-N picks (capital-constrained)
- We **log** the full graded prediction set (~200 picks/day across all classes)
- End-of-day we measure: IC, top-vs-bottom-decile spread, conviction calibration, direction hit rate, symmetric-bias check (is the model better at up-calls than down-calls?)
- ~1000 graded predictions/week → statistical signal on the *model* in ~2 weeks, independent of actual trade P&L
- The five-class scheme gives a "flat" control group — when the model says "won't move" and the stock moves a lot, that's an information failure even when no trade was implied

## Self-improvement — three nested loops

- **Daily execution loop** — the five-stage pipeline above, per active strategy
- **Daily reflection loop** — after the close, log expected-vs-actual per trade and per prediction
- **Weekly evolution loop** — meta-agent reviews the past week's metrics per strategy, then promotes / demotes / spawns variants. Always consults the lessons file first so it doesn't repeat known-bad ideas

### Species A — evolve configs and prompts, not Python source

- Each strategy is defined by a config (universe filter, scoring prompt, deep-analysis prompt, risk parameters, position count) plus its system prompts
- The meta-agent edits *those*. It does **not** write Python production code
- Safe, simple, fast to ship

### `change_requests.md` — the agent → human code-change channel

- When the meta-agent thinks Python source needs to change ("I'd benefit from a new indicator," "the universe filter should support X"), it files an entry: date, requesting strategy, proposed change, rationale, expected benefit, status (open / accepted / rejected / implemented)
- The user reviews entries weekly, accepts or rejects, and implements accepted ones manually (with Claude Code as a coding partner)
- Solves "self-modifying code is scary" entirely — the AI proposes, the human disposes

### State files

- `strategies/<id>/config.yaml` + `strategies/<id>/prompts/*.md` — strategy definition (evolvable by meta-agent)
- `ledger.jsonl` — every paper trade, tagged by strategy_id, with entry/exit/P&L
- `predictions.jsonl` — every graded daily prediction, tagged by strategy_id, with predicted vs actual
- `decisions/<date>/<strategy_id>.json` — full Claude reasoning chain per day, per strategy
- `lessons.md` — strategy-level: what's been tried and why it failed (consulted by the meta-agent before proposing changes)
- `evolution.md` — every meta-agent change with rationale
- `change_requests.md` — agent → human code-change requests
- `macro/views/YYYY-WW.md` — weekly macro snapshot
- `macro/predictions.jsonl` — macro predictions with falsification criteria + status
- `macro/lessons.md` — macro-agent track record and self-corrections

## Universe — news-driven, not fixed

- We do **not** start from a fixed list like the S&P 500
- The universe filter step (Stage 2 of the pipeline) reads today's themes from Stage 1 and dynamically narrows the broker's full catalog
- Per-tier source:
  - Tier 0 / Tier 2 (live): T212's `/api/v0/equity/metadata/instruments` list, read-only — covers US + UK + EU
  - Tier 1: Alpaca's `/v2/assets` tradable list (US equities + ETFs only) — strategies aimed at non-US instruments skip Tier 1 entirely
- Catalog is cached weekly (rarely changes); filtering happens daily

## Two parallel pipelines — one per region

UK/EU markets open ~6.5 hours before the US, so a single pipeline run can't drive both. We run two regional pipelines with the same brain code but separate schedules.

| Step | UK/EU pipeline | US pipeline |
|---|---|---|
| Pipeline run (news, universe filter, scoring, picks) | 07:00 UK | 13:00 UK |
| Entry orders (5 min after open) | 08:35 UK | 14:35 UK |
| Exit orders (30 min before close) | 16:00 UK | 20:30 UK |

- Each pipeline does its own news/macro digest with region-appropriate sources
- Each universe-filters the broker catalog to its region
- Each has its own predictions set, positions, and P&L
- Strategies are tagged with `region: US / UK / EU / ALL`
- One reflection job after both markets close (~22:00 UK)
- One weekly macro agent run on Sunday evening (~18:00 UK)
- One weekly strategy-evolution run on Saturday morning (~09:00 UK)

The 30-min exit buffer is deliberate: the final 30 min of US trading has materially higher volatility (closing-auction imbalances release 10 min before close, MOC orders pile up, volume picks up 50–100% vs midday). The 30 min also gives Tier 2 (live) time to actually tap the approve button when the notification fires.

Nine cron jobs total in GH Actions: 2 daily pipelines + 2 entries + 2 exits + 1 daily reflection + 1 weekly strategy evolution + 1 weekly macro agent.

## Runtime — Claude Code in GitHub Actions, on Max 20x

- GH Actions cron, weekdays only
- Workflow installs the Claude Code CLI, authenticates with a long-lived `CLAUDE_CODE_OAUTH_TOKEN` (generated once via `claude setup-token`, stored as a repo secret) tied to the user's Claude Max 20x plan
- Each pipeline stage runs as `claude -p "..."` with the right MCP servers attached
- We build MCP servers for broker-specific bits (Alpaca, T212 read-only, our ledger/registry/lessons)
- We get Claude Code's full toolset for free (Bash, Read, Write, WebFetch, agent spawning)
- Cost: effectively zero (covered by Max 20x). Three caveats: OAuth token rotation, rate limits, and the fact that CI use of a personal Max plan is supported but intended for research/paper — fine for what we're doing
- Model assignment: **Haiku** for wide scoring (cheap, fast), **Sonnet** for deep analysis, **Opus** for final selection on live strategies

## Trading212 API — what we can and can't do (for Phase 2)

Confirmed from T212's API Terms (last updated 2025-10-17):

- ✅ Read positions, balance, history, instrument metadata (§3.3)
- ✅ Place Market, Limit, Stop, Stop-Limit orders for live ISA accounts (§3.4)
- ❌ Algorithmic trading without human intervention (§4.2.a) — defined in §11 as orders placed "with limited to no human intervention"
- ❌ Scalping (§4.2.b.2)
- ❌ Mass / high-speed automated entry without prior written consent (§6.6)
- ⚠️ Market data is **not real-time** (§7.1.b) — fine for daily cadence, but we shouldn't use T212 for the scan data

## Tentative architecture

- Python project, monorepo
- Core modules (broker-agnostic):
  - `pipeline/` — the six-stage daily loop driver, plus the weekly macro agent
  - `strategy/` — registry of active strategies; loads configs + prompts
  - `data/` — market data (Alpaca data API), news (Finnhub free + Alpaca News + WebSearch), macro (WebSearch + WebFetch), filings (WebFetch → SEC EDGAR / Companies House)
  - `executor/` — pluggable: `ShadowExecutor` (Tier 0), `AlpacaPaperExecutor` (Tier 1), `Trading212ApproveExecutor` (Tier 2, deferred)
  - `notify/` — alerts for fills, errors, daily summary, evolution recommendations
  - `state/` — readers/writers for ledger, predictions, decisions, lessons, macro views
  - `meta/` — daily reflection, weekly strategy evolution, weekly macro agent
- MCP servers (custom): one each for Alpaca, T212 read-only, our state files. Filings access uses `WebFetch` in v1; promoted to a thin MCP if it becomes hot.
- GH Actions workflows (9 cron jobs):
  - `pipeline-uk-eu.yml` (07:00 UK weekdays)
  - `entry-uk-eu.yml` (08:35 UK weekdays)
  - `exit-uk-eu.yml` (16:00 UK weekdays)
  - `pipeline-us.yml` (13:00 UK weekdays)
  - `entry-us.yml` (14:35 UK weekdays)
  - `exit-us.yml` (20:30 UK weekdays)
  - `daily-reflection.yml` (22:00 UK weekdays)
  - `weekly-strategy-evolution.yml` (Sat 09:00 UK)
  - `weekly-macro.yml` (Sun 18:00 UK)
- Secrets: `CLAUDE_CODE_OAUTH_TOKEN`, `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `FINNHUB_API_KEY`, later `T212_API_KEY`, notification tokens

## Decisions locked in

| Item | Choice |
|---|---|
| Execution model | 3 tiers — Shadow + Alpaca-paper active; Live deferred until a strategy is chosen |
| Tier 1 broker | Alpaca paper (multiple accounts supported) |
| Tier 2 broker (deferred) | Trading212, UK Stocks & Shares ISA |
| Live execution mechanism (deferred) | Option B (one-tap approval per batch) — exact channel TBD when activated |
| Live starting capital (deferred) | £100, scaling up on demonstrated track record |
| Paper capital per strategy | £10k default, configurable per strategy |
| Position sizing | LLM-decided per pick within hard constraints; sizing style is part of strategy identity |
| Runtime | GitHub Actions + Claude Code CLI on Max 20x |
| Cadence | Buy after open, sell 30 min before close — same-day round trip |
| Regional split | Two parallel pipelines (UK/EU at 07:00 UK, US at 13:00 UK) |
| Strategy framework | Pluggable, LLM-driven, 5-stage pipeline |
| Predictions per day | ~200 graded across 5 directional classes |
| Self-improvement | Species A (evolve prompts/configs); code changes via `change_requests.md` |
| Universe | News-driven filter of broker's full catalog, not a fixed list |
| Promotion between tiers | Agent-recommended + user-approved (up); automatic on decay (down) |

## Dashboard — GitHub Pages

A static site rebuilt after every daily reflection. Read-only window into the bot's work. Hosted free on GH Pages, served from `docs/` on the main branch.

**Layout**: sidebar lists strategies in two collapsible groups — **Active** (currently running) and **Archived** (retired but with history). Click a strategy in the sidebar to focus the centre panel.

Centre panel for a focused strategy:
- Summary stats up top (total P&L, IC, hit rate, decile-spread)
- Equity curve chart
- A toggle below: **Executed** | **Uncommitted candidates**
  - **Executed** — rows from `ledger.jsonl`. Date, ticker, entry, exit, P&L, exit reason. Each row expands to show the entry thesis and a post-hoc outcome analysis (auto-generated during daily reflection).
  - **Uncommitted candidates** — rows from `predictions.jsonl` where `was_traded=false`. The wider prediction set the strategy scored but didn't execute. Same expandable fields. This is the IC / decile-spread fodder — lets you eyeball whether the strategy ranks well even on picks it didn't trade.

**Tech**: deliberately minimal — vanilla HTML, Pico.css for layout, Alpine.js for sidebar/toggle reactivity, JSON data file. No build pipeline, no framework lock-in. Python script (`src/trading_bot/dashboard/build.py`) reads state files and writes `docs/data.json`.

**Build trigger**: a step in the daily reflection workflow regenerates `docs/data.json` and commits it back to main; GH Pages serves automatically.

## Notifications

In-repo files (`decisions/`, `ledger.jsonl`, `predictions.jsonl`, `macro/views/`) are the source of truth and audit trail. Notification channels just surface things to the user; nothing is lost if a channel fails.

| Output type | Channel |
|---|---|
| Daily picks + EOD P&L summaries | Email |
| Weekly macro view | Email |
| Weekly evolution report | Email |
| Change requests (agent → human code changes) | GitHub Issue (label `change-request`) |
| Promotion proposals (Tier 0 → Tier 1, or eventually → Tier 2) | GitHub Issue (label `promotion`) |
| High-urgency errors (cron failed, OAuth expired, broker API down) | GitHub Issue + email |
| Tier 2 fill notifications & approve actions (deferred) | Telegram, activated when Tier 2 is activated |

Email transport: **Brevo** free tier (300/day, single-sender verification — no domain needed). One secret (`BREVO_API_KEY`), sender + recipient addresses configured via secrets.

## Build plan — skeleton-first, layered waves

All 5 strategies are seeded in the registry from day one (configs + prompts exist) but only activate wave-by-wave. This preserves "wider seed" while keeping each wave small enough to validate before the next.

| Wave | Adds | Risk retired |
|---|---|---|
| **1. Skeleton** | `control-rule-based` only. US region only. Shadow tier only. One cron (US daily pipeline). Tools: `get_universe`, `get_history`. Email summary. GH Pages dashboard. | End-to-end loop, cron, file state, executor abstraction, email reporting, prediction grading, dashboard rendering — all proven before spending one LLM token |
| **2. First LLM strategy** | `momentum-trader`. Alpaca paper executor (Tier 1). Bracket orders for stops/take-profits. Tools: `get_technicals`, `get_recent_news`. Daily reflection cron. | LLM pipeline, MCP tool calls, real paper fills |
| **3. Breadth — equity + first cross-asset** | `mean-reverter`, `news-reactive`, `sector-rotator`. Tools: `get_filing_summary`, `get_earnings_info`, `get_insider_trades`, `get_sector_strength`, `get_etf_relative_strength`, ETF universes (`us_etfs_sector`). | Multiple LLM strategies in parallel; ETF trading; tool library coverage |
| **4. Macro + cross-asset** | Weekly macro agent. Strategies: `macro-aligned`, `bond-cycle`, `commodity-momentum`. Tools: `get_macro_view`, `get_yield_curve`, `get_credit_spreads`, `get_dollar_index`, `get_commodity_prices`. | Two-time-scale learning loops + cross-asset universe |
| **5. Regional expansion** | UK/EU pipeline (duplicate US pipeline with adjusted schedule + region tag) | Cross-market handling |
| **6. Self-improvement** | Weekly evolution agent. `change_requests` GH Issues workflow. Lessons/evolution files wired in. | Closed-loop self-improvement |

Each wave is shippable on its own. We pause between waves to validate before adding the next.

## Still open (non-blocking)

- **Ticker blacklist** — start empty `[]`; add entries when something specific warrants it
- **Promotion criteria** — concrete thresholds (IC, hit rate, Sharpe, runway length) for Tier 0 → Tier 1 → Tier 2. Defined in Wave 6 (evolution agent) when we have real data to calibrate against
- **ISA US-stock eligibility filter list** — only needed when Tier 2 activates (deferred)

### Deferred until a strategy is chosen for live graduation

- Approve-action mechanism for Tier 2 (Telegram inline button + serverless endpoint, web page, iOS Shortcut, etc.)
- Tier 2 capital scale-up schedule
- T212 API key provisioning, secret rotation

## Open sub-discussions to schedule

1. Position sizing, risk rules, schedule
2. The Phase 2 approve-action mechanism
3. Data sources for news and macro
