# Newspaper Delivery Plan

Tracking the work to ship the multi-stage newspaper pipeline. Check items off as they land. The four pages (Dashboard / News / Macro / Evolution) share a typography stack, masthead pattern, font-picker, and shell — all driven by `docs/assets/style.css` and `src/trading_bot/dashboard/pages.py`.

## Cross-cutting constraints

- **Cost is not a concern** — Max 20x plan. Lean Sonnet where structure matters, Haiku where parallelism matters, no token-budget tradeoffs.
- **Full articles have NO word cap.** Briefs are constrained (~80-100 words for the front-page grid). Full articles run as long as the topic warrants — short when terse, deep when the story is.
- **Images are real, not stock.** Writer agents do image search (Claude WebFetch against Google/Bing image results) for the best-fitting image per article. Hot-linking is fine for personal use.
- **LLM emits JSON, Python emits HTML.** Never trust the LLM to produce final HTML — always structured JSON with markdown bodies that Python templates into HTML.
- **Bot-summary stays compressed** (~150 words) regardless of how big the publication gets — protects strategy prompt budgets.

---

## Phase 1 — Foundation ✅ (in progress)

- [x] `docs/assets/style.css` — single shared stylesheet (cream paper, accents, font-picker hooks, all section variants, full-article page, archive index)
- [x] `src/trading_bot/dashboard/pages.py` rewritten — `_shell()` injects nav + font picker + Google Fonts + shared CSS; supports per-edition subdirectories; renders index pages
- [x] `src/trading_bot/state/predictions_log.py` — JSONL persistence for News + Macro + Evolution predictions, with Prediction dataclass, append/read/mark-graded helpers
- [x] `src/trading_bot/meta/grade_predictions.py` — daily grader: walks open predictions whose target_date has passed, scores each via Haiku in parallel using a cross-asset snapshot, mutates status to proven / partial / falsified / still-open
- [x] `pipeline.py` mode `grade-predictions` wired
- [x] `.github/workflows/grade-predictions.yml` — cron-job.org-triggered workflow that runs `pipeline grade-predictions` daily
- [x] Add `grade-predictions` to `scripts/setup_cron_jobs.py` SCHEDULES (05:00 UTC, every day — just before the morning brief)
- [x] Commit + push the Phase 1 work

---

## Phase 2 — News pipeline (multi-stage agents)

The six-stage architecture. Each agent emits JSON; Python composes the HTML.

### 2A — Stage 1: Discovery agent ✅

- [x] `src/trading_bot/meta/news/discovery.py`
- [x] Sonnet + WebSearch — searches across every topic class (markets, world, tech & science, climate, health, sport, culture, beyond the tape), not just markets
- [x] Returns ~30-50 `Candidate` records: `{title, one_line, suggested_section, importance_hint, source_hints[]}`
- [x] Seeded with Alpaca News + yfinance broad-market headlines so the markets baseline is guaranteed even if WebSearch underperforms
- [x] Graceful fallback to seed-only on LLM failure

### 2B — Stage 2: Triage agent ✅

- [x] `src/trading_bot/meta/news/triage.py`
- [x] Haiku × 6 parallel — one call per candidate
- [x] Each scores 1-10 + writes a one-line angle + key facts
- [x] Filters survivors to top-N by score (no fixed cap; whatever the publisher needs)

### 2C — Stage 3: Publisher agent ✅

- [x] `src/trading_bot/meta/news/publisher.py`
- [x] Sonnet — organises survivors into sections (Front, Markets, World, Tech, Beyond, plus variable sections like Sport/Climate/Health when warranted)
- [x] Assigns bylines (persona-based), picks the lead story, sets the AI-generated humorous masthead subtitle
- [x] Emits the newspaper-plan JSON: section list, lead slug, masthead lines

### 2D — Stage 4: Brief writers (parallel) ✅

- [x] `src/trading_bot/meta/news/brief_writer.py`
- [x] Haiku × 6 parallel — one brief per story (~80-100 words, conversational, no jargon)
- [x] Includes byline, headline, body markdown, sources

### 2E — Stage 5: Full-article writers (parallel) ✅

- [x] `src/trading_bot/meta/news/article_writer.py`
- [x] Sonnet × 6 parallel — full article, **no word cap**, written to the topic
- [x] Each writer also runs an image search (Claude WebFetch / WebSearch) for the most fitting hero image — returns image_url + caption + credit
- [x] Includes inline callouts, sources block, related-articles slugs

### 2F — Stage 6: Bot-summary compressor ✅

- [x] `src/trading_bot/meta/news/bot_summary.py`
- [x] Haiku — takes the full assembled edition, produces a tight ~150-word `state/daily_news/YYYY-MM-DD.bot.md` for strategy prompts
- [x] Preserves Risk tone / Themes / Sector lean / Watchlist / Key data structure

### 2G — Trading floor section ✅

- [x] `src/trading_bot/meta/news/trading_floor.py`
- [x] Reads `state/ledger.jsonl` for yesterday's strategy P&L
- [x] Haiku × 3 parallel — writes prose pieces (winner, loser, "quieter movers") — news-style, not numbers-style
- [x] Appended into the rendered edition by the assembler (publisher must NOT include it)

### 2H — Desk's calls section ✅

- [x] `src/trading_bot/meta/news/desks_calls.py` — reads `state/predictions/news.jsonl` for graded verdicts
- [x] Generates fresh predictions (Sonnet) per horizon (tomorrow / week / month) — claim + falsification_criteria + conviction
- [x] Persists each via `append_prediction()` so the grader can score them later
- [x] Renders "Marking the homework" sub-section from the most recently graded predictions

### 2I — Assembly + sub-page generation ✅

- [x] `meta/news/render.py` — `render_news_edition()` writes `docs/news/YYYY-MM-DD/index.html` + `docs/news/YYYY-MM-DD/{slug}.html`
- [x] `pages.render_news_edition()` — thin wrapper that injects the shared shell
- [x] News archive index updated to include both legacy flat-file and structured directory editions
- [x] Wires images, "In one sentence" callouts, sources block, "related in this edition" cross-links
- [x] `news_url_for()` prefers the directory URL when the structured edition exists

### 2J — Orchestration ✅

- [x] Rewrote `src/trading_bot/meta/daily_news.py` — six stages in order, with stages 4 running its four parallel sub-stages via an outer ThreadPoolExecutor
- [x] `.github/workflows/daily-news-brief.yml` — bumped to 30-min job timeout
- [x] Pipeline state dumped to `state/daily_news/{date}.pipeline.json` for debugging / re-render
- [x] Headlines markdown still written to `state/daily_news/{date}.md` for archive compat
- [ ] Smoke-test on a single live edition (deferred — needs OAUTH token + WebSearch budget; will run via `workflow_dispatch` manually)

---

## Phase 3 — Macro pipeline ✅

Reuses the Phase 2 scaffolding (Brief, FullArticle, news article-writer) with desk-based structure rather than news-section-based.

- [x] `src/trading_bot/meta/macro_v2.py` (single module, mirrors `evolution_v2.py` shape)
  - [x] Snapshot — reuses `meta.macro._gather_snapshot()` for yield curve / credit / DXY / sectors / commodities
  - [x] Publisher (Sonnet) — desks: Editorial / Rates / FX / Credit / Sectors / Regions / Risk; desk-specific bylines
  - [x] Brief writers + Article writers — reuses the news pipeline modules (`Brief`, `FullArticle` are shape-compatible)
  - [x] Predictions (Sonnet) — horizons: month / quarter / 6mo / year / multi-year; persisted via unified `predictions_log` (source="macro")
  - [x] "For the strategies" callout (Haiku) — dark-callout bias signals + watchlist
  - [x] Renderer — writes `docs/macro/YYYY-W##/index.html` + per-piece subpages (reuses news article subpage renderer)
- [x] `meta/macro.py` now calls `run_macro_v2()` after the existing markdown view is produced (non-fatal on failure)
- [x] `weekly-macro.yml` — bumped to 40-min job timeout
- [ ] Macro weekly email with link to the front-page URL (deferred to Phase 6)
- [ ] Sector spotlight long-form pieces (deferred — publisher already allows Sectors desk features; dedicated spotlight is overkill for now)

---

## Phase 4 — Evolution restyle (editorial format) ✅

The action engine in `meta/evolution.py` already exists; this added the editorial layer.

- [x] `src/trading_bot/meta/evolution_v2.py` (single module rather than a package — small enough)
  - [x] Editorial intro (Sonnet) — "this week's read" from snapshot + applied actions
  - [x] Per-strategy report cards (Haiku × N parallel) — `{headline, what_worked, what_didnt, lessons, going_forward, config_changes}`
  - [x] Render layer — slate table + per-strategy cards + decisions chips row
- [x] `meta/evolution.py` calls `build_and_render_evolution()` after the action engine runs (non-fatal on failure)
- [x] `weekly-evolution.yml` no changes needed (existing workflow already calls run_weekly_evolution)
- [ ] Evolution weekly email with link (deferred to Phase 6 polish)

---

## Phase 5 — Dashboard restyle

CSS + small HTML edits to the existing Alpine app. No new agents, no new state.

- [ ] Edit `docs/index.html` — remove inline CSS, link to `assets/style.css`
- [ ] Apply masthead-strip pattern (the slim dashboard variant)
- [ ] Inject font picker into the Alpine app
- [ ] Numbers switch to sans + tabular-nums everywhere (overview cards, strategy cards, detail panel, trades table)
- [ ] Confirm strategy detail panel keeps Chart.js but matches the typography
- [ ] Verify all five pages (Dashboard / News / News archive / Macro / Evolution) have visually consistent shells

---

## Phase 6 — Polish ✅

- [x] Macro and Evolution emails — `render_macro_email()` + `render_evolution_email()` in `notify/email.py`, wired into `meta/macro_v2.py` and `meta/evolution.py`
- [x] Yesterday's predictions auto-grading already runs at 05:00 UTC (changed from 23:30 in Phase 1)
- [x] Dashboard "global overview" tile — verified surviving the restyle (broker + shadow overview-card pair in docs/index.html, scoped CSS in style.css)
- [x] "Today's question" — added to `NewsPlan` as a structured field, publisher prompted to generate it, rendered as a small-caps line below the masthead subtitle, passed to the bot-summary compressor as framing context
- [x] Strategy reading-time badges — `_estimate_read_minutes()` (225 wpm, markdown-noise-stripped); shown on every Read-on / Read-the-full-article link (lead + briefs) and in the article subpage meta line. Applied to both news and macro pipelines.
- [x] Smoke-test the full stack — ran both `daily-news-brief` and `weekly-macro` via `workflow_dispatch` on 2026-05-18:
  - **Daily-news:** clean. 19 pieces published across Front / Markets / World / Health, 18 articles with hero images, 3 trading-floor prose pieces, 6 predictions persisted, today's question rendered, email sent.
  - **Weekly-macro:** partial. v1 markdown landed cleanly; v2 publisher + predictions Sonnet calls timed out at 240s → bumped to 600s in `meta/macro_v2.py`, and added `docs/macro` to the workflow's `git add` (was missing). Fix in commit `9997609`. Re-run to verify.

---

## Phase 7 — Archive trimming (storage hygiene) ✅

The repo is the source of truth and is cloned by every GH Actions run. Daily news editions could each be ~250 KB of HTML (front page + ~20 sub-articles + images metadata); 1000 days × 250 KB ≈ 250 MB just for news. We trim older editions into compressed tarballs so the working tree stays slim.

- [x] `scripts/archive_old_editions.py` — walks `docs/news/YYYY-MM-DD/` and `docs/macro/YYYY-W##/`, anything older than **90 days** gets bundled into `state/archive/news-YYYY-MM.tar.gz` / `state/archive/macro-YYYY.tar.gz`, originals removed from `docs/`. Handles both directory-form (Phase 2/3) and legacy flat-file editions. Idempotent — merges into an existing tarball if you re-run on the same month.
- [x] Tar files store each edition as its top-level entry (file or dir) so a single restore command rehydrates a month if needed
- [x] `docs/news/index.html` + `docs/macro/index.html` archive lists read from `state/archive/manifest.json` so trimmed editions still appear as "archived (compressed)" links pointing at the raw GitHub blob URL of the tarball
- [x] `.github/workflows/archive-trim.yml` — weekly cron (Sunday 04:00 UTC, before grade-predictions/news brief), commits the bundled tarballs + removed-files diff. Supports `dry_run=true` input for safe preview.
- [x] `.gitattributes` — marks `state/archive/*.tar.gz` as `binary linguist-generated=true -diff` so they don't bloat GitHub's diff renders or language stats
- [x] `setup_cron_jobs.py` SCHEDULES — added `archive-trim` (Sunday 04:00 UTC)
- [x] One-time backfill skipped initially (nothing's older than 90 days yet); revisit if Pages storage approaches the 1 GB soft limit

### Future option (only if 90-day live window itself grows too large)

- [ ] Move tarballs out of git entirely — push to Cloudflare R2 (~free for personal scale), keep only the manifest in repo. Defer until tarballs sum to >500 MB.

---

## Phase 8 — Trading-quality improvements

The framework is solid; what's under-developed is the bit between
"find a candidate" and "place an order", and the post-trade learning
loop. These changes target risk-adjusted P&L, not infrastructure.

### 8A — Volatility-aware position sizing

- [ ] Add `target_daily_risk_pct` config field (default 1.0% of capital)
- [ ] Post-process LLM picks: `position_£ = target_risk_£ / (ATR_pct)`,
      then clamp to `max_position_pct` so a low-vol name doesn't blow
      past the cap
- [ ] Tell the LLM in the prompt that we vol-adjust afterwards, so the
      model focuses on direction/conviction rather than gaming `allocation_pct`

### 8B — Pre-trade FX cost gate (hard rule)

- [ ] Backstop the LLM's per-candidate cost line: after picks are
      parsed, look up `predicted_return_pct` for each pick and drop
      any where `|predicted| < 2 × round_trip_cost_pct`
- [ ] Log dropped picks with reason so the evolution agent can see
      whether the LLM repeatedly tries trades the gate would reject

### 8C — Earnings-calendar gating

- [ ] Add `get_earnings_calendar(tickers)` to `tools/` returning next
      earnings date per ticker (yfinance `Ticker(t).calendar`)
- [ ] Pre-fetch when assembling candidates; surface "earnings in 24h"
      as a per-candidate flag in the LLM prompt
- [ ] Strategy LLMs can avoid or size down; rule-based strategy gets
      a hard filter (`skip_if_earnings_in_days <= 1`)

### 8D — Trailing stops on winners ✅

- [x] `executor/alpaca_trail.py` — scans positions per slot; positions
      in profit ≥ `activation_pct` get the bracket-stop child PATCHed
      upward to `current_price × (1 - trail_pct/100)`. Doesn't lower
      stops; non-fatal on PATCH 422.
- [x] `executor/t212_trail.py` — T212's Invest API supports standalone
      STOP orders but not PATCH; trailing implemented as cancel-and-
      replace (DELETE existing stop, POST new stop with the higher
      stopPrice). Negative quantity = sell. Skipped on positions whose
      existing stop is already at or above the new target.
- [x] `scripts/midday_trail.py` runs both brokers; `--brokers alpaca`
      / `--brokers t212` switches available.
- [x] `.github/workflows/midday-trail.yml` — workflow_dispatch only,
      exposes both broker secret pairs.

### 8E — Real per-trade LLM reflection

- [ ] Replace templated `outcome_notes` / `risks_observed` text with a
      Haiku call per closed trade
- [ ] Inputs: pre-trade thesis (already stored), day's high/low/close,
      day's ticker news, the day's P&L, exit reason
- [ ] Fires from each executor's exit path; non-fatal on failure
      (templated text stays as the fallback)

### 8F — Daily portfolio-loss kill switch

- [ ] `state/halt.json` file with `{halted: bool, reason: str, set_at: iso}`
- [ ] Entry pass checks at start: if yesterday's live-tier P&L
      < -3% of total live capital, write halt.json and skip all
      live-tier entries today
- [ ] Manual unhalt by deleting / editing the file (committed via PR)
- [ ] Halt status surfaced on the dashboard + in the daily email

### 8G — Strategy-similarity detection (evolution agent add-on)

- [ ] In `evolution_v2._build_strategy_report_prompt`, compute pairwise
      Pearson correlation of daily P&L across active strategies over
      the trailing 14 days
- [ ] If any pair > 0.85, surface in the agent's editorial intro and
      in each affected strategy's report card
- [ ] LLM hypothesises *why* (overlapping universes? Same technical
      signal? Same news source?) and may suggest a `tune`/`spawn-variant`
      to differentiate

---

## Phase 10 — Trail re-entry cost + audit follow-ups

Triggered by the post-audit observation that the trailing-stop mechanism
doesn't account for re-entry stamp duty when a position fires mid-day
and the strategy re-picks the same name tomorrow.

### 10A — Stamp-duty-aware trail re-entry

- [ ] `state/trail_exits.py` — append a record every time an exit
      closes via `stop` reason (covers the trail's trigger). Stores
      ticker, region, strategy_id, exit_date in `state/trail_exits.jsonl`.
- [ ] Hook into Alpaca + T212 + shadow exit paths so any stop-fire is
      logged.
- [ ] `strategy/sizing.adjust_picks` reads recent trail-exits and
      doubles the round-trip cost estimate for any pick whose ticker
      was trailed out in the last 3 days. The existing 2× gate then
      becomes effectively 4× for re-picks.
- [ ] LLM strategy prompt grows a "recently trailed out" section so
      the model sees these tickers explicitly and can avoid re-picking
      unless conviction is high enough.
- [ ] Instrument-aware trail thresholds in `t212_trail.py`:
      - LSE non-ETF: activation 1.8%, trail 0.6% (so realised gain ≥
        ~1.2%, clears the 0.5% re-entry stamp duty by a margin)
      - LSE ETF / AIM / non-UK: defaults (1.0% / 0.8%)
      - Loaded from `state/t212_instruments.json` cache.

### 10B — Evolution-agent prompt enrichment

The agent currently sees aggregated metrics + missed-movers +
similarity pairs. The audit identified 10 additional inputs that
would materially improve its promote/demote/tune calls.

- [ ] Fee-as-share-of-gross per (strategy, region) row — flags
      strategies whose net-zero P&L is fees-driven rather than
      signal-driven.
- [ ] Cost-gate drop rate per strategy — `sizing.PickAdjustment`
      records get persisted to `state/pick_adjustments/{date}.{sid}.jsonl`
      and aggregated for the prompt.
- [ ] Falsifiable-call verdict rates per source (news / macro /
      evolution) over the trailing 4 weeks.
- [ ] Earnings-gate hit counts per strategy — track in
      `state/earnings_gate/{date}.{sid}.json`.
- [ ] Per-strategy sector concentration (top sectors by traded
      notional over the window).
- [ ] Cross-region divergence — pre-compute `divergent_strategies`
      where `|us_pnl_pct − ukeu_pnl_pct| > threshold`.
- [ ] Trail activation rate per strategy (from
      `state/trail_exits.jsonl` from 10A).
- [ ] Kill-switch history — append to `state/halt_history.jsonl` on
      every set/clear; surface count + reasons.
- [ ] News-brief subtitles for the trailing 7 days as regime context.
- [ ] Parent's `deep_analysis.md` prompt passed for `spawn-variant`
      decisions so the proposed addendum isn't written blind.

### 10C — Remaining bug fixes

- [ ] `t212_trail.py:155-169` — recover an unprotected position when
      `_cancel_order` succeeded but `_submit_stop` failed by re-posting
      a fallback stop at the old level.
- [ ] `alpaca_trail.py:167` — guard against sub-$1 stocks where
      rounding to 2dp can equal the current price.
- [ ] `evolution.py` — split `applied` vs `skipped` actions in the
      prompt instead of mixing them.
- [ ] `halt.py:62` — robust `HaltRecord` parsing via `dataclasses.fields()`
      with explicit defaults.
- [ ] `sizing.py:165` — re-order clamps so max applies last.
- [ ] `alpaca_paper.py:582` — verify slippage-tolerance comment matches
      the code direction.
- [ ] `evolution.py` — expose `PROMOTION_MIN_PNL_GBP` as a tunable,
      currently rejecting net-zero strategies hard.
- [ ] `evolution.py` — separate `PROMOTION_MIN_IC_LOWER` from
      `PROMOTION_MIN_IC` so the lower-bound check can be relaxed
      independently.

### 10D — UI follow-ups

- [ ] Dashboard missed-movers panel — surface
      `state/missed_movers/<today>.<region>.json` directly to the user.
- [ ] Fees breakdown human-readable labels (FX in / UK stamp / FR FTT
      / etc.) instead of raw `fees_breakdown` dict keys.
- [ ] News-edition `.grid-3 .brief` font sizes shrink below 760px.
- [ ] `predictions_archive` conviction filter handles both string
      ('high'/'medium'/'low') and float (0.0-1.0) representations.
- [ ] `evolution.html` per-strategy four-quadrant cards stack cleanly
      on mobile.
- [ ] "Net of fees" indicator on the overview-card P&L.
- [ ] Dashboard "today" boundary uses the latest exit_date in the
      data, not the UTC date.
- [ ] `macro/editions.json` dedups duplicate slugs from re-runs.

---

## Phase 9 — Polish + observability

Nice-to-haves and rough edges. Won't move the P&L needle as much as
Phase 8 but are worth doing once Phase 8 is bedded in.

### 9A — A/B test framework with confidence intervals ✅

- [x] `_meets_promotion()` now computes the lower 95% CI bound on IC
      via Fisher's z-transform and requires it to clear `PROMOTION_MIN_IC`,
      not just the point estimate. With n=10 graded predictions the
      lower bound on IC=0.07 is -0.59 → no promotion. With n=100 and
      IC=0.06 the lower bound is -0.14 — still no promotion. Real
      signal must be both meaningful AND sustained.

### 9B — Image-relevance check on article hero pick ✅

- [x] Second-pass Haiku verifier in `article_writer._verify_hero_images()`.
      Runs on every article that came back with an `image_url`, asks for
      yes / borderline / no given the headline + caption + first
      paragraph. Drops the image on no / borderline. Fans out parallel.
- [x] WebFetch allowed for the verifier so it can disambiguate by
      loading the URL when caption + filename aren't enough.

### 9C — De-duplicate stories across editions ✅

- [x] Publisher prompt now includes a trailing-5-days "recent leads"
      block — date, headline, slug. The publisher is explicitly told
      to relegate a story arc that's already had its day on the front.
- [x] `_recent_leads_context()` reads from `state/daily_news/*.pipeline.json`
      so it survives archive-trim (state isn't moved, only docs/).

### 9D — Sector + factor exposure analytics on the dashboard ✅

- [x] `tools/sectors.py` — cached yfinance sector lookup at
      `state/ticker_sectors.json`. Lazy fetch on misses, ~150ms
      throttle, persists.
- [x] `dashboard/build._build_sector_exposure()` computes per-sector
      notional GBP across currently-open live-tier positions.
- [x] Dashboard renders a stacked-bar of sector exposure above the
      strategy grid, with a colour-coded legend. Hash-based palette so
      every sector gets a deterministic colour across runs.
- [ ] Factor exposure (momentum / quality / value) — deferred; not
      enough data per name yet to do this cleanly.

### 9E — Mobile-responsive dashboard ✅

- [x] New `@media (max-width: 760px)` block in `style.css` — tighter
      paper padding, single-column overview cards, single-column
      strategy grid, stacked detail header, horizontal-scroll trades
      table, smaller drop caps, single-column lead-body. App nav and
      font picker stack cleanly.
- [x] Newspaper / macro pages already wrapped via the existing 900px
      breakpoints; this round was dashboard-specific.

### 9F — Bot-health alerts ✅

- [x] `scripts/health_check.py` — checks workflow failures (last 24h),
      ledger staleness, predictions log corruption, kill-switch state,
      docs/news + docs/macro directory growth.
- [x] `.github/workflows/health-check.yml` runs daily, sends a summary
      email only when findings exist. Severity tagging (error / warning
      / info) on every finding. Email subject icon reflects the worst
      severity present.

### 9G — Tax / ISA tracking (live-tier only) — deferred until live

Not relevant until live trading starts. When live: track CGT base
cost per trade for taxable accounts; track ISA contribution /
withdrawal events; emit an annual statement that maps to HMRC's CG30
form layout.

### 9H — Push notifications on trade events — deferred to live UX

Lives with the "Phase 2 / Option B" approval UX in the original
notes — Telegram / iOS-Shortcut bridge for entry placed, exit
fired, kill switch tripped. Implemented alongside the live
approval flow when the time comes.

### 9I — Searchable historical predictions ✅

- [x] `dashboard/predictions_archive.py` — reads every row from
      `state/predictions/*.jsonl`, renders to `docs/predictions/index.html`.
- [x] Client-side filter by source / horizon / status / conviction;
      live count of matching rows.
- [x] "Verdict rate by conviction" stats block at the top — does
      'high conviction' actually mean a higher proven rate? Now
      visible at a glance.
- [x] Hooked into `pages.rebuild_all_pages()` so it refreshes on every
      pipeline run.

---

## Known unknowns / risks

- **Claude Code WebSearch availability** — discovery + image search both need it. If restricted, fall back to seeded RSS feeds (Reuters / Bloomberg / FT) for news discovery and use the existing yfinance news + Alpaca News + curated UK proxies (already wired) as the discovery substrate.
- **Workflow timeout** — multi-stage pipeline could push daily-news to 20-30 min. Bump `_DEFAULT_TIMEOUT` further if needed and confirm the workflow's job timeout is generous.
- **Image hot-linking** — when an upstream site changes URL, we get a 404. Accept this; later iteration can mirror images into `docs/news/{date}/images/`.
- **Prediction grading false positives** — the grader is set to be conservative on "proven". Watch the first week of verdicts for noise; tighten the prompt if needed.

---

## File map (where everything lives)

```
src/trading_bot/
  state/
    predictions_log.py     ← Phase 1 ✅
  meta/
    grade_predictions.py   ← Phase 1 ✅
    daily_news.py          ← Phase 2 (rewrite)
    news/
      discovery.py         ← Phase 2A
      triage.py            ← Phase 2B
      publisher.py         ← Phase 2C
      brief_writer.py      ← Phase 2D
      article_writer.py    ← Phase 2E (with image search)
      bot_summary.py       ← Phase 2F
      trading_floor.py     ← Phase 2G
    macro_v2/              ← Phase 3
    evolution_v2/          ← Phase 4
  dashboard/
    pages.py               ← Phase 1 ✅ (rewritten)
docs/
  assets/
    style.css              ← Phase 1 ✅
  index.html               ← Phase 5 (restyle)
  news/
    index.html             ← Phase 1 ✅ (basic) → Phase 2I (rich)
    YYYY-MM-DD/
      index.html           ← Phase 2I
      {slug}.html          ← Phase 2I
  macro/
    YYYY-W##/
      index.html           ← Phase 3
      {slug}.html          ← Phase 3
  evolution.html           ← Phase 4
state/
  predictions/             ← Phase 1 ✅
    news.jsonl
    macro.jsonl
    evolution.jsonl
  archive/                 ← Phase 7
    news-YYYY-MM.tar.gz
    macro-YYYY.tar.gz
    manifest.json
scripts/
  archive_old_editions.py  ← Phase 7
DELIVERY_PLAN.md           ← this file
```
