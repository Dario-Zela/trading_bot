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

## Phase 9 — Polish + observability (lower priority, queued for later)

Nice-to-haves and rough edges. Won't move the P&L needle as much as
Phase 8 but are worth doing once Phase 8 is bedded in.

### 9A — A/B test framework with confidence intervals

- [ ] When the evolution agent fires a `promote` action, compute a
      proper confidence interval on the IC delta (Welch's t-test on
      the strategy's per-prediction return vs control's). Promote
      only when the lower CI bound clears the promotion threshold.
- [ ] Prevents promoting on 14 noisy trades where the IC delta is
      driven by 2-3 lucky picks.

### 9B — Image-relevance check on article hero pick

- [ ] After the article writer returns an image_url, do a second
      Haiku pass: "given this article headline + first paragraph,
      does the image at this URL look on-topic?" Returns yes/no/borderline.
- [ ] Drop hero on borderline/no; current writer occasionally picks
      something off-topic (e.g. a generic stock photo for a specific
      M&A story).

### 9C — De-duplicate stories across editions

- [ ] Discovery / publisher track the trailing 5 days of `front_lead_slug`
      and front-page headlines. If today's lead is the 4th day in a
      row of "Iran talks stalled", the publisher must pick something
      else for the front (relegate to a brief).
- [ ] Driven by a `recently_led` block in the publisher prompt — model
      sees what it's already led with and gets to decide if the story
      genuinely deserves another front.

### 9D — Sector + factor exposure analytics on the dashboard

- [ ] Compute per-day sector exposure across live tiers (using sector
      from yfinance per ticker) and surface as a stacked area on the
      dashboard overview.
- [ ] Same for factors: long-momentum, long-quality, long-value, etc.,
      via a simple rule mapping per ticker.

### 9E — Mobile-responsive dashboard

- [ ] Current dashboard is desktop-optimised. The newspaper pages
      already wrap well on mobile; the dashboard's strategy grid +
      detail panel don't (`grid-template-columns: repeat(auto-fill, ...)`
      collapses but the detail panel side-by-side stats overflow).
- [ ] Targeted CSS media-query work, no JS changes needed.

### 9F — Bot-health alerts

- [ ] A nightly check workflow that flags: workflow failures in last
      24h, ledger staleness, predictions log corruption, broker
      connectivity errors. Sends a single summary email if anything
      is amiss.
- [ ] Easier signal than scrolling GH Actions runs manually.

### 9G — Tax / ISA tracking (live-tier only)

- [ ] When we go live: track CGT base cost per trade for taxable
      accounts; track ISA contribution / withdrawal events; emit an
      annual statement that maps to HMRC's CG30 form layout.
- [ ] Not relevant until live; queued for whenever that lands.

### 9H — Push notifications on trade events

- [ ] Optional Telegram / iOS-Shortcut bridge that pings on entry
      placed, exit fired, kill switch tripped, halt issued. Already
      planned for the Phase 2 "approve" UX of going live.

### 9I — Searchable historical predictions

- [ ] The "Marking the homework" section shows the last 8 graded
      predictions. Add a `/predictions/` archive page with full
      history, filter by source/horizon/status, and a verdict-rate
      breakdown per conviction level (so we can see if 'high
      conviction' actually means something).

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
