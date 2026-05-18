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

### 2A — Stage 1: Discovery agent

- [ ] `src/trading_bot/meta/news/discovery.py`
- [ ] Sonnet + WebSearch tool — searches across *every* topic class (markets, world, tech, science, politics, culture, sport, climate, health), not just markets
- [ ] Returns ~30-50 candidates: `{title, one_line, suggested_section, importance_hint, source_hints[]}`
- [ ] Prompt template `strategies/news/prompts/discovery.md`

### 2B — Stage 2: Triage agent

- [ ] `src/trading_bot/meta/news/triage.py`
- [ ] Haiku × 6 parallel — one call per candidate
- [ ] Each scores 1-10 + writes a one-line angle + key facts
- [ ] Filters survivors to top-N by score (no fixed cap; whatever the publisher needs)

### 2C — Stage 3: Publisher agent

- [ ] `src/trading_bot/meta/news/publisher.py`
- [ ] Sonnet — organises survivors into sections (Front, Markets, World, Tech, Beyond, plus variable sections like Sport/Climate/Health when warranted)
- [ ] Assigns bylines (persona-based), picks the lead story, sets the AI-generated humorous masthead subtitle
- [ ] Emits the newspaper-plan JSON: section list, lead slug, masthead lines

### 2D — Stage 4: Brief writers (parallel)

- [ ] `src/trading_bot/meta/news/brief_writer.py`
- [ ] Haiku × 6 parallel — one brief per story (~80-100 words, conversational, no jargon)
- [ ] Includes byline, headline, body markdown, sources

### 2E — Stage 5: Full-article writers (parallel)

- [ ] `src/trading_bot/meta/news/article_writer.py`
- [ ] Sonnet × 6 parallel — full article, **no word cap**, written to the topic
- [ ] Each writer also runs an image search (Claude WebFetch against Google Images / Bing image results) for the most fitting hero image — returns image_url + caption
- [ ] Includes inline callouts, sources block, related-articles slugs

### 2F — Stage 6: Bot-summary compressor

- [ ] `src/trading_bot/meta/news/bot_summary.py`
- [ ] Haiku — takes the full assembled edition, produces a tight ~150-word `state/daily_news/YYYY-MM-DD.bot.md` for strategy prompts
- [ ] Preserves Risk tone / Themes / Sector lean / Watchlist / Key data structure

### 2G — Trading floor section

- [ ] `src/trading_bot/meta/news/trading_floor.py`
- [ ] Reads `state/ledger.jsonl` for yesterday's strategy P&L
- [ ] Haiku × 3 parallel — writes prose pieces (winner, loser, "quieter movers") — news-style, not numbers-style
- [ ] Pulled into the publisher's plan automatically

### 2H — Desk's calls section

- [ ] Reads `state/predictions/news.jsonl` for open predictions + recent verdicts
- [ ] Generates fresh predictions (Sonnet) per horizon (tomorrow / week / month) — claim + falsification_criteria + conviction
- [ ] Persists each via `append_prediction()` so the grader can score them later
- [ ] Renders "Marking the homework" sub-section from the most recently graded predictions

### 2I — Assembly + sub-page generation

- [ ] `pages.py` — `render_news_edition(date, structured_data)` writes `docs/news/YYYY-MM-DD/index.html` + `docs/news/YYYY-MM-DD/{slug}.html`
- [ ] `render_news_archive_index()` — listing of all editions
- [ ] Wires images, callouts, "related in this edition" cross-links
- [ ] Daily news email links to the front-page URL

### 2J — Orchestration

- [ ] Rewrite `src/trading_bot/meta/daily_news.py` to call the six stages in order
- [ ] Update `.github/workflows/daily-news-brief.yml` — bump timeout, expose secrets
- [ ] Smoke-test end-to-end on a single edition

---

## Phase 3 — Macro pipeline (same architecture, weekly cadence)

Reuses the multi-stage scaffolding from Phase 2, with desk-based structure rather than news-section-based.

- [ ] `src/trading_bot/meta/macro_v2/discovery.py` — data-driven: cross-asset snapshot + yield curve + sector strength + commodity prices + credit spreads
- [ ] `src/trading_bot/meta/macro_v2/publisher.py` — organises into desks (rates / FX / credit / sectors / regions), assigns desk-specific bylines
- [ ] `src/trading_bot/meta/macro_v2/brief_writer.py` — one brief per desk piece
- [ ] `src/trading_bot/meta/macro_v2/article_writer.py` — full pieces, no word cap, image search per piece
- [ ] `src/trading_bot/meta/macro_v2/sector_spotlight.py` — Sonnet × 2-3 for spotlight long-form pieces on the most actionable sectors
- [ ] `src/trading_bot/meta/macro_v2/predictions.py` — generates predictions per horizon (month / quarter / 6mo / 2027 / multi-year), persists via `append_prediction()`
- [ ] `src/trading_bot/meta/macro_v2/for_strategies.py` — Haiku-compressed "For the strategies" callout (bullet list of bias signals)
- [ ] `pages.py` — `render_macro_edition(week_id, structured_data)` writes `docs/macro/YYYY-W##/index.html` + per-piece subpages
- [ ] Update `.github/workflows/weekly-macro.yml` — bump timeout, plumb full pipeline
- [ ] Macro weekly email with link to the front-page URL

---

## Phase 4 — Evolution restyle (editorial format)

The agent already exists; this reshapes its output into the per-strategy report card format.

- [ ] `src/trading_bot/meta/evolution_v2/editorial.py` — Sonnet writes the "this week's read" intro from metrics + decisions
- [ ] `src/trading_bot/meta/evolution_v2/strategy_report.py` — Haiku × N parallel — for each strategy, write `{what_worked, what_didnt, lessons, going_forward, config_changes}` JSON
- [ ] `src/trading_bot/meta/evolution_v2/format_decisions.py` — the existing promote/demote/tune/spawn actions become the visual chips
- [ ] `pages.py` — `render_evolution(week_id, structured_data)` writes the page with the slate table + per-strategy cards + decisions row
- [ ] Update `weekly-evolution.yml` — plumb the new output structure
- [ ] Evolution weekly email with link

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

## Phase 6 — Polish

- [ ] Macro and Evolution emails (currently only News has one)
- [ ] Dashboard "global overview" tile (already done, just verify it survives the restyle)
- [ ] Strategy reading-times-it badges or progress indicators on the news front page
- [ ] "Today's question" subtitle on the news lead (the publisher generates it)
- [ ] Yesterday's predictions auto-grading runs at 23:30 UTC so verdicts are fresh for the morning brief
- [ ] Smoke-test the full stack: trigger daily news manually, verify HTML lands on Pages, email arrives, predictions persisted, grader resolves the previous day's calls

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
DELIVERY_PLAN.md           ← this file
```
