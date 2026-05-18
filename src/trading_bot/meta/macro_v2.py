"""Phase 3 — Macro pipeline.

Mirrors the Phase 2 news pipeline architecture but for weekly macro.
The existing `meta/macro.py` is a single-Sonnet-call agent that
emits a monolithic markdown view. This v2 path produces a structured
multi-desk publication:

1. **Snapshot** (no LLM) — reuses `meta.macro._gather_snapshot()`
   for the cross-asset data (yield curve, credit, DXY, sectors,
   commodities).
2. **Publisher** (Sonnet) — chooses which desks to feature, picks the
   editorial lead, generates the masthead subtitle, assigns bylines,
   emits the desk plan.
3. **Brief + Article writers** (Haiku + Sonnet × N parallel) — same
   contract as the news pipeline; we reuse `Brief` and `FullArticle`.
4. **Predictions** (Sonnet) — multi-horizon macro calls (month,
   quarter, 6mo, year, multi-year); persisted via the unified
   `predictions_log` (source="macro").
5. **For the strategies** (Haiku) — compressed bias-signal callout
   the daily news pipeline references for sector lean.
6. **Render** — assembly to `docs/macro/YYYY-W##/index.html` plus
   per-piece subpages.

Reuse decisions
===============
- `Brief`, `FullArticle` from `meta.news.*` — same shape suits macro.
- `write_briefs`, `write_articles` from `meta.news.*` — these take a
  `NewsPlan` but treat sections as opaque strings, so they work for
  desks too. We just pass them a NewsPlan with desk names where
  sections would be.
- `BYLINES` for personas — we use a macro-flavoured subset plus a few
  desk-specific aliases.
"""
from __future__ import annotations

import html
import json
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import markdown as md_lib

from trading_bot.llm.claude_code import ClaudeCodeError, run_claude, run_claude_for_json
from trading_bot.meta.news.article_writer import FullArticle, articles_to_json, write_articles
from trading_bot.meta.news.brief_writer import Brief, briefs_to_json, write_briefs
from trading_bot.meta.news.publisher import NewsPlan, PlannedPiece, plan_to_json
from trading_bot.meta.news.triage import TriagedCandidate
from trading_bot.state.paths import STATE_ROOT
from trading_bot.state.predictions_log import (
    Prediction,
    append_prediction,
    read_predictions,
)

log = logging.getLogger(__name__)

_PUBLISH_TIMEOUT = 600    # macro snapshot prompt is heavy; 240s was tight in 2026-W21
_PRED_TIMEOUT = 600       # same — multi-horizon macro reasoning needs the headroom
_STRATEGY_TIMEOUT = 240


# Macro byline roster. Different beats from the news desks.
MACRO_BYLINES: dict[str, str] = {
    "G. Mehlman":     "Rates desk. Numerate, watches the curve like a hawk.",
    "E. Castellano":  "FX desk. Patient, cross-region context.",
    "T. Larch":       "Credit desk. Sceptical, scars from 2020 still visible.",
    "I. Aresti":      "Sectors. Inside-out reasoning, sector by sector.",
    "P. Kanu":        "Regions desk. Macro through the lens of each bloc.",
    "M. Dehaene":     "Calls editor. Lives or dies by the falsifier.",
    "S. Vance":       "Risk desk. Cynic by trade, prosaic by choice.",
    "The Editor":     "Editorial board, for the weekly thesis.",
    "Bot Tribune Staff": "Collaborative or data-driven pieces.",
}

# Desks the publisher can use. The renderer maps each to a CSS accent.
_MACRO_DESKS = {
    "Editorial",
    "Rates",
    "FX",
    "Credit",
    "Sectors",
    "Regions",
    "Calls",
    "Risk",
}

# Desk → CSS modifier (must match docs/assets/style.css)
_DESK_CLASS = {
    "Editorial": "rates",       # navy editorial pieces
    "Rates":     "rates",
    "FX":        "fx",
    "Credit":    "credit",
    "Sectors":   "sectors",
    "Regions":   "regions",
    "Calls":     "calls",
    "Risk":      "risk",
}


# ---------------------------------------------------------------------------
# Public orchestration
# ---------------------------------------------------------------------------

def run_macro_v2(today: date) -> dict:
    """Full weekly macro pipeline. Returns a small summary dict.

    Skipped silently if OAUTH token unavailable (returns the existing
    legacy macro agent's behaviour for compatibility)."""
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        log.error("CLAUDE_CODE_OAUTH_TOKEN not set — skipping macro v2")
        return {"skipped": True, "reason": "no oauth token"}

    week_id = _iso_week(today)
    log.info("=== Macro v2 — week %s ===", week_id)

    # Stage 1 — Snapshot (reuse existing)
    log.info("Stage 1 / 5 — Snapshot (data)")
    snapshot = _gather_snapshot_safely()

    # Stage 2 — Publisher
    log.info("Stage 2 / 5 — Publisher (desks)")
    plan = _publish_macro_plan(today, week_id, snapshot)
    if not plan.pieces:
        log.warning("Macro publisher returned empty plan — aborting")
        return {"skipped": True, "reason": "empty plan"}

    # Build synthetic TriagedCandidates so we can reuse news brief + article writers
    triaged = _build_synthetic_triaged(plan, snapshot)

    # Stage 3 — Briefs + Articles + Predictions + Strategies callout (parallel)
    log.info("Stage 3 / 5 — Briefs + Articles + Predictions + Strategies (parallel)")
    with ThreadPoolExecutor(max_workers=4) as outer:
        f_briefs = outer.submit(write_briefs, plan, triaged, today)
        f_articles = outer.submit(write_articles, plan, triaged, today)
        f_preds = outer.submit(_generate_predictions, today, week_id, snapshot, plan)
        f_strategies = outer.submit(_compress_for_strategies, today, snapshot, plan)
        briefs = f_briefs.result()
        articles = f_articles.result()
        predictions = f_preds.result()
        for_strategies_md = f_strategies.result()

    homework = _gather_macro_homework()

    # Stage 4 — Render
    log.info("Stage 4 / 5 — Render edition")
    try:
        from trading_bot.dashboard.pages import _shell, docs_root, render_macro_pages
        edition = MacroEdition(
            week_id=week_id,
            today=today.isoformat(),
            snapshot=snapshot,
            plan=plan,
            briefs=briefs,
            articles=articles,
            predictions=predictions,
            homework=homework,
            for_strategies_md=for_strategies_md,
        )
        front_path = _render_macro_edition(edition, docs_root(), _shell)
        render_macro_pages()  # refresh archive index
        page_url = f"https://dario-zela.github.io/trading_bot/macro/{week_id}/"
    except Exception as e:
        log.warning("Macro render failed (non-fatal): %s", e)
        front_path = None
        page_url = None

    # Stage 5 — Persist pipeline state
    log.info("Stage 5 / 5 — Persist state")
    _write_pipeline_state(today, week_id, snapshot, plan, briefs, articles, predictions, homework, for_strategies_md)

    # Send the weekly macro email (non-fatal on failure)
    if page_url:
        try:
            from trading_bot.notify.email import render_macro_email, send_summary_email
            headline = next((p.headline for p in plan.pieces if p.tier == "lead"), plan.masthead_subtitle)
            subject, text_body, html_body = render_macro_email(
                week_id=week_id,
                headline=headline,
                for_strategies_md=for_strategies_md,
                full_brief_url=page_url,
            )
            send_summary_email(subject=subject, body_text=text_body, body_html=html_body)
            log.info("Macro v2: sent weekly email with link %s", page_url)
        except Exception as e:
            log.warning("Couldn't send macro email (non-fatal): %s", e)

    return {
        "week": week_id,
        "pieces": len(plan.pieces),
        "predictions": len(predictions),
        "homework_items": len(homework),
        "front_path": str(front_path) if front_path else None,
        "page_url": page_url,
    }


# ---------------------------------------------------------------------------
# Snapshot (reuses meta.macro)
# ---------------------------------------------------------------------------

def _gather_snapshot_safely() -> dict:
    """Pull cross-asset data. Returns {} if the upstream fetches fail."""
    try:
        from trading_bot.meta.macro import _gather_snapshot
        return _gather_snapshot()
    except Exception as e:
        log.warning("Macro snapshot fetch failed: %s — proceeding with empty snapshot", e)
        return {}


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------

def _publish_macro_plan(today: date, week_id: str, snapshot: dict) -> NewsPlan:
    """Run the macro publisher. Falls back to a heuristic plan on
    failure so the page still renders."""
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return _heuristic_macro_plan(today, week_id, snapshot)
    prompt = _build_publish_prompt(today, week_id, snapshot)
    try:
        response = run_claude_for_json(prompt, model="sonnet", timeout_seconds=_PUBLISH_TIMEOUT)
    except ClaudeCodeError as e:
        log.warning("Macro publisher Sonnet failed: %s — using heuristic plan", e)
        return _heuristic_macro_plan(today, week_id, snapshot)
    return _parse_macro_plan(response, today, week_id, snapshot)


def _build_publish_prompt(today: date, week_id: str, snapshot: dict) -> str:
    snap_block = json.dumps(snapshot, indent=2, default=str)[:6000]
    bylines = "\n".join(f"  - {name}: {b}" for name, b in MACRO_BYLINES.items())

    return f"""You are the publisher of The Bot Tribune's weekly Macro
publication for week ending {today.isoformat()} (ISO week {week_id}).

Macro is a peer publication to the daily News — *not* a weaker cousin.
It is published Sunday evenings and read by the strategies through
the rest of the week. The reader expects a multi-desk weekly view
with a clear editorial thesis and falsifiable calls.

## This week's data snapshot

```
{snap_block}
```

## Byline roster (use these names, don't invent new ones)

{bylines}

## Desks (use these names, don't invent new ones)

- **Editorial** — exactly one piece, the week's thesis. Mark tier="lead".
- **Rates** — 1-3 pieces on the curve, central banks, sovereign bonds.
- **FX** — 1-3 pieces on currency cross-currents.
- **Credit** — 1-2 pieces on spreads, defaults, IG vs HY.
- **Sectors** — 0-3 pieces on individual sectors that warrant a deep
  look this week. (The full sector ratings table is appended after,
  outside the LLM — you don't need to enumerate.)
- **Regions** — 1-4 pieces on regional macro (US, UK/EU, China, EM).
- **Risk** — 0-2 pieces on tail-risks worth flagging.
- **Calls** is appended automatically — do NOT include.

## What you emit

For each piece, a slot record. The downstream writers will fill the
actual prose. Your job: pick what the week's topics are, who covers
what, and what headlines / kickers tell the reader the most.

## Required output

Return JSON only:

```json
{{
  "masthead_subtitle": "<one descriptive sentence ≤120 chars on this week's headline theme>",
  "pieces": [
    {{
      "desk": "Editorial" | "Rates" | "FX" | "Credit" | "Sectors" | "Regions" | "Risk",
      "headline": "<rewritten or unchanged>",
      "kicker": "<DESK · TOPIC>",
      "byline": "<one of the names above>",
      "one_line": "<single-sentence standfirst>",
      "angle": "<the actual story — what makes this piece interesting>",
      "key_facts": ["<concrete fact a writer can hang the piece on>", "<another>", ...],
      "tier": "lead" | "feature" | "brief",
      "slug": "<short-hyphenated-slug>"
    }}
  ],
  "notes": "<one-line internal note for debugging>"
}}
```

## Rules

- Exactly one piece with `desk="Editorial"` and `tier="lead"`.
- Render pieces in order: Editorial first, then Rates, FX, Credit,
  Sectors, Regions, Risk.
- Headlines are sentence-case, no clickbait, no "in which…".
- Subtitle is descriptive — what the week is about. Not arch.
- Slug is short, lowercase, hyphenated.
- Each piece gets 2-4 key_facts that a writer can lean on.
- Volume: aim for 8-14 pieces total this week. Use fewer on a quiet
  week — don't pad to fill desks that don't have a story.
"""


def _parse_macro_plan(response: dict | list, today: date, week_id: str, snapshot: dict) -> NewsPlan:
    if isinstance(response, list):
        response = {"pieces": response}
    if not isinstance(response, dict):
        return _heuristic_macro_plan(today, week_id, snapshot)

    raw_pieces = response.get("pieces") or []
    if not isinstance(raw_pieces, list):
        return _heuristic_macro_plan(today, week_id, snapshot)

    used_slugs: set[str] = set()
    pieces: list[PlannedPiece] = []
    lead_slug = ""

    for i, raw in enumerate(raw_pieces):
        if not isinstance(raw, dict):
            continue
        desk = str(raw.get("desk") or "").strip()
        if desk not in _MACRO_DESKS:
            continue
        byline = str(raw.get("byline") or "").strip()
        if byline not in MACRO_BYLINES:
            byline = "Bot Tribune Staff"
        tier = str(raw.get("tier") or "brief").strip().lower()
        if tier not in {"lead", "feature", "brief"}:
            tier = "brief"
        slug = _normalise_slug(raw.get("slug") or raw.get("headline") or f"piece-{i}")
        slug = _unique_slug(slug, used_slugs)
        used_slugs.add(slug)

        piece = PlannedPiece(
            slug=slug,
            section=desk,                                 # reuse PlannedPiece.section for desk
            headline=str(raw.get("headline") or "").strip(),
            kicker=str(raw.get("kicker") or "").strip().upper()[:60],
            byline=byline,
            one_line=str(raw.get("one_line") or "").strip()[:320],
            tier=tier,
            triage_index=i,
        )
        # Attach angle + key_facts via a sidecar dict; the brief/article writers
        # read these through the synthetic triaged list.
        piece.__dict__["_macro_angle"] = str(raw.get("angle") or piece.one_line).strip()
        piece.__dict__["_macro_key_facts"] = list(raw.get("key_facts") or [])
        piece.__dict__["_macro_why"] = ""
        pieces.append(piece)
        if tier == "lead" and not lead_slug:
            lead_slug = slug

    if not lead_slug and pieces:
        # Promote the first piece if no lead was explicitly marked
        pieces[0].tier = "lead"
        pieces[0].section = "Editorial"
        lead_slug = pieces[0].slug

    subtitle = str(response.get("masthead_subtitle") or "").strip()
    if not subtitle and pieces:
        subtitle = pieces[0].one_line[:120]

    return NewsPlan(
        edition_date=today.isoformat(),
        masthead_subtitle=subtitle[:200],
        front_lead_slug=lead_slug,
        pieces=pieces,
        notes=str(response.get("notes") or "")[:280],
    )


def _heuristic_macro_plan(today: date, week_id: str, snapshot: dict) -> NewsPlan:
    """Minimal fallback when the LLM is unavailable. One editorial
    placeholder + a Rates brief + a Sectors brief from the snapshot."""
    pieces: list[PlannedPiece] = []

    pieces.append(PlannedPiece(
        slug=f"editorial-{week_id}", section="Editorial",
        headline=f"Week ending {today.isoformat()} — the macro picture",
        kicker="EDITORIAL · WEEKLY", byline="The Editor",
        one_line="Editorial placeholder (LLM unavailable).",
        tier="lead", triage_index=0,
    ))
    pieces[-1].__dict__["_macro_angle"] = "Weekly macro view (heuristic fallback)."
    pieces[-1].__dict__["_macro_key_facts"] = []
    pieces[-1].__dict__["_macro_why"] = ""

    yc = snapshot.get("yield_curve") or {}
    if yc:
        y10 = _fmt_num(yc.get("10Y"), digits=2, suffix="%")
        spread = _fmt_num(yc.get("spread_3m10y"), digits=2, suffix="%")
        pieces.append(PlannedPiece(
            slug="rates-curve", section="Rates",
            headline=f"The curve at {y10} for the 10y",
            kicker="RATES · CURVE", byline="G. Mehlman",
            one_line=f"3m10y spread at {spread}.",
            tier="feature", triage_index=1,
        ))
        pieces[-1].__dict__["_macro_angle"] = "The shape of the curve this week."
        pieces[-1].__dict__["_macro_key_facts"] = [f"10Y at {y10}"]
        pieces[-1].__dict__["_macro_why"] = ""

    sectors = snapshot.get("sector_strength") or []
    if sectors:
        top = sectors[0]
        r5 = _fmt_num(top.get("5d"), digits=2, suffix="%")
        pieces.append(PlannedPiece(
            slug="sectors-leader", section="Sectors",
            headline=f"{top.get('label', 'top sector')} leads the week",
            kicker="SECTORS · LEAD", byline="I. Aresti",
            one_line=f"{top.get('label', '')} 5d return {r5}.",
            tier="feature", triage_index=2,
        ))
        pieces[-1].__dict__["_macro_angle"] = "Sector leadership this week."
        pieces[-1].__dict__["_macro_key_facts"] = [f"{top.get('label', '')}: 5d {r5}"]
        pieces[-1].__dict__["_macro_why"] = ""

    return NewsPlan(
        edition_date=today.isoformat(),
        masthead_subtitle=f"Heuristic macro view — LLM unavailable for week {week_id}.",
        front_lead_slug=pieces[0].slug if pieces else "",
        pieces=pieces,
        notes="macro heuristic plan",
    )


def _build_synthetic_triaged(plan: NewsPlan, snapshot: dict) -> list[TriagedCandidate]:
    """Build a TriagedCandidate per plan piece so the news brief +
    article writers (which take a triaged list) can be reused as-is.

    The publisher already wrote angle + key_facts when planning each
    piece; we just lift them onto the triaged record."""
    triaged: list[TriagedCandidate] = []
    for piece in plan.pieces:
        angle = piece.__dict__.get("_macro_angle") or piece.one_line
        key_facts = piece.__dict__.get("_macro_key_facts") or []
        why = piece.__dict__.get("_macro_why") or ""
        triaged.append(TriagedCandidate(
            title=piece.headline,
            one_line=piece.one_line,
            suggested_section=piece.section,
            importance_hint=8 if piece.tier == "lead" else (7 if piece.tier == "feature" else 5),
            source_hints=[],
            score=8 if piece.tier == "lead" else (7 if piece.tier == "feature" else 5),
            angle=angle,
            key_facts=[str(f) for f in key_facts][:5],
            why_it_matters=why,
            suggested_section_final=piece.section,
            failed=False,
        ))
    return triaged


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def _generate_predictions(today: date, week_id: str, snapshot: dict, plan: NewsPlan) -> list[Prediction]:
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return []
    if not plan.pieces:
        return []

    prompt = _build_predictions_prompt(today, week_id, snapshot, plan)
    try:
        response = run_claude_for_json(prompt, model="sonnet", timeout_seconds=_PRED_TIMEOUT)
    except ClaudeCodeError as e:
        log.warning("Macro predictions Sonnet failed: %s", e)
        return []

    predictions = _parse_macro_predictions(response, today)
    for p in predictions:
        try:
            append_prediction(p)
        except Exception as e:
            log.warning("Failed to persist macro prediction %s: %s", p.id, e)
    log.info("Macro predictions: %d persisted", len(predictions))
    return predictions


def _build_predictions_prompt(today: date, week_id: str, snapshot: dict, plan: NewsPlan) -> str:
    plan_block = "\n".join(
        f"  - [{p.section}] {p.headline} — {p.one_line}"
        for p in plan.pieces[:15]
    ) or "  (empty plan)"
    snap_block = json.dumps(snapshot, indent=2, default=str)[:4000]

    month_end = today + timedelta(days=30)
    quarter_end = today + timedelta(days=90)
    half_end = today + timedelta(days=180)
    year_end = today + timedelta(days=365)

    return f"""You are the calls editor for The Bot Tribune's weekly
Macro publication. Week ending {today.isoformat()}.

Write 6-10 falsifiable macro predictions across multiple horizons.
Each call must be concrete enough to grade against market data on the
target date. Multi-year calls are encouraged when they're genuinely
defensible from public data.

## This week's snapshot

```
{snap_block}
```

## The pieces in this week's edition

{plan_block}

## Horizons + target dates

- **this-month** — target_date ≤ {month_end.isoformat()}
- **this-quarter** — target_date ≤ {quarter_end.isoformat()}
- **this-half** — target_date ≤ {half_end.isoformat()}
- **this-year** — target_date ≤ {year_end.isoformat()}
- **multi-year** — target_date is the end of a calendar year, 2027-2030

## Distribution

- 2-3 this-month
- 2-3 this-quarter
- 1-2 this-half / this-year
- 0-2 multi-year (only when defensible)

## Rules

- Concrete: cite the indicator, threshold, and date.
- Falsifiable: state EXACTLY what disproves the call.
- Asymmetric: prefer calls where the contrarian case is
  default-priced. Boring consensus calls add nothing.
- Honest conviction: low / medium / high. A high-conviction multi-
  year call should be defensible from public information today.

## Required output

```json
{{
  "predictions": [
    {{
      "claim": "<call in plain English ≤200 chars>",
      "horizon": "this-month" | "this-quarter" | "this-half" | "this-year" | "multi-year",
      "target_date": "YYYY-MM-DD",
      "falsification_criteria": "<one sentence — what disproves this>",
      "conviction": "low" | "medium" | "high",
      "rationale": "<1-2 sentence reasoning>"
    }}
  ]
}}
```
"""


def _parse_macro_predictions(response: dict | list, today: date) -> list[Prediction]:
    if isinstance(response, list):
        response = {"predictions": response}
    if not isinstance(response, dict):
        return []
    raw = response.get("predictions") or []
    if not isinstance(raw, list):
        return []

    out: list[Prediction] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    valid_horizons = {"this-month", "this-quarter", "this-half", "this-year", "multi-year"}
    for item in raw:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim", "")).strip()
        if not claim:
            continue
        horizon = str(item.get("horizon", "")).strip()
        if horizon not in valid_horizons:
            horizon = "this-quarter"
        target = str(item.get("target_date", "")).strip()
        if not _looks_like_iso_date(target):
            target = _default_target_for(horizon, today)
        falsifier = str(item.get("falsification_criteria", "")).strip()
        if not falsifier:
            continue
        conv = str(item.get("conviction", "medium")).strip().lower()
        if conv not in {"low", "medium", "high"}:
            conv = "medium"
        rationale = str(item.get("rationale", "")).strip()[:600]

        out.append(Prediction(
            id=str(uuid.uuid4()),
            source="macro",
            made_at=now_iso,
            claim=claim[:240],
            horizon=horizon,
            target_date=target,
            falsification_criteria=falsifier[:280],
            conviction=conv,
            source_section="Calls",
            source_slug=f"macro-{_iso_week(today)}-calls",
            status="open",
            grading_note=rationale,   # stash rationale here for now
        ))
    return out


def _gather_macro_homework() -> list[Prediction]:
    all_macro = read_predictions(source="macro")
    graded = [p for p in all_macro if p.status in {"proven", "partial", "falsified", "still-open"} and p.graded_at]
    graded.sort(key=lambda p: p.graded_at or "", reverse=True)
    return graded[:8]


# ---------------------------------------------------------------------------
# For the strategies — compressed bias-signal callout
# ---------------------------------------------------------------------------

def _compress_for_strategies(today: date, snapshot: dict, plan: NewsPlan) -> str:
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return _fallback_for_strategies(snapshot)

    plan_block = "\n".join(
        f"- [{p.section}] {p.headline} — {p.one_line}"
        for p in plan.pieces[:15]
    ) or "(empty edition)"
    snap_block = json.dumps(snapshot, indent=2, default=str)[:3000]

    prompt = f"""You are compressing this week's Macro edition into a
"For the strategies" callout for week ending {today.isoformat()}.

The strategies read this each Sunday and lean their sector / region
biases for the week ahead. Keep it tight — 100-150 words. Markdown.

## This week's pieces

{plan_block}

## Data snapshot

```
{snap_block}
```

## Required output — markdown, this exact shape

```
**Bias for the week ahead:**

- _Rates:_ <one line on duration / curve bias>
- _FX:_ <one line on the dollar / cross-currents>
- _Credit:_ <one line on spread bias>
- _Sectors:_ <one or two lines on sector lean>
- _Regions:_ <one line on regional lean>
- _Risk:_ <one line — what to watch as a downside trigger>

**Watchlist:** <8-12 tickers across regions, comma-separated>
```

Output the markdown only, no preamble.
"""
    try:
        result = run_claude(prompt, model="haiku", timeout_seconds=_STRATEGY_TIMEOUT)
    except ClaudeCodeError as e:
        log.warning("For-strategies Haiku failed: %s", e)
        return _fallback_for_strategies(snapshot)
    text = (result.text or "").strip()
    return text or _fallback_for_strategies(snapshot)


def _fallback_for_strategies(snapshot: dict) -> str:
    return (
        "**Bias for the week ahead:**\n\n"
        "- _Rates:_ neutral (LLM unavailable)\n"
        "- _FX:_ neutral\n"
        "- _Credit:_ neutral\n"
        "- _Sectors:_ neutral\n"
        "- _Regions:_ neutral\n"
        "- _Risk:_ none flagged\n\n"
        "**Watchlist:** SPY, QQQ, IWM, TLT, HYG, UUP, GLD, EZU, EWJ, EEM\n"
    )


# ---------------------------------------------------------------------------
# Edition data + render
# ---------------------------------------------------------------------------

@dataclass
class MacroEdition:
    week_id: str
    today: str
    snapshot: dict
    plan: NewsPlan
    briefs: dict[str, Brief]
    articles: dict[str, FullArticle]
    predictions: list[Prediction]
    homework: list[Prediction]
    for_strategies_md: str


def _render_macro_edition(edition: MacroEdition, docs_root: Path, shell_fn) -> Path:
    """Render the front page + per-piece subpages for the macro edition."""
    edition_dir = docs_root / "macro" / edition.week_id
    edition_dir.mkdir(parents=True, exist_ok=True)

    # Per-piece subpages (reuse the news article renderer)
    from trading_bot.meta.news.render import _render_article_subpage
    pieces_by_slug = {p.slug: p for p in edition.plan.pieces}
    for piece in edition.plan.pieces:
        art = edition.articles.get(piece.slug)
        if not art:
            continue
        subpage_body = _render_article_subpage(piece, art, edition.plan, pieces_by_slug, datetime.fromisoformat(edition.today + "T00:00:00").date())
        page = shell_fn(
            title=f"{piece.headline} — Macro {edition.week_id}",
            body_html=subpage_body,
            current="macro",
            depth=2,
            page_class="macro",
        )
        (edition_dir / f"{piece.slug}.html").write_text(page)

    # Front page
    front_body = _render_macro_front(edition, edition_dir)
    front_page = shell_fn(
        title=f"Macro — {edition.week_id}",
        body_html=front_body,
        current="macro",
        depth=2,
        page_class="macro",
    )
    front_path = edition_dir / "index.html"
    front_path.write_text(front_page)
    # Update docs/macro/latest.html → newest edition
    from trading_bot.meta.news.render import _write_latest_redirect
    _write_latest_redirect(edition_dir.parent, edition_dir.name)
    return front_path


def _render_macro_front(edition: MacroEdition, edition_dir: Path) -> str:
    parts: list[str] = ['<main class="paper">']

    # Prev / next from the on-disk sibling weeks
    prev_id, next_id = _neighbouring_weeks(edition.week_id, edition_dir.parent)
    from trading_bot.meta.news.render import _edition_nav_html
    nav_html = _edition_nav_html(
        prev_url=f"../{prev_id}/" if prev_id else "",
        next_url=f"../{next_id}/" if next_id else "",
        latest_url="../latest.html",
        prev_label=prev_id or "",
        next_label=next_id or "",
    )

    # Masthead
    parts.append(
        '<header class="masthead">'
        '  <h1><a href="../latest.html" class="masthead-link">The Bot Tribune</a>'
        f'<span class="sub">— Macro, {html.escape(edition.week_id)}</span></h1>'
        + nav_html
        + (f'  <div class="subtitle">{html.escape(edition.plan.masthead_subtitle)}</div>' if edition.plan.masthead_subtitle else '')
        + '</header>'
    )

    # Masthead strip
    parts.append(
        '<div class="masthead-strip">'
        f'  <span><strong>Macro</strong></span>'
        f'  <span>Week ending {html.escape(edition.today)}</span>'
        f'  <span>{len(edition.plan.pieces)} piece{"s" if len(edition.plan.pieces) != 1 else ""} · {len(edition.predictions)} fresh call{"s" if len(edition.predictions) != 1 else ""}</span>'
        '</div>'
    )

    # Cross-asset snapshot grid
    parts.append(_render_snapshot_grid(edition.snapshot))

    # Group pieces by desk
    by_desk: dict[str, list[PlannedPiece]] = {}
    for p in edition.plan.pieces:
        by_desk.setdefault(p.section, []).append(p)

    # Editorial first (the lead)
    editorial_pieces = by_desk.get("Editorial", [])
    if editorial_pieces:
        parts.append(_render_macro_editorial(editorial_pieces[0], edition.briefs.get(editorial_pieces[0].slug), edition.articles.get(editorial_pieces[0].slug)))

    # Then each desk in canonical order
    desk_order = ["Rates", "FX", "Credit", "Sectors", "Regions", "Risk"]
    for desk in desk_order:
        pieces = by_desk.get(desk, [])
        if not pieces:
            continue
        parts.append(_render_macro_desk(desk, pieces, edition.briefs, edition.articles))

    # Calls + Marking the homework
    if edition.predictions or edition.homework:
        parts.append(_render_macro_calls(edition.predictions, edition.homework))

    # For the strategies callout (dark callout)
    parts.append(_render_for_strategies(edition.for_strategies_md))

    parts.append(
        '<footer class="colophon">'
        f'Auto-generated by the weekly-macro pipeline · {html.escape(_generated_at())}'
        '</footer>'
    )
    parts.append('</main>')
    return "\n".join(parts)


def _fmt_num(v, *, digits: int = 2, suffix: str = "") -> str:
    """Format a snapshot number to a fixed number of decimal places.
    Falls back to em-dash if the value isn't numeric."""
    if isinstance(v, (int, float)):
        return f"{v:.{digits}f}{suffix}"
    return "—"


def _render_snapshot_grid(snapshot: dict) -> str:
    """Cross-asset snap cards — yield, credit, DXY, top sector."""
    if not snapshot:
        return ""
    cards: list[str] = []

    yc = snapshot.get("yield_curve") or {}
    if yc:
        cards.append(_snap_card(
            "10y yield",
            _fmt_num(yc.get("10Y"), digits=2, suffix="%"),
            f"3m10y {_fmt_num(yc.get('spread_3m10y'), digits=2, suffix='%')}",
            "rates",
        ))
    cs = snapshot.get("credit_spreads") or {}
    if cs:
        diff = cs.get("hy_vs_ig_5d_diff")
        delta = f"{diff:+.2f}%" if isinstance(diff, (int, float)) else "—"
        cards.append(_snap_card("HY-IG spread (5d)", delta, "HYG vs LQD", "credit"))
    dxy = snapshot.get("dollar_index") or {}
    if dxy:
        r5 = dxy.get("return_5d_pct")
        delta = f"{r5:+.2f}%" if isinstance(r5, (int, float)) else "—"
        cards.append(_snap_card(
            "DXY 5d",
            delta,
            f"level {_fmt_num(dxy.get('level'), digits=2)}",
            "fx",
        ))
    sectors = snapshot.get("sector_strength") or []
    if sectors:
        top = sectors[0]
        r5 = top.get("5d")
        delta = f"{r5:+.2f}%" if isinstance(r5, (int, float)) else "—"
        cards.append(_snap_card(f"{top.get('label', 'Top sector')}", delta, "5d strongest", "rates"))

    if not cards:
        return ""

    return (
        '<div class="section-label rates">'
        '  <span>The snapshot</span>'
        '  <span class="ord">cross-asset, week-on-week</span>'
        '</div>'
        f'<div class="snapshot-grid">{"".join(cards)}</div>'
    )


def _snap_card(label: str, val: str, delta: str, cls: str) -> str:
    return (
        f'<div class="snap-card {cls}">'
        f'  <div class="label">{html.escape(label)}</div>'
        f'  <div class="val">{html.escape(str(val))}</div>'
        f'  <div class="delta">{html.escape(delta)}</div>'
        '</div>'
    )


def _render_macro_editorial(piece: PlannedPiece, brief: Brief | None, article: FullArticle | None) -> str:
    body_md = (brief.body_md if brief else piece.one_line) or ""
    body_html = _md_to_html(body_md)
    dek = ""
    if article and article.in_one_sentence:
        dek = f'<p class="dek">{html.escape(article.in_one_sentence)}</p>'
    elif piece.one_line:
        dek = f'<p class="dek">{html.escape(piece.one_line)}</p>'
    read_more = ""
    if article:
        from trading_bot.meta.news.render import _estimate_read_minutes, _read_badge
        minutes = _estimate_read_minutes(article.body_md)
        read_more = (
            f'<a class="read-more lead" href="{html.escape(piece.slug)}.html">'
            f'Read the full thesis →{_read_badge(minutes)}</a>'
        )
    return (
        '<div class="section-label rates">'
        '  <span>The week\'s thesis</span>'
        '  <span class="ord">editorial</span>'
        '</div>'
        '<article class="editorial">'
        f'  <p class="meta rates"><span class="accent">{html.escape(piece.kicker or "EDITORIAL · WEEKLY")}</span>'
        f'    <span class="dot">·</span> By {html.escape(piece.byline)}</p>'
        f'  <h2>{html.escape(piece.headline)}</h2>'
        f'  {dek}'
        f'  <div class="editorial-body">{body_html}</div>'
        f'  {read_more}'
        '</article>'
    )


def _render_macro_desk(desk: str, pieces: list[PlannedPiece], briefs: dict[str, Brief], articles: dict[str, FullArticle]) -> str:
    cls = _DESK_CLASS.get(desk, "rates")
    out = [
        f'<div class="section-label {cls}">'
        f'  <span>{html.escape(desk)} desk</span>'
        f'  <span class="ord">{len(pieces)} piece{"s" if len(pieces) != 1 else ""}</span>'
        '</div>'
    ]

    features = [p for p in pieces if p.tier == "feature"]
    briefs_only = [p for p in pieces if p.tier == "brief"]

    if features:
        grid_cls = "grid-2" if len(features) >= 2 else ""
        if grid_cls:
            out.append(f'<div class="{grid_cls}">')
        for p in features:
            out.append(_render_macro_brief(p, briefs.get(p.slug), articles.get(p.slug), cls))
        if grid_cls:
            out.append('</div>')

    if briefs_only:
        grid_cls = "grid-3" if len(briefs_only) >= 2 else ""
        if grid_cls:
            out.append(f'<div class="{grid_cls}">')
        for p in briefs_only:
            out.append(_render_macro_brief(p, briefs.get(p.slug), articles.get(p.slug), cls))
        if grid_cls:
            out.append('</div>')

    return "\n".join(out)


def _render_macro_brief(piece: PlannedPiece, brief: Brief | None, article: FullArticle | None, cls: str) -> str:
    body_md = (brief.body_md if brief else piece.one_line) or ""
    body_html = _md_to_html(body_md)
    read_more = ""
    if article:
        from trading_bot.meta.news.render import _estimate_read_minutes, _read_badge
        minutes = _estimate_read_minutes(article.body_md)
        read_more = (
            f'<a class="read-more small {cls}" href="{html.escape(piece.slug)}.html">'
            f'Read on →{_read_badge(minutes)}</a>'
        )
    return (
        f'<article class="brief {cls}">'
        f'  <p class="meta {cls}"><span class="accent">{html.escape(piece.kicker)}</span>'
        f'    <span class="dot">·</span> By {html.escape(piece.byline)}</p>'
        f'  <h3>{html.escape(piece.headline)}</h3>'
        f'  {body_html}'
        f'  <div class="brief-footer">{read_more}</div>'
        f'</article>'
    )


def _render_macro_calls(predictions: list[Prediction], homework: list[Prediction]) -> str:
    out = [
        '<div class="section-label calls">'
        '  <span>The desk\'s calls</span>'
        f'  <span class="ord">{len(predictions)} fresh, {len(homework)} graded</span>'
        '</div>'
    ]

    by_horizon: dict[str, list[Prediction]] = {}
    for p in predictions:
        by_horizon.setdefault(p.horizon, []).append(p)

    horizon_labels = [
        ("this-month", "This month"),
        ("this-quarter", "This quarter"),
        ("this-half", "Next six months"),
        ("this-year", "This year"),
        ("multi-year", "Multi-year"),
    ]
    for hkey, hlabel in horizon_labels:
        items = by_horizon.get(hkey, [])
        if not items:
            continue
        out.append(f'<div class="subsection">{html.escape(hlabel)}</div>')
        for p in items:
            rationale = f'<p>{html.escape(p.grading_note)}</p>' if p.grading_note else ""
            out.append(
                '<article class="pred">'
                f'  <p class="meta calls"><span class="accent">{html.escape(p.conviction.upper())} CONVICTION</span>'
                f'    <span class="dot">·</span> Target {html.escape(p.target_date)}</p>'
                f'  <h3>{html.escape(p.claim)}</h3>'
                f'  {rationale}'
                f'  <div class="falsif"><strong>Falsified if:</strong> {html.escape(p.falsification_criteria)}</div>'
                '</article>'
            )

    if homework:
        out.append('<div class="subsection">Marking the homework</div>')
        out.append('<div class="grid-2">')
        for p in homework:
            status_cls = p.status if p.status in {"proven", "partial", "falsified", "still-open"} else "open"
            verdict_label = {"proven": "Proven", "partial": "Partial", "falsified": "Falsified", "still-open": "Still open"}.get(p.status, p.status)
            note = html.escape(p.grading_note or "(no grading note recorded)")
            out.append(
                '<article class="pred">'
                f'  <p><span class="verdict {status_cls}">{html.escape(verdict_label)}</span>'
                f'    <span style="font-family: var(--sans); font-size: 0.74rem; color: var(--ink-muted);">'
                f'    Made {html.escape(p.made_at[:10])} · target {html.escape(p.target_date)}</span></p>'
                f'  <h3 style="font-size: 1.18rem;">{html.escape(p.claim)}</h3>'
                f'  <p>{note}</p>'
                '</article>'
            )
        out.append('</div>')

    return "\n".join(out)


def _render_for_strategies(md_text: str) -> str:
    """The 'For the strategies' dark callout box."""
    body_html = _md_to_html(md_text)
    return (
        '<div class="section-label sectors" style="margin-top: 3rem;">'
        '  <span>For the strategies</span>'
        '  <span class="ord">bias signals for the week ahead</span>'
        '</div>'
        '<aside style="background: #1f1f1f; color: #e8e8e8; padding: 1.6rem 1.8rem; border-radius: 4px; margin-top: 1rem; font-size: 1.02rem; line-height: 1.65;">'
        '<style>aside strong { color: var(--c-action-spawn); }</style>'
        f'{body_html}'
        '</aside>'
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_pipeline_state(today: date, week_id: str, snapshot: dict, plan: NewsPlan,
                          briefs: dict[str, Brief], articles: dict[str, FullArticle],
                          predictions: list[Prediction], homework: list[Prediction],
                          for_strategies_md: str) -> None:
    out_dir = STATE_ROOT / "macro" / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "week_id": week_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot": snapshot,
        "plan": plan_to_json(plan),
        "briefs": briefs_to_json(briefs),
        "articles": articles_to_json(articles),
        "predictions": [asdict(p) for p in predictions],
        "homework": [asdict(p) for p in homework],
        "for_strategies_md": for_strategies_md,
    }
    (out_dir / f"{week_id}.pipeline.json").write_text(json.dumps(state, indent=2, default=str))


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _normalise_slug(raw: str) -> str:
    s = (raw or "").lower()
    s = _SLUG_RE.sub("-", s).strip("-")
    if not s:
        s = "piece"
    if len(s) > 60:
        s = s[:60].rstrip("-")
    return s


def _unique_slug(base: str, used: set[str]) -> str:
    if base not in used:
        return base
    n = 2
    while f"{base}-{n}" in used:
        n += 1
    return f"{base}-{n}"


def _iso_week(d: date) -> str:
    year, week, _ = d.isocalendar()
    return f"{year}-W{week:02d}"


_WEEK_RE = re.compile(r"^\d{4}-W\d{2}$")


def _neighbouring_weeks(week_id: str, macro_dir: Path) -> tuple[str | None, str | None]:
    """Return (previous_week, next_week) IDs for navigation, based on
    the on-disk YYYY-W## sibling dirs."""
    if not macro_dir.exists():
        return None, None
    weeks: list[str] = []
    for child in macro_dir.iterdir():
        if not child.is_dir():
            continue
        if not _WEEK_RE.match(child.name):
            continue
        if (child / "index.html").exists() or child.name == week_id:
            weeks.append(child.name)
    if week_id not in weeks:
        weeks.append(week_id)
    weeks.sort()
    i = weeks.index(week_id)
    prev_id = weeks[i - 1] if i > 0 else None
    next_id = weeks[i + 1] if i < len(weeks) - 1 else None
    return prev_id, next_id


def _md_to_html(md_text: str) -> str:
    return md_lib.markdown(md_text or "", extensions=["tables", "fenced_code", "sane_lists", "nl2br"])


def _looks_like_iso_date(s: str) -> bool:
    try:
        datetime.fromisoformat(s).date()
        return True
    except ValueError:
        return False


def _default_target_for(horizon: str, today: date) -> str:
    deltas = {"this-month": 30, "this-quarter": 90, "this-half": 180, "this-year": 365}
    if horizon == "multi-year":
        # Default to end of next calendar year
        return date(today.year + 1, 12, 31).isoformat()
    return (today + timedelta(days=deltas.get(horizon, 90))).isoformat()


def _generated_at() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
