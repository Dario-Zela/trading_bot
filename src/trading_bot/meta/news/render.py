"""Phase 2I — Assembly. Render the structured news edition to HTML.

Writes to `docs/news/YYYY-MM-DD/`:
- `index.html` — the front page, full newspaper layout
- `{slug}.html` — one per piece in the plan, the full-article page

Output co-exists with the legacy `docs/news/YYYY-MM-DD.html` flat-file
renderer (in `pages.py`). When both exist, the dir-based form is
preferred and `news_url_for(date)` points there.

The HTML structure matches the desktop mockups (`Desktop/trading_bot_*_mockup.html`):
shared masthead-strip, cream paper, accent-coloured section rules,
drop caps on the lead and the first paragraph of each brief.
"""
from __future__ import annotations

import html
import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path

import markdown as md_lib

from trading_bot.meta.news.brief_writer import Brief
from trading_bot.meta.news.article_writer import FullArticle
from trading_bot.meta.news.desks_calls import DesksCalls
from trading_bot.meta.news.publisher import NewsPlan, PlannedPiece
from trading_bot.meta.news.trading_floor import FloorBrief
from trading_bot.meta.news.triage import TriagedCandidate

log = logging.getLogger(__name__)


# Section → CSS modifier class. Used on .section-label, .meta, .brief
# and .read-more. Must match what's defined in docs/assets/style.css.
_SECTION_CLASS: dict[str, str] = {
    "Front":           "front",
    "Markets":         "markets",
    "World":           "world",
    "Tech & science":  "tech",
    "Climate":         "world",     # reuse navy — no dedicated palette yet
    "Health":          "tech",      # reuse emerald
    "Sport":           "beyond",
    "Culture":         "beyond",
    "Beyond the tape": "beyond",
    # Synthetic sections (we add these in assembly, not in the plan)
    "Trading floor":   "desk",
    "Desk's calls":    "calls",
}

# Section render order for the front page. Synthetic sections come last.
_SECTION_ORDER = [
    "Front",
    "Markets",
    "World",
    "Tech & science",
    "Climate",
    "Health",
    "Sport",
    "Culture",
    "Beyond the tape",
    "Trading floor",
    "Desk's calls",
]


def render_news_edition(
    today: date,
    *,
    plan: NewsPlan,
    briefs: dict[str, Brief],
    articles: dict[str, FullArticle],
    triaged: list[TriagedCandidate],
    floor: list[FloorBrief],
    desks: DesksCalls,
    docs_root: Path,
    shell_fn,                          # callable: (title, body_html, current, depth, page_class) -> str
) -> Path:
    """Render the front page + every article subpage. Returns the path
    to the front page (the canonical edition URL).

    `shell_fn` is `pages._shell` — we accept it as a parameter to avoid
    a hard import cycle with the dashboard package."""
    edition_dir = docs_root / "news" / today.isoformat()
    edition_dir.mkdir(parents=True, exist_ok=True)

    # Build a lookup: slug -> piece (used everywhere)
    pieces_by_slug = {p.slug: p for p in plan.pieces}

    # 1. Per-article subpages
    for piece in plan.pieces:
        art = articles.get(piece.slug)
        if not art:
            continue
        subpage_html = _render_article_subpage(piece, art, plan, pieces_by_slug, today)
        page = shell_fn(
            title=f"{piece.headline} — {today.isoformat()}",
            body_html=subpage_html,
            current="news",
            depth=2,            # docs/news/YYYY-MM-DD/{slug}.html
            page_class="news",
        )
        (edition_dir / f"{piece.slug}.html").write_text(page)

    # 2. Front page
    front_html = _render_front_page(plan, briefs, articles, floor, desks, today)
    front_page = shell_fn(
        title=f"News — {today.isoformat()}",
        body_html=front_html,
        current="news",
        depth=2,
        page_class="news",
    )
    front_path = edition_dir / "index.html"
    front_path.write_text(front_page)

    log.info("Rendered news edition → %s (%d articles)", front_path, len(plan.pieces))
    return front_path


def _md_to_html(md_text: str) -> str:
    return md_lib.markdown(
        md_text or "",
        extensions=["tables", "fenced_code", "sane_lists", "nl2br"],
    )


def _section_class(section: str) -> str:
    return _SECTION_CLASS.get(section, "beyond")


def _generated_at() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------
# Front page
# ---------------------------------------------------------------------------

def _render_front_page(
    plan: NewsPlan,
    briefs: dict[str, Brief],
    articles: dict[str, FullArticle],
    floor: list[FloorBrief],
    desks: DesksCalls,
    today: date,
) -> str:
    # Group pieces by section, in plan order
    sections: dict[str, list[PlannedPiece]] = {}
    for p in plan.pieces:
        sections.setdefault(p.section, []).append(p)

    parts: list[str] = []
    parts.append('<main class="paper">')

    # Masthead
    parts.append(_masthead_html(plan, today))

    # Masthead strip (the meta bar under the masthead)
    parts.append(_masthead_strip_html(today, plan))

    # 1. The lead (Front section, "lead" tier)
    front_pieces = sections.get("Front", [])
    lead = next((p for p in front_pieces if p.tier == "lead"), None)
    if lead is None and front_pieces:
        lead = front_pieces[0]
    if lead is not None:
        parts.append(_render_lead(lead, briefs.get(lead.slug), articles.get(lead.slug)))

    # 2. Each non-Front section (in canonical order)
    for section in _SECTION_ORDER:
        if section == "Front":
            continue
        if section == "Trading floor":
            if floor:
                parts.append(_render_floor_section(floor))
            continue
        if section == "Desk's calls":
            if desks.fresh_predictions or desks.homework_items:
                parts.append(_render_desks_section(desks))
            continue
        pieces = sections.get(section, [])
        if not pieces:
            continue
        parts.append(_render_section(section, pieces, briefs, articles, today))

    # Colophon
    parts.append(
        '<footer class="colophon">'
        f'Auto-generated by the trading-bot daily news pipeline · {html.escape(_generated_at())}'
        '</footer>'
    )
    parts.append('</main>')
    return "\n".join(parts)


def _masthead_html(plan: NewsPlan, today: date) -> str:
    formatted = today.strftime("%A, %d %B %Y")
    subtitle = html.escape(plan.masthead_subtitle or "")
    return (
        '<header class="masthead">'
        f'  <h1>The Bot Tribune<span class="sub">— Daily, {html.escape(formatted)}</span></h1>'
        + (f'  <div class="subtitle">{subtitle}</div>' if subtitle else '')
        + '</header>'
    )


def _masthead_strip_html(today: date, plan: NewsPlan) -> str:
    return (
        '<div class="masthead-strip">'
        f'  <span><strong>Vol. {today.strftime("%y")}</strong> · No. {today.strftime("%j")}</span>'
        f'  <span>{html.escape(today.isoformat())}</span>'
        f'  <span>{len(plan.pieces)} piece{"s" if len(plan.pieces) != 1 else ""} today</span>'
        '</div>'
    )


def _render_lead(piece: PlannedPiece, brief: Brief | None, article: FullArticle | None) -> str:
    section_cls = "front"
    # Section header
    out = [
        '<div class="section-label front">'
        '  <span>Front page</span>'
        '  <span class="ord">The lead</span>'
        '</div>'
    ]
    # Lead kicker + headline + dek + body
    kicker = html.escape(piece.kicker or "LEAD")
    body_md = (brief.body_md if brief else piece.one_line) or ""
    body_html = _md_to_html(body_md)
    dek = ""
    if article and article.in_one_sentence:
        dek = f'<p class="dek">{html.escape(article.in_one_sentence)}</p>'
    elif piece.one_line:
        dek = f'<p class="dek">{html.escape(piece.one_line)}</p>'

    read_more = (
        f'<a class="read-more lead" href="{html.escape(piece.slug)}.html">Read the full article →</a>'
        if article else ""
    )
    out.append(
        '<article class="lead">'
        f'  <p class="meta lead"><span class="accent">{kicker}</span>'
        f'    <span class="dot">·</span> By {html.escape(piece.byline)}</p>'
        f'  <h2>{html.escape(piece.headline)}</h2>'
        f'  {dek}'
        f'  <div class="lead-body">{body_html}</div>'
        f'  {read_more}'
        '</article>'
    )
    return "\n".join(out)


def _render_section(
    section: str,
    pieces: list[PlannedPiece],
    briefs: dict[str, Brief],
    articles: dict[str, FullArticle],
    today: date,
) -> str:
    cls = _section_class(section)
    # Section label
    out = [
        f'<div class="section-label {cls}">'
        f'  <span>{html.escape(section)}</span>'
        f'  <span class="ord">{len(pieces)} piece{"s" if len(pieces) != 1 else ""}</span>'
        '</div>'
    ]

    # Split into features (larger) and briefs (smaller)
    features = [p for p in pieces if p.tier == "feature"]
    briefs_only = [p for p in pieces if p.tier == "brief"]

    if features:
        # Features in a 2-col grid (or 1-col if there's only one)
        grid_cls = "grid-2" if len(features) >= 2 else ""
        if grid_cls:
            out.append(f'<div class="{grid_cls}">')
        for p in features:
            out.append(_render_brief_card(p, briefs.get(p.slug), articles.get(p.slug), cls, kicker_fallback=section.upper()))
        if grid_cls:
            out.append('</div>')

    if briefs_only:
        # Briefs in a 3-col grid
        grid_cls = "grid-3" if len(briefs_only) >= 2 else ""
        if grid_cls:
            out.append(f'<div class="{grid_cls}">')
        for p in briefs_only:
            out.append(_render_brief_card(p, briefs.get(p.slug), articles.get(p.slug), cls, kicker_fallback=section.upper()))
        if grid_cls:
            out.append('</div>')

    return "\n".join(out)


def _render_brief_card(
    piece: PlannedPiece,
    brief: Brief | None,
    article: FullArticle | None,
    section_cls: str,
    *,
    kicker_fallback: str,
) -> str:
    kicker = html.escape(piece.kicker or kicker_fallback)
    body_md = (brief.body_md if brief else piece.one_line) or ""
    body_html = _md_to_html(body_md)
    read_more = ""
    if article:
        read_more = (
            f'<a class="read-more small {section_cls}" href="{html.escape(piece.slug)}.html">'
            f'Read on →</a>'
        )
    sources = ""
    if brief and brief.sources_used:
        sources = '<span class="sources">' + html.escape("Sources: " + ", ".join(brief.sources_used[:3])) + '</span>'

    return (
        f'<article class="brief {section_cls}">'
        f'  <p class="meta {section_cls}"><span class="accent">{kicker}</span>'
        f'    <span class="dot">·</span> By {html.escape(piece.byline)}</p>'
        f'  <h3>{html.escape(piece.headline)}</h3>'
        f'  {body_html}'
        f'  <div class="brief-footer">{read_more}{sources}</div>'
        f'</article>'
    )


def _render_floor_section(floor: list[FloorBrief]) -> str:
    out = [
        '<div class="section-label desk">'
        '  <span>The trading floor</span>'
        '  <span class="ord">yesterday\'s P&amp;L, in prose</span>'
        '</div>'
    ]
    grid_cls = "grid-3" if len(floor) >= 2 else ""
    if grid_cls:
        out.append(f'<div class="{grid_cls}">')
    for fb in floor:
        body_html = _md_to_html(fb.body_md)
        out.append(
            f'<article class="brief desk">'
            f'  <p class="meta desk"><span class="accent">{html.escape(fb.kicker)}</span>'
            f'    <span class="dot">·</span> By {html.escape(fb.byline)}</p>'
            f'  <h3>{html.escape(fb.headline)}</h3>'
            f'  {body_html}'
            f'</article>'
        )
    if grid_cls:
        out.append('</div>')
    return "\n".join(out)


def _render_desks_section(desks: DesksCalls) -> str:
    out = [
        '<div class="section-label calls">'
        '  <span>The desk\'s calls</span>'
        f'  <span class="ord">{len(desks.fresh_predictions)} fresh '
        f'{"call" if len(desks.fresh_predictions) == 1 else "calls"}'
        + (f', {len(desks.homework_items)} graded' if desks.homework_items else "") +
        '</span>'
        '</div>'
    ]

    # Group fresh predictions by horizon
    by_horizon: dict[str, list] = {"tomorrow": [], "this-week": [], "this-month": []}
    for p in desks.fresh_predictions:
        by_horizon.setdefault(p.horizon, []).append(p)

    horizon_labels = [("tomorrow", "Tomorrow"), ("this-week", "This week"), ("this-month", "This month")]
    for hkey, hlabel in horizon_labels:
        items = by_horizon.get(hkey, [])
        if not items:
            continue
        out.append(f'<div class="subsection">{html.escape(hlabel)}</div>')
        for p in items:
            falsif = html.escape(p.falsification_criteria)
            target = html.escape(p.target_date)
            conv = html.escape(p.conviction).upper()
            rationale = ""
            if p.grading_note:
                # The publisher stashed rationale in grading_note; surface it
                rationale = f'<p>{html.escape(p.grading_note)}</p>'
            out.append(
                '<article class="pred">'
                f'  <p class="meta calls"><span class="accent">{conv} CONVICTION</span>'
                f'    <span class="dot">·</span> Target {target}</p>'
                f'  <h3>{html.escape(p.claim)}</h3>'
                f'  {rationale}'
                f'  <div class="falsif"><strong>Falsified if:</strong> {falsif}</div>'
                '</article>'
            )

    # Marking the homework
    if desks.homework_items:
        out.append('<div class="subsection">Marking the homework</div>')
        out.append('<div class="grid-2">')
        for p in desks.homework_items:
            status_cls = p.status if p.status in {"proven", "partial", "falsified", "still-open"} else "open"
            verdict_label = {
                "proven": "Proven",
                "partial": "Partial",
                "falsified": "Falsified",
                "still-open": "Still open",
            }.get(p.status, p.status)
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


# ---------------------------------------------------------------------------
# Per-article subpage
# ---------------------------------------------------------------------------

def _render_article_subpage(
    piece: PlannedPiece,
    article: FullArticle,
    plan: NewsPlan,
    pieces_by_slug: dict[str, PlannedPiece],
    today: date,
) -> str:
    kicker = html.escape(piece.kicker or piece.section.upper())
    body_html = _md_to_html(article.body_md)

    # Hero image
    hero_html = ""
    if article.image_url:
        cap_parts = []
        if article.image_caption:
            cap_parts.append(html.escape(article.image_caption))
        if article.image_credit:
            cap_parts.append(f'<em>{html.escape(article.image_credit)}</em>')
        caption = " — ".join(cap_parts) if cap_parts else ""
        caption_html = f'<div class="hero-caption">{caption}</div>' if caption else ""
        hero_html = (
            f'<img class="hero" src="{html.escape(article.image_url)}" '
            f'alt="{html.escape(article.image_caption or piece.headline)}">'
            + caption_html
        )

    # In-one-sentence callout
    callout_html = ""
    if article.in_one_sentence:
        callout_html = (
            '<div class="callout" style="border-left-color: var(--c-front);">'
            '<div class="callout-label" style="color: var(--c-front);">In one sentence</div>'
            f'<div>{html.escape(article.in_one_sentence)}</div>'
            '</div>'
        )

    # Sources block
    sources_html = ""
    if article.sources:
        items = []
        for s in article.sources:
            title = html.escape(s.get("title", "") or "untitled")
            url = s.get("url") or ""
            if url:
                items.append(f'<li><a href="{html.escape(url)}" rel="nofollow noopener" target="_blank">{title}</a></li>')
            else:
                items.append(f'<li>{title}</li>')
        sources_html = (
            '<section class="sources-block">'
            '<h4>Sources</h4>'
            f'<ol>{"".join(items)}</ol>'
            '</section>'
        )

    # Related articles (within this edition)
    related_html = ""
    related = []
    for slug in article.related_slugs:
        p = pieces_by_slug.get(slug)
        if p and slug != piece.slug:
            related.append(p)
    if related:
        items = []
        for p in related[:4]:
            section_tag = html.escape(p.section.upper())
            items.append(
                f'<a href="{html.escape(p.slug)}.html">'
                f'<span style="display:block;font-family:var(--sans);font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--ink-muted);margin-bottom:0.2rem;">{section_tag}</span>'
                f'{html.escape(p.headline)}'
                '</a>'
            )
        related_html = (
            '<section class="related">'
            '<h4>Related in this edition</h4>'
            f'{"".join(items)}'
            '</section>'
        )

    return (
        '<main class="paper article-page">'
        '<a class="back-link" href="index.html">← Back to the front page</a>'
        f'<div class="article-meta">{kicker} · By {html.escape(piece.byline)} · {html.escape(today.isoformat())}</div>'
        f'<h1>{html.escape(piece.headline)}</h1>'
        f'{hero_html}'
        f'{callout_html}'
        f'<div class="body">{body_html}</div>'
        f'{sources_html}'
        f'{related_html}'
        '<footer class="colophon">'
        f'The Bot Tribune · {html.escape(today.isoformat())} · '
        f'<a href="index.html">Back to the edition</a>'
        '</footer>'
        '</main>'
    )
