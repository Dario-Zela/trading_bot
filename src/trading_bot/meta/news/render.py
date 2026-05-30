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
    "Front":            "front",
    "Standing watches": "world",    # reuse navy until a dedicated palette lands
    "Markets":          "markets",
    "World":            "world",
    "Tech & science":   "tech",
    "Climate":          "world",     # reuse navy — no dedicated palette yet
    "Health":           "tech",      # reuse emerald
    "Sport":            "beyond",
    "Culture":          "beyond",
    "Beyond the tape":  "beyond",
    # Synthetic sections (we add these in assembly, not in the plan)
    "Trading floor":    "desk",
    "Desk's calls":     "calls",
}

# Section render order for the front page. Standing watches sits right
# below Front so the reader sees "what's NEW" (Front) then "what's
# STILL HAPPENING" (Standing watches) before drilling into Markets etc.
_SECTION_ORDER = [
    "Front",
    "Standing watches",
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

    # Compute prev/next once so every subpage carries the same nav
    prev_iso, next_iso = _neighbouring_news_dates(today, edition_dir.parent)
    prev_url = f"../{prev_iso}/" if prev_iso else ""
    next_url = f"../{next_iso}/" if next_iso else ""

    # 1. Per-article subpages
    for piece in plan.pieces:
        art = articles.get(piece.slug)
        if not art:
            continue
        subpage_html = _render_article_subpage(
            piece, art, plan, pieces_by_slug, today,
            prev_url=prev_url, next_url=next_url,
            prev_label=prev_iso or "", next_label=next_iso or "",
        )
        page = shell_fn(
            title=f"{piece.headline} — {today.isoformat()}",
            body_html=subpage_html,
            current="news",
            depth=2,            # docs/news/YYYY-MM-DD/{slug}.html
            page_class="news",
        )
        (edition_dir / f"{piece.slug}.html").write_text(page)

    # 2. Front page
    front_html = _render_front_page(plan, briefs, articles, floor, desks, today, edition_dir)

    # 3. Update `docs/news/latest.html` so the masthead always links to the
    # newest edition (one-click "latest" from any page in the publication).
    _write_latest_redirect(edition_dir.parent, edition_dir.name)

    # 4. Refresh the sidecar editions.json that the JS upgrade reads on
    # every page load to keep prev/next current.
    _write_editions_index(edition_dir.parent, kind="news")
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


_MARKDOWN_NOISE_RE = re.compile(r"[#`*_>\[\]()!\-]+")
_WORDS_PER_MINUTE = 225
_URL_RE = re.compile(r"https?://([^/\s]+)(?:/.*)?", re.IGNORECASE)


def _shorten_source(s: str) -> str:
    """Collapse 'https://www.foo.com/some/long/path...' → 'foo.com'.
    Leave non-URL source names alone (they're already short)."""
    m = _URL_RE.fullmatch(s.strip())
    if not m:
        return s.strip()
    host = m.group(1).lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _estimate_read_minutes(md_text: str) -> int:
    """Estimate minutes-to-read from a markdown body. Strips markdown
    syntax noise, counts whitespace-separated tokens, divides by 225
    words/min, rounds up to the nearest minute (minimum 1)."""
    if not md_text:
        return 0
    cleaned = _MARKDOWN_NOISE_RE.sub(" ", md_text)
    n_words = len(cleaned.split())
    if n_words <= 0:
        return 0
    # Round up so a 230-word piece reads "2 min" not "1 min"
    return max(1, (n_words + _WORDS_PER_MINUTE - 1) // _WORDS_PER_MINUTE)


def _read_badge(minutes: int) -> str:
    """Small inline badge — '· 4 min'. Empty string for 0 minutes."""
    if minutes <= 0:
        return ""
    return f' <span style="opacity:0.7;font-weight:400;">· {minutes} min</span>'


def _section_class(section: str) -> str:
    return _SECTION_CLASS.get(section, "beyond")


def _generated_at() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _write_editions_index(parent_dir: Path, *, kind: str) -> None:
    """List sibling edition directories newest-first to a JSON file that
    the in-page nav JS fetches to keep prev/next current.

    News editions are dated YYYY-MM-DD; macro editions are ISO weeks
    YYYY-W##. We sort lexicographically, which works for both."""
    if kind == "news":
        pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    elif kind == "macro":
        pattern = re.compile(r"^\d{4}-W\d{2}$")
    else:
        return

    # Phase 10D — dedup by `id`. iterdir() should already be unique on
    # disk but a defensive set() also handles symlinks pointing at the
    # same week which can happen if a macro re-run wrote to a slightly
    # different dir name and then someone normalised.
    seen_ids: set[str] = set()
    editions: list[dict[str, str]] = []
    if parent_dir.exists():
        for child in sorted(parent_dir.iterdir(), reverse=True):
            if not child.is_dir():
                continue
            if not pattern.match(child.name):
                continue
            if not (child / "index.html").exists():
                continue
            if child.name in seen_ids:
                continue
            seen_ids.add(child.name)
            editions.append({"id": child.name, "url": f"../{child.name}/"})

    out_path = parent_dir / "editions.json"
    import json as _json
    out_path.write_text(_json.dumps(editions, separators=(",", ":")))


def _write_latest_redirect(parent_dir: Path, target_name: str) -> None:
    """Write parent_dir/latest.html as a meta-refresh redirect to
    target_name/. Idempotent — re-running for the same date is a no-op,
    re-running for a newer date overwrites with the newer target."""
    latest_path = parent_dir / "latest.html"
    # Only overwrite if the new target is lexically newer than the
    # existing one (so re-rendering an OLD edition doesn't repoint
    # latest backward).
    if latest_path.exists():
        try:
            existing = latest_path.read_text()
            # Pull the existing target out of the meta refresh
            existing_m = re.search(r'url=([^/"\']+)/?', existing)
            if existing_m and existing_m.group(1) > target_name:
                return
        except OSError:
            pass
    redirect_html = (
        '<!DOCTYPE html>\n'
        '<html><head>\n'
        f'<meta http-equiv="refresh" content="0; url={html.escape(target_name)}/">\n'
        f'<link rel="canonical" href="{html.escape(target_name)}/">\n'
        f'<title>Redirecting to latest edition…</title>\n'
        '</head><body>\n'
        f'<p>Redirecting to <a href="{html.escape(target_name)}/">{html.escape(target_name)}</a>…</p>\n'
        '</body></html>\n'
    )
    latest_path.write_text(redirect_html)


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
    edition_dir: Path,
) -> str:
    # Group pieces by section, in plan order
    sections: dict[str, list[PlannedPiece]] = {}
    for p in plan.pieces:
        sections.setdefault(p.section, []).append(p)

    parts: list[str] = []
    parts.append('<main class="paper">')

    # Masthead
    parts.append(_masthead_html(plan, today, edition_dir))

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


def _masthead_html(plan: NewsPlan, today: date, edition_dir: Path) -> str:
    formatted = today.strftime("%A, %d %B %Y")
    subtitle = html.escape(plan.masthead_subtitle or "")
    question = html.escape(plan.todays_question or "")
    question_html = ""
    if question:
        # Class only — margin/typography lives in style.css so the
        # subtitle adjacent-sibling rule can space it properly.
        question_html = (
            '  <div class="subtitle todays-question">'
            f'<span class="label">Today\'s question · </span>{question}'
            '</div>'
        )

    # Prev / next arrows from the on-disk listing of edition dirs.
    prev_iso, next_iso = _neighbouring_news_dates(today, edition_dir.parent)
    nav_html = _edition_nav_html(
        prev_url=f"../{prev_iso}/" if prev_iso else "",
        next_url=f"../{next_iso}/" if next_iso else "",
        latest_url="../latest.html",
        prev_label=prev_iso or "",
        next_label=next_iso or "",
        edition_id=today.isoformat(),
        kind="news",
    )

    return (
        '<header class="masthead">'
        f'  <h1><a href="../latest.html" class="masthead-link">The Bot Tribune</a>'
        f'<span class="sub">— Daily, {html.escape(formatted)}</span></h1>'
        + nav_html
        + (f'  <div class="subtitle">{subtitle}</div>' if subtitle else '')
        + question_html
        + '</header>'
    )


def _neighbouring_news_dates(today: date, news_dir: Path) -> tuple[str | None, str | None]:
    """Return (previous_iso, next_iso) for navigation from `today`,
    based on the on-disk YYYY-MM-DD directories in `news_dir`. Returns
    (None, None) if `news_dir` doesn't exist yet."""
    if not news_dir.exists():
        return None, None
    dates = []
    for child in news_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            datetime.strptime(child.name, "%Y-%m-%d")
        except ValueError:
            continue
        if (child / "index.html").exists() or child.name == today.isoformat():
            dates.append(child.name)
    dates.sort()
    iso = today.isoformat()
    if iso not in dates:
        dates.append(iso)
        dates.sort()
    i = dates.index(iso)
    prev_iso = dates[i - 1] if i > 0 else None
    next_iso = dates[i + 1] if i < len(dates) - 1 else None
    return prev_iso, next_iso


def _edition_nav_html(*, prev_url: str, next_url: str, latest_url: str,
                      prev_label: str, next_label: str,
                      archive_url: str = "../index.html",
                      edition_id: str = "",
                      kind: str = "news") -> str:
    """A small arrow strip under the masthead H1. Prev/next are seeded
    from the on-disk state at render time, BUT each anchor carries a
    data-role attribute so a tiny JS snippet can upgrade them on page
    load by fetching `../editions.json`. That way an old edition's
    'next →' picks up newer editions published after it without
    having to re-render the old HTML."""
    def _arrow(role: str, direction: str, url: str, label: str) -> str:
        if not url:
            return (
                f'<a class="edition-nav-link disabled" data-role="{role}" '
                f'href="#" aria-disabled="true">{direction}</a>'
            )
        return (
            f'<a class="edition-nav-link" data-role="{role}" '
            f'href="{html.escape(url)}" title="{html.escape(label)}">{direction}</a>'
        )
    return (
        f'<div class="edition-nav" data-edition="{html.escape(edition_id)}" data-kind="{html.escape(kind)}">'
        + _arrow("prev", "← prev", prev_url, prev_label)
        + f'<a class="edition-nav-link latest" href="{html.escape(latest_url)}">latest ↑</a>'
        + _arrow("next", "next →", next_url, next_label)
        + f'<a class="edition-nav-link all" href="{html.escape(archive_url)}">all</a>'
        + '</div>'
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

    read_more = ""
    if article:
        minutes = _estimate_read_minutes(article.body_md)
        read_more = (
            f'<a class="read-more lead" href="{html.escape(piece.slug)}.html">'
            f'Read the full article →{_read_badge(minutes)}</a>'
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
        minutes = _estimate_read_minutes(article.body_md)
        read_more = (
            f'<a class="read-more small {section_cls}" href="{html.escape(piece.slug)}.html">'
            f'Read on →{_read_badge(minutes)}</a>'
        )
    sources = ""
    if brief and brief.sources_used:
        # Dedupe + shorten URL-shaped sources to their hostname so long URLs
        # don't clip into neighbouring text.
        shortened: list[str] = []
        seen: set[str] = set()
        for s in brief.sources_used[:6]:
            short = _shorten_source(s)
            if short and short.lower() not in seen:
                seen.add(short.lower())
                shortened.append(short)
            if len(shortened) >= 3:
                break
        if shortened:
            sources = '<span class="sources">' + html.escape("Sources: " + ", ".join(shortened)) + '</span>'

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
    *,
    prev_url: str = "",
    next_url: str = "",
    prev_label: str = "",
    next_label: str = "",
    edition_id: str = "",
    kind: str = "news",
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

    minutes = _estimate_read_minutes(article.body_md)
    read_time = f" · {minutes} min read" if minutes > 0 else ""
    # Edition-nav strip — same arrows the front page has, so the user
    # can move between editions without backing out. The data-edition
    # attribute lets the upgrade JS keep prev/next current as new
    # editions land.
    subpage_nav = _edition_nav_html(
        prev_url=prev_url,
        next_url=next_url,
        latest_url="../latest.html",
        prev_label=prev_label,
        next_label=next_label,
        archive_url="../index.html",
        edition_id=edition_id or today.isoformat(),
        kind=kind,
    )
    return (
        '<main class="paper article-page">'
        + subpage_nav
        + '<a class="back-link" href="index.html">← Back to the front page</a>'
        f'<div class="article-meta">{kicker} · By {html.escape(piece.byline)} · {html.escape(today.isoformat())}{html.escape(read_time)}</div>'
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
