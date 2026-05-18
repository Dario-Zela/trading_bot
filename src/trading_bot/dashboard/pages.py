"""Static HTML page rendering for the trading-bot site.

Renders four kinds of page from structured data produced upstream:

- News editions (multi-stage agent output) → `docs/news/YYYY-MM-DD/index.html`
  plus per-article subpages at `docs/news/YYYY-MM-DD/{slug}.html`.
- Macro views (weekly multi-desk output) → `docs/macro/YYYY-W##/index.html`
  plus per-piece subpages at `docs/macro/YYYY-W##/{slug}.html`.
- Evolution log (weekly per-strategy reports) → `docs/evolution.html`.
- Archive indices listing all editions / views.

The render layer is HTML-template based: agents produce structured JSON
(headlines, markdown bodies, sections, predictions). This module turns
that into HTML. The LLM never emits final HTML — too risky for layout
stability.

Shared shell (masthead-strip, app nav, font picker, colophon) is provided
by `_shell()`. Pages link to `docs/assets/style.css` for everything else.

For backwards compatibility this module still exposes `render_news_pages`,
`render_macro_pages`, `render_evolution_page`, and `rebuild_all_pages` —
they now drive the new shell + structure but operate on the existing
markdown files in `state/`. Once Phase 2/3 land, the multi-stage agents
will write structured JSON and these legacy entry points will pivot.
"""
from __future__ import annotations

import html
import json
import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path

import markdown as md_lib

from trading_bot.state.paths import STATE_ROOT


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

_NAV_ITEMS = (
    ("dashboard", "Dashboard", "index.html"),
    ("news",      "News",      "news/index.html"),
    ("macro",     "Macro",     "macro/index.html"),
    ("evolution", "Evolution", "evolution.html"),
)

_FONTS_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400'
    '&family=Source+Serif+4:ital,wght@0,400;0,600;1,400'
    '&family=Source+Sans+3:wght@400;500;600;700'
    '&family=Newsreader:ital,wght@0,400;0,600;0,800;1,400'
    '&family=Crimson+Pro:ital,wght@0,400;0,600;0,800;1,400'
    '&family=Lora:ital,wght@0,400;0,600;0,700;1,400'
    '&family=Inter:wght@400;500;600;700'
    '&family=IBM+Plex+Sans:wght@400;500;600;700'
    '&family=Lato:wght@400;700'
    '&display=swap" rel="stylesheet">'
)

_FONT_PICKER_SCRIPT = """
<script>
const BOT_FONTS = {
  classic:    { head: "'Playfair Display', Georgia, serif", body: "'Source Serif 4', Georgia, serif", sans: "'Source Sans 3', sans-serif" },
  newsreader: { head: "'Newsreader', Georgia, serif",       body: "'Newsreader', Georgia, serif",      sans: "'Inter', sans-serif" },
  crimson:    { head: "'Crimson Pro', Georgia, serif",      body: "'Crimson Pro', Georgia, serif",     sans: "'IBM Plex Sans', sans-serif" },
  lora:       { head: "'Lora', Georgia, serif",             body: "'Lora', Georgia, serif",            sans: "'Lato', sans-serif" },
};
function applyBotFont(name) {
  const f = BOT_FONTS[name] || BOT_FONTS.classic;
  const r = document.documentElement.style;
  r.setProperty('--serif-head', f.head);
  r.setProperty('--serif-body', f.body);
  r.setProperty('--sans', f.sans);
  try { localStorage.setItem('botFont', name); } catch(e) {}
  document.querySelectorAll('.font-picker button').forEach(b => {
    b.classList.toggle('active', b.dataset.font === name);
  });
}
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.font-picker button').forEach(b => {
    b.addEventListener('click', () => applyBotFont(b.dataset.font));
  });
  let saved = 'classic';
  try { saved = localStorage.getItem('botFont') || 'classic'; } catch(e) {}
  applyBotFont(saved);
});
</script>
"""


def docs_root() -> Path:
    return Path(__file__).resolve().parents[3] / "docs"


def _up_to_root(depth: int) -> str:
    return "../" * depth if depth > 0 else "./"


def _nav_html(current: str, depth: int) -> str:
    up = _up_to_root(depth)
    items = []
    for ident, label, path in _NAV_ITEMS:
        cls = "nav-tab active" if ident == current else "nav-tab"
        items.append(f'<a class="{cls}" href="{up}{path}">{label}</a>')
    return "\n".join(items)


def _font_picker_html() -> str:
    return (
        '<div class="font-picker">'
        '<span class="label">Font</span>'
        '<button data-font="classic">Classic</button>'
        '<button data-font="newsreader">Newsreader</button>'
        '<button data-font="crimson">Crimson</button>'
        '<button data-font="lora">Lora</button>'
        "</div>"
    )


def _shell(
    *,
    title: str,
    body_html: str,
    current: str,
    depth: int,
    page_class: str = "",
    extra_nav: str = "",
) -> str:
    """Wrap a page body in the shared shell — head, nav, font picker,
    Google Fonts, stylesheet link, colophon. `depth` controls how many
    `../` we prepend to relative paths (0 = file is in docs/, 1 = in
    docs/news/, 2 = in docs/news/2026-05-18/)."""
    up = _up_to_root(depth)
    body_class = f"page-{page_class}" if page_class else ""
    return f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{html.escape(title)} — Trading Bot</title>
  {_FONTS_LINK}
  <link rel="stylesheet" href="{up}assets/style.css">
</head>
<body class="{body_class}">
  <nav class="app-nav">
    <span class="brand">⚡ TRADING BOT</span>
    {_nav_html(current, depth)}
    <span class="spacer"></span>
    {extra_nav}
    {_font_picker_html()}
  </nav>
  {body_html}
  {_FONT_PICKER_SCRIPT}
</body>
</html>
"""


def _generated_at() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _md_to_html(md_text: str) -> str:
    return md_lib.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "nl2br"],
    )


# ---------------------------------------------------------------------------
# Daily news pages — Phase 1 backwards-compat path
# Reads existing state/daily_news/*.md and renders them with the new shell.
# Phase 2 replaces this with structured-JSON-driven rendering.
# ---------------------------------------------------------------------------

def _news_md_dir() -> Path:
    return STATE_ROOT / "daily_news"


def _news_out_dir() -> Path:
    p = docs_root() / "news"
    p.mkdir(parents=True, exist_ok=True)
    return p


def render_news_pages() -> int:
    """Render every daily-news markdown file. Returns the number of brief
    pages written (not counting the archive index)."""
    src_dir = _news_md_dir()
    out_dir = _news_out_dir()

    entries: list[tuple[str, Path]] = []
    if src_dir.exists():
        for src in sorted(src_dir.glob("*.md")):
            if src.stem.endswith(".bot"):
                continue  # skip bot-summary companions
            date_str = src.stem
            out_path = out_dir / f"{date_str}.html"
            body = _md_to_html(src.read_text())
            paper = (
                f'<main class="paper">'
                f'<header class="masthead">'
                f'  <h1>The Bot Tribune<span class="sub">— Daily, {date_str}</span></h1>'
                f'</header>'
                f'<div class="masthead-strip">'
                f'  <span><strong>News</strong></span>'
                f'  <span>{html.escape(date_str)}</span>'
                f'  <span>Rendered {html.escape(_generated_at())}</span>'
                f'</div>'
                f'<article>{body}</article>'
                f'<footer class="colophon">Auto-generated by the trading-bot daily news pipeline.</footer>'
                f'</main>'
            )
            page = _shell(
                title=f"News — {date_str}",
                body_html=paper,
                current="news",
                depth=1,
                page_class="news",
            )
            out_path.write_text(page)
            entries.append((date_str, out_path))
    _write_news_index(entries)
    log.info("Rendered %d daily news pages → %s", len(entries), out_dir)
    return len(entries)


def _write_news_index(entries: list[tuple[str, Path]]) -> None:
    out_dir = _news_out_dir()
    if not entries:
        body = (
            '<main class="paper">'
            '<header class="masthead"><h1>The Bot Tribune<span class="sub">— News archive</span></h1></header>'
            '<p style="text-align:center;font-style:italic;color:var(--ink-muted);margin-top:2rem;">'
            'No daily news briefs yet. The agent runs each weekday morning.</p>'
            '</main>'
        )
        (out_dir / "index.html").write_text(
            _shell(title="News archive", body_html=body, current="news", depth=1, page_class="news")
        )
        return
    entries_desc = sorted(entries, key=lambda e: e[0], reverse=True)
    items = "\n".join(
        f'<li><a href="{html.escape(p.name)}">{html.escape(d)}</a>'
        f'<span class="muted">daily news brief</span></li>'
        for d, p in entries_desc
    )
    body = (
        '<main class="paper">'
        '<header class="masthead">'
        '  <h1>The Bot Tribune<span class="sub">— News archive</span></h1>'
        f'  <div class="subtitle">{len(entries)} brief{"s" if len(entries) != 1 else ""} on file. Most recent first.</div>'
        '</header>'
        '<div class="masthead-strip">'
        '  <span><strong>News archive</strong></span>'
        f'  <span>{html.escape(_generated_at())}</span>'
        '  <span></span>'
        '</div>'
        f'<ul class="index-list">{items}</ul>'
        '</main>'
    )
    (out_dir / "index.html").write_text(
        _shell(title="News archive", body_html=body, current="news", depth=1, page_class="news")
    )


# ---------------------------------------------------------------------------
# Weekly macro pages — same backwards-compat shape
# ---------------------------------------------------------------------------

def _macro_md_dir() -> Path:
    return STATE_ROOT / "macro" / "views"


def _macro_out_dir() -> Path:
    p = docs_root() / "macro"
    p.mkdir(parents=True, exist_ok=True)
    return p


def render_macro_pages() -> int:
    src_dir = _macro_md_dir()
    out_dir = _macro_out_dir()

    entries: list[tuple[str, Path]] = []
    if src_dir.exists():
        for src in sorted(src_dir.glob("*.md")):
            week_id = src.stem
            out_path = out_dir / f"{week_id}.html"
            body = _md_to_html(src.read_text())
            paper = (
                '<main class="paper">'
                '<header class="masthead">'
                f'  <h1>The Bot Tribune<span class="sub">— Macro, {html.escape(week_id)}</span></h1>'
                '</header>'
                '<div class="masthead-strip">'
                '  <span><strong>Macro</strong></span>'
                f'  <span>{html.escape(week_id)}</span>'
                f'  <span>Rendered {html.escape(_generated_at())}</span>'
                '</div>'
                f'<article>{body}</article>'
                '<footer class="colophon">Auto-generated by the weekly-macro agent.</footer>'
                '</main>'
            )
            page = _shell(
                title=f"Macro — {week_id}",
                body_html=paper,
                current="macro",
                depth=1,
                page_class="macro",
            )
            out_path.write_text(page)
            entries.append((week_id, out_path))
    _write_macro_index(entries)
    log.info("Rendered %d macro view pages → %s", len(entries), out_dir)
    return len(entries)


def _write_macro_index(entries: list[tuple[str, Path]]) -> None:
    out_dir = _macro_out_dir()
    if not entries:
        body = (
            '<main class="paper">'
            '<header class="masthead"><h1>The Bot Tribune<span class="sub">— Macro archive</span></h1></header>'
            '<p style="text-align:center;font-style:italic;color:var(--ink-muted);margin-top:2rem;">'
            'No macro views yet. The agent runs every Sunday evening.</p>'
            '</main>'
        )
        (out_dir / "index.html").write_text(
            _shell(title="Macro archive", body_html=body, current="macro", depth=1, page_class="macro")
        )
        return
    entries_desc = sorted(entries, key=lambda e: e[0], reverse=True)
    items = "\n".join(
        f'<li><a href="{html.escape(p.name)}">{html.escape(d)}</a>'
        f'<span class="muted">macro view</span></li>'
        for d, p in entries_desc
    )
    body = (
        '<main class="paper">'
        '<header class="masthead">'
        '  <h1>The Bot Tribune<span class="sub">— Macro archive</span></h1>'
        f'  <div class="subtitle">{len(entries)} view{"s" if len(entries) != 1 else ""} on file. Most recent first.</div>'
        '</header>'
        '<div class="masthead-strip">'
        '  <span><strong>Macro archive</strong></span>'
        f'  <span>{html.escape(_generated_at())}</span>'
        '  <span></span>'
        '</div>'
        f'<ul class="index-list">{items}</ul>'
        '</main>'
    )
    (out_dir / "index.html").write_text(
        _shell(title="Macro archive", body_html=body, current="macro", depth=1, page_class="macro")
    )


# ---------------------------------------------------------------------------
# Evolution log
# ---------------------------------------------------------------------------

def _evolution_md_path() -> Path:
    return STATE_ROOT / "evolution.md"


def render_evolution_page() -> bool:
    src = _evolution_md_path()
    out = docs_root() / "evolution.html"
    if not src.exists():
        body = (
            '<main class="paper">'
            '<header class="masthead"><h1>The Bot Tribune<span class="sub">— Evolution log</span></h1></header>'
            '<p style="text-align:center;font-style:italic;color:var(--ink-muted);margin-top:2rem;">'
            'No evolution entries yet. The agent runs every Saturday morning.</p>'
            '</main>'
        )
        out.write_text(_shell(
            title="Evolution log", body_html=body, current="evolution",
            depth=0, page_class="evolution",
        ))
        return False
    body = _md_to_html(src.read_text())
    paper = (
        '<main class="paper">'
        '<header class="masthead">'
        '  <h1>The Bot Tribune<span class="sub">— Evolution log</span></h1>'
        '</header>'
        '<div class="masthead-strip">'
        '  <span><strong>Evolution</strong></span>'
        f'  <span>Rendered {html.escape(_generated_at())}</span>'
        '  <span></span>'
        '</div>'
        f'<article>{body}</article>'
        '<footer class="colophon">Auto-generated by the weekly-evolution agent.</footer>'
        '</main>'
    )
    out.write_text(_shell(
        title="Evolution log", body_html=paper, current="evolution",
        depth=0, page_class="evolution",
    ))
    log.info("Rendered evolution.html")
    return True


# ---------------------------------------------------------------------------
# Public composite
# ---------------------------------------------------------------------------

def rebuild_all_pages() -> dict:
    """Re-render every static page in one call. Called from dashboard.build
    after every pipeline run so news/macro/evolution stay synchronised with
    the latest state."""
    n_news = render_news_pages()
    n_macro = render_macro_pages()
    have_evolution = render_evolution_page()
    return {"news": n_news, "macro": n_macro, "evolution": have_evolution}


def pages_url(path: str) -> str:
    base = "https://dario-zela.github.io/trading_bot"
    return f"{base}/{path.lstrip('/')}"


def news_url_for(d: date) -> str:
    return pages_url(f"news/{d.isoformat()}.html")
