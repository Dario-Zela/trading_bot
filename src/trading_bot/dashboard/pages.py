"""Static HTML page rendering for content stored as markdown in state/.

Three artifacts get web pages on GitHub Pages:
- Daily news briefs (state/daily_news/*.md → docs/news/*.html)
- Weekly macro views (state/macro/views/*.md → docs/macro/*.html)
- Evolution log (state/evolution.md → docs/evolution.html)

Each page shares a small HTML shell with nav tabs that match the main
dashboard's header so the four views feel like one site. Markdown is
rendered server-side (Python `markdown` lib) into HTML so the pages
work without JavaScript.
"""
from __future__ import annotations

import html
import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path

import markdown as md_lib

from trading_bot.state.paths import STATE_ROOT


log = logging.getLogger(__name__)


_NAV_ITEMS = (
    ("dashboard", "Dashboard"),
    ("news", "News"),
    ("macro", "Macro"),
    ("evolution", "Evolution"),
)


def docs_root() -> Path:
    # src/trading_bot/dashboard/pages.py → repo root is 4 parents up
    return Path(__file__).resolve().parents[3] / "docs"


def _nav_html(current: str, depth: int) -> str:
    """Build the nav-tabs HTML for a given section + nesting depth.
    `depth` is how many `../` to prepend (0 = file is in docs/, 1 = in
    docs/news/, etc.)."""
    up = "../" * depth
    hrefs = {
        "dashboard": f"{up}index.html",
        "news":      f"{up}news/index.html",
        "macro":     f"{up}macro/index.html",
        "evolution": f"{up}evolution.html",
    }
    out = []
    for ident, label in _NAV_ITEMS:
        cls = "nav-tab active" if ident == current else "nav-tab"
        out.append(f'<a class="{cls}" href="{hrefs[ident]}">{label}</a>')
    return '<nav class="nav-tabs">\n' + "\n".join(out) + "\n</nav>"


def _shell(*, title: str, body_html: str, current: str, depth: int, subtitle: str = "") -> str:
    """Wrap rendered markdown in the standard page shell."""
    sub = f'<div class="page-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    nav = _nav_html(current, depth)
    return f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{html.escape(title)} — Trading Bot</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
  <style>
    :root {{
      --positive: #16a34a;
      --negative: #dc2626;
      --accent: var(--pico-primary);
    }}
    body {{ margin: 0; }}
    .app {{ max-width: 880px; margin: 0 auto; padding: 1.5rem 2rem 4rem; }}
    .topbar {{
      display: flex; align-items: center; gap: 1rem;
      padding-bottom: 1rem; border-bottom: 1px solid var(--pico-muted-border-color);
      margin-bottom: 0.5rem;
    }}
    .topbar h1 {{ margin: 0; font-size: 1.3rem; letter-spacing: -0.01em; }}
    .topbar .spacer {{ flex: 1; }}
    .nav-tabs {{ display: flex; gap: 0; }}
    .nav-tab {{
      padding: 0.4rem 0.9rem;
      font-size: 0.85rem;
      font-weight: 500;
      color: var(--pico-muted-color);
      text-decoration: none;
      border-bottom: 2px solid transparent;
      margin-bottom: -1px;
    }}
    .nav-tab:hover {{ color: var(--accent); }}
    .nav-tab.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
    .page-subtitle {{
      color: var(--pico-muted-color);
      font-size: 0.85rem;
      margin: 0.5rem 0 1.5rem;
    }}
    .page-content {{ font-size: 1rem; line-height: 1.65; }}
    .page-content h1 {{ font-size: 1.8rem; letter-spacing: -0.01em; margin-top: 2rem; }}
    .page-content h2 {{ font-size: 1.35rem; margin-top: 2rem; }}
    .page-content h3 {{ font-size: 1.1rem; margin-top: 1.5rem; }}
    .page-content h4 {{
      font-size: 1rem; margin-top: 1.25rem;
      color: var(--pico-h3-color);
    }}
    .page-content table {{ font-size: 0.9rem; margin: 1rem 0; }}
    .page-content table th {{ background: var(--pico-muted-border-color); }}
    .page-content blockquote {{
      border-left: 3px solid var(--accent);
      margin-left: 0; padding-left: 1rem;
      color: var(--pico-muted-color);
    }}
    .page-content code {{ font-size: 0.85em; }}
    .page-content hr {{ border-top: 1px solid var(--pico-muted-border-color); margin: 2rem 0; }}
    .index-list {{ list-style: none; padding: 0; }}
    .index-list li {{
      padding: 0.65rem 0;
      border-bottom: 1px solid var(--pico-muted-border-color);
    }}
    .index-list li a {{ font-weight: 600; text-decoration: none; }}
    .index-list .muted {{ color: var(--pico-muted-color); font-size: 0.85rem; margin-left: 0.4rem; }}
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <h1>Trading Bot</h1>
      <div class="spacer"></div>
      {nav}
    </header>
    <h2 style="margin: 1.5rem 0 0;">{html.escape(title)}</h2>
    {sub}
    <main class="page-content">
      {body_html}
    </main>
  </div>
</body>
</html>
"""


def _md_to_html(md_text: str) -> str:
    """Render markdown with tables, fenced code, smart links."""
    return md_lib.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "nl2br"],
    )


def _generated_at() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------
# Daily news pages
# ---------------------------------------------------------------------------

def _news_md_dir() -> Path:
    return STATE_ROOT / "daily_news"


def _news_out_dir() -> Path:
    p = docs_root() / "news"
    p.mkdir(parents=True, exist_ok=True)
    return p


def render_news_pages() -> int:
    """Render every daily-news markdown file into docs/news/*.html plus an
    index. Returns the number of brief pages written (not counting index)."""
    src_dir = _news_md_dir()
    if not src_dir.exists():
        log.info("No daily_news markdown directory — skipping news pages")
        _write_empty_news_index()
        return 0

    entries: list[tuple[str, Path]] = []
    for src in sorted(src_dir.glob("*.md")):
        date_str = src.stem
        out_path = _news_out_dir() / f"{date_str}.html"
        body_html = _md_to_html(src.read_text())
        page = _shell(
            title=f"Daily news brief — {date_str}",
            body_html=body_html,
            current="news",
            depth=1,
            subtitle=f"Generated by the daily-news-brief agent · rendered {_generated_at()}",
        )
        out_path.write_text(page)
        entries.append((date_str, out_path))

    _write_news_index(entries)
    log.info("Rendered %d daily news pages → %s", len(entries), _news_out_dir())
    return len(entries)


def _write_news_index(entries: list[tuple[str, Path]]) -> None:
    if not entries:
        _write_empty_news_index()
        return
    entries_desc = sorted(entries, key=lambda e: e[0], reverse=True)
    items = "\n".join(
        f'<li><a href="{html.escape(p.name)}">{html.escape(d)}</a>'
        f'<span class="muted">daily news brief</span></li>'
        for d, p in entries_desc
    )
    body = f'<ul class="index-list">\n{items}\n</ul>'
    page = _shell(
        title="Daily news briefs",
        body_html=body,
        current="news",
        depth=1,
        subtitle=f"{len(entries)} brief{'s' if len(entries) != 1 else ''} archived",
    )
    (_news_out_dir() / "index.html").write_text(page)


def _write_empty_news_index() -> None:
    body = '<p class="page-content">No daily news briefs yet. The agent runs each weekday morning.</p>'
    page = _shell(title="Daily news briefs", body_html=body, current="news", depth=1)
    (_news_out_dir() / "index.html").write_text(page)


# ---------------------------------------------------------------------------
# Weekly macro pages
# ---------------------------------------------------------------------------

def _macro_md_dir() -> Path:
    return STATE_ROOT / "macro" / "views"


def _macro_out_dir() -> Path:
    p = docs_root() / "macro"
    p.mkdir(parents=True, exist_ok=True)
    return p


def render_macro_pages() -> int:
    src_dir = _macro_md_dir()
    if not src_dir.exists():
        log.info("No macro views directory — skipping macro pages")
        _write_empty_macro_index()
        return 0

    entries: list[tuple[str, Path]] = []
    for src in sorted(src_dir.glob("*.md")):
        week_id = src.stem
        out_path = _macro_out_dir() / f"{week_id}.html"
        body_html = _md_to_html(src.read_text())
        page = _shell(
            title=f"Weekly macro view — {week_id}",
            body_html=body_html,
            current="macro",
            depth=1,
            subtitle=f"Generated by the weekly-macro agent · rendered {_generated_at()}",
        )
        out_path.write_text(page)
        entries.append((week_id, out_path))

    _write_macro_index(entries)
    log.info("Rendered %d macro view pages → %s", len(entries), _macro_out_dir())
    return len(entries)


def _write_macro_index(entries: list[tuple[str, Path]]) -> None:
    if not entries:
        _write_empty_macro_index()
        return
    entries_desc = sorted(entries, key=lambda e: e[0], reverse=True)
    items = "\n".join(
        f'<li><a href="{html.escape(p.name)}">{html.escape(d)}</a>'
        f'<span class="muted">macro view</span></li>'
        for d, p in entries_desc
    )
    body = f'<ul class="index-list">\n{items}\n</ul>'
    page = _shell(
        title="Weekly macro views",
        body_html=body,
        current="macro",
        depth=1,
        subtitle=f"{len(entries)} view{'s' if len(entries) != 1 else ''} archived",
    )
    (_macro_out_dir() / "index.html").write_text(page)


def _write_empty_macro_index() -> None:
    body = '<p class="page-content">No macro views yet. The agent runs every Sunday evening.</p>'
    page = _shell(title="Weekly macro views", body_html=body, current="macro", depth=1)
    (_macro_out_dir() / "index.html").write_text(page)


# ---------------------------------------------------------------------------
# Evolution log page
# ---------------------------------------------------------------------------

def _evolution_md_path() -> Path:
    return STATE_ROOT / "evolution.md"


def render_evolution_page() -> bool:
    """Render state/evolution.md to docs/evolution.html. Returns True if
    rendered, False if the source file doesn't exist."""
    src = _evolution_md_path()
    out = docs_root() / "evolution.html"
    if not src.exists():
        page = _shell(
            title="Strategy evolution log",
            body_html='<p class="page-content">No evolution log yet. The agent runs every Saturday morning.</p>',
            current="evolution",
            depth=0,
        )
        out.write_text(page)
        log.info("Wrote empty evolution.html")
        return False
    body_html = _md_to_html(src.read_text())
    page = _shell(
        title="Strategy evolution log",
        body_html=body_html,
        current="evolution",
        depth=0,
        subtitle=f"Generated by the weekly-evolution agent · rendered {_generated_at()}",
    )
    out.write_text(page)
    log.info("Rendered evolution.html (%d chars)", len(body_html))
    return True


# ---------------------------------------------------------------------------
# Composite + public URL helper
# ---------------------------------------------------------------------------

def rebuild_all_pages() -> dict:
    """Render every static page. Called from dashboard.build at the end of
    every pipeline so news/macro/evolution stay fresh alongside data.json."""
    n_news = render_news_pages()
    n_macro = render_macro_pages()
    have_evolution = render_evolution_page()
    return {"news": n_news, "macro": n_macro, "evolution": have_evolution}


def pages_url(path: str) -> str:
    """Best-effort public URL for a relative-to-docs path. Uses the
    repo's GitHub Pages convention. Returns '' if we can't construct one."""
    # GitHub Pages from main branch /docs serves at
    # https://<user>.github.io/<repo>/. Repo name is hard-coded since it's
    # in the cron-job.org schedule too — single source of truth.
    base = "https://dario-zela.github.io/trading_bot"
    return f"{base}/{path.lstrip('/')}"


def news_url_for(d: date) -> str:
    return pages_url(f"news/{d.isoformat()}.html")
