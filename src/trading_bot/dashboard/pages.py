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
    # News + Macro point at the latest-edition redirect so clicking the
    # ribbon tab opens today's edition directly. The full archive lives
    # at news/index.html / macro/index.html and is reachable from the
    # "all" link in each edition's nav strip.
    ("news",      "News",      "news/latest.html"),
    ("macro",     "Macro",     "macro/latest.html"),
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

_EDITION_NAV_SCRIPT = """
<script>
/* Edition-nav upgrade: on every news/macro page, find the .edition-nav
   strip, read the current edition id from its data attribute, fetch the
   sibling editions.json, and update the prev/next anchor hrefs. This
   keeps an older edition's "next →" live without re-rendering its HTML
   every time a new edition is published. */
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.edition-nav[data-edition]').forEach(async (nav) => {
    const cur = nav.dataset.edition;
    if (!cur) return;
    try {
      const resp = await fetch('../editions.json', { cache: 'no-cache' });
      if (!resp.ok) return;
      const list = await resp.json();
      if (!Array.isArray(list) || list.length === 0) return;
      // editions.json is newest-first; sort defensively in case
      const ids = [...list].sort((a, b) => (a.id < b.id ? 1 : -1));
      const idx = ids.findIndex(e => e.id === cur);
      if (idx < 0) return;
      const newer = idx > 0 ? ids[idx - 1] : null;        // 'next →'
      const older = idx < ids.length - 1 ? ids[idx + 1] : null; // '← prev'
      const setLink = (role, target) => {
        const a = nav.querySelector(`.edition-nav-link[data-role="${role}"]`);
        if (!a) return;
        if (target) {
          a.href = target.url;
          a.title = target.id;
          a.classList.remove('disabled');
          a.removeAttribute('aria-disabled');
          if (role === 'prev') a.textContent = '← prev';
          if (role === 'next') a.textContent = 'next →';
        } else {
          a.href = '#';
          a.classList.add('disabled');
          a.setAttribute('aria-disabled', 'true');
        }
      };
      setLink('prev', older);
      setLink('next', newer);
    } catch (e) { /* offline / no manifest — keep render-time links */ }
  });
});
</script>
"""

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


def _archive_manifest_path() -> Path:
    return Path(__file__).resolve().parents[3] / "state" / "archive" / "manifest.json"


def _read_archived_news_dates() -> list[dict]:
    """Read the Phase 7 archive manifest, return per-edition records
    suitable for inclusion in the news archive index. Each record is
    `{date, url}` where url points at the tarball blob (the user
    downloads to inspect)."""
    return _read_archived_records(kind="news")


def _read_archived_macro_weeks() -> list[dict]:
    return _read_archived_records(kind="macro")


def _read_archived_records(*, kind: str) -> list[dict]:
    manifest_path = _archive_manifest_path()
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return []
    out: list[dict] = []
    for bundle in manifest.get(kind, []) or []:
        if not isinstance(bundle, dict):
            continue
        url = bundle.get("url", "")
        for entry in (bundle.get("entries") or []):
            out.append({"date": entry, "url": url})
    return out


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
  {_EDITION_NAV_SCRIPT}
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
    pages written (not counting the archive index).

    This is the legacy flat-file path. The new structured pipeline
    (Phase 2) calls `render_news_edition()` which writes a directory
    per edition; both forms appear in the archive index."""
    src_dir = _news_md_dir()
    out_dir = _news_out_dir()

    legacy_entries: list[tuple[str, Path]] = []
    if src_dir.exists():
        for src in sorted(src_dir.glob("*.md")):
            if src.stem.endswith(".bot"):
                continue  # skip bot-summary companions
            date_str = src.stem
            # Skip the legacy render if a dir-based edition exists for the same date
            if (out_dir / date_str / "index.html").exists():
                continue
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
            legacy_entries.append((date_str, out_path))

    # Discover dir-based editions too (Phase 2 output)
    dir_entries: list[tuple[str, Path]] = []
    for child in sorted(out_dir.glob("*")):
        if not child.is_dir():
            continue
        if not _ISO_DATE_RE.match(child.name):
            continue
        idx = child / "index.html"
        if idx.exists():
            dir_entries.append((child.name, idx))

    _write_news_index(legacy_entries, dir_entries)
    _ensure_latest_redirect(out_dir, dir_entries, legacy_entries, "news")
    from trading_bot.meta.news.render import _write_editions_index
    _write_editions_index(out_dir, kind="news")
    total = len(legacy_entries) + len(dir_entries)
    log.info("Rendered %d daily news pages → %s (%d legacy, %d structured)",
             total, out_dir, len(legacy_entries), len(dir_entries))
    return total


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def render_news_edition(today: date, **kwargs) -> Path:
    """Render the structured Phase 2 newspaper edition for `today`.

    Thin wrapper around `meta.news.render.render_news_edition`. Kept on
    `pages.py` so that callers can use a single rendering entry point.

    Accepts keyword args: plan, briefs, articles, triaged, floor, desks.
    See `meta.news.render.render_news_edition` for the full signature."""
    from trading_bot.meta.news.render import render_news_edition as _render
    return _render(
        today,
        docs_root=docs_root(),
        shell_fn=_shell,
        **kwargs,
    )


def _write_news_index(
    legacy_entries: list[tuple[str, Path]],
    dir_entries: list[tuple[str, Path]] | None = None,
) -> None:
    """Render the News archive index. Both flat (legacy) and directory
    (Phase 2) editions are listed; when both forms exist for a date the
    directory form wins (and the legacy is suppressed at render time).
    Phase 7 — also lists trimmed editions from state/archive/manifest.json
    as compressed-archive links."""
    dir_entries = dir_entries or []
    out_dir = _news_out_dir()

    # Combine, deduplicating by date — directory wins, archive fills gaps
    combined: dict[str, tuple[str, str, str]] = {}  # date -> (href, label, kind)
    for d, p in legacy_entries:
        combined[d] = (p.name, d, "legacy")
    for d, p in dir_entries:
        combined[d] = (f"{d}/", d, "structured")

    # Phase 7: pull trimmed editions from the archive manifest
    for arch in _read_archived_news_dates():
        if arch["date"] not in combined:
            combined[arch["date"]] = (arch["url"], arch["date"], "archived")

    if not combined:
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

    entries_desc = sorted(combined.items(), key=lambda kv: kv[0], reverse=True)
    kind_label = {
        "structured": "newspaper edition",
        "legacy":     "daily brief",
        "archived":   "archived (compressed)",
    }
    items = "\n".join(
        f'<li><a href="{html.escape(href)}">{html.escape(label)}</a>'
        f'<span class="muted">{kind_label.get(kind, "daily brief")}</span></li>'
        for _, (href, label, kind) in entries_desc
    )
    n = len(combined)
    body = (
        '<main class="paper">'
        '<header class="masthead">'
        '  <h1>The Bot Tribune<span class="sub">— News archive</span></h1>'
        f'  <div class="subtitle">{n} edition{"s" if n != 1 else ""} on file. Most recent first.</div>'
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


def _ensure_latest_redirect(
    out_dir: Path,
    dir_entries: list[tuple[str, Path]],
    legacy_entries: list[tuple[str, Path]],
    kind: str,
) -> None:
    """Make sure `out_dir/latest.html` exists. Phase 2/3 editions write
    it on each render, but the ribbon tab points there unconditionally —
    so we bootstrap on archive rebuild too. Falls back to the legacy
    flat-file when no directory edition exists yet, and only writes a
    placeholder if neither form is present."""
    latest_path = out_dir / "latest.html"
    target: str = ""
    if dir_entries:
        newest = sorted(dir_entries, key=lambda e: e[0], reverse=True)[0][0]
        target = f"{newest}/"
    elif legacy_entries:
        newest = sorted(legacy_entries, key=lambda e: e[0], reverse=True)[0]
        target = newest[1].name
    if not target:
        # No editions at all — write a friendly placeholder so the
        # ribbon tab never 404s.
        latest_path.write_text(
            '<!DOCTYPE html><html><head><meta charset="UTF-8">'
            f'<title>{kind.capitalize()} — no editions yet</title>'
            '<link rel="stylesheet" href="../assets/style.css"></head>'
            f'<body class="page-{kind}"><main class="paper">'
            '<header class="masthead">'
            f'<h1>The Bot Tribune<span class="sub">— {kind.capitalize()}</span></h1>'
            '<div class="subtitle">No editions yet.</div></header>'
            f'<p style="text-align:center;font-style:italic;color:var(--ink-muted);margin-top:2rem;">'
            f'The {kind} agent runs on its schedule; the first edition will appear here.</p>'
            '</main></body></html>'
        )
        return
    # Don't downgrade an existing latest.html to point to an older edition
    if latest_path.exists():
        try:
            existing = latest_path.read_text()
            m = re.search(r'url=([^/"\']+)/?', existing)
            if m and m.group(1) >= target.rstrip("/"):
                return
        except OSError:
            pass
    latest_path.write_text(
        '<!DOCTYPE html><html><head>'
        f'<meta http-equiv="refresh" content="0; url={html.escape(target)}">'
        f'<link rel="canonical" href="{html.escape(target)}">'
        f'<title>Redirecting to latest {kind} edition…</title>'
        '</head><body>'
        f'<p>Redirecting to <a href="{html.escape(target)}">{html.escape(target)}</a>…</p>'
        '</body></html>'
    )


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
    # Discover directory-form macro editions (Phase 3 output)
    dir_entries: list[tuple[str, Path]] = []
    for child in sorted(out_dir.glob("*")):
        if not child.is_dir():
            continue
        if not re.match(r"^\d{4}-W\d{2}$", child.name):
            continue
        idx = child / "index.html"
        if idx.exists():
            dir_entries.append((child.name, idx))

    _write_macro_index(entries)
    _ensure_latest_redirect(out_dir, dir_entries, entries, "macro")
    from trading_bot.meta.news.render import _write_editions_index
    _write_editions_index(out_dir, kind="macro")
    total = len(entries) + len(dir_entries)
    log.info("Rendered %d macro pages → %s (%d legacy, %d structured)",
             total, out_dir, len(entries), len(dir_entries))
    return total


def _write_macro_index(entries: list[tuple[str, Path]]) -> None:
    out_dir = _macro_out_dir()
    # Combine live + archived. Live entries store (week, Path) where Path
    # is the on-disk file (we use p.name). Archived store the absolute URL.
    live_weeks = {d for d, _ in entries}
    combined: list[tuple[str, str, str]] = []   # (week, href, kind)
    for d, p in entries:
        # If p is a directory we prefer the directory URL; otherwise it's
        # the legacy flat-file rendered to docs/macro/<week>.html.
        href = f"{d}/" if (out_dir / d / "index.html").exists() else p.name
        combined.append((d, href, "view"))
    for arch in _read_archived_macro_weeks():
        if arch["date"] in live_weeks:
            continue
        combined.append((arch["date"], arch["url"], "archived"))

    if not combined:
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
    entries_desc = sorted(combined, key=lambda e: e[0], reverse=True)
    kind_label = {"view": "macro view", "archived": "archived (compressed)"}
    items = "\n".join(
        f'<li><a href="{html.escape(href)}">{html.escape(d)}</a>'
        f'<span class="muted">{kind_label.get(kind, "macro view")}</span></li>'
        for d, href, kind in entries_desc
    )
    body = (
        '<main class="paper">'
        '<header class="masthead">'
        '  <h1>The Bot Tribune<span class="sub">— Macro archive</span></h1>'
        f'  <div class="subtitle">{len(combined)} view{"s" if len(combined) != 1 else ""} on file. Most recent first.</div>'
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
    # Phase 9I — predictions archive page
    try:
        from trading_bot.dashboard.predictions_archive import render_predictions_archive
        render_predictions_archive(docs_root(), _shell)
    except Exception as e:
        log.warning("Predictions archive render failed (non-fatal): %s", e)
    return {"news": n_news, "macro": n_macro, "evolution": have_evolution}


def pages_url(path: str) -> str:
    base = "https://dario-zela.github.io/trading_bot"
    return f"{base}/{path.lstrip('/')}"


def news_url_for(d: date) -> str:
    """Canonical URL for the edition on date `d`. Prefers the
    directory form (Phase 2 structured edition) when both exist."""
    iso = d.isoformat()
    if (_news_out_dir() / iso / "index.html").exists():
        return pages_url(f"news/{iso}/")
    return pages_url(f"news/{iso}.html")
