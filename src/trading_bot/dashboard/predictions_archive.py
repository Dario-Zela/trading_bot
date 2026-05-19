"""Phase 9I — searchable predictions archive page.

Reads `state/predictions/{news,macro,evolution}.jsonl` and renders a
single page at `docs/predictions/index.html` showing every graded
prediction with client-side filter by source / horizon / status /
conviction. Also includes a "verdict rate by conviction" stats block
so we can see whether 'high conviction' actually means something.

Rendered fresh by `render_predictions_archive()` from the main pages
build path.
"""
from __future__ import annotations

import html
import json
import logging
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from trading_bot.state.paths import STATE_ROOT


log = logging.getLogger(__name__)


_KNOWN_STATUSES = ("proven", "partial", "falsified", "still-open", "open")
_KNOWN_CONVICTIONS = ("high", "medium", "low")


def render_predictions_archive(docs_root: Path, shell_fn) -> Path:
    """Render the archive page. Returns the output path."""
    preds_dir = STATE_ROOT / "predictions"
    rows: list[dict] = []
    if preds_dir.exists():
        for p in sorted(preds_dir.glob("*.jsonl")):
            source = p.stem
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rec.setdefault("source", source)
                    rows.append(rec)

    # Newest-first
    rows.sort(key=lambda r: r.get("made_at", ""), reverse=True)

    stats = _compute_stats(rows)
    body = _build_body(rows, stats)

    page = shell_fn(
        title="Predictions archive",
        body_html=body,
        current="news",        # appears under news in the nav highlight
        depth=1,
        page_class="news",
    )
    out_dir = docs_root / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "index.html"
    out.write_text(page)
    log.info("predictions-archive: %d rows → %s", len(rows), out)
    return out


def _conviction_bucket(c) -> str:
    """Normalise conviction to one of 'high' / 'medium' / 'low' / ''.

    Strings ('high'/'medium'/'low') pass through; floats (0.0-1.0,
    legacy per-trade) get bucketed by quantile-ish thresholds."""
    if c is None or c == "":
        return ""
    if isinstance(c, str):
        s = c.lower().strip()
        return s if s in {"high", "medium", "low"} else ""
    try:
        f = float(c)
    except (TypeError, ValueError):
        return ""
    if f >= 0.7:   return "high"
    if f >= 0.4:   return "medium"
    return "low"


def _compute_stats(rows: list[dict]) -> dict:
    """Verdict rate by conviction (and overall). For 'graded' we only
    count rows that have a terminal verdict (proven/partial/falsified).
    still-open and open aren't 'graded yet'."""
    terminal = {"proven", "partial", "falsified"}
    by_conviction: dict[str, dict] = defaultdict(lambda: {
        "proven": 0, "partial": 0, "falsified": 0, "still-open": 0,
        "open": 0, "graded": 0, "total": 0,
    })
    for r in rows:
        conv = _conviction_bucket(r.get("conviction")) or "unknown"
        status = (r.get("status") or "open").lower()
        bucket = by_conviction[conv]
        bucket["total"] += 1
        if status in bucket:
            bucket[status] += 1
        if status in terminal:
            bucket["graded"] += 1
    # Add overall totals
    overall = {"proven": 0, "partial": 0, "falsified": 0, "still-open": 0,
               "open": 0, "graded": 0, "total": 0}
    for bucket in by_conviction.values():
        for k, v in bucket.items():
            overall[k] += v
    return {"by_conviction": dict(by_conviction), "overall": overall}


def _build_body(rows: list[dict], stats: dict) -> str:
    n_total = len(rows)
    overall = stats["overall"]
    n_graded = overall.get("graded", 0)
    win_rate = (overall.get("proven", 0) / n_graded * 100.0) if n_graded else 0.0
    partial_rate = (overall.get("partial", 0) / n_graded * 100.0) if n_graded else 0.0
    fals_rate = (overall.get("falsified", 0) / n_graded * 100.0) if n_graded else 0.0

    # Stats table
    stats_rows = []
    for conv in _KNOWN_CONVICTIONS:
        b = stats["by_conviction"].get(conv)
        if not b or b.get("graded", 0) == 0:
            continue
        graded = b["graded"]
        proven_pct = b["proven"] / graded * 100.0
        partial_pct = b["partial"] / graded * 100.0
        fals_pct = b["falsified"] / graded * 100.0
        stats_rows.append(
            f'<tr>'
            f'<td class="sector"><strong>{html.escape(conv.upper())}</strong></td>'
            f'<td class="lean">{graded} graded / {b["total"]} total</td>'
            f'<td class="driver">'
            f'<span class="up">Proven {proven_pct:.0f}%</span> · '
            f'Partial {partial_pct:.0f}% · '
            f'<span class="down">Falsified {fals_pct:.0f}%</span>'
            f'</td>'
            f'</tr>'
        )
    stats_table = (
        '<div class="sectors-table"><h3>Verdict rate by conviction</h3>'
        '<table><tbody>'
        + "".join(stats_rows)
        + '</tbody></table></div>'
    ) if stats_rows else ""

    # Row markup — emit data-* attrs the filter JS uses
    row_html = []
    for r in rows:
        rid = html.escape(r.get("id", ""))
        source = html.escape(r.get("source", ""))
        horizon = html.escape(r.get("horizon", ""))
        # Phase 10D — conviction may be a string (high/medium/low) or
        # a float (0.0-1.0 on per-trade predictions). Normalise to a
        # bucket so the filter matches both.
        conviction = html.escape(_conviction_bucket(r.get("conviction")))
        status = html.escape((r.get("status") or "open").lower())
        made_at = html.escape((r.get("made_at") or "")[:10])
        target_date = html.escape(r.get("target_date", ""))
        claim = html.escape(r.get("claim", ""))
        falsifier = html.escape(r.get("falsification_criteria", ""))
        note = html.escape((r.get("grading_note") or "")[:300])
        verdict_class = status if status in {"proven", "partial", "falsified", "still-open"} else "open"
        row_html.append(
            f'<article class="pred" data-source="{source}" data-horizon="{horizon}" '
            f'data-status="{status}" data-conviction="{conviction}" '
            f'style="margin-bottom: 1.2rem; padding-bottom: 0.8rem; border-bottom: 1px solid var(--hairline);">'
            f'<p><span class="verdict {verdict_class}">{html.escape(status.upper())}</span>'
            f'<span style="font-family:var(--sans);font-size:0.74rem;color:var(--ink-muted);">'
            f' [{source} · {horizon} · {conviction or "—"} conviction] · '
            f'made {made_at} · target {target_date}</span></p>'
            f'<h3 style="font-size: 1.15rem; margin: 0.3rem 0;">{claim}</h3>'
            f'<div class="falsif" style="margin: 0.4rem 0;"><strong>Falsified if:</strong> {falsifier}</div>'
            + (f'<p style="font-family:var(--serif-body); color: var(--ink-soft); margin: 0.3rem 0;">{note}</p>' if note else '')
            + '</article>'
        )
    list_html = "\n".join(row_html) if row_html else "<p>No predictions on file yet.</p>"

    # Filter controls
    filter_controls = """
<div style="margin: 1.5rem 0; padding: 1rem; background: var(--paper-dark); border-radius: 4px; display: flex; gap: 1rem; flex-wrap: wrap; font-family: var(--sans); font-size: 0.85rem;">
  <label>Source
    <select id="filter-source" style="margin-left: 0.4rem; padding: 0.2rem 0.5rem;">
      <option value="">all</option>
      <option value="news">news</option>
      <option value="macro">macro</option>
      <option value="evolution">evolution</option>
    </select>
  </label>
  <label>Horizon
    <select id="filter-horizon" style="margin-left: 0.4rem; padding: 0.2rem 0.5rem;">
      <option value="">all</option>
      <option value="tomorrow">tomorrow</option>
      <option value="this-week">this-week</option>
      <option value="this-month">this-month</option>
      <option value="this-quarter">this-quarter</option>
      <option value="this-half">this-half</option>
      <option value="this-year">this-year</option>
      <option value="multi-year">multi-year</option>
    </select>
  </label>
  <label>Status
    <select id="filter-status" style="margin-left: 0.4rem; padding: 0.2rem 0.5rem;">
      <option value="">all</option>
      <option value="proven">proven</option>
      <option value="partial">partial</option>
      <option value="falsified">falsified</option>
      <option value="still-open">still-open</option>
      <option value="open">open</option>
    </select>
  </label>
  <label>Conviction
    <select id="filter-conviction" style="margin-left: 0.4rem; padding: 0.2rem 0.5rem;">
      <option value="">all</option>
      <option value="high">high</option>
      <option value="medium">medium</option>
      <option value="low">low</option>
    </select>
  </label>
  <span id="filter-count" style="margin-left: auto; color: var(--ink-muted); align-self: center;"></span>
</div>
<script>
(function() {
  const filters = ['source', 'horizon', 'status', 'conviction'];
  function applyFilters() {
    const vals = {};
    filters.forEach(f => vals[f] = document.getElementById('filter-' + f).value);
    let shown = 0, total = 0;
    document.querySelectorAll('article.pred').forEach(el => {
      total++;
      const match = filters.every(f => {
        const want = vals[f];
        const got = el.dataset[f] || '';
        return !want || got === want;
      });
      el.style.display = match ? '' : 'none';
      if (match) shown++;
    });
    document.getElementById('filter-count').textContent = `${shown} / ${total}`;
  }
  filters.forEach(f => document.getElementById('filter-' + f).addEventListener('change', applyFilters));
  applyFilters();
})();
</script>
"""

    return (
        '<main class="paper">'
        '<header class="masthead">'
        '<h1>The Bot Tribune<span class="sub">— Predictions archive</span></h1>'
        f'<div class="subtitle">{n_total} predictions on file · {n_graded} graded · '
        f'Proven {win_rate:.0f}% · Partial {partial_rate:.0f}% · Falsified {fals_rate:.0f}%</div>'
        '</header>'
        '<div class="masthead-strip">'
        '<span><strong>All calls</strong></span>'
        f'<span>Rendered {html.escape(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))}</span>'
        '<span></span>'
        '</div>'
        + stats_table
        + filter_controls
        + '<div class="section-label calls"><span>Every call on file</span>'
        f'<span class="ord">newest first</span></div>'
        + list_html
        + '<footer class="colophon">Auto-generated by the predictions archive renderer.</footer>'
        + '</main>'
    )
