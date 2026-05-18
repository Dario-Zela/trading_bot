"""Daily market news brief — the multi-stage newspaper pipeline.

Orchestration:

1. **Discovery** (Sonnet + WebSearch) — surface ~30-50 candidates.
2. **Triage** (Haiku × 6 parallel) — score, sharpen angle, pull facts.
3. **Publisher** (Sonnet) — lead, sections, bylines, masthead subtitle.
4. *Parallel block:*
   - **Brief writers** (Haiku × 6 parallel) — the front-page bodies.
   - **Article writers** (Sonnet × 6 parallel) — full articles + images.
   - **Trading floor** (Haiku × 3 parallel) — yesterday's P&L in prose.
   - **Desk's calls** (Sonnet) — fresh predictions + Marking the homework.
5. **Bot summary** (Haiku) — ~150-word strategy briefing.
6. **Render** — assembly to `docs/news/YYYY-MM-DD/index.html`
   + per-article subpages, plus the index update.
7. **Email** — morning brief with a link to the front page.

The pipeline preserves the existing contract with downstream strategies:
the bot summary still lands at `state/daily_news/{date}.bot.md` in the
familiar Risk tone / Themes / Sector lean / Watchlist / Key data shape.
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

from trading_bot.meta.news.article_writer import articles_to_json, write_articles
from trading_bot.meta.news.bot_summary import compress_bot_summary
from trading_bot.meta.news.brief_writer import briefs_to_json, write_briefs
from trading_bot.meta.news.desks_calls import build_desks_calls
from trading_bot.meta.news.discovery import candidates_to_json, discover_stories
from trading_bot.meta.news.publisher import plan_edition, plan_to_json
from trading_bot.meta.news.trading_floor import write_floor_briefs
from trading_bot.meta.news.triage import triage_candidates, triaged_to_json
from trading_bot.state.paths import STATE_ROOT


log = logging.getLogger(__name__)


def run_daily_news_brief(today: date) -> dict:
    """Run the full multi-stage newspaper pipeline for `today`.

    Returns a summary dict with stage counts + the public URL.
    """
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        log.error("CLAUDE_CODE_OAUTH_TOKEN not set — skipping daily news brief")
        return {"skipped": True, "reason": "no oauth token"}

    pipeline_state: dict = {"date": today.isoformat(), "stages": {}}

    # Stage 1 — Discovery
    log.info("=== Stage 1 / 6 — Discovery ===")
    candidates = discover_stories(today)
    pipeline_state["stages"]["discovery"] = {
        "count": len(candidates),
        "candidates": candidates_to_json(candidates),
    }
    if not candidates:
        log.warning("Discovery returned no candidates — aborting brief for %s", today.isoformat())
        return {"skipped": True, "reason": "no candidates"}

    # Stage 2 — Triage
    log.info("=== Stage 2 / 6 — Triage (×%d) ===", len(candidates))
    triaged = triage_candidates(candidates, today)
    pipeline_state["stages"]["triage"] = {
        "count": len(triaged),
        "triaged": triaged_to_json(triaged),
    }

    # Stage 3 — Publisher
    log.info("=== Stage 3 / 6 — Publisher ===")
    plan = plan_edition(triaged, today)
    pipeline_state["stages"]["publisher"] = plan_to_json(plan)
    if not plan.pieces:
        log.warning("Publisher returned empty plan — aborting brief for %s", today.isoformat())
        return {"skipped": True, "reason": "empty plan"}

    # Stage 4 — Briefs + Articles + Floor + Desks (in parallel)
    log.info("=== Stage 4 / 6 — Briefs + Articles + Floor + Desks (parallel) ===")
    with ThreadPoolExecutor(max_workers=4) as outer:
        f_briefs = outer.submit(write_briefs, plan, triaged, today)
        f_articles = outer.submit(write_articles, plan, triaged, today)
        f_floor = outer.submit(write_floor_briefs, today)
        f_desks = outer.submit(build_desks_calls, plan, triaged, today)
        briefs = f_briefs.result()
        articles = f_articles.result()
        floor = f_floor.result()
        desks = f_desks.result()

    pipeline_state["stages"]["briefs"] = briefs_to_json(briefs)
    pipeline_state["stages"]["articles"] = articles_to_json(articles)
    pipeline_state["stages"]["floor"] = [asdict(f) for f in floor]
    pipeline_state["stages"]["desks"] = {
        "fresh_predictions": [asdict(p) for p in desks.fresh_predictions],
        "homework_items": [asdict(p) for p in desks.homework_items],
    }

    # Stage 5 — Bot summary
    log.info("=== Stage 5 / 6 — Bot summary compression ===")
    bot_summary = compress_bot_summary(plan, briefs, floor, desks, today)
    pipeline_state["stages"]["bot_summary"] = bot_summary

    # Persist the pipeline state for debugging / archive
    _write_pipeline_state(today, pipeline_state)

    # Persist the bot summary (strategy prompts read this)
    bot_path = _bot_brief_path(today)
    bot_path.parent.mkdir(parents=True, exist_ok=True)
    bot_path.write_text(bot_summary + "\n")
    log.info("Wrote bot summary → %s (%d chars)", bot_path, len(bot_summary))

    # Persist a human-readable headlines markdown for archive compat
    _write_headlines_markdown(today, plan, briefs)

    # Stage 6 — Render + page URL + email
    log.info("=== Stage 6 / 6 — Render edition + send email ===")
    page_url: str | None = None
    try:
        from trading_bot.dashboard.pages import (
            news_url_for, render_news_edition, render_news_pages,
        )
        render_news_edition(
            today,
            plan=plan, briefs=briefs, articles=articles,
            triaged=triaged, floor=floor, desks=desks,
        )
        render_news_pages()  # refresh archive index to include the new edition
        page_url = news_url_for(today)
    except Exception as e:
        log.warning("Render failed (non-fatal): %s", e)

    if page_url:
        try:
            from trading_bot.notify.email import (
                render_news_brief_email, send_summary_email,
            )
            subject, text_body, html_body = render_news_brief_email(
                run_date=today,
                bot_summary_md=bot_summary,
                full_brief_url=page_url,
            )
            send_summary_email(subject=subject, body_text=text_body, body_html=html_body)
            log.info("daily-news-brief: sent email with link %s", page_url)
        except Exception as e:
            log.warning("Couldn't send news brief email (non-fatal): %s", e)

    return {
        "date": today.isoformat(),
        "discovery_count": len(candidates),
        "triaged_count": len(triaged),
        "pieces": len(plan.pieces),
        "articles": len(articles),
        "floor_pieces": len(floor),
        "fresh_predictions": len(desks.fresh_predictions),
        "homework_items": len(desks.homework_items),
        "bot_summary_chars": len(bot_summary),
        "page_url": page_url,
    }


def _write_pipeline_state(today: date, state: dict) -> None:
    """Dump the full multi-stage state for inspection / re-render."""
    path = STATE_ROOT / "daily_news" / f"{today.isoformat()}.pipeline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    state["generated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state, indent=2, default=str))
    log.info("Wrote pipeline state → %s", path)


def _write_headlines_markdown(today: date, plan, briefs) -> None:
    """Save a human-readable headlines list to state/daily_news/{date}.md.
    The legacy renderer + archive tooling expect this file to exist."""
    path = _brief_path(today)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# The Bot Tribune — {today.isoformat()}",
        "",
        f"*{plan.masthead_subtitle}*" if plan.masthead_subtitle else "",
        "",
        "## Today's pieces",
        "",
    ]
    by_section: dict[str, list] = {}
    for p in plan.pieces:
        by_section.setdefault(p.section, []).append(p)
    for section, pieces in by_section.items():
        lines.append(f"### {section}")
        lines.append("")
        for p in pieces:
            lines.append(f"- **{p.headline}** — {p.one_line}  ")
            lines.append(f"  *By {p.byline} · {p.kicker}*")
        lines.append("")
    path.write_text("\n".join(lines))


def _brief_path(d: date) -> Path:
    return STATE_ROOT / "daily_news" / f"{d.isoformat()}.md"


def _bot_brief_path(d: date) -> Path:
    return STATE_ROOT / "daily_news" / f"{d.isoformat()}.bot.md"
