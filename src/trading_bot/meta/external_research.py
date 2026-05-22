"""Weekly external research scan.

Runs alongside the weekly evolution pipeline. Uses Claude Sonnet
with WebSearch + WebFetch to look at what's being researched and
written about in quantitative finance / algorithmic trading
recently, then writes a structured brief the evolution agent reads
when it decides actions (spawn-variant in particular benefits from
seeing whether a paper recently validated the kind of edge we'd
be cloning).

Output: `state/external_research/YYYY-WW.md` (one per ISO week).
Also surfaced on the dashboard.

The brief is structured per theme so the evolution agent can scan
it quickly: each theme gets a one-line description of what was
researched + a one-line implication for our slate. The agent then
sees this in its prompt as the "external context" section.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

from trading_bot.llm.claude_code import ClaudeCodeError, run_claude_for_json
from trading_bot.state.paths import STATE_ROOT


log = logging.getLogger(__name__)


_RESEARCH_DIR = "external_research"
_TIMEOUT = 600  # 10 min — research with WebSearch takes time
# Equivalent of how the news pipeline calls claude with tools enabled.
# Sonnet + WebSearch + WebFetch lets the agent pull abstracts + a
# handful of full pages.
_CLAUDE_EXTRA_ARGS = (
    "--allowed-tools", "WebSearch,WebFetch",
)


@dataclass
class ResearchBrief:
    """Structured weekly research brief. The markdown body is what
    gets included in the evolution prompt + rendered on the
    dashboard; the per-theme list lets us index / preview."""
    week_iso: str               # e.g. "2026-W21"
    generated_at: str           # ISO timestamp
    headline: str               # one-line summary of the week's research
    themes: list[dict]          # [{theme, summary, implication}]
    body_md: str                # full structured markdown
    sources: list[str]          # URLs cited


def _path_for_week(today: date) -> Path:
    iso_year, iso_week, _ = today.isocalendar()
    p = STATE_ROOT / _RESEARCH_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{iso_year}-W{iso_week:02d}.json"


def latest_brief() -> ResearchBrief | None:
    """Read the most-recent research brief, if any."""
    d = STATE_ROOT / _RESEARCH_DIR
    if not d.exists():
        return None
    files = sorted(d.glob("*.json"))
    if not files:
        return None
    import json
    try:
        raw = json.loads(files[-1].read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return ResearchBrief(
        week_iso=raw.get("week_iso", ""),
        generated_at=raw.get("generated_at", ""),
        headline=raw.get("headline", ""),
        themes=list(raw.get("themes") or []),
        body_md=raw.get("body_md", ""),
        sources=list(raw.get("sources") or []),
    )


def run_external_research(today: date) -> dict:
    """Generate this week's research brief via Sonnet+WebSearch.
    Skips silently if CLAUDE_CODE_OAUTH_TOKEN is missing."""
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        log.warning("external_research: OAuth token missing — skipping")
        return {"skipped": True, "reason": "no oauth token"}

    log.info("external_research: starting weekly scan for %s", today.isoformat())
    prompt = _build_prompt(today)
    try:
        response = run_claude_for_json(
            prompt,
            model="sonnet",
            timeout_seconds=_TIMEOUT,
            extra_args=_CLAUDE_EXTRA_ARGS,
            retries=2,
        )
    except ClaudeCodeError as e:
        log.error("external_research: Claude call failed: %s", e)
        return {"error": str(e)}

    if not isinstance(response, dict):
        log.error("external_research: response wasn't a JSON object")
        return {"error": "bad response shape"}

    brief = _response_to_brief(response, today)
    if brief is None:
        return {"error": "could not parse response"}

    _save(brief, today)
    log.info(
        "external_research: saved brief for %s — %d themes, %d sources",
        brief.week_iso, len(brief.themes), len(brief.sources),
    )
    return {
        "week_iso": brief.week_iso,
        "n_themes": len(brief.themes),
        "n_sources": len(brief.sources),
    }


def _build_prompt(today: date) -> str:
    iso_year, iso_week, _ = today.isocalendar()
    return f"""You are the external-research analyst for an algorithmic
trading bot's weekly evolution pipeline. Today is {today.isoformat()}
(ISO week {iso_year}-W{iso_week:02d}).

Your job: scan the recent quantitative-finance / systematic-trading
research landscape and produce a structured brief the evolution
agent will read when deciding what strategies to spawn, tune, or
retire next week.

## What we already run

The bot operates a slate of seven strategies in two regions (US,
UK/EU):

  - control-rule-based — buys top previous-day gainers, no filter
  - commodity-momentum — USO-led oil / commodity bias
  - momentum-trader — multi-day momentum continuation
  - mean-reverter — RSI <30 + distance from SMA20
  - macro-aligned — regime-based with yield-curve / DXY signals
  - news-reactive — morning-brief-driven event picks
  - sector-rotator — sector ETF relative-strength

Each is mid-frequency (1-5 day hold) and long-only. Costs are
realistic for UK retail (T212 / Alpaca paper). We don't run options,
futures, FX direct, or anything that requires more than ~£10k of
capital per strategy.

## What to find

Use WebSearch + WebFetch to surface, from research published or
indexed in the last 4-8 weeks:

1. **Active research themes** — what topics are quants currently
   writing about? (factor decay? alternative data? regime models?
   transformer-based predictors? volatility surfaces?). Aim for
   3-5 themes, prioritise novelty over textbook restatements.

2. **Concrete findings** — for each theme, one or two papers /
   notes with a specific finding (a beta, a Sharpe, a hit-rate)
   not just a "study suggests". Cite the source URL.

3. **Implications for our slate** — for each theme, one line on
   whether this looks like:
     - **fits existing**: tune one of our current strategies
     - **spawn-candidate**: justifies a new strategy variant
     - **out of scope**: interesting but needs capital / capability
       we don't have (options, leverage, intraday)

## Search strategy

Mix academic + practitioner sources. Suggested starting queries:

  - "site:arxiv.org quantitative finance 2026"
  - "site:ssrn.com trading strategy"
  - "site:papers.ssrn.com momentum mean reversion 2026"
  - "AQR research" / "Two Sigma research" / "Man Group research"
  - "Quantocracy" / "Hudson and Thames" / "Newfound Research"
  - Practitioner blogs: Robot Wealth, Quantitative Investments
  - Twitter / X for fresh ideas: but only if you can verify the
    source's track record

If a search returns mostly noise (how-to articles, brokerages,
listicles), refine. Prefer original research over aggregator
summaries.

## Output

Return JSON ONLY:

```json
{{
  "headline": "<one-line summary of the week's research themes>",
  "themes": [
    {{
      "theme": "<short title>",
      "summary": "<2-3 sentences, what's being researched and the
                   strongest finding>",
      "implication": "fits existing: <strategy_id>" |
                     "spawn-candidate: <one-line variant idea>" |
                     "out of scope: <why>"
    }},
    ... 3 to 5 of these
  ],
  "sources": ["<url1>", "<url2>", ...]
}}
```

Hard rules:
- 3 to 5 themes total — quality over breadth.
- Every theme must cite at least one verifiable source URL.
- No "research suggests" without a named source.
- Don't restate textbook material; we already know about momentum
  and mean-reversion. Surface what's NEW or counter-intuitive.
"""


def _response_to_brief(response: dict, today: date) -> ResearchBrief | None:
    from datetime import datetime, timezone

    headline = str(response.get("headline") or "").strip()
    themes_raw = response.get("themes") or []
    sources_raw = response.get("sources") or []
    if not headline or not isinstance(themes_raw, list):
        log.error("external_research: missing headline or themes")
        return None
    themes: list[dict] = []
    for t in themes_raw:
        if not isinstance(t, dict):
            continue
        theme = str(t.get("theme") or "").strip()
        summary = str(t.get("summary") or "").strip()
        implication = str(t.get("implication") or "").strip()
        if theme and summary:
            themes.append({
                "theme": theme,
                "summary": summary,
                "implication": implication,
            })
    if not themes:
        log.error("external_research: themes list was empty after filtering")
        return None
    sources = [str(s).strip() for s in sources_raw if isinstance(s, str) and str(s).strip()]
    iso_year, iso_week, _ = today.isocalendar()
    week_iso = f"{iso_year}-W{iso_week:02d}"
    body_md = _build_body_md(headline, themes, sources)
    return ResearchBrief(
        week_iso=week_iso,
        generated_at=datetime.now(timezone.utc).isoformat(),
        headline=headline,
        themes=themes,
        body_md=body_md,
        sources=sources,
    )


def _build_body_md(headline: str, themes: list[dict], sources: list[str]) -> str:
    lines: list[str] = [f"## {headline}", ""]
    for t in themes:
        lines.append(f"### {t['theme']}")
        lines.append("")
        lines.append(t["summary"])
        impl = t.get("implication") or ""
        if impl:
            lines.append("")
            lines.append(f"**Implication:** {impl}")
        lines.append("")
    if sources:
        lines.append("### Sources")
        lines.append("")
        for s in sources:
            lines.append(f"- {s}")
    return "\n".join(lines)


def _save(brief: ResearchBrief, today: date) -> None:
    import json
    path = _path_for_week(today)
    path.write_text(json.dumps(asdict(brief), indent=2))
