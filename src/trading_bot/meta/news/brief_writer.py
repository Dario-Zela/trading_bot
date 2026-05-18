"""Phase 2D — Brief writers.

Haiku × 6 parallel. Each call writes the short (~80-100 word) brief
that appears on the front page or section page. The brief's job is
*enough to get the reader to click through to the full article* —
not to be the article itself.

Tone: conversational, lay the conditions out before getting technical,
no jargon at this stage. The full article (Phase 2E) is where the
piece can get specialised.

We also fan out by piece, not by section, so brief writing is
embarrassingly parallel — Haiku eats this for breakfast.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date

from trading_bot.llm.claude_code import ClaudeCodeError, run_claude_for_json
from trading_bot.meta.news.publisher import BYLINES, NewsPlan, PlannedPiece
from trading_bot.meta.news.triage import TriagedCandidate

log = logging.getLogger(__name__)


_MAX_PARALLEL = 6
_BRIEF_TIMEOUT = 240
_BRIEF_WORDS_TARGET = 90      # aim — writer has leeway ±15


@dataclass
class Brief:
    """The brief slot for one piece. Renders into the front-page or
    section grid. Full article body lives on its own sub-page."""
    slug: str
    headline: str                   # may have been further sharpened
    kicker: str
    byline: str
    body_md: str                    # ~80-100 words of markdown
    sources_used: list[str] = field(default_factory=list)
    failed: bool = False            # true if Haiku errored — body is a fallback line


def write_briefs(
    plan: NewsPlan,
    triaged: list[TriagedCandidate],
    today: date,
) -> dict[str, Brief]:
    """Write briefs for every piece in the plan. Returns a slug→Brief
    map so assembly can render them straight."""
    if not plan.pieces:
        return {}
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        log.error("CLAUDE_CODE_OAUTH_TOKEN not set — using fallback briefs for all pieces")
        return {p.slug: _fallback_brief(p, triaged) for p in plan.pieces}

    log.info("Briefs: writing %d pieces (max %d in parallel)", len(plan.pieces), _MAX_PARALLEL)
    briefs: dict[str, Brief] = {}
    with ThreadPoolExecutor(max_workers=_MAX_PARALLEL) as pool:
        futures = {
            pool.submit(_write_one, p, triaged[p.triage_index], today): p
            for p in plan.pieces
            if 0 <= p.triage_index < len(triaged)
        }
        for fut in as_completed(futures):
            piece = futures[fut]
            try:
                briefs[piece.slug] = fut.result()
            except Exception as e:
                log.warning("Brief failed for %r: %s — using fallback", piece.slug, e)
                briefs[piece.slug] = _fallback_brief(piece, triaged)

    # Cover any piece the executor skipped (out-of-range triage_index)
    for p in plan.pieces:
        if p.slug not in briefs:
            briefs[p.slug] = _fallback_brief(p, triaged)

    n_failed = sum(1 for b in briefs.values() if b.failed)
    log.info("Briefs complete: %d written, %d fallback", len(briefs) - n_failed, n_failed)
    return briefs


def _write_one(piece: PlannedPiece, triaged: TriagedCandidate, today: date) -> Brief:
    prompt = _build_prompt(piece, triaged, today)
    try:
        response = run_claude_for_json(prompt, model="haiku", timeout_seconds=_BRIEF_TIMEOUT)
    except ClaudeCodeError as e:
        log.warning("Brief Haiku failed for %r: %s", piece.slug, e)
        return _fallback_brief(piece, [triaged])
    return _parse_brief(piece, response)


def _build_prompt(piece: PlannedPiece, triaged: TriagedCandidate, today: date) -> str:
    persona = BYLINES.get(piece.byline, "Staff writer.")
    facts_block = "\n".join(f"  - {f}" for f in triaged.key_facts) if triaged.key_facts else "  (no triage facts — work from the angle)"
    sources_block = "\n".join(f"  - {s}" for s in triaged.source_hints) if triaged.source_hints else "  (none provided)"
    tier_guidance = {
        "lead": "This is the FRONT PAGE LEAD. Set the stage clearly; the reader will continue to the full article. ~110-140 words.",
        "feature": "Section feature. A solid mid-size piece. ~90-120 words.",
        "brief": "Brief. Quick capture of what happened and why it matters. ~70-90 words.",
    }.get(piece.tier, "Brief. ~80 words.")

    return f"""You are {piece.byline} writing for The Bot Tribune on {today.isoformat()}.
Your beat persona: {persona}

Write a single newspaper brief for the piece below. The reader will
click through to the full article for depth — your job is the
*standfirst experience*: enough context to understand what happened
and why it matters, in plain English.

## The story

- **Headline:** {piece.headline}
- **Section:** {piece.section}
- **Tier:** {piece.tier} — {tier_guidance}
- **Angle (the hook):** {triaged.angle}
- **One-line summary:** {triaged.one_line}
- **Why it matters:** {triaged.why_it_matters}

## Key facts to use

{facts_block}

## Sources

{sources_block}

## Writing rules

1. **Open with the situation, not the technical detail.** A reader who
   doesn't already know this beat should be able to follow the first
   sentence. Examples:
   - GOOD: "The Federal Reserve held rates steady on Wednesday, the
     ninth consecutive meeting without a change."
   - BAD: "The dot plot shifted higher again as the SEP came in
     hawkish."
2. **Then earn the technical detail.** Specifics — numbers, named
   actors, hard quotes — should appear by the second or third sentence.
3. **Avoid editorialising in the brief.** The full article can have
   more colour; this is the headline experience.
4. **No clichés.** "Watershed moment", "uncharted territory", "in a
   stunning move" — cut them.
5. **No author intrusion.** Don't address the reader, don't say
   "we'll see what happens."
6. **End with a forward-look or implication.** What does this set up?
   What's worth watching?

## Required output

Return JSON only:

```json
{{
  "body_md": "<the brief as plain markdown — paragraphs separated by blank lines. NO headline or byline in the body.>",
  "sources_used": ["<source name or URL>", ...]
}}
```
"""


def _parse_brief(piece: PlannedPiece, response: dict | list) -> Brief:
    if isinstance(response, list) and response:
        response = response[0] if isinstance(response[0], dict) else {}
    if not isinstance(response, dict):
        return _fallback_brief(piece, [])

    body = str(response.get("body_md") or response.get("body") or "").strip()
    sources_raw = response.get("sources_used") or response.get("sources") or []
    if isinstance(sources_raw, str):
        sources_raw = [sources_raw]
    sources = [str(s).strip() for s in sources_raw if str(s).strip()][:8]

    if not body:
        return _fallback_brief(piece, [])

    return Brief(
        slug=piece.slug,
        headline=piece.headline,
        kicker=piece.kicker,
        byline=piece.byline,
        body_md=body,
        sources_used=sources,
        failed=False,
    )


def _fallback_brief(piece: PlannedPiece, triaged: list[TriagedCandidate] | TriagedCandidate | None) -> Brief:
    """Fallback when Haiku is unavailable or fails. We still need
    *something* on the page — show the one-line from triage."""
    body = piece.one_line or "(no brief available)"
    if isinstance(triaged, list) and triaged:
        t = triaged[piece.triage_index] if 0 <= piece.triage_index < len(triaged) else None
        if t and t.why_it_matters:
            body = f"{body}\n\n{t.why_it_matters}"
    elif isinstance(triaged, TriagedCandidate):
        if triaged.why_it_matters:
            body = f"{body}\n\n{triaged.why_it_matters}"
    return Brief(
        slug=piece.slug,
        headline=piece.headline,
        kicker=piece.kicker,
        byline=piece.byline,
        body_md=body,
        sources_used=[],
        failed=True,
    )


def briefs_to_json(briefs: dict[str, Brief]) -> dict[str, dict]:
    return {slug: asdict(b) for slug, b in briefs.items()}


def briefs_from_json(data: dict[str, dict]) -> dict[str, Brief]:
    out: dict[str, Brief] = {}
    for slug, b in data.items():
        if not isinstance(b, dict):
            continue
        out[slug] = Brief(
            slug=str(b.get("slug", slug)),
            headline=str(b.get("headline", "")),
            kicker=str(b.get("kicker", "")),
            byline=str(b.get("byline", "Bot Tribune Staff")),
            body_md=str(b.get("body_md", "")),
            sources_used=list(b.get("sources_used") or []),
            failed=bool(b.get("failed", False)),
        )
    return out
