"""Phase 2C — Publisher agent.

Single Sonnet call. Reads the triaged roster, organises survivors into
sections, picks the lead, assigns a byline persona to each piece, and
writes the daily masthead subtitle (descriptive — not the old arch
"in which..." conceit).

The publisher's output is the *plan* — slot names, section ordering,
which story goes where. The brief and full-article writers consume
that plan to produce the actual prose.

A note on sections
==================
We always render Front, Markets, World, Tech & science, and Beyond the
tape (dropped if empty). Other sections (Climate, Health, Sport,
Culture) appear only when the day actually warrants them — the
publisher decides. Trading floor and Desk's calls are appended by
their own dedicated stages outside the LLM, so the publisher should
NOT include them.

A note on bylines
=================
Each piece gets a byline persona. The persona is the writer's voice
hint — markets pieces get a markets desk byline, tech pieces get the
tech desk byline, and so on. Bylines persist across days so the
"newspaper" reads like a paper, not a slop generator.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import date

from trading_bot.llm.claude_code import ClaudeCodeError, run_claude_for_json
from trading_bot.meta.news.triage import TriagedCandidate

log = logging.getLogger(__name__)


_PUBLISH_TIMEOUT = 240


# Canonical byline roster. The publisher MUST choose from these — we
# don't want it inventing new staff each day. Each persona's blurb
# becomes the writer's voice hint downstream.
BYLINES: dict[str, str] = {
    "M. Halloway":        "Markets desk. Dry, numerate, prefers facts to drama.",
    "C. Tanaka":          "World affairs. Patient, contextualises events.",
    "R. Okonkwo":         "Tech & science. Plain English, allergic to hype.",
    "S. Whitfield":       "Climate & environment. Measured, evidence-led.",
    "Dr. P. Aaronson":    "Health & medicine. Cautious, footnotes everything.",
    "K. Brennan":         "Sport. Crisp, occasionally wry.",
    "L. Bordeaux":        "Culture & beyond the tape. Lighter touch, a little colour.",
    "The Editor":         "House editorials and judgement calls.",
    "Bot Tribune Staff":  "Collaborative or wires-driven pieces.",
}

# Valid sections the publisher can use. Trading floor + Desk's calls
# are appended by separate stages — publisher must not include them.
_LLM_SECTIONS = {
    "Front",
    "Markets",
    "World",
    "Tech & science",
    "Climate",
    "Health",
    "Sport",
    "Culture",
    "Beyond the tape",
}


@dataclass
class PlannedPiece:
    """One story slot in the day's plan. Brief/article writers consume
    this; they get the triage record separately for the facts."""
    slug: str                           # URL slug for the sub-page filename
    section: str
    headline: str                       # publisher may rewrite triage's title
    kicker: str                         # small-caps line above headline
    byline: str                         # one of BYLINES.keys()
    one_line: str                       # ~one-sentence summary for the front page
    tier: str                           # "lead" | "feature" | "brief"
    triage_index: int                   # back-reference into the triage list


@dataclass
class NewsPlan:
    """The publisher's output. Brief/article writers fan out across
    pieces; assembly stitches them into the final HTML."""
    edition_date: str                   # ISO date
    masthead_subtitle: str              # short descriptive line below the masthead
    front_lead_slug: str                # slug of the front lead piece
    pieces: list[PlannedPiece]          # in render order
    notes: str = ""                     # publisher's internal notes (debug-only)


def plan_edition(
    triaged: list[TriagedCandidate],
    today: date,
) -> NewsPlan:
    """Run the publisher. Returns the newspaper plan downstream stages
    consume. On LLM failure we fall back to a heuristic plan so the
    pipeline keeps producing an edition."""
    if not triaged:
        log.warning("Publisher: no triaged candidates, returning empty plan")
        return NewsPlan(
            edition_date=today.isoformat(),
            masthead_subtitle="A quiet day on the wires.",
            front_lead_slug="",
            pieces=[],
            notes="empty triage",
        )

    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        log.error("CLAUDE_CODE_OAUTH_TOKEN not set — publisher cannot run, using heuristic plan")
        return _heuristic_plan(triaged, today)

    prompt = _build_prompt(triaged, today)
    try:
        response = run_claude_for_json(prompt, model="sonnet", timeout_seconds=_PUBLISH_TIMEOUT)
    except ClaudeCodeError as e:
        log.warning("Publisher LLM call failed: %s — using heuristic plan", e)
        return _heuristic_plan(triaged, today)

    plan = _parse_plan(response, triaged, today)
    if not plan.pieces:
        log.warning("Publisher returned no pieces — using heuristic plan")
        return _heuristic_plan(triaged, today)
    return plan


def _build_prompt(triaged: list[TriagedCandidate], today: date) -> str:
    roster_lines = []
    for i, t in enumerate(triaged):
        roster_lines.append(
            f"[{i}] (score {t.score}/10, suggested: {t.suggested_section_final}) "
            f"{t.title}\n"
            f"      angle: {t.angle}\n"
            f"      one-line: {t.one_line}"
        )
    roster = "\n\n".join(roster_lines)

    byline_block = "\n".join(
        f"  - {name}: {blurb}" for name, blurb in BYLINES.items()
    )

    return f"""You are the publisher of The Bot Tribune for {today.isoformat()}.
The triage desk has scored every candidate. Your job: pick the lead,
organise the rest into sections, assign a byline, and write the day's
masthead subtitle.

## The roster (already sorted by triage score, highest first)

{roster}

## Byline roster (you MUST use these — do not invent new names)

{byline_block}

## Rules

1. **Front lead** — exactly one. Usually the highest-scoring piece with
   broad implications, but you can prefer a lower-scored piece if it
   reads better as the day's headline. Mark it tier="lead", section="Front".
2. **Sections** — always render: Markets, World, Tech & science,
   Beyond the tape (drop if a section has zero pieces). Render
   Climate, Health, Sport, Culture only when you have ≥1 piece that
   genuinely fits — don't pad to fill them.
3. **Do NOT include** "Trading floor" or "Desk's calls" sections — those
   are added by separate stages after you.
4. **Tier** — each piece is "lead" (front only), "feature" (mid-size,
   usually score 7+), or "brief" (small slot). Section pages mix
   features and briefs naturally.
5. **Headline** — you may rewrite the triage headline if it's weak or
   tabloid. Keep it accurate. Sentence-case with proper nouns; no
   ALL-CAPS, no clickbait.
6. **Kicker** — small-caps line above the headline. Usually the
   section + a single-word topic tag. Examples: "MARKETS · RATES",
   "WORLD · GAZA", "TECH · AI POLICY". Front lead's kicker is just
   the topic in caps, no "FRONT".
7. **Byline** — match the section to the persona where natural.
   Multiple pieces by the same byline in one section is fine; it's a
   real newspaper.
8. **One-line** — a single sentence the front page can use as the
   stand-first under the headline. Briefs use it as the brief body.
9. **Slug** — short, hyphenated, lowercase, URL-safe, unique within
   the edition. E.g., "fed-holds-rates", "uk-cpi-prints-hot".
10. **Volume** — no fixed cap per section. Use as many or as few as
    triage produced and the day deserves. A lazy day is allowed to be
    a lazy day.

## Masthead subtitle

One descriptive sentence (≤90 chars) that summarises the day, leaning
on the lead piece. **Dry, not arch.** No "in which…" or theatrical
framing. Examples of the right register:

- "Fed holds rates as the labour market wobbles; oil firms, sterling soft."
- "Israeli cabinet stalls on hostage vote; tech earnings split the tape."
- "Quiet day on the wires, with a chunky biotech deal as the exception."

## Required output

Return JSON only:

```json
{{
  "masthead_subtitle": "<one-sentence subtitle>",
  "pieces": [
    {{
      "triage_index": <int — index into the roster above>,
      "section": "Front" | "Markets" | "World" | "Tech & science" | "Climate" | "Health" | "Sport" | "Culture" | "Beyond the tape",
      "headline": "<rewritten or unchanged>",
      "kicker": "<SECTION · TOPIC or TOPIC>",
      "byline": "<one of the byline names above>",
      "one_line": "<single-sentence standfirst>",
      "tier": "lead" | "feature" | "brief",
      "slug": "<short-hyphenated-slug>"
    }}
  ],
  "notes": "<one-line internal note — what shape the day took, for debugging>"
}}
```

Render pieces in the order they should appear in the paper: Front first,
then each section in roster order (Markets, World, Tech & science,
then any variable sections, then Beyond the tape last). Within a
section, features before briefs.
"""


def _parse_plan(response: dict | list, triaged: list[TriagedCandidate], today: date) -> NewsPlan:
    if isinstance(response, list):
        response = {"pieces": response}
    if not isinstance(response, dict):
        return _heuristic_plan(triaged, today)

    raw_pieces = response.get("pieces") or []
    if not isinstance(raw_pieces, list):
        raw_pieces = []

    used_slugs: set[str] = set()
    pieces: list[PlannedPiece] = []
    lead_slug = ""

    for raw in raw_pieces:
        if not isinstance(raw, dict):
            continue
        try:
            idx = int(raw.get("triage_index", -1))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(triaged):
            continue

        section = str(raw.get("section") or "").strip()
        if section not in _LLM_SECTIONS:
            # Snap unknown sections to the triage suggestion
            section = triaged[idx].suggested_section_final
        if section not in _LLM_SECTIONS:
            section = "Beyond the tape"

        byline = str(raw.get("byline") or "").strip()
        if byline not in BYLINES:
            byline = "Bot Tribune Staff"

        tier = str(raw.get("tier") or "brief").strip().lower()
        if tier not in {"lead", "feature", "brief"}:
            tier = "brief"

        slug = _normalise_slug(raw.get("slug") or triaged[idx].title)
        slug = _unique_slug(slug, used_slugs)
        used_slugs.add(slug)

        piece = PlannedPiece(
            slug=slug,
            section=section if tier != "lead" else "Front",
            headline=str(raw.get("headline") or triaged[idx].title).strip(),
            kicker=str(raw.get("kicker") or "").strip().upper()[:60],
            byline=byline,
            one_line=str(raw.get("one_line") or triaged[idx].one_line).strip()[:320],
            tier=tier,
            triage_index=idx,
        )
        pieces.append(piece)
        if tier == "lead" and not lead_slug:
            lead_slug = piece.slug

    # If the model didn't mark anyone as lead, promote the first Front
    # piece or just the first piece.
    if not lead_slug and pieces:
        front_first = next((p for p in pieces if p.section == "Front"), None)
        if front_first is None:
            front_first = pieces[0]
            front_first.section = "Front"
            front_first.tier = "lead"
        lead_slug = front_first.slug

    subtitle = str(response.get("masthead_subtitle") or "").strip()
    if not subtitle and pieces:
        subtitle = pieces[0].one_line[:90]

    return NewsPlan(
        edition_date=today.isoformat(),
        masthead_subtitle=subtitle[:160],
        front_lead_slug=lead_slug,
        pieces=pieces,
        notes=str(response.get("notes") or "")[:280],
    )


def _heuristic_plan(triaged: list[TriagedCandidate], today: date) -> NewsPlan:
    """Fallback when the LLM is unavailable. Promotes the top-scored
    piece to the lead, groups the rest by their triage section, and
    keeps the paper alive."""
    if not triaged:
        return NewsPlan(
            edition_date=today.isoformat(),
            masthead_subtitle="A quiet day on the wires.",
            front_lead_slug="",
            pieces=[],
            notes="heuristic: empty roster",
        )

    # The top item is the lead
    section_persona = {
        "Markets":         "M. Halloway",
        "World":           "C. Tanaka",
        "Tech & science":  "R. Okonkwo",
        "Climate":         "S. Whitfield",
        "Health":          "Dr. P. Aaronson",
        "Sport":           "K. Brennan",
        "Culture":         "L. Bordeaux",
        "Beyond the tape": "L. Bordeaux",
    }

    used_slugs: set[str] = set()
    pieces: list[PlannedPiece] = []

    for i, t in enumerate(triaged):
        section = t.suggested_section_final if t.suggested_section_final in _LLM_SECTIONS else "Beyond the tape"
        is_lead = (i == 0)
        tier = "lead" if is_lead else ("feature" if t.score >= 7 else "brief")
        section_for_piece = "Front" if is_lead else section
        kicker_topic = section.upper() if not is_lead else "LEAD"

        slug = _normalise_slug(t.title)
        slug = _unique_slug(slug, used_slugs)
        used_slugs.add(slug)

        pieces.append(PlannedPiece(
            slug=slug,
            section=section_for_piece,
            headline=t.title,
            kicker=kicker_topic,
            byline=section_persona.get(section, "Bot Tribune Staff"),
            one_line=t.one_line or t.angle,
            tier=tier,
            triage_index=i,
        ))

    # Reorder: Front first, then a fixed section order, then everything else
    section_order = ["Front", "Markets", "World", "Tech & science", "Climate", "Health", "Sport", "Culture", "Beyond the tape"]
    pieces.sort(key=lambda p: (section_order.index(p.section) if p.section in section_order else 99, 0 if p.tier == "feature" else 1))

    lead_slug = pieces[0].slug if pieces else ""
    subtitle = (triaged[0].one_line or triaged[0].angle)[:90] if triaged else ""

    return NewsPlan(
        edition_date=today.isoformat(),
        masthead_subtitle=subtitle,
        front_lead_slug=lead_slug,
        pieces=pieces,
        notes="heuristic plan (LLM unavailable)",
    )


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _normalise_slug(raw: str) -> str:
    s = (raw or "").lower()
    s = _SLUG_RE.sub("-", s).strip("-")
    if not s:
        s = "story"
    # Cap to a sane length
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


# Round-trip helpers
def plan_to_json(plan: NewsPlan) -> dict:
    return {
        "edition_date": plan.edition_date,
        "masthead_subtitle": plan.masthead_subtitle,
        "front_lead_slug": plan.front_lead_slug,
        "notes": plan.notes,
        "pieces": [asdict(p) for p in plan.pieces],
    }


def plan_from_json(data: dict) -> NewsPlan:
    pieces_raw = data.get("pieces") or []
    pieces: list[PlannedPiece] = []
    for p in pieces_raw:
        if not isinstance(p, dict):
            continue
        pieces.append(PlannedPiece(
            slug=str(p.get("slug", "")),
            section=str(p.get("section", "Beyond the tape")),
            headline=str(p.get("headline", "")),
            kicker=str(p.get("kicker", "")),
            byline=str(p.get("byline", "Bot Tribune Staff")),
            one_line=str(p.get("one_line", "")),
            tier=str(p.get("tier", "brief")),
            triage_index=int(p.get("triage_index", -1)),
        ))
    return NewsPlan(
        edition_date=str(data.get("edition_date", "")),
        masthead_subtitle=str(data.get("masthead_subtitle", "")),
        front_lead_slug=str(data.get("front_lead_slug", "")),
        pieces=pieces,
        notes=str(data.get("notes", "")),
    )
