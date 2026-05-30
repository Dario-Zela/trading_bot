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


_PUBLISH_TIMEOUT = 600    # heavier on busy days (40+ candidates); macro hit
                          # the same wall last week, news caught it next.


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
# "Standing watches" carries ongoing situations (Iran/Hormuz, US-China
# trade, UK fiscal) as 2-line state-of-play instead of letting them
# crowd Front every day.
_LLM_SECTIONS = {
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
}

# Coarse region tags. The publisher tags each piece with one of these so
# the geographic-floor check can enforce non-US presence on Front.
_REGIONS = {"us", "uk", "eu", "asia", "global"}
_NON_US_REGIONS = {"uk", "eu", "asia", "global"}

# Allowed `kind` values. "new" = genuinely new event today;
# "rolling-update" = ongoing situation with marginal development.
# Standing watches usually carries rolling-updates of fatigued themes.
_KINDS = {"new", "rolling-update"}

# Theme-fatigue threshold: a theme that's been Front this many times
# in the trailing 7 days is fatigued and should NOT lead again unless
# the piece is flagged `kind: "new"` (genuine escalation/resolution).
_FATIGUE_LIMIT = 3


@dataclass
class PlannedPiece:
    """One story slot in the day's plan. Brief/article writers consume
    this; they get the triage record(s) separately for the facts.

    Stories can MERGE multiple triaged candidates into a single piece —
    `triage_indices` is the merged set, `triage_index` is the primary
    (used for back-compat with older code paths that consume one
    record at a time). The article writer reads all merged sources
    and synthesises across them so the reader isn't given ten
    overlapping Fed pieces; the brief writer reads just the primary."""
    slug: str                           # URL slug for the sub-page filename
    section: str
    headline: str                       # publisher may rewrite triage's title
    kicker: str                         # small-caps line above headline
    byline: str                         # one of BYLINES.keys()
    one_line: str                       # ~one-sentence summary for the front page
    tier: str                           # "lead" | "feature" | "brief"
    triage_index: int                   # primary source — back-reference into triage list
    triage_indices: list = field(default_factory=list)  # all merged sources
    theme: str = ""                     # coarse theme tag (kebab-case, e.g. "iran-hormuz",
                                        # "fed-rates", "uk-fiscal"). Powers the fatigue gate.
    region: str = "global"              # one of _REGIONS — for the geographic floor
    kind: str = "new"                   # one of _KINDS — "new" or "rolling-update"


@dataclass
class NewsPlan:
    """The publisher's output. Brief/article writers fan out across
    pieces; assembly stitches them into the final HTML."""
    edition_date: str                   # ISO date
    masthead_subtitle: str              # short descriptive line below the masthead
    front_lead_slug: str                # slug of the front lead piece
    pieces: list[PlannedPiece]          # in render order
    notes: str = ""                     # publisher's internal notes (debug-only)
    todays_question: str = ""           # the framing question the strategies should keep in mind


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
    _enforce_geographic_floor(plan)
    return plan


def _enforce_geographic_floor(plan: NewsPlan) -> None:
    """Guarantee at least 1 non-US piece in the top 3 (lead + first two
    features), regardless of what the LLM did. If the top 3 are all
    `region: "us"`, swap in the highest-tier non-US piece from later
    in the order. Mutates `plan.pieces` in place.

    No-op when there are <3 pieces or when there's already a non-US
    piece in the top 3.
    """
    if len(plan.pieces) < 3:
        return
    top = plan.pieces[:3]
    if any(p.region in _NON_US_REGIONS for p in top):
        return
    # Find the earliest non-US piece outside the top 3. Prefer
    # features over briefs so we don't promote a low-quality piece.
    swap_idx = -1
    for i in range(3, len(plan.pieces)):
        p = plan.pieces[i]
        if p.region in _NON_US_REGIONS and p.tier != "brief":
            swap_idx = i
            break
    if swap_idx < 0:
        for i in range(3, len(plan.pieces)):
            if plan.pieces[i].region in _NON_US_REGIONS:
                swap_idx = i
                break
    if swap_idx < 0:
        log.info("Geographic floor: no non-US piece available to swap in")
        return
    # Demote the weakest of the top 3 (the third one); promote the
    # non-US piece into position 2 (first feature slot) so the lead
    # stays as-is.
    promoted = plan.pieces.pop(swap_idx)
    demoted = plan.pieces.pop(2)
    plan.pieces.insert(2, promoted)
    plan.pieces.append(demoted)
    log.info(
        "Geographic floor enforced: promoted '%s' (region=%s) into top 3",
        promoted.slug, promoted.region,
    )


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

    # Trailing 7-day context: per-theme Front counts (drives the
    # fatigue gate) + per-region totals (drives the geographic floor)
    # + recent lead headlines.
    recent_leads_block = _recent_activity_context(today)

    return f"""You are the publisher of The Bot Tribune for {today.isoformat()}.
The triage desk has scored every candidate. Your job: pick the lead,
organise the rest into sections, assign a byline, and write the day's
masthead subtitle.

## The roster (already sorted by triage score, highest first)

{roster}
{recent_leads_block}

## Byline roster (you MUST use these — do not invent new names)

{byline_block}

## Rules

1. **Front lead** — exactly one. Usually the highest-scoring piece with
   broad implications, but you can prefer a lower-scored piece if it
   reads better as the day's headline. Mark it tier="lead", section="Front".
2. **Cluster aggressively.** If multiple triaged candidates cover the
   same underlying story — same central event, same actor, same
   directional implication — MERGE them into a single piece. List
   every member of the cluster in `triage_indices` (in priority
   order, primary first). Example: ten triage rows about the Fed all
   become **one** Markets piece with all ten indices listed.
   Heuristics for "same story":
   - Same headline noun phrase (Fed, BoE, Gaza, Apple earnings, …)
     with different angles → merge.
   - Same company across different outlets → merge.
   - Same regulatory action with different reaction pieces → merge.
   - Two genuinely independent angles on the same actor (e.g. an
     earnings beat AND a leadership reshuffle) → keep separate.
   The reader gets richer, longer pieces with multiple sourced
   angles instead of repetitive listicles. The writer downstream is
   told to synthesise — your job is just the clustering.
3. **Sections** — always render: Markets, World, Tech & science,
   Beyond the tape (drop if a section has zero pieces). Render
   Climate, Health, Sport, Culture only when you have ≥1 piece that
   genuinely fits — don't pad to fill them.

   **Standing watches** is for ongoing situations (Iran/Hormuz,
   US-China trade, UK fiscal squeeze, gilt-market stress) where the
   theme has been live for many days and today's "update" is
   marginal. Render these as `tier: "brief"` 2-line state-of-play
   pieces, NOT as front-page leads. A theme belongs here if (a) it's
   appeared on Front 3+ times in the last 7 days per the recent-
   activity block AND (b) today's development is incremental (a new
   speaker comment, a fresh skirmish, another deal-near rumour) rather
   than a genuine inflection (signed agreement, escalation,
   collapse). The reader wants to know the situation is still live
   without re-reading yesterday's headline.
4. **Do NOT include** "Trading floor" or "Desk's calls" sections — those
   are added by separate stages after you.
5. **Tier** — each piece is "lead" (front only), "feature" (mid-size,
   usually score 7+), or "brief" (small slot). Section pages mix
   features and briefs naturally. A clustered piece carrying 5+
   merged sources almost always rates "feature" or "lead", not "brief".
6. **Headline** — you may rewrite the triage headline if it's weak or
   tabloid. Keep it accurate. Sentence-case with proper nouns; no
   ALL-CAPS, no clickbait. For a clustered piece, the headline
   should capture the whole arc, not just the top member.
7. **Kicker** — small-caps line above the headline. Usually the
   section + a single-word topic tag. Examples: "MARKETS · RATES",
   "WORLD · GAZA", "TECH · AI POLICY". Front lead's kicker is just
   the topic in caps, no "FRONT".
8. **Byline** — match the section to the persona where natural.
   Multiple pieces by the same byline in one section is fine; it's a
   real newspaper.
9. **One-line** — a single sentence the front page can use as the
   stand-first under the headline. Briefs use it as the brief body.
10. **Slug** — short, hyphenated, lowercase, URL-safe, unique within
    the edition. E.g., "fed-holds-rates", "uk-cpi-prints-hot".
11. **Volume** — no fixed cap per section. Use as many or as few as
    triage produced and the day deserves, AFTER clustering. If you
    end up with 25 candidates and they cluster into 8 distinct
    stories, ship 8 pieces — not 25. A lazy day is allowed to be a
    lazy day.

## Masthead subtitle

One descriptive sentence (≤90 chars) that summarises the day, leaning
on the lead piece. **Dry, not arch.** No "in which…" or theatrical
framing. Examples of the right register:

- "Fed holds rates as the labour market wobbles; oil firms, sterling soft."
- "Israeli cabinet stalls on hostage vote; tech earnings split the tape."
- "Quiet day on the wires, with a chunky biotech deal as the exception."

## Today's question

One question (≤120 chars) the trading strategies should keep in mind
as they read the brief. The question should be falsifiable in
principle and rooted in today's lead. Examples:

- "Does Fed credibility survive a print this hot without a verbal shock?"
- "Is the Gaza ceasefire window real enough for energy to fade today?"
- "Will the chip-sector pullback hold below the 50-day, or bounce on AMD?"

Dry. No "we wonder" / "could it be that..." padding.

## Per-piece tagging (required)

Every piece you emit MUST carry three tags so the fatigue gate, the
geographic floor, and tomorrow's run can do their jobs:

- **`theme`** — a coarse kebab-case tag for the story arc. Same
  story across days = same theme. Examples: `iran-hormuz`,
  `fed-rates`, `us-china-trade`, `uk-fiscal`, `uk-politics`,
  `boe-rates`, `ai-capex`, `oil`, `gilts`, `nvidia-earnings`,
  `bp-governance`, `spacex-ipo`. Pick the tightest tag that
  groups multiple days' coverage of the SAME underlying arc.
  *Don't invent a new theme tag just because today's angle is
  slightly different.* "Iran nuclear talks" yesterday and
  "Hormuz ceasefire" today share the theme `iran-hormuz`.
- **`region`** — one of `us`, `uk`, `eu`, `asia`, `global`.
  Use `global` for cross-cutting or geopolitical stories that
  aren't anchored to one capital (oil, multilateral diplomacy,
  cross-asset macro).
- **`kind`** — `"new"` or `"rolling-update"`. `"new"` = a
  genuine first-day event or a material inflection in an ongoing
  story (signing, escalation, resignation, surprise data
  print). `"rolling-update"` = incremental update of an
  ongoing situation (another speaker comment, another "deal
  near" leak, marginal market move).

## Theme fatigue (don't crowd Front with the same situation)

The recent-activity block above shows which themes have been on
Front in the last 7 days. **A theme that's already led Front 3+
times in 7 days is FATIGUED.** Fatigued themes do NOT belong on
Front again unless the piece is genuinely `kind: "new"` (signed
agreement, escalation, collapse, major surprise). Marginal
updates on fatigued themes go to **Standing watches** as a
2-line state-of-play brief.

This is not "ban the topic" — the topic stays in the paper. It's
"stop re-leading the same slow burn." The reader has been reading
about Iran/Hormuz/Fed-speak/UK-fiscal for days; today's marginal
turn doesn't deserve the masthead.

## Geographic floor (Front cannot be all-US)

This is a UK-focused publication. **At least 1 of the top 3 pieces
(lead + first 2 features) MUST be tagged `region` other than `us`.**
If the top-scored stories are all US, pick a high-scored UK / EU /
Asia / global piece and promote it into the top 3. A fresh non-US
angle is worth one rank point of triage score for breaking
out of US-skew.

## Required output

Return JSON only:

```json
{{
  "masthead_subtitle": "<one-sentence subtitle>",
  "todays_question": "<one falsifiable question ≤120 chars>",
  "pieces": [
    {{
      "triage_indices": [<int>, <int>, ...],  // primary first; one index for a solo story, many for a merged cluster
      "section": "Front" | "Standing watches" | "Markets" | "World" | "Tech & science" | "Climate" | "Health" | "Sport" | "Culture" | "Beyond the tape",
      "headline": "<rewritten or unchanged>",
      "kicker": "<SECTION · TOPIC or TOPIC>",
      "byline": "<one of the byline names above>",
      "one_line": "<single-sentence standfirst>",
      "tier": "lead" | "feature" | "brief",
      "slug": "<short-hyphenated-slug>",
      "theme": "<kebab-case theme tag — match prior days' tags when same arc>",
      "region": "us" | "uk" | "eu" | "asia" | "global",
      "kind": "new" | "rolling-update"
    }}
  ],
  "notes": "<one-line internal note — what shape the day took, for debugging>"
}}
```

Render pieces in the order they should appear in the paper: Front first,
then Standing watches, then each section in roster order (Markets, World,
Tech & science, then any variable sections, then Beyond the tape last).
Within a section, features before briefs.
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
        # Accept both new shape (triage_indices: [int, ...]) and legacy
        # shape (triage_index: int). Normalise to a deduped list of
        # in-range indices; the first one is the "primary".
        indices_raw = raw.get("triage_indices")
        if not isinstance(indices_raw, list):
            indices_raw = [raw.get("triage_index", -1)]
        idx_list: list[int] = []
        seen_idx: set[int] = set()
        for v in indices_raw:
            try:
                vi = int(v)
            except (TypeError, ValueError):
                continue
            if vi < 0 or vi >= len(triaged):
                continue
            if vi in seen_idx:
                continue
            seen_idx.add(vi)
            idx_list.append(vi)
        if not idx_list:
            continue
        idx = idx_list[0]

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

        theme_raw = str(raw.get("theme") or "").strip().lower()
        # Normalise to kebab-case: spaces/underscores → "-", strip non
        # [a-z0-9-]. Empty theme is permitted (falls back to "").
        theme = re.sub(r"[^a-z0-9]+", "-", theme_raw).strip("-")[:40]

        region = str(raw.get("region") or "global").strip().lower()
        if region not in _REGIONS:
            region = "global"

        kind = str(raw.get("kind") or "new").strip().lower()
        if kind not in _KINDS:
            kind = "new"

        piece = PlannedPiece(
            slug=slug,
            section=section if tier != "lead" else "Front",
            headline=str(raw.get("headline") or triaged[idx].title).strip(),
            kicker=str(raw.get("kicker") or "").strip().upper()[:60],
            byline=byline,
            one_line=str(raw.get("one_line") or triaged[idx].one_line).strip()[:320],
            tier=tier,
            triage_index=idx,
            triage_indices=idx_list,
            theme=theme,
            region=region,
            kind=kind,
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
        todays_question=str(response.get("todays_question") or "").strip()[:200],
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
            triage_indices=[i],
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
        todays_question="",
    )


def _recent_activity_context(today: date) -> str:
    """Pull the trailing 7 days of publisher plans and compute two
    summaries the publisher LLM uses to avoid theme fatigue and US
    skew:

    1. A per-theme count of Front appearances (and total appearances).
       Themes that have led Front 3+ times in 7 days are flagged
       FATIGUED — those should drop to Standing watches unless
       genuinely escalated.
    2. A per-region count of total pieces. Lets the LLM see the
       running geographic mix so it can rebalance proactively.
    3. The list of recent front-page lead headlines (legacy
       behaviour — kept so the LLM still sees the actual stories,
       not just tags).
    """
    import json as _json
    from collections import Counter
    from datetime import timedelta
    from trading_bot.state.paths import STATE_ROOT
    state_dir = STATE_ROOT / "daily_news"
    if not state_dir.exists():
        return ""

    front_theme_days: Counter = Counter()
    total_theme: Counter = Counter()
    region_total: Counter = Counter()
    recent_leads: list[tuple[str, str, str, str]] = []  # (date, headline, slug, theme)

    for offset in range(1, 8):
        d = (today - timedelta(days=offset)).isoformat()
        p = state_dir / f"{d}.pipeline.json"
        if not p.exists():
            continue
        try:
            payload = _json.loads(p.read_text())
        except _json.JSONDecodeError:
            continue
        plan = payload.get("stages", {}).get("publisher", {}) or {}
        lead_slug = plan.get("front_lead_slug", "")
        pieces = plan.get("pieces", []) or []
        for pc in pieces:
            theme = (pc.get("theme") or "").strip().lower()
            region = (pc.get("region") or "global").strip().lower()
            if theme:
                total_theme[theme] += 1
                if pc.get("section") == "Front":
                    front_theme_days[theme] += 1
            if region:
                region_total[region] += 1
        lead = next((pc for pc in pieces if pc.get("slug") == lead_slug), None)
        if lead:
            recent_leads.append((
                d, lead.get("headline", "(unknown)"),
                lead_slug, (lead.get("theme") or "").lower(),
            ))

    if not recent_leads and not front_theme_days:
        return ""

    lines = ["", "## Recent activity (last 7 days)", ""]

    if front_theme_days:
        lines.append("### Theme appearances on Front (out of last 7 editions)")
        lines.append("")
        lines.append("| theme | front-days | total-pieces | fatigued? |")
        lines.append("|---|---:|---:|:---:|")
        for theme, n_front in sorted(front_theme_days.items(), key=lambda x: -x[1]):
            n_total = total_theme[theme]
            fatigued = "**YES**" if n_front >= _FATIGUE_LIMIT else "no"
            lines.append(f"| `{theme}` | {n_front} | {n_total} | {fatigued} |")
        lines.append("")

    if region_total:
        lines.append("### Region mix (last 7 days, all sections)")
        lines.append("")
        total = sum(region_total.values()) or 1
        for region in ("us", "uk", "eu", "asia", "global"):
            n = region_total.get(region, 0)
            pct = 100 * n / total
            lines.append(f"- **{region}** — {n} pieces ({pct:.0f}%)")
        lines.append("")

    if recent_leads:
        lines.append("### Recent front-page leads")
        lines.append("")
        for d, headline, slug, theme in recent_leads:
            theme_tag = f" · `{theme}`" if theme else ""
            lines.append(f"- **{d}** — *{headline}*  (slug: `{slug}`{theme_tag})")
        lines.append("")

    return "\n".join(lines)


# Legacy alias retained for any external caller; new code should use
# _recent_activity_context.
_recent_leads_context = _recent_activity_context


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
        "todays_question": plan.todays_question,
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
        region = str(p.get("region", "global")).lower()
        if region not in _REGIONS:
            region = "global"
        kind = str(p.get("kind", "new")).lower()
        if kind not in _KINDS:
            kind = "new"
        pieces.append(PlannedPiece(
            slug=str(p.get("slug", "")),
            section=str(p.get("section", "Beyond the tape")),
            headline=str(p.get("headline", "")),
            kicker=str(p.get("kicker", "")),
            byline=str(p.get("byline", "Bot Tribune Staff")),
            one_line=str(p.get("one_line", "")),
            tier=str(p.get("tier", "brief")),
            triage_index=int(p.get("triage_index", -1)),
            triage_indices=[int(x) for x in (p.get("triage_indices") or [p.get("triage_index", -1)]) if isinstance(x, (int, float))],
            theme=str(p.get("theme", "")).lower()[:40],
            region=region,
            kind=kind,
        ))
    return NewsPlan(
        edition_date=str(data.get("edition_date", "")),
        masthead_subtitle=str(data.get("masthead_subtitle", "")),
        front_lead_slug=str(data.get("front_lead_slug", "")),
        pieces=pieces,
        notes=str(data.get("notes", "")),
        todays_question=str(data.get("todays_question", "")),
    )
