"""Phase 2E — Full-article writers.

Sonnet × 6 parallel. Each call writes the *full* article — no word
cap. The writer is also responsible for finding a fitting hero image
via WebSearch / WebFetch.

Tone progresses naturally from the brief: open with the situation in
plain English, then earn the technical depth. A full piece may run
short (terse story) or long (story warrants depth) — the writer
decides, no cap from us.

Image policy
============
We allow hot-linking — this is a personal newspaper, not a commercial
publication. The writer is told to prefer Wikipedia/Wikimedia Commons
(stable URLs, hot-link-friendly) but may pick any direct image URL it
believes will load.

Output
======
The article emits structured JSON — markdown body, image_url,
image_caption, image_credit, an "in one sentence" pull-callout, a
sources block, and slugs of other pieces in this edition that are
worth cross-linking. Assembly templates this into HTML.
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
_ARTICLE_TIMEOUT = 600        # full Sonnet article + image search can take several minutes
# Tools the article writer needs: WebSearch (find image candidates,
# verify facts) and WebFetch (load a candidate image URL to confirm
# it returns an image). Comma-separated to match Claude Code CLI.
_WRITER_TOOLS = ["--allowedTools", "WebSearch,WebFetch"]


@dataclass
class FullArticle:
    """One full article. Renders into docs/news/YYYY-MM-DD/{slug}.html."""
    slug: str
    headline: str
    kicker: str
    byline: str
    body_md: str                                # full article markdown
    in_one_sentence: str = ""                   # pull-callout: TL;DR for the impatient reader
    image_url: str = ""
    image_caption: str = ""
    image_credit: str = ""
    sources: list[dict] = field(default_factory=list)   # [{"title": ..., "url": ...}]
    related_slugs: list[str] = field(default_factory=list)
    failed: bool = False


def write_articles(
    plan: NewsPlan,
    triaged: list[TriagedCandidate],
    today: date,
) -> dict[str, FullArticle]:
    """Write a full article per planned piece. Returns slug→FullArticle.

    If the OAUTH token is missing we return shallow fallbacks so the
    edition still has *something* on the per-article subpages.
    """
    if not plan.pieces:
        return {}
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        log.error("CLAUDE_CODE_OAUTH_TOKEN not set — using fallback articles for all pieces")
        return {p.slug: _fallback_article(p, triaged) for p in plan.pieces}

    log.info("Articles: writing %d pieces (max %d in parallel)", len(plan.pieces), _MAX_PARALLEL)
    articles: dict[str, FullArticle] = {}
    related_pool = [(p.slug, p.headline, p.section) for p in plan.pieces]

    with ThreadPoolExecutor(max_workers=_MAX_PARALLEL) as pool:
        futures = {}
        for p in plan.pieces:
            # Use the merged-source list when present; fall back to the
            # singleton primary for older plans / heuristic output. Both
            # filtered to in-range indices.
            indices = p.triage_indices or ([p.triage_index] if p.triage_index >= 0 else [])
            sources = [triaged[i] for i in indices if 0 <= i < len(triaged)]
            if not sources:
                continue
            futures[pool.submit(_write_one, p, sources, today, related_pool)] = p
        for fut in as_completed(futures):
            piece = futures[fut]
            try:
                articles[piece.slug] = fut.result()
            except Exception as e:
                log.warning("Article failed for %r: %s — using fallback", piece.slug, e)
                articles[piece.slug] = _fallback_article(piece, triaged)

    for p in plan.pieces:
        if p.slug not in articles:
            articles[p.slug] = _fallback_article(p, triaged)

    # Phase 9B — image relevance check. The writer's first-pass image
    # search occasionally returns something off-topic (generic stock
    # photo, wrong company). Fan out a fast Haiku verifier; drop the
    # image when it comes back "no". Fast + cheap; failures are silent.
    _verify_hero_images(articles, plan, triaged)

    n_failed = sum(1 for a in articles.values() if a.failed)
    log.info("Articles complete: %d written, %d fallback", len(articles) - n_failed, n_failed)
    return articles


def _verify_hero_images(
    articles: dict[str, FullArticle],
    plan: NewsPlan,
    triaged: list[TriagedCandidate],
) -> None:
    """Phase 9B — second-pass relevance check. For every article whose
    writer returned an image_url, ask Haiku whether the image looks
    on-topic given the headline + caption + first paragraph. Verdicts:
    'yes' keeps it, 'borderline' / 'no' drops the hero so the page
    renders without a misleading photo."""
    pieces_with_images = [
        p for p in plan.pieces
        if p.slug in articles and articles[p.slug].image_url and not articles[p.slug].failed
    ]
    if not pieces_with_images:
        return
    log.info("Image relevance: verifying %d hero images", len(pieces_with_images))
    with ThreadPoolExecutor(max_workers=_MAX_PARALLEL) as pool:
        futures = {
            pool.submit(_verify_one_image, p, articles[p.slug], triaged): p
            for p in pieces_with_images
        }
        n_dropped = 0
        for fut in as_completed(futures):
            piece = futures[fut]
            try:
                verdict = fut.result()
            except Exception as e:
                log.debug("Image verifier failed for %r: %s — keeping image", piece.slug, e)
                continue
            if verdict in ("no", "borderline"):
                art = articles[piece.slug]
                log.info("Image dropped for %r (verdict %s): %s",
                         piece.slug, verdict, art.image_url[:90])
                art.image_url = ""
                art.image_caption = ""
                art.image_credit = ""
                n_dropped += 1
        if n_dropped:
            log.info("Image relevance: %d dropped, %d kept",
                     n_dropped, len(pieces_with_images) - n_dropped)


_IMAGE_CHECK_TIMEOUT = 60


def _verify_one_image(piece, article, triaged) -> str:
    """Single Haiku verifier. Returns 'yes' / 'borderline' / 'no'."""
    body_first_para = ""
    if article.body_md:
        body_first_para = article.body_md.split("\n\n")[0][:400]
    prompt = f"""You verify whether an article's hero image looks
on-topic. Brief judgment call — yes / borderline / no — based on the
headline, caption, and opening paragraph.

## The piece

- Headline: {piece.headline}
- Section: {piece.section}
- First paragraph: {body_first_para}

## The image

- URL: {article.image_url}
- Caption (writer's): {article.image_caption}
- Credit: {article.image_credit}

## Rules

- "yes" — the image clearly relates to the story's subject (the named
  company, the named person, the named event). Includes generic-but-
  appropriate (a stock chart for a markets piece, a refinery for an
  energy piece) if the headline genuinely warrants it.
- "borderline" — the image is loosely related but generic where a
  specific image would be expected (e.g., a generic skyline for a
  named company's headquarters story).
- "no" — clearly off-topic (wrong company, wrong country, wildly
  generic on a specific story).

Use WebFetch to load the URL only if needed to disambiguate; usually
the caption + URL filename is enough.

## Required output

```json
{{
  "verdict": "yes" | "borderline" | "no",
  "note": "<one line on why>"
}}
```
"""
    try:
        response = run_claude_for_json(
            prompt, model="haiku",
            timeout_seconds=_IMAGE_CHECK_TIMEOUT,
            extra_args=["--allowedTools", "WebFetch"],
        )
    except ClaudeCodeError:
        return "yes"     # on failure, keep the image — better an iffy pic than none
    if not isinstance(response, dict):
        return "yes"
    v = (response.get("verdict") or "").strip().lower()
    return v if v in {"yes", "borderline", "no"} else "yes"


def _write_one(
    piece: PlannedPiece,
    sources: list[TriagedCandidate],
    today: date,
    related_pool: list[tuple[str, str, str]],
) -> FullArticle:
    prompt = _build_prompt(piece, sources, today, related_pool)
    try:
        response = run_claude_for_json(
            prompt,
            model="sonnet",
            timeout_seconds=_ARTICLE_TIMEOUT,
            extra_args=_WRITER_TOOLS,
        )
    except ClaudeCodeError as e:
        log.warning("Article Sonnet failed for %r: %s", piece.slug, e)
        return _fallback_article(piece, sources)
    return _parse_article(piece, response)


def _build_prompt(
    piece: PlannedPiece,
    sources: list[TriagedCandidate],
    today: date,
    related_pool: list[tuple[str, str, str]],
) -> str:
    persona = BYLINES.get(piece.byline, "Staff writer.")
    primary = sources[0]
    is_cluster = len(sources) > 1

    # Build the facts / sources blocks across every merged source. The
    # writer is told to synthesise rather than write a list of mini-
    # articles — the cluster gives breadth, the synthesis gives depth.
    fact_lines: list[str] = []
    src_lines: list[str] = []
    for i, t in enumerate(sources, start=1):
        if is_cluster:
            fact_lines.append(f"### Angle {i}: {t.title}")
            if t.angle:
                fact_lines.append(f"_{t.angle}_")
                fact_lines.append("")
        for f in (t.key_facts or []):
            fact_lines.append(f"- {f}")
        if is_cluster:
            fact_lines.append("")
        for s in (t.source_hints or []):
            src_lines.append(f"  - {s}")
    facts_block = "\n".join(fact_lines).strip() or "(no triage facts — work from the angle and what you find via web search)"
    sources_block = "\n".join(src_lines) if src_lines else "  (none provided — find your own)"

    if is_cluster:
        cluster_note = (
            f"\n## Cluster context\n\n"
            f"This piece merges **{len(sources)} triaged candidates** "
            f"into one story. The angles below all touch the same "
            f"underlying event — synthesise across them rather than "
            f"writing a list of separate beats. Lead with the strongest "
            f"angle, weave in supporting threads, end on the implication "
            f"that ties them together.\n"
        )
    else:
        cluster_note = ""

    # Cross-link pool: pieces in this edition the writer might reference
    related_block = "\n".join(
        f"  - {slug}: {headline} ({section})"
        for slug, headline, section in related_pool
        if slug != piece.slug
    ) or "  (none)"

    tier_guidance = {
        # Word ranges are tightened from earlier versions — the previous
        # 1500-2000 target was hitting Sonnet's output-token ceiling and
        # truncating the closing JSON fence, which broke the parser. These
        # ceilings keep the body comfortably below ~3,000 tokens including
        # the surrounding JSON fields.
        "lead": "FRONT PAGE LEAD. Write at the depth a serious reader would want — 700-1100 words. Use 3-4 sub-headings (## ...) to break the body into clear movements. A clustered lead can stretch to 1200 if necessary, but DON'T pad to fill space.",
        "feature": "Feature piece. 500-900 words; clustered features can reach 1000. Use 2-3 sub-headings. Substance over length.",
        "brief": "Brief slot — the full article is still a proper read. 350-550 words; clustered briefs up to 700. One sub-heading is fine; two is the ceiling.",
    }.get(piece.tier, "Write to the length the story deserves — 500-800 words is a reasonable default.")

    return f"""You are {piece.byline} writing the full article for The Bot
Tribune on {today.isoformat()}.
Your beat persona: {persona}

## The piece

- **Headline:** {piece.headline}
- **Kicker:** {piece.kicker}
- **Section:** {piece.section}
- **Tier:** {piece.tier} — {tier_guidance}
- **Angle:** {primary.angle}
- **One-line:** {primary.one_line}
- **Why it matters:** {primary.why_it_matters}
{cluster_note}
## Key facts (from triage)

{facts_block}

## Source hints

{sources_block}

## Other pieces in this edition (for cross-linking via related_slugs)

{related_block}

## You have these tools available

- **WebSearch** — use to verify facts, find quotes, check named actors,
  confirm dates, and locate fitting images.
- **WebFetch** — use to load specific URLs (a candidate image URL to
  confirm it returns a real image, a source article to read it
  properly, etc.).

You SHOULD use web search aggressively before writing. The triage
facts are a starting point, not the final word — verify them, expand
them, find quotes and concrete numbers, and reach for sources beyond
the hints. If something doesn't check out, drop it.

## Images — hero + inline

Pieces are meant to read like a real newspaper section. The hero
image goes at the top of the article; inline images break up the
body and visualise specific moments in the story. Aim for:

- **lead / feature pieces with clusters**: 2-4 inline images on top
  of the hero. A relevant chart, a named person, a building / city,
  a product shot — let the visuals do narrative work.
- **solo features**: 1-2 inline images plus the hero.
- **briefs**: hero only is fine; one inline only if it carries
  meaning the prose doesn't.

For each image find a stable direct URL (ending .jpg / .png / .webp):
- **Wikipedia / Wikimedia Commons** images first (stable, licensed,
  e.g. `https://upload.wikimedia.org/...`)
- Major-publication photo URLs if you find them on a search result
  (Reuters, AP, FT, etc.) — hot-linking works in practice
- Stock-photo URLs (Unsplash, Pexels) when the topic is generic

Avoid:
- Image search RESULT pages (these don't render; you need the actual
  image URL ending in .jpg / .png / .webp)
- Images behind login walls (Twitter image CDN often fails)
- Generic stock photos when a specific topical image exists

Use WebFetch to verify each URL returns an image. If a slot would
require a bad/wrong photo, just leave it empty — fewer good images
beats one misleading one. Place inline images inside the body_md
where they fit narratively (use the `![caption](url)` markdown).

## Writing rules

1. **Open with the situation, in plain English.** A reader new to this
   beat should be able to follow the first paragraph.
2. **Then earn the technical depth.** Numbers, named actors, quotes,
   regulatory specifics — they belong in the body, not the lede.
3. **Show your reasoning.** When you make an interpretive claim
   ("this suggests..."), point at the evidence that supports it.
4. **Concrete over abstract.** "The bill passed 52-48" beats "the bill
   passed narrowly."
5. **Voice matters.** This is a newspaper, not a wire feed. Use
   varied sentence rhythms, the occasional turn of phrase, a scene-
   setting paragraph where it lands. Per the byline persona: not
   purple, but not flat. Think mid-broadsheet — FT magazine,
   Economist long-read, NYT business feature.
6. **No clichés, no breathless framing, no clickbait.** "Stunning",
   "watershed", "everything we knew is wrong" — cut them.
7. **Don't address the reader.** No "we'll see", no "your portfolio",
   no "as we noted yesterday."
8. **Sub-headings as structural beats.** For longer pieces, use 2-5
   `## Sub-heading` lines that mark distinct movements — context →
   the event → reaction → implication. Don't bury everything in one
   wall of paragraphs.
9. **End with what's next.** A natural forward-look — what to watch,
   what's at stake, when the next data point lands. Avoid "stay
   tuned" — name the specific catalyst.

## Cluster synthesis (when this piece merges multiple sources)

If the cluster context above shows multiple angles, your job is to
weave them, not list them. Devote a sub-heading per major thread,
return to the spine of the central story at each transition, and
in the closer tie the threads together into the implication that
covers them all. The reader should finish thinking "I got the
whole shape of this story", not "I read seven mini-articles".

## "In one sentence" callout

A single sentence (≤180 chars) — the TL;DR for the impatient reader.
Goes in a callout box near the top of the article. Not the same as
the kicker or headline — it's the substance compressed to one line.

## Hard contract

Your response MUST begin with the `{{` of the JSON object. Do NOT write
any preamble, narration, planning, or explanation before the JSON.
Don't say "Let me compose…" or "Now I have enough…" or "Here is the
output:" — those phrases waste output tokens and have repeatedly caused
the JSON to truncate mid-body when Sonnet's response budget runs short.
Open with `{{`, close with `}}`, nothing else.

If you find yourself wanting to explain what you did, resist — that's
output budget the closing `}}` needs to fit. Write the article in
`body_md` as the article itself, and let it speak for you.

## Required output

```json
{{
  "body_md": "<the full article as markdown. Sub-headings use ##, blockquotes for quotes, em-dashes — like this — not '--'. Inline images go in the body via ![caption](url). NO headline or byline in the body.>",
  "in_one_sentence": "<one sentence TL;DR, ≤180 chars>",
  "image_url": "<direct hero image URL, or empty string>",
  "image_caption": "<one-line caption describing the hero image>",
  "image_credit": "<photographer / outlet / Wikipedia author — whoever should be credited>",
  "sources": [
    {{"title": "<source title or outlet>", "url": "<source url>"}},
    ...
  ],
  "related_slugs": ["<slug from the cross-link pool above that's actually relevant — or empty list>"]
}}
```
"""


def _parse_article(piece: PlannedPiece, response: dict | list) -> FullArticle:
    if isinstance(response, list) and response:
        response = response[0] if isinstance(response[0], dict) else {}
    if not isinstance(response, dict):
        return _fallback_article(piece, [])

    body = str(response.get("body_md") or response.get("body") or "").strip()
    if not body:
        return _fallback_article(piece, [])

    sources_raw = response.get("sources") or []
    sources: list[dict] = []
    if isinstance(sources_raw, list):
        for s in sources_raw:
            if isinstance(s, dict):
                title = str(s.get("title") or s.get("name") or "").strip()
                url = str(s.get("url") or s.get("href") or "").strip()
                if title or url:
                    sources.append({"title": title or url, "url": url})
            elif isinstance(s, str) and s.strip():
                # Bare string — treat as title with no URL
                sources.append({"title": s.strip(), "url": ""})

    related_raw = response.get("related_slugs") or []
    if isinstance(related_raw, str):
        related_raw = [related_raw]
    related = [str(r).strip() for r in related_raw if str(r).strip() and str(r).strip() != piece.slug][:6]

    return FullArticle(
        slug=piece.slug,
        headline=piece.headline,
        kicker=piece.kicker,
        byline=piece.byline,
        body_md=body,
        in_one_sentence=str(response.get("in_one_sentence") or "").strip()[:260],
        image_url=str(response.get("image_url") or "").strip(),
        image_caption=str(response.get("image_caption") or "").strip()[:240],
        image_credit=str(response.get("image_credit") or "").strip()[:160],
        sources=sources[:12],
        related_slugs=related,
        failed=False,
    )


def _fallback_article(piece: PlannedPiece, triaged: list[TriagedCandidate] | TriagedCandidate | None) -> FullArticle:
    """Used when the writer is unavailable or fails. We surface the
    triage facts so the page still says something concrete.

    `triaged` accepts either:
      • the full triage list (we index by piece.triage_index), or
      • the per-piece merged source list (we use the first entry as
        the primary), or
      • a single TriagedCandidate."""
    t = None
    if isinstance(triaged, list) and triaged:
        # Try index lookup first (full triage list); fall back to
        # treating the list as the per-piece merged sources.
        if 0 <= piece.triage_index < len(triaged):
            t = triaged[piece.triage_index]
        else:
            t = triaged[0]
    elif isinstance(triaged, TriagedCandidate):
        t = triaged

    parts = [piece.one_line] if piece.one_line else []
    if t:
        if t.why_it_matters:
            parts.append(t.why_it_matters)
        if t.key_facts:
            parts.append("Key facts:\n" + "\n".join(f"- {f}" for f in t.key_facts))

    body = "\n\n".join(parts) or "(no article body available)"

    return FullArticle(
        slug=piece.slug,
        headline=piece.headline,
        kicker=piece.kicker,
        byline=piece.byline,
        body_md=body,
        in_one_sentence=piece.one_line[:260],
        image_url="",
        image_caption="",
        image_credit="",
        sources=[],
        related_slugs=[],
        failed=True,
    )


def articles_to_json(articles: dict[str, FullArticle]) -> dict[str, dict]:
    return {slug: asdict(a) for slug, a in articles.items()}


def articles_from_json(data: dict[str, dict]) -> dict[str, FullArticle]:
    out: dict[str, FullArticle] = {}
    for slug, a in data.items():
        if not isinstance(a, dict):
            continue
        sources_raw = a.get("sources") or []
        sources: list[dict] = []
        for s in sources_raw:
            if isinstance(s, dict):
                sources.append({"title": str(s.get("title", "")), "url": str(s.get("url", ""))})
        out[slug] = FullArticle(
            slug=str(a.get("slug", slug)),
            headline=str(a.get("headline", "")),
            kicker=str(a.get("kicker", "")),
            byline=str(a.get("byline", "Bot Tribune Staff")),
            body_md=str(a.get("body_md", "")),
            in_one_sentence=str(a.get("in_one_sentence", "")),
            image_url=str(a.get("image_url", "")),
            image_caption=str(a.get("image_caption", "")),
            image_credit=str(a.get("image_credit", "")),
            sources=sources,
            related_slugs=list(a.get("related_slugs") or []),
            failed=bool(a.get("failed", False)),
        )
    return out
