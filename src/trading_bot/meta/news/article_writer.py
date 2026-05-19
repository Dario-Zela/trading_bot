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
        futures = {
            pool.submit(_write_one, p, triaged[p.triage_index], today, related_pool): p
            for p in plan.pieces
            if 0 <= p.triage_index < len(triaged)
        }
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
    triaged: TriagedCandidate,
    today: date,
    related_pool: list[tuple[str, str, str]],
) -> FullArticle:
    prompt = _build_prompt(piece, triaged, today, related_pool)
    try:
        response = run_claude_for_json(
            prompt,
            model="sonnet",
            timeout_seconds=_ARTICLE_TIMEOUT,
            extra_args=_WRITER_TOOLS,
        )
    except ClaudeCodeError as e:
        log.warning("Article Sonnet failed for %r: %s", piece.slug, e)
        return _fallback_article(piece, [triaged])
    return _parse_article(piece, response)


def _build_prompt(
    piece: PlannedPiece,
    triaged: TriagedCandidate,
    today: date,
    related_pool: list[tuple[str, str, str]],
) -> str:
    persona = BYLINES.get(piece.byline, "Staff writer.")
    facts_block = "\n".join(f"  - {f}" for f in triaged.key_facts) if triaged.key_facts else "  (none — work from the angle and what you find via web search)"
    sources_block = "\n".join(f"  - {s}" for s in triaged.source_hints) if triaged.source_hints else "  (none provided — find your own)"

    # Cross-link pool: pieces in this edition the writer might reference
    related_block = "\n".join(
        f"  - {slug}: {headline} ({section})"
        for slug, headline, section in related_pool
        if slug != piece.slug
    ) or "  (none)"

    tier_guidance = {
        "lead": "FRONT PAGE LEAD. The defining piece of the day — go long if the story warrants, but don't pad. Aim for the depth a serious reader would want; a tight 600 words can beat a sprawling 2000.",
        "feature": "Feature piece. Write to the length the story deserves — anywhere from 400 to 1500+ words. Don't artificially extend.",
        "brief": "This piece is tier=brief — but the full article still gets the writer's attention. 300-500 words is a reasonable target; longer if there's substance.",
    }.get(piece.tier, "Write to the length the story deserves.")

    return f"""You are {piece.byline} writing the full article for The Bot
Tribune on {today.isoformat()}.
Your beat persona: {persona}

## The piece

- **Headline:** {piece.headline}
- **Kicker:** {piece.kicker}
- **Section:** {piece.section}
- **Tier:** {piece.tier} — {tier_guidance}
- **Angle:** {triaged.angle}
- **One-line:** {triaged.one_line}
- **Why it matters:** {triaged.why_it_matters}

## Key facts (from triage)

{facts_block}

## Source hints

{sources_block}

## Other pieces in this edition (for cross-linking via related_slugs)

{related_block}

## You have these tools available

- **WebSearch** — use to verify facts, find quotes, check named actors,
  confirm dates, and locate a fitting image.
- **WebFetch** — use to load specific URLs (a candidate image URL to
  confirm it returns a real image, a source article to read it
  properly, etc.).

You SHOULD use web search aggressively before writing. The triage
facts are a starting point, not the final word — verify them, expand
them, find quotes and concrete numbers, and reach for sources beyond
the hints. If something doesn't check out, drop it.

## Hero image — finding one

Use WebSearch to find a strong, topical image. Prefer:
- **Wikipedia / Wikimedia Commons** images (stable hot-link URLs,
  unambiguous licensing, e.g. `https://upload.wikimedia.org/...`)
- Major-publication photo URLs if you find them on a search result
  (Reuters, AP, FT, etc.) — hot-linking works in practice
- Stock-photo URLs (Unsplash, etc.) when the topic is generic

Avoid:
- Image search RESULT pages (these don't render; you need the actual
  image URL ending in .jpg / .png / .webp)
- Images behind login walls (Twitter image CDN often fails)
- Generic stock photos when a specific topical image exists

Use WebFetch to verify the URL returns an image. If you can't find a
good image, leave `image_url` empty rather than ship a bad one.

## Writing rules

1. **Open with the situation, in plain English.** A reader new to this
   beat should be able to follow the first paragraph.
2. **Then earn the technical depth.** Numbers, named actors, quotes,
   regulatory specifics — they belong in the body, not the lede.
3. **Show your reasoning.** When you make an interpretive claim
   ("this suggests..."), point at the evidence that supports it.
4. **Concrete over abstract.** "The bill passed 52-48" beats "the bill
   passed narrowly."
5. **No clichés, no breathless framing, no clickbait.** "Stunning",
   "watershed", "everything we knew is wrong" — cut them.
6. **Don't address the reader.** No "we'll see", no "your portfolio",
   no "as we noted yesterday."
7. **Variety in sentence length.** Real newspapers don't read like
   marketing copy.
8. **End with what's next.** A natural forward-look — what to watch,
   what's at stake, when the next data point lands.
9. **No word cap.** Write as long as the topic warrants. Don't pad;
   don't truncate. A short piece on a small story is correct.

## "In one sentence" callout

A single sentence (≤180 chars) — the TL;DR for the impatient reader.
Goes in a callout box near the top of the article. Not the same as
the kicker or headline — it's the substance compressed to one line.

## Required output

Return JSON only:

```json
{{
  "body_md": "<the full article as markdown. Use ## for sub-headings if useful, blockquotes for quotes, em-dashes — like this — not '--'. NO headline or byline in the body.>",
  "in_one_sentence": "<one sentence TL;DR, ≤180 chars>",
  "image_url": "<direct image URL, or empty string>",
  "image_caption": "<one-line caption describing what the image shows>",
  "image_credit": "<photographer / outlet / Wikipedia author — whoever should be credited>",
  "sources": [
    {{"title": "<source title or outlet>", "url": "<source url>"}},
    ...
  ],
  "related_slugs": ["<slug from the cross-link pool above that's actually relevant — or empty list>"]
}}
```

Do not include any text outside the JSON.
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
    triage facts so the page still says something concrete."""
    t = None
    if isinstance(triaged, list) and triaged:
        if 0 <= piece.triage_index < len(triaged):
            t = triaged[piece.triage_index]
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
