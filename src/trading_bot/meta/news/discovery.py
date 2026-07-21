"""Phase 2A — Discovery agent.

Single Sonnet call with WebSearch enabled. Its job: find ~30-50 candidate
stories from the past 24 hours, across every topic the Tribune covers,
and emit them as structured JSON for the triage stage to score.

The Tribune covers Markets / World / Tech & science / Climate / Health /
Sport / Culture / Beyond the tape — not just finance. Discovery's role
is to cast the wide net; triage decides what gets through.

Output: list of Candidate records. Each downstream stage works from this.

Implementation notes
====================
- Seed the agent with Alpaca News + yfinance broad-market headlines so
  the markets baseline is guaranteed even if WebSearch underperforms.
- The agent is explicitly asked to use web search to expand beyond the
  seed, especially for non-markets coverage.
- On LLM failure we fall back to the seed alone — markets-only coverage
  rather than nothing. Triage handles whichever shape we deliver.
"""
from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import date

import requests

from trading_bot.llm.claude_code import ClaudeCodeError, run_claude_for_json
from trading_bot.tools.news import get_recent_news


log = logging.getLogger(__name__)


# Broad-market tickers we use to seed Alpaca News with a guaranteed
# markets-side starting set. Discovery is asked to extend beyond these
# via web search. The user is UK-based, so the UK/Europe seed is the
# fuller list and the US seed is the smaller (US news is easy to find
# via web search; UK/Europe news is what discovery tends to miss).
_BROAD_TICKERS = ("SPY", "QQQ", "IWM")  # SPY+QQQ+IWM cover broad US; DIA dropped
_UK_PROXIES = (
    # FTSE 100 majors across sectors
    "BP.L", "SHEL.L", "HSBA.L", "BARC.L", "LLOY.L", "NWG.L",
    "AZN.L", "GSK.L", "ULVR.L", "VOD.L", "BT-A.L", "RIO.L",
    "GLEN.L", "AAL.L", "TSCO.L", "SBRY.L", "NG.L", "RR.L",
    # FTSE 250 / wider UK proxies
    "BABA.L", "MNG.L", "AVV.L", "EZJ.L",
)
_EU_PROXIES = (
    # Major European names (cross-listed or pan-EU exposure)
    "SAP", "ASML", "MC.PA", "OR.PA", "AIR.PA", "SIE.DE", "NESN.SW",
)

# Non-market RSS feeds — fill the UK-politics / world / tech / sport
# gap that the ticker-based seed structurally can't reach. Fetched
# directly (not via WebSearch) so they arrive regardless of the LLM's
# search behaviour.
_RSS_FEEDS: tuple[tuple[str, str], ...] = (
    ("BBC UK", "http://feeds.bbci.co.uk/news/uk/rss.xml"),
    ("BBC Politics", "http://feeds.bbci.co.uk/news/politics/rss.xml"),
    ("BBC World", "http://feeds.bbci.co.uk/news/world/rss.xml"),
    ("BBC Business", "http://feeds.bbci.co.uk/news/business/rss.xml"),
    ("BBC Technology", "http://feeds.bbci.co.uk/news/technology/rss.xml"),
    ("BBC Sport", "http://feeds.bbci.co.uk/sport/rss.xml"),
    ("Guardian Politics", "https://www.theguardian.com/politics/rss"),
    ("Guardian UK", "https://www.theguardian.com/uk-news/rss"),
)

# Cap per-feed to keep the seed block from ballooning; the LLM sees a
# balanced non-market starter set rather than 80 BBC-World bullets.
_RSS_ITEMS_PER_FEED = 6

# Target count for the discovery output. The agent is told to aim for
# this; triage handles whatever it gets.
_TARGET_CANDIDATES = 40

# If more than this share of candidates tag as Markets, we retry once
# with a stronger diversity nudge. The seed and prompt already push for
# 40-50% UK/EU + 30-40% US + 10-20% RoW across all sections, so a
# >70% Markets result means the LLM anchored on the tickers.
_MARKETS_BIAS_RETRY_THRESHOLD = 0.70


@dataclass
class Candidate:
    """One story the discovery agent surfaced. Each field is what the
    triage agent needs to decide a score + angle."""
    title: str
    one_line: str
    suggested_section: str  # Markets / World / Tech & science / etc.
    importance_hint: int    # 1-10 — discovery's first-pass impression
    source_hints: list[str] = field(default_factory=list)


def discover_stories(today: date) -> list[Candidate]:
    """Run the discovery stage. Returns the candidates (~30-50) the rest
    of the pipeline will triage."""
    market_seed = _gather_seed_headlines()
    rss_seed = _gather_rss_headlines()
    log.info(
        "Discovery: seed = %d market + %d non-market (RSS) headlines",
        len(market_seed), len(rss_seed),
    )

    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        log.error("CLAUDE_CODE_OAUTH_TOKEN not set — discovery cannot run, returning seed only")
        return _candidates_from_seed_only(market_seed + rss_seed)

    prompt = _build_prompt(today, market_seed, rss_seed)
    try:
        response = run_claude_for_json(
            prompt,
            model="sonnet",
            extra_args=["--allowedTools", "WebSearch,WebFetch"],
        )
    except ClaudeCodeError as e:
        log.warning("Discovery LLM call failed: %s — falling back to seed-only", e)
        return _candidates_from_seed_only(market_seed + rss_seed)

    candidates = _parse_candidates(response)
    log.info("Discovery: %d candidates returned", len(candidates))

    # Markets-bias retry: if the LLM anchored on the market seed and
    # returned a >70% Markets slate, ask again with a stronger diversity
    # nudge and merge the two rounds (triage picks from the union).
    if candidates and _markets_share(candidates) > _MARKETS_BIAS_RETRY_THRESHOLD:
        markets_n = sum(1 for c in candidates if c.suggested_section == "Markets")
        log.warning(
            "Discovery: %d/%d candidates tagged Markets (%.0f%%) — retrying for section diversity",
            markets_n, len(candidates), 100 * markets_n / len(candidates),
        )
        candidates = _retry_for_diversity(today, market_seed, rss_seed, candidates)

    if len(candidates) < 5:
        # Anything below 5 is suspicious — agent probably hit a parsing or
        # search failure. Augment with seed to keep the pipeline alive.
        log.warning("Discovery returned only %d candidates — augmenting with seed", len(candidates))
        seed_cands = _candidates_from_seed_only(market_seed + rss_seed)
        existing_titles = {c.title.lower() for c in candidates}
        for sc in seed_cands:
            if sc.title.lower() not in existing_titles:
                candidates.append(sc)
                if len(candidates) >= 20:
                    break
    return candidates


def _markets_share(candidates: list[Candidate]) -> float:
    if not candidates:
        return 0.0
    markets = sum(1 for c in candidates if c.suggested_section == "Markets")
    return markets / len(candidates)


def _retry_for_diversity(
    today: date,
    market_seed: list[dict],
    rss_seed: list[dict],
    first_round: list[Candidate],
) -> list[Candidate]:
    """Second-pass discovery call when round one skewed too Markets-heavy.
    Returns the union of both rounds, deduped by title."""
    retry_prompt = _build_retry_prompt(today, market_seed, rss_seed, first_round)
    try:
        response = run_claude_for_json(
            retry_prompt,
            model="sonnet",
            extra_args=["--allowedTools", "WebSearch,WebFetch"],
        )
    except ClaudeCodeError as e:
        log.warning("Discovery retry failed: %s — keeping round-one candidates", e)
        return first_round

    extra = _parse_candidates(response)
    log.info("Discovery retry: %d additional candidates", len(extra))
    seen = {c.title.lower() for c in first_round}
    merged = list(first_round)
    for c in extra:
        key = c.title.lower()
        if key not in seen:
            merged.append(c)
            seen.add(key)
    return merged


def _gather_seed_headlines() -> list[dict]:
    """Pull broad-market headlines from Alpaca News + UK proxies. Same
    function the legacy daily-news brief uses — we reuse it so the seed
    is consistent."""
    seen_urls: set[str] = set()
    out: list[dict] = []
    try:
        # UK first in the list — when limit caps the count, UK news survives.
        by_ticker = get_recent_news(
            list(_UK_PROXIES) + list(_EU_PROXIES) + list(_BROAD_TICKERS),
            days=1, limit=40,
        )
    except Exception as e:
        log.warning("Seed news fetch failed: %s", e)
        return out
    for items in by_ticker.values():
        for item in items:
            if item.url and item.url in seen_urls:
                continue
            if item.url:
                seen_urls.add(item.url)
            out.append({
                "title": item.headline,
                "summary": (item.summary or "")[:220],
                "source": item.source,
                "url": item.url,
                "timestamp": item.timestamp,
                "tickers": list(item.tickers),
            })
    out.sort(key=lambda h: h.get("timestamp", ""), reverse=True)
    return out[:30]


def _gather_rss_headlines() -> list[dict]:
    """Pull top headlines from BBC + Guardian RSS feeds.

    The ticker-based seed is Markets-only by construction; this fills the
    UK-politics / world / tech / sport gap. Fetched directly (no LLM tool
    dependency) so it arrives even if WebSearch degrades. Each feed is
    capped at _RSS_ITEMS_PER_FEED to keep the prompt balanced."""
    out: list[dict] = []
    seen: set[str] = set()
    for source, url in _RSS_FEEDS:
        try:
            r = requests.get(url, timeout=8, headers={"User-Agent": "trading-bot-discovery/1.0"})
            r.raise_for_status()
            root = ET.fromstring(r.content)
        except (requests.RequestException, ET.ParseError) as e:
            log.warning("RSS fetch failed for %s: %s", source, e)
            continue
        items = root.findall(".//item")[:_RSS_ITEMS_PER_FEED]
        for it in items:
            title = (it.findtext("title") or "").strip()
            link = (it.findtext("link") or "").strip()
            if not title or link in seen:
                continue
            seen.add(link)
            desc = (it.findtext("description") or "").strip()
            # Strip trivial HTML tags that Guardian embeds in <description>
            desc = _strip_html_tags(desc)
            out.append({
                "title": title,
                "summary": desc[:220],
                "source": source,
                "url": link,
                "timestamp": (it.findtext("pubDate") or "").strip(),
                "tickers": [],
            })
    return out


def _strip_html_tags(text: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", text).strip()


def _format_seed_block(market_seed: list[dict], rss_seed: list[dict]) -> str:
    """Build the seed section of the discovery prompt. Markets and
    non-market bullets are listed separately so the LLM can see that
    UK-politics / world / sport / tech starters have already been
    surfaced and it should extend both tracks, not just markets."""
    parts: list[str] = []
    if market_seed:
        bullets = "\n".join(
            f"- [{(h.get('source') or 'src')[:25]}] {h.get('title', '')}"
            for h in market_seed
        )
        parts.append(
            "\n## Markets seed headlines (ticker-based; starting point, "
            "not exhaustive — extend well beyond these)\n\n" + bullets
        )
    if rss_seed:
        bullets = "\n".join(
            f"- [{(h.get('source') or 'src')[:25]}] {h.get('title', '')}"
            for h in rss_seed
        )
        parts.append(
            "\n## Non-market seed headlines (BBC / Guardian RSS; UK politics, "
            "world, sport, tech — verify freshness and extend with WebSearch "
            "for developments in the last few hours)\n\n" + bullets
        )
    return "\n".join(parts) + ("\n" if parts else "")


def _build_prompt(today: date, market_seed: list[dict], rss_seed: list[dict]) -> str:
    seed_block = _format_seed_block(market_seed, rss_seed)

    return f"""You are the discovery editor for The Bot Tribune, a
UK-based algorithmic trading bot's daily newspaper. Today is
{today.isoformat()}. The publication's primary reader is in the UK
and trades both UK/EU and US sessions, so coverage should reflect a
UK perspective on the global day.

Your job: produce a candidate list of ~{_TARGET_CANDIDATES} newsworthy
stories from the past 24 hours, across every topic the Tribune covers.
The next stage of the pipeline (triage) will score each candidate;
your job is breadth and quality of capture, not curation.

## Geographic balance (important)

This is a UK-centred publication. Aim for roughly:

- **40-50% UK / Europe** — Westminster, Threadneedle Street, BoE,
  Brussels, ECB, FTSE 100/250, major continental names, EU
  regulation, UK / European politics and policy
- **30-40% US** — Fed, US macro, S&P/Nasdaq, major US corporates,
  Washington politics — but only the genuinely consequential
  stories. Not every Trump truth-social post.
- **10-20% rest-of-world** — China, Japan, EM, geopolitics where it
  affects markets

Within UK coverage, *prefer* UK-domestic stories that a US-only
discovery would miss: BoE speakers, gilt auctions, FTSE earnings,
UK retail data, NHS / planning / energy policy, City regulatory
news, Scottish politics where consequential.

## Coverage scope

Aim for breadth and a mix of importance levels — not everything is a 10/10.

- **Markets / Finance / Business** — central banks (BoE, ECB, Fed),
  macro data, M&A, earnings, regulatory, major deals
- **World affairs / Politics / Geopolitics** — UK and European politics
  first, then US, then global; elections, diplomacy, conflicts

  **UK political coverage is required, not optional.** Across the last
  fortnight this newspaper produced zero UK-politics candidates — that
  is a structural failure of discovery, not a quiet political week.
  Each day's candidate set MUST include UK political signal where any
  exists. Search explicitly for:
  - UK by-elections (Burnham/Makerfield, any seat in motion or
    upcoming ballot)
  - Labour leadership pressure / Starmer's standing in polling
  - Reform UK polling trends and seat threats
  - UK fiscal policy, OBR commentary, Treasury announcements
  - Gilt-market signals (auctions, yields, BoE QT path)
  - BoE speaker calendar, MPC vote splits, dissents
  - Westminster select-committee actions touching banks, energy,
    pharma, telecoms (anything that touches FTSE 100/250 constituents)
  If you genuinely can't find UK political news on a given day, say so
  in the `notes` field — but a UK-focused publication going 14 days
  in a row without UK politics is the failure mode this rule exists
  to prevent.
- **Tech & science** — AI, biotech, frontier research, major product
  launches, regulatory developments (UK/EU AI rules included)
- **Climate / Environment** — significant climate events, policy
  decisions, important scientific findings
- **Health / Medicine** — major studies, public health, drug approvals
- **Culture** — arts, music, books, film, design — but only when
  substantive
- **Sport** — major events, governance shifts, large deals (Premier
  League / rugby / cricket count more than NFL for this reader)
- **Beyond the tape** — interesting long-tail oddments worth knowing

## How to discover

Use web search aggressively. Search for:
- "top UK news today" / "{today.isoformat()} UK news" first
- Then "top US news today" / "Europe news {today.isoformat()}"
- UK / European publications: BBC, FT, Guardian, Times, Telegraph,
  Reuters UK, Le Monde, Spiegel, Handelsblatt, El País, Politico EU
- US publications: WSJ, NYT, Bloomberg, Reuters, AP, Politico
- Asia: Nikkei, SCMP, Bloomberg Asia (for the cross-asset stories
  that affect UK/US markets)
- Topic-specific queries to fill obvious gaps (e.g., "BoE rate
  decision", "Premier League results", "EU AI Act")
- Independent verification of any single-source claim where reasonable
{seed_block}

## Quality bar

- Each candidate is a story that could plausibly become its own article.
- Avoid: pure celebrity gossip, unverified rumors, clickbait, single
  inflammatory tweets, opinion pieces with no underlying news.
- Include: anything substantive — even if it's "small" news in a niche.
- A "boring" story with real implications outscores a dramatic story
  with nothing under it.

## Required output

Return JSON only, no preamble:

```json
{{
  "candidates": [
    {{
      "title": "<full headline as you would write it>",
      "one_line": "<one sentence summary capturing what happened>",
      "suggested_section": "Markets" | "World" | "Tech & science" | "Climate" | "Health" | "Sport" | "Culture" | "Beyond the tape",
      "importance_hint": <integer 1-10. 10 is front-page-worthy. 7-8 is a strong section piece. 4-6 is a moderately interesting brief. 1-3 is borderline.>,
      "source_hints": ["<publication name>", "<url if you found one>"]
    }}
  ]
}}
```

Aim for ~{_TARGET_CANDIDATES} candidates. Distribute importance hints
honestly — most days have 2-4 genuine 9-10s and a long tail of 4-7s.
"""


def _build_retry_prompt(
    today: date,
    market_seed: list[dict],
    rss_seed: list[dict],
    first_round: list[Candidate],
) -> str:
    """Retry prompt when round one skewed too Markets-heavy. Shows the
    LLM its own output so it can see the imbalance, and demands
    non-Markets sections be filled from web search."""
    seed_block = _format_seed_block(market_seed, rss_seed)
    first_round_bullets = "\n".join(
        f"- [{c.suggested_section}] {c.title}" for c in first_round[:30]
    )
    markets_n = sum(1 for c in first_round if c.suggested_section == "Markets")
    total = len(first_round)
    pct = round(100 * markets_n / total) if total else 0

    return f"""You are the discovery editor for The Bot Tribune (UK-based
algorithmic trading bot's daily newspaper). Today is {today.isoformat()}.

Your previous discovery attempt was too Markets-heavy: {markets_n} of
{total} candidates ({pct}%) tagged as `Markets`. The Tribune is a
general-interest publication with a UK-first lens, and its readers
expect substantive non-Markets coverage every day.

## Round-one output (for your reference — do NOT re-submit these)

{first_round_bullets}

## Your task now

Return an ADDITIONAL ~20-30 candidates that specifically fill the gaps
above. Do NOT emit stories already listed. Target the following
sections:

- **World** — UK politics (PMQs, byelections, party leadership,
  Reform UK, Labour internal, Conservative rebuild, Scottish/Welsh
  government), European politics, US politics beyond markets impact,
  major diplomacy / conflicts
- **Tech & science** — AI research and product news, biotech, space,
  frontier science, regulatory (UK/EU AI Act, DMA), major consumer
  tech
- **Sport** — Premier League / Championship / cricket / rugby / F1 /
  tennis / Olympics / major football transfers or governance
- **Culture** — arts, books, film, TV, music with substance (releases,
  awards, industry moves) — no red-carpet gossip
- **Health / Medicine** — NHS news, drug approvals, major studies,
  public-health developments
- **Climate** — significant climate events, policy decisions, notable
  research
- **Beyond the tape** — long-tail oddments that are genuinely
  interesting

**Explicit UK-politics discovery is required.** Search for:
- "UK politics news {today.isoformat()}"
- "PMQs today", "House of Commons today"
- "Andy Burnham", "Keir Starmer", "Nigel Farage", "Reform UK",
  "Labour leadership", "byelection"
- BBC Politics, Guardian Politics, Politico UK, Sky News Politics

Use WebSearch aggressively — the non-market RSS seed above is a
starting point, extend with searches for developments in the last few
hours.

## Constraints

- Do NOT emit any story listed under "Round-one output" above.
- At least 60% of your response MUST be non-Markets. If you can't hit
  that from the last 24 hours, extend the window to 36 hours rather
  than filling with Markets padding.
- Return valid JSON in the same schema as round one.
{seed_block}

## Required output

```json
{{
  "candidates": [
    {{
      "title": "<full headline>",
      "one_line": "<one sentence>",
      "suggested_section": "Markets" | "World" | "Tech & science" | "Climate" | "Health" | "Sport" | "Culture" | "Beyond the tape",
      "importance_hint": <1-10>,
      "source_hints": ["<publication>", "<url>"]
    }}
  ]
}}
```
"""


def _parse_candidates(response: dict | list) -> list[Candidate]:
    """Extract candidate records from the LLM response. Tolerant of
    minor shape drift — discovery is the noisiest stage."""
    if isinstance(response, list):
        raw = response
    elif isinstance(response, dict):
        raw = response.get("candidates") or response.get("stories") or []
    else:
        return []

    out: list[Candidate] = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        title = str(c.get("title", "")).strip()
        if not title:
            continue
        try:
            importance = int(c.get("importance_hint", 5))
        except (TypeError, ValueError):
            importance = 5
        importance = max(1, min(10, importance))
        sources_raw = c.get("source_hints") or c.get("sources") or []
        if isinstance(sources_raw, str):
            sources_raw = [sources_raw]
        sources = [str(s).strip() for s in sources_raw if str(s).strip()]
        out.append(Candidate(
            title=title,
            one_line=str(c.get("one_line") or c.get("summary") or "").strip()[:320],
            suggested_section=str(c.get("suggested_section") or "Beyond the tape").strip(),
            importance_hint=importance,
            source_hints=sources,
        ))
    return out


def _candidates_from_seed_only(seed: list[dict]) -> list[Candidate]:
    """Fallback when the LLM call fails entirely. Better to have
    markets-only coverage than no edition at all."""
    out: list[Candidate] = []
    for h in seed[:25]:
        title = h.get("title", "")
        if not title:
            continue
        sources: list[str] = []
        if h.get("source"):
            sources.append(h["source"])
        if h.get("url"):
            sources.append(h["url"])
        out.append(Candidate(
            title=title,
            one_line=(h.get("summary") or title)[:240],
            suggested_section="Markets",
            importance_hint=5,
            source_hints=sources,
        ))
    return out


# Public helper for downstream stages — converts candidates back to a
# JSON-ready form. The pipeline pickles intermediate stage outputs to
# state/daily_news/YYYY-MM-DD.pipeline.json for debugging.
def candidates_to_json(candidates: list[Candidate]) -> list[dict]:
    return [asdict(c) for c in candidates]


def candidates_from_json(items: list[dict]) -> list[Candidate]:
    return _parse_candidates({"candidates": items})
