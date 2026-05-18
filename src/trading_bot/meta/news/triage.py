"""Phase 2B — Triage agent.

Per-candidate Haiku call, max 6 in parallel via ThreadPoolExecutor. Each
call scores a single candidate 1-10, writes the one-line angle that
makes the story interesting, lists 3-5 key facts, and a why-it-matters
sentence the publisher can lean on.

Per-candidate isolation matters: a noisy or controversial candidate
shouldn't poison the score of a neighbouring story. Batching would be
cheaper but risks exactly that — discovery is already noisy enough.

Output: list of TriagedCandidate, ordered by score descending. The
publisher decides which survive; triage just delivers the ranked
roster + the writing hooks each story needs.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date

from trading_bot.llm.claude_code import ClaudeCodeError, run_claude_for_json
from trading_bot.meta.news.discovery import Candidate

log = logging.getLogger(__name__)


_MAX_PARALLEL = 6
_TRIAGE_TIMEOUT = 180  # per-candidate Haiku call


@dataclass
class TriagedCandidate:
    """A discovery Candidate enriched with the triage agent's verdict.
    The publisher reads these directly to pick the lead, sections, and
    which stories survive."""
    # Carried over from discovery
    title: str
    one_line: str
    suggested_section: str
    importance_hint: int
    source_hints: list[str]
    # Added by triage
    score: int                          # 1-10 — triage's own verdict
    angle: str                          # the hook: what's the actual story?
    key_facts: list[str]                # 3-5 concrete facts a writer can lean on
    why_it_matters: str                 # 1-2 sentences on significance
    suggested_section_final: str        # may override discovery's hint
    # Failure marker
    failed: bool = False                # true if Haiku call errored — keep at score=importance_hint


def triage_candidates(
    candidates: list[Candidate],
    today: date,
) -> list[TriagedCandidate]:
    """Run triage in parallel. Returns the full set, sorted by score
    descending. Publisher applies its own cut."""
    if not candidates:
        return []
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        log.error("CLAUDE_CODE_OAUTH_TOKEN not set — triage cannot run, returning candidates as-is")
        return [_fallback_triaged(c) for c in candidates]

    log.info("Triage: scoring %d candidates (max %d in parallel)", len(candidates), _MAX_PARALLEL)
    results: list[TriagedCandidate] = []
    with ThreadPoolExecutor(max_workers=_MAX_PARALLEL) as pool:
        futures = {pool.submit(_triage_one, c, today): c for c in candidates}
        for fut in as_completed(futures):
            cand = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                log.warning("Triage failed for %r: %s — using fallback", cand.title[:60], e)
                results.append(_fallback_triaged(cand))

    results.sort(key=lambda t: t.score, reverse=True)
    n_failed = sum(1 for r in results if r.failed)
    log.info("Triage complete: %d scored, %d failed → fallback", len(results) - n_failed, n_failed)
    return results


def _triage_one(cand: Candidate, today: date) -> TriagedCandidate:
    prompt = _build_prompt(cand, today)
    try:
        response = run_claude_for_json(prompt, model="haiku", timeout_seconds=_TRIAGE_TIMEOUT)
    except ClaudeCodeError as e:
        log.warning("Triage Haiku call failed for %r: %s", cand.title[:60], e)
        return _fallback_triaged(cand)
    return _parse_triage(cand, response)


def _build_prompt(cand: Candidate, today: date) -> str:
    sources_str = ", ".join(cand.source_hints) if cand.source_hints else "(none provided)"
    return f"""You are a triage editor for The Bot Tribune, an algorithmic
trading bot's daily newspaper. Today is {today.isoformat()}.

A discovery editor surfaced the candidate story below. Score it,
sharpen the angle, and pull out the facts a writer would need.

## The candidate

- **Headline:** {cand.title}
- **One-line:** {cand.one_line}
- **Suggested section:** {cand.suggested_section}
- **Discovery's importance hint:** {cand.importance_hint}/10
- **Sources:** {sources_str}

## What I need from you

1. **score** (integer 1-10) — your honest read of how worth-covering this
   is *today*. Anchor: 10 = front-page lead (huge, surprising, with broad
   implications). 7-8 = strong section piece. 4-6 = a moderate brief.
   1-3 = barely worth a line. Disagree with the discovery hint freely.
2. **angle** — one sentence on what the *story* is, not just what
   happened. "Fed cut rates" is not a story; "Fed cut rates *despite*
   the hottest CPI in a year" is.
3. **key_facts** — 3 to 5 concrete, verifiable facts a writer can hang
   the piece on. Numbers, names, dates, quotes. No editorialising.
4. **why_it_matters** — 1-2 sentences on the significance. Who cares,
   and what does this change?
5. **suggested_section_final** — confirm the discovery suggestion, or
   override it. Valid: Markets, World, Tech & science, Climate, Health,
   Sport, Culture, Beyond the tape.

## Quality bar

- Be skeptical. A breathless headline with no facts under it scores
  low even if the topic is hot.
- Reward stories with implications beyond their immediate subject.
- Punish duplication: if the angle is "yet another rate-cut take", say so.
- Don't pad key_facts to hit 5. Three solid facts beats five vague ones.

## Required output

Return JSON only:

```json
{{
  "score": <integer 1-10>,
  "angle": "<one-sentence story hook>",
  "key_facts": ["<fact>", "<fact>", "<fact>"],
  "why_it_matters": "<1-2 sentence significance>",
  "suggested_section_final": "<section name>"
}}
```
"""


def _parse_triage(cand: Candidate, response: dict | list) -> TriagedCandidate:
    """Build a TriagedCandidate from the LLM response. Tolerant — if a
    field is missing we substitute discovery's hint."""
    if isinstance(response, list) and response:
        response = response[0] if isinstance(response[0], dict) else {}
    if not isinstance(response, dict):
        return _fallback_triaged(cand)

    try:
        score = int(response.get("score", cand.importance_hint))
    except (TypeError, ValueError):
        score = cand.importance_hint
    score = max(1, min(10, score))

    facts_raw = response.get("key_facts") or []
    if isinstance(facts_raw, str):
        facts_raw = [facts_raw]
    facts = [str(f).strip() for f in facts_raw if str(f).strip()][:5]

    section_final = str(response.get("suggested_section_final") or cand.suggested_section).strip()

    return TriagedCandidate(
        title=cand.title,
        one_line=cand.one_line,
        suggested_section=cand.suggested_section,
        importance_hint=cand.importance_hint,
        source_hints=list(cand.source_hints),
        score=score,
        angle=str(response.get("angle") or cand.one_line).strip()[:320],
        key_facts=facts,
        why_it_matters=str(response.get("why_it_matters") or "").strip()[:400],
        suggested_section_final=section_final,
        failed=False,
    )


def _fallback_triaged(cand: Candidate) -> TriagedCandidate:
    """When Haiku fails for a candidate, keep it in the pool with
    discovery's hint as the score. The publisher can still use it; we
    just don't have the enriched fields."""
    return TriagedCandidate(
        title=cand.title,
        one_line=cand.one_line,
        suggested_section=cand.suggested_section,
        importance_hint=cand.importance_hint,
        source_hints=list(cand.source_hints),
        score=cand.importance_hint,
        angle=cand.one_line,
        key_facts=[],
        why_it_matters="",
        suggested_section_final=cand.suggested_section,
        failed=True,
    )


# Round-trip helpers for the pipeline debug dump
def triaged_to_json(triaged: list[TriagedCandidate]) -> list[dict]:
    return [asdict(t) for t in triaged]


def triaged_from_json(items: list[dict]) -> list[TriagedCandidate]:
    out: list[TriagedCandidate] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("title"):
            continue
        out.append(TriagedCandidate(
            title=str(item.get("title", "")),
            one_line=str(item.get("one_line", "")),
            suggested_section=str(item.get("suggested_section", "Beyond the tape")),
            importance_hint=int(item.get("importance_hint", 5)),
            source_hints=list(item.get("source_hints") or []),
            score=int(item.get("score", item.get("importance_hint", 5))),
            angle=str(item.get("angle", "")),
            key_facts=list(item.get("key_facts") or []),
            why_it_matters=str(item.get("why_it_matters", "")),
            suggested_section_final=str(item.get("suggested_section_final", item.get("suggested_section", ""))),
            failed=bool(item.get("failed", False)),
        ))
    return out
