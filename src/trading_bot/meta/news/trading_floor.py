"""Phase 2G — Trading floor section.

Reads `state/ledger.jsonl` for the most recent trading day, aggregates
P&L per strategy, and produces three prose pieces:

- **The winner** — the strategy that did best
- **The loser**  — the strategy that did worst
- **Quieter movers** — everyone else worth mentioning

We deliberately write in prose, not in numbers. The dashboard already
has the numbers. The newspaper's job is to say what the day *meant*
for each strategy — was it a good idea well-executed, a bad idea
saved by luck, a known-bad day, an early signal of regime change?

Output shape mirrors `Brief` so the renderer can mix these in with
the other section briefs.
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from trading_bot.llm.claude_code import ClaudeCodeError, run_claude_for_json
from trading_bot.state.paths import ledger_path

log = logging.getLogger(__name__)

_MAX_PARALLEL = 3       # winner / loser / quieter — three pieces
_TF_TIMEOUT = 180


@dataclass
class FloorBrief:
    """One prose piece in the Trading floor section."""
    slug: str
    headline: str
    kicker: str                       # always "TRADING FLOOR · WINNER/LOSER/QUIETER"
    byline: str = "Bot"               # the bot speaks for the trading floor
    body_md: str = ""
    failed: bool = False


@dataclass
class _StrategyDay:
    """Aggregate stats for one strategy on the floor's reporting day."""
    strategy_id: str
    region: str
    tier: str
    n_closed: int
    pnl_gbp_total: float
    pnl_pct_avg: float                # mean of trade-level pct
    n_winners: int
    n_losers: int
    tickers: list[str]
    notable_winners: list[dict]       # top 3 by pnl_pct
    notable_losers: list[dict]        # bottom 3 by pnl_pct
    sample_thesis: str                # any one trade's thesis, for narrative


def write_floor_briefs(today: date) -> list[FloorBrief]:
    """Produce the trading floor section for {today}. Looks at trades
    that closed yesterday (or the most recent day with closes).

    Returns 0-3 briefs depending on what the day had. If no strategies
    closed trades, returns an empty list — the section is dropped.
    """
    day_to_use, by_strategy = _aggregate_recent_closes(today)
    if not by_strategy:
        log.info("Trading floor: no closes in recent history, skipping section")
        return []

    log.info("Trading floor: reporting day %s, %d strategies", day_to_use, len(by_strategy))

    # Rank by total P&L; pick winner / loser / quieter pool
    ranked = sorted(by_strategy.values(), key=lambda s: s.pnl_gbp_total, reverse=True)
    winner = ranked[0] if ranked else None
    loser = ranked[-1] if len(ranked) >= 2 and ranked[-1].pnl_gbp_total < 0 else None
    if loser is winner:
        loser = None
    quieter_pool = [s for s in ranked if s is not winner and s is not loser]

    pieces_spec: list[tuple[str, str, _StrategyDay | list[_StrategyDay] | None]] = []
    if winner:
        pieces_spec.append(("winner", "WINNER", winner))
    if loser:
        pieces_spec.append(("loser", "LOSER", loser))
    if quieter_pool:
        pieces_spec.append(("quieter", "QUIETER MOVERS", quieter_pool))

    # Missed-movers piece — sits alongside winner/loser/quieter. Reads
    # state/missed_movers/{date}.{region}.json which the exit pipeline
    # writes after every region's exit pass.
    missed_payload = _gather_missed_movers(today)
    if missed_payload:
        pieces_spec.append(("missed", "WHAT WE MISSED", missed_payload))

    if not pieces_spec:
        return []

    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        log.error("CLAUDE_CODE_OAUTH_TOKEN not set — using fallback floor briefs")
        return [_fallback_floor_brief(kind, kicker_tag, data, day_to_use) for kind, kicker_tag, data in pieces_spec]

    briefs: list[FloorBrief] = []
    with ThreadPoolExecutor(max_workers=_MAX_PARALLEL) as pool:
        futures = {
            pool.submit(_write_floor_piece, kind, kicker_tag, data, day_to_use): (kind, kicker_tag, data)
            for kind, kicker_tag, data in pieces_spec
        }
        for fut in as_completed(futures):
            kind, kicker_tag, data = futures[fut]
            try:
                briefs.append(fut.result())
            except Exception as e:
                log.warning("Trading floor %s piece failed: %s — using fallback", kind, e)
                briefs.append(_fallback_floor_brief(kind, kicker_tag, data, day_to_use))

    # Stable ordering: winner, loser, quieter, missed
    order = {"winner": 0, "loser": 1, "quieter": 2, "missed": 3}
    briefs.sort(key=lambda b: order.get(b.slug.split("-")[-1], 99))
    return briefs


def _gather_missed_movers(today: date) -> dict | None:
    """Read state/missed_movers/{today}.{region}.json for every region
    and assemble a payload for the missed-movers piece. Walks back up
    to 7 days so weekend / market-closed days still surface the most
    recent analysis."""
    from trading_bot.meta.missed_movers import load_report
    from datetime import timedelta

    aggregated: dict[str, dict] = {}
    use_date = today
    for offset in range(0, 7):
        d = today - timedelta(days=offset)
        for region in ("us", "uk-eu"):
            rep = load_report(d, region)
            if not rep or region in aggregated:
                continue
            aggregated[region] = rep
        if aggregated:
            use_date = d
            break

    if not aggregated:
        return None
    return {
        "as_of": use_date.isoformat(),
        "by_region": aggregated,
    }


def _aggregate_recent_closes(today: date) -> tuple[str | None, dict[str, _StrategyDay]]:
    """Walk back from today up to 7 days, return the first day with
    closed trades + a strategy_id-keyed aggregate."""
    path = ledger_path()
    if not path.exists():
        return None, {}

    by_day: dict[str, list[dict]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            exit_date = rec.get("exit_date")
            if not exit_date:
                continue
            by_day.setdefault(exit_date, []).append(rec)

    if not by_day:
        return None, {}

    # Most recent exit_date present (looking back up to 7 days from today)
    candidate_days = sorted(by_day.keys(), reverse=True)
    use_day = None
    for d in candidate_days:
        try:
            dt = datetime.fromisoformat(d).date()
        except ValueError:
            continue
        if (today - dt).days <= 7:
            use_day = d
            break

    if not use_day:
        return None, {}

    trades = by_day[use_day]
    by_strategy: dict[str, _StrategyDay] = {}
    by_strategy_trades: dict[str, list[dict]] = {}
    for t in trades:
        sid = t.get("strategy_id") or "(unknown)"
        by_strategy_trades.setdefault(sid, []).append(t)

    for sid, ts in by_strategy_trades.items():
        n = len(ts)
        pnl_total = sum((t.get("pnl_gbp") or 0.0) for t in ts)
        pcts = [t.get("pnl_pct") for t in ts if isinstance(t.get("pnl_pct"), (int, float))]
        pnl_pct_avg = (sum(pcts) / len(pcts)) if pcts else 0.0
        winners = sum(1 for t in ts if (t.get("pnl_gbp") or 0.0) > 0)
        losers = sum(1 for t in ts if (t.get("pnl_gbp") or 0.0) < 0)
        tickers = sorted({t.get("ticker") or "" for t in ts if t.get("ticker")})[:12]
        ts_sorted = sorted(
            (t for t in ts if isinstance(t.get("pnl_pct"), (int, float))),
            key=lambda t: t.get("pnl_pct", 0.0),
            reverse=True,
        )
        notable_winners = [_compact_trade(t) for t in ts_sorted[:3]]
        notable_losers  = [_compact_trade(t) for t in ts_sorted[-3:][::-1]] if len(ts_sorted) >= 2 else []
        sample = next((t.get("thesis") or "" for t in ts if t.get("thesis")), "")
        first = ts[0]
        by_strategy[sid] = _StrategyDay(
            strategy_id=sid,
            region=first.get("region", "?"),
            tier=first.get("tier", "?"),
            n_closed=n,
            pnl_gbp_total=pnl_total,
            pnl_pct_avg=pnl_pct_avg,
            n_winners=winners,
            n_losers=losers,
            tickers=tickers,
            notable_winners=notable_winners,
            notable_losers=notable_losers,
            sample_thesis=sample[:240],
        )

    return use_day, by_strategy


def _compact_trade(t: dict) -> dict:
    # Long-only P&L can't exceed -100%; ledger sometimes has bad rows
    # (cancelled trades with non-zero recorded numbers, data glitches).
    # Clip to [-100, +500]% so the LLM doesn't get nonsensical inputs
    # that read as confident facts.
    raw_pct = float(t.get("pnl_pct") or 0.0)
    clipped_pct = max(-100.0, min(500.0, raw_pct))
    return {
        "ticker": t.get("ticker", ""),
        "pnl_gbp": float(t.get("pnl_gbp") or 0.0),
        "pnl_pct": clipped_pct,
        "exit_reason": t.get("exit_reason", ""),
        "outcome_notes": (t.get("outcome_notes") or "")[:300],
    }


def _write_floor_piece(
    kind: str,
    kicker_tag: str,
    data: _StrategyDay | list[_StrategyDay],
    day_iso: str,
) -> FloorBrief:
    prompt = _build_prompt(kind, data, day_iso)
    try:
        response = run_claude_for_json(prompt, model="haiku", timeout_seconds=_TF_TIMEOUT)
    except ClaudeCodeError as e:
        log.warning("Trading floor Haiku failed for %s: %s", kind, e)
        return _fallback_floor_brief(kind, kicker_tag, data, day_iso)

    body = ""
    headline = ""
    if isinstance(response, dict):
        body = str(response.get("body_md") or response.get("body") or "").strip()
        headline = str(response.get("headline") or "").strip()

    if not body:
        return _fallback_floor_brief(kind, kicker_tag, data, day_iso)

    if not headline:
        headline = _default_headline(kind, data)

    return FloorBrief(
        slug=f"floor-{day_iso}-{kind}",
        headline=headline,
        kicker=f"TRADING FLOOR · {kicker_tag}",
        byline="Bot",
        body_md=body,
        failed=False,
    )


def _build_prompt(kind: str, data, day_iso: str) -> str:
    if kind == "missed":
        return _prompt_missed(data, day_iso)
    if kind == "winner":
        d = data  # type: ignore[assignment]
        return _prompt_single(kind, d, day_iso, role="the trading day's best performer")
    if kind == "loser":
        d = data  # type: ignore[assignment]
        return _prompt_single(kind, d, day_iso, role="the trading day's worst performer")
    # quieter
    return _prompt_quieter(data, day_iso)  # type: ignore[arg-type]


def _prompt_single(kind: str, s: _StrategyDay, day_iso: str, *, role: str) -> str:
    winners_block = "\n".join(
        f"  - {t['ticker']}: £{t['pnl_gbp']:+,.2f} ({t['pnl_pct']:+.2f}%)  exit: {t['exit_reason']}"
        for t in s.notable_winners
    ) or "  (no clear standout winners)"
    losers_block = "\n".join(
        f"  - {t['ticker']}: £{t['pnl_gbp']:+,.2f} ({t['pnl_pct']:+.2f}%)  exit: {t['exit_reason']}"
        for t in s.notable_losers
    ) or "  (no clear standout losers)"
    notes_block = ""
    for t in s.notable_winners[:1] + s.notable_losers[:1]:
        if t.get("outcome_notes"):
            notes_block += f"\n  - {t['ticker']}: {t['outcome_notes']}"
    if not notes_block:
        notes_block = "  (no specific trade-level notes)"

    return f"""You are the trading floor reporter for The Bot Tribune.
Today is {day_iso}. You are writing about {role} from the bot's
ledger.

You write in prose, not in numbers. The reader already sees the
numbers on the dashboard — your job is the *narrative*: what kind of
day was this, was the thesis right but the timing wrong (or the
reverse), did one trade carry the day, was there an obvious lesson?

## The strategy

- **Strategy:** {s.strategy_id}
- **Region / tier:** {s.region} · {s.tier}
- **Trades closed:** {s.n_closed}  (winners: {s.n_winners} · losers: {s.n_losers})
- **Total P&L:** £{s.pnl_gbp_total:+,.2f}
- **Avg P&L per trade:** {s.pnl_pct_avg:+.2f}%
- **Names traded:** {', '.join(s.tickers) or '(none recorded)'}
- **Sample thesis:** {s.sample_thesis or '(none recorded)'}

## Best individual trades

{winners_block}

## Worst individual trades

{losers_block}

## Trade-level notes (from the bot's own outcome analysis)

{notes_block}

## Writing rules

- Lead with the *character* of the day, not the bottom line.
- Specific over generic — name names, cite the actual trade that
  shaped the result.
- Be honest. Don't congratulate luck. Don't blame the market for a
  bad call.
- 90-130 words. Conversational. Dry.
- No clichés ("knocked it out of the park", "took a hit", "tough
  day at the office") — find better phrasing.
- End with a forward-look: what does the day suggest about tomorrow?
- The byline is "Bot" — write in the bot's own voice, but with the
  detachment of a beat reporter covering itself. Avoid "I" / "me" /
  "we".

## Required output

```json
{{
  "headline": "<a short headline, ≤80 chars>",
  "body_md": "<the prose piece — markdown, no headline or byline embedded>"
}}
```
"""


def _prompt_quieter(strategies: list[_StrategyDay], day_iso: str) -> str:
    lines = []
    for s in strategies[:6]:  # cap so the prompt doesn't bloat
        lines.append(
            f"- {s.strategy_id} ({s.region}/{s.tier}): {s.n_closed} closed, "
            f"£{s.pnl_gbp_total:+,.2f} total ({s.pnl_pct_avg:+.2f}% avg). "
            f"Names: {', '.join(s.tickers[:6]) or '(n/a)'}."
        )
    body = "\n".join(lines)

    return f"""You are the trading floor reporter for The Bot Tribune.
Today is {day_iso}. You're writing the "quieter movers" piece — the
strategies that weren't the day's biggest winner or loser, but are
worth a line each.

## The quieter strategies (each gets ~1 sentence)

{body}

## Writing rules

- One paragraph, 80-120 words total.
- Cover each strategy with one or two phrases — what kind of day
  they had, what's notable.
- No need to mention every name; pick the ones that matter.
- Crisp. The reader is scanning.
- Byline is "Bot" — third-person tone about the strategies; no
  "we" / "I".

## Required output

```json
{{
  "headline": "<short headline, ≤80 chars — e.g., 'Mid-tier strategies tread water'>",
  "body_md": "<one paragraph>"
}}
```
"""


def _prompt_missed(payload: dict, day_iso: str) -> str:
    """Build the 'what we missed' prompt from a missed-movers payload —
    the per-region top-mover lists with catalyst + miss-reason."""
    by_region = payload.get("by_region", {})
    as_of = payload.get("as_of", day_iso)

    blocks = []
    for region, rep in by_region.items():
        movers = rep.get("top_movers", []) or []
        if not movers:
            continue
        lines = [f"### {region.upper()} — {len(movers)} biggest movers"]
        for m in movers[:8]:
            traded = ", ".join(m.get("was_traded_by") or []) or "no strategy"
            in_uni = ", ".join(m.get("in_universe_of") or []) or "outside any universe"
            catalyst = m.get("catalyst") or "(no catalyst found)"
            miss = m.get("miss_reason") or "(no reason hypothesis)"
            lines.append(
                f"- **{m['ticker']} {m['move_pct']:+.2f}%** — "
                f"traded by: {traded}; universes: {in_uni}. "
                f"Catalyst: {catalyst}. Why missed: {miss}"
            )
        blocks.append("\n".join(lines))
    body = "\n\n".join(blocks) if blocks else "(no movers data this run)"

    return f"""You are the trading floor reporter for The Bot Tribune.
Today is {day_iso}. You're writing the "what we missed" piece for
the trading floor section — the biggest movers in the bot's tradable
universes that we did NOT take a position in today (as of {as_of}).

The point of this piece is honest self-criticism: surface the tickers
that ran without us, identify whether they were in our coverage at
all, and call out the filter that excluded them. This piece feeds
straight into the weekly evolution agent's Lessons section.

## Today's biggest movers we missed (or held)

{body}

## Writing rules

- One paragraph, 110-150 words.
- Lead with the punchiest miss — biggest absolute move, named upfront.
- Name 3-5 tickers with their moves and the catalysts you have data
  for. If the catalyst is "no obvious news", say so — don't fabricate.
- Distinguish "filtered out" from "outside our universe" — these are
  different problems that need different fixes.
- If we actually held a top mover, mention it briefly — credit where due.
- No clichés, no "could've", no hand-wringing. Dry, factual.
- Byline is "Bot" — third-person about the strategies.

## Required output

```json
{{
  "headline": "<short headline, ≤80 chars — e.g., 'NVDA ran 6% on Q1 beat; control held back by RSI filter'>",
  "body_md": "<one paragraph>"
}}
```
"""


def _default_headline(kind: str, data: _StrategyDay | list[_StrategyDay] | dict) -> str:
    if isinstance(data, _StrategyDay):
        if kind == "winner":
            return f"{data.strategy_id} carries the day"
        if kind == "loser":
            return f"{data.strategy_id} stumbles"
    if kind == "missed":
        return "What the filters let pass"
    return "Quieter movers across the floor"


def _fallback_floor_brief(
    kind: str,
    kicker_tag: str,
    data,
    day_iso: str | None,
) -> FloorBrief:
    """Minimal prose if Haiku is unavailable. Surfaces the headline
    facts without trying to be writerly."""
    if kind == "missed" and isinstance(data, dict):
        # Walk the per-region payload, surface the top mover from each
        pieces = []
        for region, rep in (data.get("by_region") or {}).items():
            movers = rep.get("top_movers") or []
            if not movers:
                continue
            biggest = max(movers, key=lambda m: abs(m.get("move_pct", 0)))
            traded = ", ".join(biggest.get("was_traded_by") or []) or "no strategy"
            pieces.append(
                f"{region.upper()}: {biggest['ticker']} {biggest['move_pct']:+.2f}% "
                f"(traded by {traded})"
            )
        body = "Biggest movers we missed today — " + "; ".join(pieces) + "." if pieces else "No movers analysed."
        head = _default_headline(kind, data)
    elif isinstance(data, _StrategyDay):
        body = (
            f"{data.strategy_id} closed {data.n_closed} trades for a total "
            f"P&L of £{data.pnl_gbp_total:+,.2f}, averaging "
            f"{data.pnl_pct_avg:+.2f}% per trade. Names included "
            f"{', '.join(data.tickers[:5]) or '(none recorded)'}."
        )
        head = _default_headline(kind, data)
    elif isinstance(data, list):
        body = "Quieter movers: " + "; ".join(
            f"{s.strategy_id} £{s.pnl_gbp_total:+,.2f}"
            for s in data[:6]
        )
        head = "Quieter movers across the floor"
    else:
        body = "No closes recorded."
        head = "Quiet day on the floor"

    day_part = day_iso or "today"
    return FloorBrief(
        slug=f"floor-{day_part}-{kind}",
        headline=head,
        kicker=f"TRADING FLOOR · {kicker_tag}",
        byline="Bot",
        body_md=body,
        failed=True,
    )
