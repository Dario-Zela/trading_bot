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

    # Stable ordering: winner, loser, quieter
    order = {"winner": 0, "loser": 1, "quieter": 2}
    briefs.sort(key=lambda b: order.get(b.slug.split("-")[-1], 99))
    return briefs


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
    return {
        "ticker": t.get("ticker", ""),
        "pnl_gbp": float(t.get("pnl_gbp") or 0.0),
        "pnl_pct": float(t.get("pnl_pct") or 0.0),
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


def _build_prompt(kind: str, data: _StrategyDay | list[_StrategyDay], day_iso: str) -> str:
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
        f"  - {t['ticker']}: £{t['pnl_gbp']:+,.2f} ({t['pnl_pct']*100:+.2f}%)  exit: {t['exit_reason']}"
        for t in s.notable_winners
    ) or "  (no clear standout winners)"
    losers_block = "\n".join(
        f"  - {t['ticker']}: £{t['pnl_gbp']:+,.2f} ({t['pnl_pct']*100:+.2f}%)  exit: {t['exit_reason']}"
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
- **Avg P&L per trade:** {s.pnl_pct_avg*100:+.2f}%
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
            f"£{s.pnl_gbp_total:+,.2f} total ({s.pnl_pct_avg*100:+.2f}% avg). "
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


def _default_headline(kind: str, data: _StrategyDay | list[_StrategyDay]) -> str:
    if isinstance(data, _StrategyDay):
        if kind == "winner":
            return f"{data.strategy_id} carries the day"
        if kind == "loser":
            return f"{data.strategy_id} stumbles"
    return "Quieter movers across the floor"


def _fallback_floor_brief(
    kind: str,
    kicker_tag: str,
    data: _StrategyDay | list[_StrategyDay] | None,
    day_iso: str | None,
) -> FloorBrief:
    """Minimal prose if Haiku is unavailable. Surfaces the headline
    facts without trying to be writerly."""
    if isinstance(data, _StrategyDay):
        body = (
            f"{data.strategy_id} closed {data.n_closed} trades for a total "
            f"P&L of £{data.pnl_gbp_total:+,.2f}, averaging "
            f"{data.pnl_pct_avg*100:+.2f}% per trade. Names included "
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
