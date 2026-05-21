"""Daily missed-movers analysis.

After exits land, scan the union of strategy universes for the day's
biggest movers, cross-reference against what we actually traded, and
ask an LLM (Sonnet + WebSearch) why each missed one moved and why our
filters likely excluded it.

Saved output (`state/missed_movers/YYYY-MM-DD.{region}.json`) feeds two
downstream consumers:
- The daily news brief's "trading floor" section gets a "What we
  missed" piece each day.
- The weekly evolution agent reads the trailing 5-7 days of these to
  populate the "Lessons" quadrant of each strategy's report card with
  specific tickers the strategy's filter excluded.

Cost: 1 Sonnet call per missed mover (5-10 per region per day) with
WebSearch enabled so the model can identify the catalyst.
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from trading_bot.llm.claude_code import ClaudeCodeError, run_claude_for_json
from trading_bot.state.paths import STATE_ROOT
from trading_bot.tools import get_history
from trading_bot.tools.universe import get_universe

log = logging.getLogger(__name__)


_MAX_PARALLEL = 6
_PER_MOVER_TIMEOUT = 240
# How many of the day's biggest movers we examine per region. We look at
# *both* the top and the bottom (gainers + losers) so the analysis
# catches "we should have shorted X" cases too (once we run short).
_TOP_N_GAINERS = 6
_TOP_N_LOSERS = 4
_CLASSIFIER_TOOLS = ["--allowedTools", "WebSearch,WebFetch"]


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

@dataclass
class MissedMover:
    """One ticker that moved big today, with context on whether any
    strategy held / considered it and why we likely missed."""
    ticker: str
    move_pct: float                                 # close-to-close % change
    close: float
    in_universe_of: list[str] = field(default_factory=list)   # strategy ids
    was_traded_by: list[str] = field(default_factory=list)    # strategy ids that traded today
    catalyst: str = ""                              # one-line news driver
    miss_reason: str = ""                           # one-line filter hypothesis
    suggestion: str = ""                            # one-line change that would catch a future analogue
    failed: bool = False                            # true if LLM classification errored


@dataclass
class MissedMoversReport:
    date: str
    region: str
    universes_checked: list[str]
    n_tickers_checked: int
    top_movers: list[MissedMover] = field(default_factory=list)
    summary: str = ""                               # one-paragraph synthesis for downstream consumers


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

# Decoupled missed-movers universe per region. The strategies' actual
# universes (now t212_isa_* derivations with ~6.9k-8.5k names) are far
# too wide for this diagnostic — fetching history for 8500 tickers
# every exit was the main hang point in today's UK-EU exit run. The
# "missed" movers we care about are the ones liquid enough that a
# competent strategy COULD have caught them; that's almost always
# in the major-index blue-chip list.
_MISSED_MOVERS_UNIVERSE_BY_REGION = {
    "us": "sp500",                # ~500 names
    "uk-eu": "uk_eu_blue_chips",  # FTSE100 + DAX40 + CAC40 + AEX25 = ~205 names
}


def analyze_missed_movers(today: date, region: str) -> MissedMoversReport:
    """Run the missed-movers analysis for `region` on `today`.

    Returns the report and writes it to state/missed_movers/<iso>.<region>.json.
    Silent on missing OAUTH — emits a report without `catalyst` / `miss_reason`
    (the data is still useful even without LLM classification).
    """
    from trading_bot.strategy.registry import load_active_strategies

    strategies = load_active_strategies(region=region)
    if not strategies:
        log.info("missed-movers: no active strategies for region %s — skipping", region)
        return MissedMoversReport(date=today.isoformat(), region=region,
                                  universes_checked=[], n_tickers_checked=0)

    # The set of strategy IDs is still needed downstream so we can label
    # each missed mover with "which strategy could have caught this".
    universe_to_strategies: dict[str, list[str]] = {}
    for s in strategies:
        u = s.config.universe or ""
        if not u:
            continue
        universe_to_strategies.setdefault(u, []).append(s.config.id)

    # 1) Curated missed-movers universe. Far smaller than the strategy
    # candidate universes — see _MISSED_MOVERS_UNIVERSE_BY_REGION above.
    universe_id = _MISSED_MOVERS_UNIVERSE_BY_REGION.get(region)
    if not universe_id:
        log.warning("missed-movers: no curated universe for region %r — skipping", region)
        return MissedMoversReport(date=today.isoformat(), region=region,
                                  universes_checked=[], n_tickers_checked=0)
    universes_checked = [universe_id]
    all_tickers: set[str] = set()
    ticker_to_universes: dict[str, list[str]] = {}
    try:
        tickers = get_universe(universe_id)
    except Exception as e:
        log.warning("missed-movers: failed to load universe %r: %s", universe_id, e)
        return MissedMoversReport(date=today.isoformat(), region=region,
                                  universes_checked=[], n_tickers_checked=0)
    for t in tickers:
        all_tickers.add(t)
        ticker_to_universes.setdefault(t, []).append(universe_id)

    if not all_tickers:
        return MissedMoversReport(date=today.isoformat(), region=region,
                                  universes_checked=universes_checked, n_tickers_checked=0)

    log.info("missed-movers: %d unique tickers across %d universes for %s",
             len(all_tickers), len(universes_checked), region)

    # 2) Fetch yesterday + today bars for each. get_history paginates over
    # tickers; on a wide universe this is the slow step (~30-60s for 500
    # tickers via yfinance).
    try:
        history = get_history(sorted(all_tickers), lookback_days=3, end_date=today)
    except Exception as e:
        log.warning("missed-movers: history fetch failed: %s", e)
        return MissedMoversReport(date=today.isoformat(), region=region,
                                  universes_checked=universes_checked,
                                  n_tickers_checked=len(all_tickers))

    # 3) Compute close-to-close move per ticker for today
    moves: list[tuple[str, float, float]] = []  # (ticker, move_pct, close)
    for ticker, bars in history.items():
        if not bars or len(bars) < 2:
            continue
        latest = bars[-1]
        prev = bars[-2]
        if prev.close <= 0:
            continue
        move_pct = (latest.close / prev.close - 1.0) * 100.0
        moves.append((ticker, move_pct, latest.close))

    if not moves:
        return MissedMoversReport(date=today.isoformat(), region=region,
                                  universes_checked=universes_checked,
                                  n_tickers_checked=len(all_tickers))

    moves.sort(key=lambda x: x[1])
    # Filter so "losers" are actually negative and "gainers" actually positive.
    # On strong-up days we'd otherwise call the smallest gainers "losers".
    actual_losers = [m for m in moves if m[1] < 0][:_TOP_N_LOSERS]
    actual_gainers = [m for m in reversed(moves) if m[1] > 0][:_TOP_N_GAINERS]
    candidates = list(actual_gainers) + list(actual_losers)
    log.info("missed-movers: top %d gainers + %d losers selected",
             len(actual_gainers), len(actual_losers))

    # 4) Look up what the bot actually traded today
    traded_by = _traded_today(today, region)

    # 5) Build the MissedMover records (without LLM classification yet)
    movers: list[MissedMover] = []
    for ticker, move_pct, close in candidates:
        in_universes = ticker_to_universes.get(ticker, [])
        in_strategies = sorted({
            sid for u in in_universes
            for sid in universe_to_strategies.get(u, [])
        })
        movers.append(MissedMover(
            ticker=ticker, move_pct=round(move_pct, 3), close=round(close, 4),
            in_universe_of=in_strategies,
            was_traded_by=traded_by.get(ticker, []),
        ))

    # 6) LLM-classify each mover (catalyst + miss reason) in parallel.
    # Skip on missing OAUTH — leaves catalyst/miss_reason empty.
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") and movers:
        _classify_all(movers, today, region)

    # 7) Synthesise a one-paragraph summary for downstream consumers
    summary = _build_summary(movers, today, region)

    report = MissedMoversReport(
        date=today.isoformat(), region=region,
        universes_checked=universes_checked,
        n_tickers_checked=len(all_tickers),
        top_movers=movers,
        summary=summary,
    )
    _save_report(report)
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _traded_today(today: date, region: str) -> dict[str, list[str]]:
    """Map ticker → list of strategy ids that opened a position in this
    ticker on `today` for `region`. Uses the ledger directly so we
    catch shadow + paper trades alike."""
    from trading_bot.state.paths import ledger_path
    path = ledger_path()
    if not path.exists():
        return {}
    iso = today.isoformat()
    out: dict[str, list[str]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("region") != region:
                continue
            if rec.get("entry_date") != iso:
                continue
            ticker = rec.get("ticker")
            sid = rec.get("strategy_id")
            if not ticker or not sid:
                continue
            out.setdefault(ticker, [])
            if sid not in out[ticker]:
                out[ticker].append(sid)
    return out


def _classify_all(movers: list[MissedMover], today: date, region: str) -> None:
    """Fan out one Sonnet+WebSearch call per mover to identify the
    catalyst and hypothesise why our filters missed it. Mutates the
    list in place."""
    log.info("missed-movers: classifying %d movers via Sonnet+WebSearch", len(movers))
    with ThreadPoolExecutor(max_workers=_MAX_PARALLEL) as pool:
        futures = {pool.submit(_classify_one, m, today, region): m for m in movers}
        for fut in as_completed(futures):
            m = futures[fut]
            try:
                catalyst, reason, suggestion = fut.result()
                m.catalyst = catalyst
                m.miss_reason = reason
                m.suggestion = suggestion
            except Exception as e:
                log.warning("missed-movers: classification failed for %s: %s", m.ticker, e)
                m.failed = True


def _classify_one(m: MissedMover, today: date, region: str) -> tuple[str, str, str]:
    prompt = _build_classification_prompt(m, today, region)
    response = run_claude_for_json(
        prompt, model="sonnet",
        timeout_seconds=_PER_MOVER_TIMEOUT,
        extra_args=_CLASSIFIER_TOOLS,
    )
    if not isinstance(response, dict):
        return "(could not classify)", "(no reason hypothesis)", ""
    catalyst = str(response.get("catalyst") or "").strip()[:280]
    reason = str(response.get("miss_reason") or "").strip()[:320]
    suggestion = str(response.get("suggestion") or "").strip()[:320]
    return (
        catalyst or "(no clear catalyst found)",
        reason or "(no reason hypothesis)",
        suggestion,
    )


def _build_classification_prompt(m: MissedMover, today: date, region: str) -> str:
    universes_block = ", ".join(m.in_universe_of) or "(no strategy's universe includes this)"
    traded_block = ", ".join(m.was_traded_by) or "(no strategy traded it)"
    direction = "gainer" if m.move_pct > 0 else "loser"

    return f"""You are the post-trade analyst for an algorithmic trading bot.
Today is {today.isoformat()}; region under review: {region}.

The bot has identified {m.ticker} as one of the day's biggest {direction}
in its tradable universe — but its strategies did NOT take a position
in this name today. Your job: explain why the stock moved and
hypothesise why the bot's filters likely passed on it.

## The mover

- **Ticker:** {m.ticker}
- **Close-to-close move:** {m.move_pct:+.2f}%
- **Today's close:** {m.close}
- **In the universe of strategies:** {universes_block}
- **Was traded by:** {traded_block}

## Tools available

- **WebSearch** — find news / earnings / regulatory / M&A drivers from
  today specifically. Try queries like '{m.ticker} {today.isoformat()}
  news' or '{m.ticker} earnings' or '{m.ticker} catalyst'.
- **WebFetch** — pull a specific article URL when the search points at
  a strong source (Reuters, Bloomberg, FT, CNBC, company press release).

## What to produce

A three-field verdict:

1. **catalyst** (≤200 chars) — one sentence on WHAT moved the stock.
   Examples: "Q1 revenue beat $44B vs $43B est; raised FY guidance" /
   "Activist Elliott discloses 3% stake, pushes for board changes" /
   "FDA breakthrough designation for lead oncology candidate".
   If you genuinely can't find a clear driver, say "no obvious news
   catalyst — likely technical / sector move".

2. **miss_reason** (≤240 chars) — one or two sentences hypothesising
   why a momentum / mean-reversion / news-reactive bot likely missed
   this. Consider:
   - It was in the universe but a technical filter excluded it (RSI
     too high, gap too large, volume profile)
   - It was a catalyst-driven move (earnings surprise, M&A) where
     pre-market positioning beats post-market signals
   - It was outside the universe entirely (smaller cap, ADR-only)
   - The strategies that hold it were saturated on position count

   If `in_universe_of` is empty above, lead with "outside any active
   strategy's universe" and explain what cap/sector category it falls
   into. If non-empty, focus on what filter likely excluded it.

3. **suggestion** (≤240 chars) — one sentence on a concrete change
   that would catch the *next* analogue of this miss without breaking
   the rest of the strategy. Examples:
   - "Earnings-gate should release names with post-results gap ≥
     +X% within Y minutes of open"
   - "Add a `dollar_index_short_proxy` so commodity strategies can
     express USD-weakness via gold even on long-only constraints"
   - "Bump momentum-trader's volume floor; the gap-up filter missed
     this because it requires ≥1.5× avg volume in the prior session"
   - "(no actionable change — outside risk tolerance / requires
     short capability we don't have)" when an honest no-op fits.

   Stay concrete (cite a parameter or rule), don't generalise.

## Required output

Return JSON only:

```json
{{
  "catalyst": "<one sentence>",
  "miss_reason": "<one-two sentences>",
  "suggestion": "<one sentence>"
}}
```
"""


def _build_summary(movers: list[MissedMover], today: date, region: str) -> str:
    """Multi-line synopsis: a one-line headline followed by themed
    bullets. Newlines are preserved by the dashboard (and any other
    consumer that respects `white-space: pre-line`).

    Themes the bullets surface, when applicable:
      1. Upside misses split by gainer count and their biggest names.
      2. Downside misses called out for the long-only blind spot.
      3. The single most-instructive miss (largest |move|) with its
         catalyst — useful for the daily news brief.
    """
    if not movers:
        return f"No notable movers identified in {region} on {today.isoformat()}."

    region_label = {"us": "US", "uk-eu": "UK/EU"}.get(region, region.upper())
    traded = [m for m in movers if m.was_traded_by]
    untraded = [m for m in movers if not m.was_traded_by]
    gainers = [m for m in untraded if m.move_pct > 0]
    losers = [m for m in untraded if m.move_pct < 0]

    # Headline
    headline = (
        f"Held {len(traded)} of today's {len(movers)} biggest {region_label} movers."
    )

    bullets: list[str] = []

    # Upside misses — the strategies passed on these despite the
    # universe coverage. Name the top 5 by move size for cite-ability
    # in the news brief.
    if gainers:
        top_g = sorted(gainers, key=lambda m: m.move_pct, reverse=True)[:5]
        tag = ", ".join(
            f"{m.ticker} {m.move_pct:+.0f}%" for m in top_g
        )
        bullets.append(
            f"• {len(gainers)} upside miss{'es' if len(gainers) != 1 else ''} — "
            f"{tag}. These cleared the universe screen but no strategy "
            f"picked them up; the morning brief either filtered them on "
            f"a catalyst/RSI/earnings rule or had nothing positive to "
            f"trigger an entry."
        )

    # Downside misses — long-only constraint
    if losers:
        top_l = sorted(losers, key=lambda m: m.move_pct)[:4]
        tag = ", ".join(f"{m.ticker} {m.move_pct:+.0f}%" for m in top_l)
        bullets.append(
            f"• {len(losers)} downside move{'s' if len(losers) != 1 else ''} "
            f"({tag}) are unreachable on a long-only basket — no mechanism "
            f"to express a short or pair-trade."
        )

    # The single most-instructive miss
    if untraded:
        biggest = max(untraded, key=lambda m: abs(m.move_pct))
        line = (
            f"• Most instructive miss: {biggest.ticker} "
            f"({biggest.move_pct:+.2f}%)"
        )
        if biggest.catalyst:
            line += f" — {biggest.catalyst}"
        if biggest.miss_reason:
            line += f" Why we passed: {biggest.miss_reason}"
        bullets.append(line)

    if not bullets:
        return headline
    return headline + "\n\n" + "\n\n".join(bullets)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _missed_movers_dir() -> Path:
    p = STATE_ROOT / "missed_movers"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _report_path(today: date, region: str) -> Path:
    return _missed_movers_dir() / f"{today.isoformat()}.{region}.json"


def _save_report(report: MissedMoversReport) -> None:
    path = _report_path(date.fromisoformat(report.date), report.region)
    payload = {
        "date": report.date,
        "region": report.region,
        "universes_checked": report.universes_checked,
        "n_tickers_checked": report.n_tickers_checked,
        "summary": report.summary,
        "top_movers": [asdict(m) for m in report.top_movers],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2))
    log.info("missed-movers: wrote %s (%d movers)", path, len(report.top_movers))


def load_report(today: date, region: str) -> dict | None:
    """Read a saved report. Returns None if the file doesn't exist."""
    path = _report_path(today, region)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def load_recent_reports(days: int = 7) -> list[dict]:
    """Load every missed-movers report from the last `days` days, both
    regions. Used by the weekly evolution agent."""
    from datetime import timedelta
    out: list[dict] = []
    d = _missed_movers_dir()
    if not d.exists():
        return out
    cutoff_iso = (date.today() - timedelta(days=days)).isoformat()
    for p in sorted(d.glob("*.json")):
        try:
            payload = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if (payload.get("date") or "") < cutoff_iso:
            continue
        out.append(payload)
    # Newest-first
    out.sort(key=lambda r: r.get("date", ""), reverse=True)
    return out
