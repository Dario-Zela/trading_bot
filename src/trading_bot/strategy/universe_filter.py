"""Per-strategy LLM universe filter.

Replaces the strategy-agnostic Python `_prefilter` (which sorts by
|return_5d_pct| and biases every strategy toward the biggest movers).

For each LLM strategy, this calls Sonnet with the full universe + the
strategy's lens + today's macro/news context, and asks it to rank the
most relevant N candidates for THAT strategy's edge. mean-reverter
sees a different filtered set than momentum-trader from the same
universe — by design.

Runs inside `LLMStrategy.select_picks`, so the existing
ThreadPoolExecutor in `pipeline.run_entry` already parallelises the
calls across strategies (max 4 concurrent). Wall-clock cost is
roughly the time of the slowest strategy's filter call, not the sum.

Strategy config `prefilter_mode` field selects behaviour:
  - "llm"   : this module (Sonnet)
  - "python": the legacy Python sort by |return_5d_pct|
  - "off"   : no filter, the full universe goes downstream
             (only safe for small universes or rule-based strategies)

Graceful degradation: if the Sonnet call fails, returns None and the
caller falls back to the Python heuristic so the pipeline still ships
picks rather than going dark.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from trading_bot.llm.claude_code import ClaudeCodeError, run_claude_for_json
from trading_bot.strategy.base import StrategyConfig


log = logging.getLogger(__name__)


_TIMEOUT_SECONDS = 300        # Bounded budget per pre-filter call
_DEFAULT_TOP_N = 300          # Returned to the strategy for downstream technicals fetch


def _strategies_dir() -> Path:
    # Import-local to avoid the circular import the registry module triggers.
    from trading_bot.strategy.registry import _strategies_dir as _d
    return _d()


def _load_prefilter_prompt(strategy_id: str) -> str:
    """Strategy's `prompts/prefilter.md` — its lens for this stage.
    Returns empty string if missing; caller can decide to fall back."""
    p = _strategies_dir() / strategy_id / "prompts" / "prefilter.md"
    if not p.exists():
        return ""
    return p.read_text()


def _load_ticker_sectors() -> dict[str, str]:
    """Best-effort sector lookup from the cached file."""
    try:
        from trading_bot.state.paths import STATE_ROOT
        path = STATE_ROOT / "ticker_sectors.json"
        if path.exists():
            return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _build_filter_prompt(
    cfg: StrategyConfig,
    universe: list[str],
    on_date: date,
    *,
    top_n: int,
    strategy_lens: str,
) -> str:
    """Assemble the universe filter prompt.

    We include just (ticker, sector) for each candidate — keeping the
    table small so Sonnet can scan 8k+ names in one pass. The strategy
    lens prompt + today's macro/news brief give the LLM the semantic
    context it needs; technicals come in the next pipeline stage."""
    sectors = _load_ticker_sectors()
    universe_lines = []
    for tkr in universe:
        sec = sectors.get(tkr) or sectors.get(tkr.split(".")[0]) or "?"
        universe_lines.append(f"{tkr} | {sec}")

    macro_block = ""
    try:
        from trading_bot.tools.macro_view import get_macro_view
        view = get_macro_view()
        if view:
            macro_block = f"\n## Macro view\n\n{view}\n"
    except Exception:
        pass

    news_block = ""
    try:
        from trading_bot.tools.daily_news import get_daily_news_brief
        brief = get_daily_news_brief(on_date)
        if brief:
            news_block = f"\n## Today's news brief\n\n{brief}\n"
    except Exception:
        pass

    return f"""You are pre-filtering the universe for strategy `{cfg.id}` (region {cfg.region}) on trading day {on_date.isoformat()}.

The downstream pipeline will fetch 70 days of OHLCV technicals for the candidates you return and run them through Stage-1 Haiku scoring + Stage-2/3 Sonnet deep analysis. Your job is to NARROW {len(universe)} universe names down to the top {top_n} most relevant candidates for THIS strategy's edge, before the expensive technicals fetch.

## Strategy lens

{strategy_lens.strip() if strategy_lens else "(no lens prompt — fall back to general selection by liquidity and relevance to typical trading-day signals)"}
{macro_block}{news_block}
## Universe ({len(universe)} names — ticker | sector)

{chr(10).join(universe_lines)}

## Hard contract

Your response MUST begin with the `{{` of the JSON object. NO preamble, narration, "Let me think...", or explanation outside the JSON. Open with `{{`, close with `}}`.

## Required output

```json
{{
  "tickers": ["TICKER1", "TICKER2", ...]
}}
```

Rules:
- Return exactly {top_n} tickers (or fewer if you genuinely can't find that many relevant names — but err on the side of MORE so downstream stages have material to work with).
- Tickers MUST be drawn from the universe list above. Don't invent symbols.
- Order matters: rank most-relevant-to-this-strategy first.
- Prefer breadth across sectors when the strategy's lens doesn't prescribe a sector bias — diversity gives the Stage-2 analysis room to find a winner.
- Filter OUT names that are obviously not in this strategy's wheelhouse: e.g., the mean-reverter strategy should drop strongly-trending growth names; the momentum strategy should drop deep-value low-volatility names; news-reactive should prefer names with known fresh catalysts.

Open the JSON now.
"""


def llm_universe_filter(
    cfg: StrategyConfig,
    universe: list[str],
    on_date: date,
    *,
    top_n: int = _DEFAULT_TOP_N,
) -> list[str] | None:
    """Return a strategy-aware filtered subset of the universe.

    Returns None on failure (caller falls back to the Python heuristic).
    The empty list is a valid return — "filter returned nothing" — and
    callers should treat it as "the LLM thinks no name is worth scoring
    today", not as a failure.
    """
    if not universe:
        return []

    strategy_lens = _load_prefilter_prompt(cfg.id)
    prompt = _build_filter_prompt(
        cfg, universe, on_date, top_n=top_n, strategy_lens=strategy_lens,
    )

    log.info(
        "%s: LLM universe filter — %d universe → top %d (prompt %d chars)",
        cfg.id, len(universe), top_n, len(prompt),
    )
    try:
        response = run_claude_for_json(
            prompt,
            model="sonnet",
            timeout_seconds=_TIMEOUT_SECONDS,
        )
    except ClaudeCodeError as e:
        log.warning("%s: LLM universe filter failed: %s — caller should fall back", cfg.id, e)
        return None

    if not isinstance(response, dict):
        log.warning("%s: LLM filter returned non-dict shape: %r", cfg.id, type(response).__name__)
        return None

    tickers_raw = response.get("tickers") or []
    if not isinstance(tickers_raw, list):
        log.warning("%s: LLM filter 'tickers' field not a list", cfg.id)
        return None

    universe_set = set(universe)
    out: list[str] = []
    seen: set[str] = set()
    for t in tickers_raw:
        if not isinstance(t, str):
            continue
        t = t.strip().upper()
        if not t or t in seen:
            continue
        # Try the exact name first, then case-fixed against the universe.
        if t in universe_set:
            out.append(t)
            seen.add(t)
            continue
        # yfinance tickers are case-sensitive on suffix (e.g. 'BRK-B' vs 'brk-b'),
        # but the universe entries are already upper-cased.
        # Skip silently — Sonnet hallucinated a name not in the universe.
    log.info("%s: LLM filter accepted %d/%d returned tickers", cfg.id, len(out), len(tickers_raw))
    return out
