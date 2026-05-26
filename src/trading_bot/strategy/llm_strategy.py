"""LLM-driven strategy runtime.

Wave 2b initial scope: a single Claude call per pipeline run. The Python
harness pre-fetches everything Claude needs (technicals + recent news for the
pre-filtered candidates), packs it into a structured prompt alongside the
strategy's deep_analysis.md + final_select.md prompts, and asks Claude to
return the final trade list as JSON.

This is intentionally one-shot rather than the full multi-stage pipeline
described in sparknotes. The multi-stage version (wide scoring → deep
analysis → final selection) is a follow-up — once the wiring here is proven,
we split into stages for cost efficiency and easier debugging.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from trading_bot.executor.base import TradeIntent
from trading_bot.llm.claude_code import ClaudeCodeError, run_claude_for_json
from trading_bot.state.predictions import PredictionRecord, append_prediction
from trading_bot.strategy.base import Strategy
from trading_bot.tools import (
    get_commodity_prices,
    get_credit_spreads,
    get_dollar_index,
    get_earnings_info,
    get_filing_summary,
    get_insider_trades,
    get_macro_view,
    get_recent_news,
    get_sector_strength,
    get_technicals,
    get_universe,
    get_yield_curve,
)


log = logging.getLogger(__name__)


# Pre-filter parameters — keep a wide directional spread so the LLM scores
# rising, falling, and flat candidates (not just trend-following winners).
# Liquidity comes from the universe (FTSE 350, S&P 1500, etc.); we don't
# enforce an extra volume floor at filter time because today's intraday
# bar (which yfinance returns mid-session) reports only partial volume,
# yielding misleadingly low volume_ratio for morning runs.
_PREFILTER_TOP_N = 100  # candidates handed to the LLM for multi-class scoring


# Python-prefilter sort keys. Each rank function returns a comparable
# value; sentinels are used so missing technicals always sort to the
# bottom regardless of direction (with `reverse=True`, -inf sinks; with
# `reverse=False`, +inf sinks).
def _prefilter_rank_fn(key: str):
    import math
    if key == "abs_return_5d":
        return lambda t: abs(t.return_5d_pct) if t.return_5d_pct is not None else -math.inf
    if key == "abs_return_20d":
        return lambda t: abs(t.return_20d_pct) if t.return_20d_pct is not None else -math.inf
    if key == "rsi_14_asc":
        # Ascending sort; None pushed to the back via +inf.
        return lambda t: t.rsi_14 if t.rsi_14 is not None else math.inf
    if key == "volume_ratio_desc":
        return lambda t: t.volume_ratio if t.volume_ratio is not None else -math.inf
    if key == "dollar_volume_desc":
        # sma_20 × avg_volume_20 ≈ 20-day average dollar volume. Large-
        # caps and ETFs dwarf microcaps on this metric, which is the
        # point — strategies that should be playing sector vehicles
        # (macro-aligned, bond-cycle) get those vehicles up top instead
        # of whatever microcap had the biggest recent move.
        def _dv(t):
            if t.sma_20 is None or t.avg_volume_20 is None:
                return -math.inf
            return t.sma_20 * t.avg_volume_20
        return _dv
    # Unknown key → fall back to the legacy default rather than crashing.
    return lambda t: abs(t.return_5d_pct) if t.return_5d_pct is not None else -math.inf


# Sort direction per key. True = descending (largest first). The
# ascending case is `rsi_14_asc` (most oversold first).
_PREFILTER_DESC = {
    "abs_return_5d":      True,
    "abs_return_20d":     True,
    "rsi_14_asc":         False,
    "volume_ratio_desc":  True,
    "dollar_volume_desc": True,
}


def _safe_float(val, *, default: float) -> float:
    try:
        f = float(val)
        if f != f:  # NaN
            return default
        return f
    except (TypeError, ValueError):
        return default


def _fmt(val, spec: str, na: str = "?") -> str:
    """Format an optional numeric for a prompt line; None → `na`. Deep-analysis
    candidates legitimately carry None technicals (e.g. sma_50 / MACD on a
    <55-bar history), and an unguarded f-string format on None raises
    TypeError and kills the whole strategy's pick run."""
    return format(val, spec) if val is not None else na


def _classify(predicted_pct: float, rsi: float | None) -> str:
    """Bucket a candidate into one of the five directional classes used for
    prediction grading. Crude version based on the 5d return + RSI; replaced
    with explicit LLM scoring per candidate in a later refactor."""
    if predicted_pct >= 8.0 or (rsi is not None and rsi >= 70 and predicted_pct >= 4.0):
        return "strong_up"
    if predicted_pct >= 2.0:
        return "mild_up"
    if predicted_pct <= -8.0 or (rsi is not None and rsi <= 30 and predicted_pct <= -4.0):
        return "strong_down"
    if predicted_pct <= -2.0:
        return "mild_down"
    return "flat"


def _build_recent_self_trades_block(strategy_id: str, region: str, *,
                                    n_trades: int = 5, days: int = 14) -> str:
    """Phase 11G — return a markdown block summarising the strategy's
    last few real trades, so the LLM can see what its own recent
    decisions led to before making today's pick.

    Skips templated reflections (the fallback that fires when the
    Haiku reflection agent failed). Empty string when there's nothing
    to surface."""
    import json as _json
    from datetime import date as _date, timedelta
    from trading_bot.state.paths import ledger_path
    p = ledger_path()
    if not p.exists():
        return ""
    cutoff = (_date.today() - timedelta(days=days)).isoformat()
    matched: list[dict] = []
    try:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if rec.get("strategy_id") != strategy_id:
                    continue
                if rec.get("region") != region:
                    continue
                ed = rec.get("exit_date") or ""
                if not ed or ed < cutoff:
                    continue
                if rec.get("exit_reason") in ("cancelled", "cleared"):
                    continue
                matched.append(rec)
    except OSError:
        return ""
    if not matched:
        return ""
    matched.sort(key=lambda r: r.get("exit_date") or "", reverse=True)
    rows = []
    for r in matched[:n_trades]:
        ticker = r.get("ticker", "?")
        pct = float(r.get("pnl_pct") or 0)
        pct = max(-100.0, min(500.0, pct))
        reason = r.get("exit_reason", "?")
        notes = (r.get("outcome_notes") or "").strip()
        # Skip the templated-fallback marker
        if "(No LLM reflection — fallback text.)" in notes:
            notes = ""
        risks = (r.get("risks_observed") or "").strip()
        if "(reflection agent did not run on this trade)" in risks:
            risks = ""
        line = f"- **{ticker} {pct:+.2f}%** ({r.get('exit_date')}, {reason})"
        if notes:
            line += f"\n  - Outcome: {notes[:240]}"
        if risks:
            line += f"\n  - Risks: {risks[:240]}"
        rows.append(line)
    return (
        "## Your recent trades (last 14 days, top "
        f"{min(n_trades, len(matched))} most recent)\n\n"
        "These are your own decisions and their outcomes. Today's pick "
        "should reflect what you've just learned — if the same setup "
        "lost money 3 times this week, expect it to keep losing today.\n\n"
        + "\n".join(rows)
    )


def _build_open_positions_block(strategy_id: str, region: str) -> str:
    """Phase 12G — surface currently-held multi-day positions so the LLM
    can reason about its exposure before picking today's names. Already-
    held tickers are auto-dropped by the pipeline pre-entry, but the LLM
    seeing them lets it: (a) avoid wasting a pick slot on a duplicate,
    (b) factor existing exposure into sector / correlation reasoning.
    Empty string when there are no open positions."""
    from trading_bot.state import read_open_trades
    from datetime import date as _date

    try:
        open_trades = read_open_trades(strategy_id=strategy_id, region=region)
    except Exception:
        return ""
    if not open_trades:
        return ""

    today_iso = _date.today().isoformat()
    rows: list[str] = []
    for t in open_trades:
        ticker = t.get("ticker", "?")
        entry = t.get("entry_date", "?")
        target = t.get("target_exit_date") or "(same-day)"
        thesis = (t.get("thesis") or "").strip()
        hold = t.get("hold_days") or 1
        line = f"- **{ticker}** entered {entry}, target exit {target} (hold {hold}d)"
        if thesis:
            line += f" — {thesis[:200]}"
        rows.append(line)

    return (
        "## Currently-held positions (multi-day carryover)\n\n"
        f"As of {today_iso}, you have {len(open_trades)} open position(s) "
        "from prior sessions still within their hold window. Today's picks "
        "are screened against this list — you cannot re-pick a ticker you "
        "already hold. Use this view to keep your sector / correlation "
        "exposure balanced when selecting new names:\n\n"
        + "\n".join(rows)
    )


class LLMStrategy(Strategy):
    """Strategy backed by Claude Code in single-call mode."""

    # Top-N from Stage 1 to hand to Stage 2's deep-analysis call.
    # 20 gives the Sonnet stage plenty of optionality while keeping
    # per-candidate tool fetches (filings / earnings / insider) cheap.
    _STAGE2_TOP_N = 20

    def select_picks(self, on_date: date) -> list[TradeIntent]:
        cfg = self.config

        # Fail fast if auth isn't configured — pre-fetching yfinance data for
        # 500 tickers takes ~40s and we'd rather skip the strategy cleanly.
        import os
        if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            log.error(
                "%s: CLAUDE_CODE_OAUTH_TOKEN not set — skipping LLM strategy. "
                "Run `claude setup-token` locally and add the token as a repo secret.",
                cfg.id,
            )
            return []

        prompts = self._load_prompts()

        # Universe pre-filter. Three modes (set on cfg.prefilter_mode):
        #   - "llm":    Sonnet pre-filter with strategy-specific lens.
        #               Replaces both the upstream universe fetch (we now
        #               only fetch technicals for the LLM's shortlist) AND
        #               the bias-toward-big-movers of the Python sort.
        #   - "python": legacy — fetch all universe technicals, sort by |5d|.
        #   - "off":    no pre-filter (only safe for small universes).
        # On LLM failure we fall back to Python so the pipeline degrades
        # gracefully rather than going dark.
        tickers = get_universe(cfg.universe)
        log.info("%s: %d universe candidates, mode=%s",
                 cfg.id, len(tickers), cfg.prefilter_mode)

        prefilter_mode = (cfg.prefilter_mode or "python").lower().strip()
        if prefilter_mode == "llm":
            from trading_bot.strategy.universe_filter import llm_universe_filter
            shortlist = llm_universe_filter(
                cfg, tickers, on_date,
                top_n=max(int(cfg.prefilter_top_n or 100), 20),
            )
            if shortlist is None:
                log.warning("%s: LLM pre-filter failed — falling back to Python sort",
                            cfg.id)
                techs = get_technicals(tickers, end_date=on_date)
                candidates = self._prefilter(techs)
            elif not shortlist:
                log.info("%s: LLM pre-filter returned empty shortlist — no picks today", cfg.id)
                return []
            else:
                log.info("%s: LLM pre-filter returned %d names — fetching technicals",
                         cfg.id, len(shortlist))
                techs = get_technicals(shortlist, end_date=on_date)
                # The shortlist is already strategy-aware; we just need to
                # drop anything yfinance couldn't price + cap to top_n.
                candidates = [techs[t] for t in shortlist if t in techs and techs[t].rsi_14 is not None]
        elif prefilter_mode == "off":
            techs = get_technicals(tickers, end_date=on_date)
            candidates = [t for t in techs.values() if t.rsi_14 is not None]
        else:  # "python" or anything unrecognised
            techs = get_technicals(tickers, end_date=on_date)
            candidates = self._prefilter(techs)

        log.info("%s: %d candidates passed pre-filter", cfg.id, len(candidates))
        if not candidates:
            return []

        # Phase 8C — earnings gating. If the strategy has set
        # `skip_if_earnings_in_days > 0`, drop candidates with a known
        # earnings date inside that window. Saves tokens AND avoids
        # binary post-print drawdowns the technical signal can't see.
        if cfg.skip_if_earnings_in_days > 0:
            n_before = len(candidates)
            candidates = self._filter_pre_earnings(
                candidates, on_date=on_date,
                window_days=cfg.skip_if_earnings_in_days,
            )
            n_after = len(candidates)
            log.info("%s: %d candidates after earnings gate (%d-day window)",
                     cfg.id, n_after, cfg.skip_if_earnings_in_days)
            # Phase 10B — persist for evolution agent
            try:
                self._persist_earnings_gate(cfg.id, on_date, n_before - n_after, n_before)
            except Exception as e:
                log.debug("persist earnings-gate failed: %s", e)
            if not candidates:
                return []

        # Gather news for the surviving candidates in one batched call
        candidate_tickers = [t.ticker for t in candidates]
        try:
            news = get_recent_news(candidate_tickers, days=3, limit=50)
        except Exception as e:
            log.warning("News fetch failed (continuing without news): %s", e)
            news = {ticker: [] for ticker in candidate_tickers}

        # Stage 1 — Haiku wide-scoring on every candidate. Cheap broad
        # 5-class prediction pass; the goal is to surface the most
        # informative top-N for the expensive Stage 2 call.
        stage1_predictions = self._stage1_wide_score(cfg, on_date, candidates, news, prompts)

        # Narrow to top-N for Stage 2. Falls back to ALL candidates if
        # Stage 1 returned nothing (e.g., Haiku call failed) so the
        # pipeline degrades gracefully instead of going dark.
        stage2_candidates = self._select_stage2_candidates(candidates, stage1_predictions)
        log.info(
            "%s: stage 1 scored %d, handing %d to stage 2 deep analysis",
            cfg.id, len(stage1_predictions) or len(candidates), len(stage2_candidates),
        )

        prompt = self._build_prompt(
            on_date=on_date,
            candidates=stage2_candidates,
            news=news,
            deep_analysis_prompt=prompts["deep_analysis"],
            final_select_prompt=prompts["final_select"],
        )

        try:
            response = run_claude_for_json(
                prompt,
                model=cfg.model_assignment.get("final_select", "sonnet"),
            )
        except ClaudeCodeError as e:
            log.error("%s: stage-2 LLM call failed: %s", cfg.id, e)
            self._log_predictions(candidates, llm_predictions=stage1_predictions, picked_intents=[], on_date=on_date)
            return []

        stage2_predictions, picks_raw = self._extract_predictions_and_picks(response)

        # Phase 8A + 8B — vol-aware sizing + FX cost gate. Applied
        # BEFORE _parse_picks so the TradeIntents go out with the
        # adjusted allocations. Adjustment log gets emitted to INFO.
        from trading_bot.strategy.sizing import adjust_picks, format_adjustment_log, persist_adjustments
        picks_raw, adjustments = adjust_picks(
            picks_raw,
            candidates=stage2_candidates,
            predictions=stage2_predictions,
            cfg=cfg,
        )
        log.info("%s: pick post-processing\n%s", cfg.id, format_adjustment_log(adjustments))
        # Phase 10B — persist for the evolution agent to read
        try:
            persist_adjustments(cfg.id, on_date, adjustments)
        except Exception as e:
            log.debug("persist_adjustments failed: %s", e)

        intents = self._parse_picks(picks_raw)
        # Stage 2 predictions (richer reasoning) override Stage 1 for the
        # top-N; Stage 1 fills the rest of the universe for IC computation.
        merged_predictions = {**stage1_predictions, **stage2_predictions}
        self._log_predictions(
            candidates,
            llm_predictions=merged_predictions,
            picked_intents=intents,
            on_date=on_date,
        )
        return intents

    def _stage1_wide_score(
        self,
        cfg,
        on_date: date,
        candidates: list,
        news: dict,
        prompts: dict[str, str],
    ) -> dict[str, dict]:
        """Cheap Haiku call: predict 5-class direction for every candidate
        with minimal context (technicals + 1 headline per ticker). Returns
        {ticker: prediction_dict}. Empty on failure — caller falls back
        to using all candidates for Stage 2."""
        prompt = self._build_stage1_prompt(cfg, on_date, candidates, news, prompts)
        model = cfg.model_assignment.get("wide_scoring", "haiku")
        try:
            response = run_claude_for_json(prompt, model=model)
        except ClaudeCodeError as e:
            log.warning("%s: stage-1 (wide scoring) failed: %s — degrading to single-stage", cfg.id, e)
            return {}
        preds, _ = self._extract_predictions_and_picks(response)
        return preds

    def _build_stage1_prompt(
        self,
        cfg,
        on_date: date,
        candidates: list,
        news: dict,
        prompts: dict[str, str],
    ) -> str:
        """Compact Stage-1 prompt: one row per candidate (ticker, technicals
        snapshot, top headline), strategy bias section, ask for 5-class
        predictions on EVERY candidate in JSON."""
        bias = (prompts.get("deep_analysis") or "").strip()
        # Crude tail-trim — Stage 1 just needs the strategy bias, not the
        # full deep_analysis. Keep first 1500 chars.
        if len(bias) > 1500:
            bias = bias[:1500] + "\n\n[bias truncated for stage 1]"

        rows: list[str] = []
        for c in candidates:
            top_headline = ""
            ticker_news = news.get(c.ticker) or []
            if ticker_news:
                top_headline = f" · news: {ticker_news[0].headline[:120]}"
            rsi = f"{c.rsi_14:.0f}" if c.rsi_14 is not None else "—"
            ret5 = f"{c.return_5d_pct:+.2f}%" if c.return_5d_pct is not None else "—"
            ret20 = f"{c.return_20d_pct:+.2f}%" if c.return_20d_pct is not None else "—"
            rows.append(f"- {c.ticker}: close={c.close:.2f} 5d={ret5} 20d={ret20} RSI={rsi}{top_headline}")

        return (
            f"You are running stage 1 (wide scoring) for trading strategy `{cfg.id}` on "
            f"{on_date.isoformat()}. Region: {cfg.region}. This is a CHEAP broad pass — "
            f"score every candidate, no picks yet, no allocations.\n\n"
            f"## Strategy bias / approach (your edge)\n\n{bias or '(no bias prompt — score by technicals)'}\n\n"
            f"## Candidates ({len(candidates)})\n\n" + "\n".join(rows) + "\n\n"
            "## Hard contract\n\n"
            "Work with the candidate data already provided above. Do NOT ask for "
            "additional tools or data sources — `get_macro_view`, `get_sector_strength`, "
            "`get_recent_news`, etc. are NOT available in this call. If a piece of "
            "context you'd ideally have is missing, score from technicals alone "
            "(RSI, MACD, SMA distance, momentum) and lower your conviction to reflect "
            "the uncertainty. Return the JSON below regardless — listing questions "
            "or requests for data is a contract failure.\n\n"
            "## Required output\n\n"
            "Score every candidate above with a 5-class prediction. Output the JSON "
            "object immediately — no narration, no preamble, no markdown around it:\n\n"
            "```json\n"
            "{\n"
            "  \"predictions\": [\n"
            "    {\n"
            "      \"ticker\": \"...\",\n"
            "      \"predicted_class\": \"strong_up\" | \"mild_up\" | \"flat\" | \"mild_down\" | \"strong_down\",\n"
            "      \"predicted_return_pct\": <float estimate of today's intraday return>,\n"
            "      \"conviction\": <0.0-1.0>,\n"
            "      \"rationale\": \"<1 short sentence>\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n\n"
            "Be honest about flat and falling names — those scores feed prediction-grading "
            "metrics and matter as much as the up calls. Do not include `picks` in this response."
        )

    def _select_stage2_candidates(
        self, candidates: list, stage1_predictions: dict[str, dict]
    ) -> list:
        """Rank by Stage 1's conviction × expected magnitude, keep top N.
        Falls back to original candidates list if Stage 1 produced nothing."""
        if not stage1_predictions:
            return candidates

        def score(c) -> float:
            p = stage1_predictions.get(c.ticker)
            if not p:
                return 0.0
            try:
                conv = float(p.get("conviction") or 0)
            except (TypeError, ValueError):
                conv = 0.0
            try:
                mag = abs(float(p.get("predicted_return_pct") or 0))
            except (TypeError, ValueError):
                mag = 0.0
            return conv * mag

        ranked = sorted(candidates, key=score, reverse=True)
        return ranked[: self._STAGE2_TOP_N]

    # ---- internals ---------------------------------------------------------

    def _load_prompts(self) -> dict[str, str]:
        from trading_bot.strategy.registry import _strategies_dir

        prompts_dir = _strategies_dir() / self.config.id / "prompts"
        out: dict[str, str] = {}
        for name in ("deep_analysis", "final_select"):
            path = prompts_dir / f"{name}.md"
            out[name] = path.read_text() if path.exists() else ""
        # final_select is mandatory; deep_analysis is recommended but optional
        if not out["final_select"]:
            log.warning(
                "%s: no final_select.md prompt found at %s — Claude has no instructions",
                self.config.id, prompts_dir,
            )
        return out

    def _prefilter(self, techs: dict) -> list:
        """Python pre-filter. Ranker selected by cfg.prefilter_sort_key.

        Keeps any ticker with the minimum usable technicals (`rsi_14` +
        `return_5d_pct` both computable), then ranks by the configured
        sort key. The legacy default is `abs_return_5d`, which biases
        every strategy toward the biggest movers — fine for momentum,
        wrong for macro / sector / mean-reversion lenses. The evolution
        agent can tune this per-strategy via the `tune` action.

        Liquidity is enforced via universe selection upstream, not here
        — a today-vs-avg volume ratio is unreliable mid-session when
        today's bar only carries partial volume.
        """
        keep = []
        for ticker, t in techs.items():
            if t.rsi_14 is None or t.return_5d_pct is None:
                continue
            keep.append(t)
        sort_key = (getattr(self.config, "prefilter_sort_key", None) or "abs_return_5d").lower().strip()
        keep.sort(key=_prefilter_rank_fn(sort_key), reverse=_PREFILTER_DESC.get(sort_key, True))
        return keep[:_PREFILTER_TOP_N]

    def _build_prompt(
        self,
        *,
        on_date: date,
        candidates: list,
        news: dict,
        deep_analysis_prompt: str,
        final_select_prompt: str,
    ) -> str:
        cfg = self.config
        sections: list[str] = []

        sections.append(
            f"You are running strategy `{cfg.id}` for trading day {on_date.isoformat()}.\n"
            f"Region: {cfg.region}. Capital allocation: £{cfg.capital_gbp:.0f}.\n"
            f"Hard constraints from strategy config:\n"
            f"- max_positions: {cfg.max_positions}\n"
            f"- max_position_pct: {cfg.max_position_pct}%\n"
            f"- min_position_size: £{cfg.min_position_gbp}\n"
            f"- target_daily_risk: {cfg.target_daily_risk_pct}% of capital per position\n"
            f"- use_stops: {cfg.use_stops}\n"
            f"- use_take_profits: {cfg.use_take_profits}\n"
            f"\n"
            f"## Sizing note (important)\n"
            f"Your `allocation_pct` on each pick is treated as a CONVICTION "
            f"signal, not the final position size. After you return picks, "
            f"a post-processor rewrites the allocation to risk-parity sizing "
            f"so each position carries the same {cfg.target_daily_risk_pct}%-of-"
            f"capital exposure to a 1-ATR adverse move. High-volatility names "
            f"end up smaller, low-volatility names end up larger, both "
            f"clamped to the max/min above. Set `allocation_pct` to express "
            f"your relative confidence (e.g. 10 vs 5 = 'twice as confident') "
            f"and don't worry about ATR — that's handled afterwards.\n"
            f"\n"
            f"## Cost gate (hard rule)\n"
            f"Picks where `|predicted_return_pct|` is less than "
            f"{cfg.cost_gate_multiplier:.1f}× the round-trip cost shown per "
            f"candidate below will be dropped automatically. Don't pick "
            f"trades whose expected return doesn't meaningfully clear the "
            f"fee floor — they cost the strategy money even when the price "
            f"call is right.\n"
            f"\n"
            f"**Multi-day holds raise the bar.** The gate's threshold is "
            f"multiplied by `hold_days` — a 5-day pick must beat "
            f"5 × {cfg.cost_gate_multiplier:.1f}× round-trip cost. So if "
            f"the cost is 0.4%, a 1-day pick needs >{cfg.cost_gate_multiplier * 0.4:.1f}% "
            f"predicted return; a 5-day pick needs >{cfg.cost_gate_multiplier * 0.4 * 5:.1f}%. "
            f"Don't reach for a longer hold to get more conviction — pick "
            f"the shortest horizon the thesis allows.\n"
        )

        # Trading-cost awareness — strategies are routed through
        # Trading 212 (live) or shadowed against the same fee schedule
        # (paper). Costs are non-trivial for short-horizon trades and
        # should be subtracted from expected return when scoring picks.
        from trading_bot.tools.fees import FEE_SCHEDULE_BRIEF
        sections.append("## Trading costs (subtract from expected return)\n" + FEE_SCHEDULE_BRIEF)

        # Optional macro context — only injected when the strategy lists
        # get_macro_view in its tools list.
        tools_set = set(cfg.tools or [])
        if "get_macro_view" in tools_set:
            view = get_macro_view()
            if view:
                sections.append("## Current macro view (treat as the regime backdrop)\n" + view)

        # Optional daily news brief — injected when the strategy lists
        # get_daily_news_brief. Today's market-moving headlines themed
        # into a short markdown brief by the daily-news-brief agent.
        if "get_daily_news_brief" in tools_set:
            from trading_bot.tools.daily_news import get_daily_news_brief
            brief = get_daily_news_brief(on_date)
            if brief:
                sections.append("## Today's market news brief (themes from this morning's headlines)\n" + brief)

        # Universe-level / cross-asset snapshots — fetched once and rendered
        # compactly before the candidates. Each block is opt-in via cfg.tools.
        cross_asset_block = self._build_cross_asset_block(tools_set)
        if cross_asset_block:
            sections.append(cross_asset_block)

        # Phase 11G — self-P&L feedback. The strategy sees its 5 most-
        # recent trades with the LLM-written outcome / risks notes so
        # it can adjust today's call based on what it just learned —
        # without waiting for the weekly evolution loop.
        recent_trades_block = _build_recent_self_trades_block(cfg.id, cfg.region)
        if recent_trades_block:
            sections.append(recent_trades_block)

        # Phase 12G — show open multi-day positions so the LLM knows what
        # the strategy is already exposed to before adding new names.
        open_positions_block = _build_open_positions_block(cfg.id, cfg.region)
        if open_positions_block:
            sections.append(open_positions_block)

        sections.append("## Your strategy bias / approach\n" + deep_analysis_prompt.strip())
        sections.append("## Final selection instructions\n" + final_select_prompt.strip())

        # Per-candidate fundamentals data — only fetched if strategy lists them
        # in tools (so equity-only strategies don't pay the latency for tools
        # they'd ignore).
        per_candidate_earnings = self._maybe_fetch_earnings(tools_set, candidates)
        per_candidate_insiders = self._maybe_fetch_insiders(tools_set, candidates)
        per_candidate_filings = self._maybe_fetch_filings(tools_set, candidates)

        # Compute the typical position size once so the per-candidate cost
        # estimate is realistic for THIS strategy's sizing — not a generic
        # number the model has to convert from %.
        from trading_bot.tools.fees import (
            estimate_round_trip_cost_pct, infer_instrument_type,
            yf_ticker_classify,
        )
        typical_position_gbp = cfg.capital_gbp * (cfg.max_position_pct / 100.0)

        # Phase 10A — tickers we trailed out of recently. Re-picking
        # these pays the entry fees again (stamp duty especially on
        # UK shares), so we flag them inline AND the post-processor
        # doubles the cost gate for them.
        from trading_bot.state.trail_exits import load_recent_trail_exits
        trailed_recently = load_recent_trail_exits(days=3)
        if trailed_recently:
            trail_lines = []
            for tkr, recs in sorted(trailed_recently.items()):
                latest = max(recs, key=lambda r: r.exit_date)
                trail_lines.append(
                    f"- **{tkr}** — trailed out {latest.exit_date} "
                    f"({latest.pnl_pct:+.2f}%, by {latest.strategy_id})"
                )
            sections.append(
                "## Recently trailed out (re-entries pay full fees again)\n\n"
                + "\n".join(trail_lines)
                + "\n\nThe post-processor adds ONE extra round-trip cost on "
                  "top of the normal cost-gate threshold for these tickers "
                  "automatically (i.e. predicted return must beat "
                  "`cost_gate_multiplier × cost + cost`, not just "
                  "`cost_gate_multiplier × cost`). Only re-pick one if your "
                  "predicted return clears that bar AND you have a fresh "
                  "thesis — yesterday's setup losing money via stop today is "
                  "usually a reason to step away for a few sessions."
            )

        sections.append("## Candidates (pre-filtered to those in a healthy uptrend)\n")
        for c in candidates:
            ticker_news = news.get(c.ticker, [])
            news_lines = ""
            if ticker_news:
                # Cap to 5 most recent per ticker to keep the prompt bounded
                bullets = "\n".join(
                    f"  - [{item.timestamp[:10]}] {item.headline}" + (f" — {item.summary[:150]}" if item.summary else "")
                    for item in ticker_news[:5]
                )
                news_lines = f"\n  Recent news:\n{bullets}"

            earnings_line = ""
            ei = per_candidate_earnings.get(c.ticker) if per_candidate_earnings else None
            if ei and (ei.next_earnings_date or ei.last_surprise_pct is not None):
                bits = []
                if ei.next_earnings_date:
                    bits.append(f"next earnings: {ei.next_earnings_date}")
                if ei.last_surprise_pct is not None:
                    bits.append(f"last surprise: {ei.last_surprise_pct:+.1f}%")
                earnings_line = f"\n- earnings: {' · '.join(bits)}"

            insider_line = ""
            ins = per_candidate_insiders.get(c.ticker) if per_candidate_insiders else None
            if ins and (ins.n_buys + ins.n_sells > 0):
                insider_line = (
                    f"\n- insiders ({ins.lookback_days}d): "
                    f"{ins.n_buys} buys / {ins.n_sells} sells — **{ins.net_signal}**"
                )

            filings_lines = ""
            filings = per_candidate_filings.get(c.ticker) if per_candidate_filings else None
            if filings:
                bullets = []
                # Cap to 4 most recent so the prompt doesn't blow up
                for f in filings[:4]:
                    items = f" — items {', '.join(f.items)}" if f.items else ""
                    bullets.append(f"  - [{f.filing_date}] {f.form_type}{items}")
                    if f.excerpt:
                        # First ~250 chars only — full text is too long
                        bullets.append(f"    > {f.excerpt[:250]}...")
                filings_lines = "\n  Recent filings:\n" + "\n".join(bullets)

            exch, ccy = yf_ticker_classify(c.ticker)
            cost = estimate_round_trip_cost_pct(
                tier="trading212-paper", currency=ccy, exchange=exch,
                instrument_type=infer_instrument_type(c.ticker),
                notional_gbp=typical_position_gbp,
                quantity=typical_position_gbp / max(c.close, 1.0),
            )
            # Surface the actionable hurdle alongside the raw cost — the
            # LLM otherwise has to mentally combine the schedule, the
            # candidate's cost, cfg.cost_gate_multiplier, and hold_days
            # to know whether a pick has a chance. Show the 1-day and
            # 5-day return floors directly so it picks names that have
            # the headroom to clear them BEFORE writing predicted_return_pct.
            cost_pct = (cost.get("total_pct") or 0.0) * 100.0
            mult = float(cfg.cost_gate_multiplier or 2.0)
            hurdle_1d = cost_pct * mult
            hurdle_5d = cost_pct * mult * 5
            cost_line = (
                f"\n- round-trip cost: {cost['note']}"
                f"\n- minimum predicted_return_pct to clear cost gate: "
                f"**{hurdle_1d:+.2f}%** (1-day hold) / "
                f"**{hurdle_5d:+.2f}%** (5-day hold). "
                f"Picks below this floor will be auto-dropped."
            )

            # Phase 11F — relative strength line.
            rel_line = ""
            if c.rel_strength_5d is not None or c.rel_strength_20d is not None:
                bench = c.benchmark or "?"
                r5 = f"{c.rel_strength_5d:+.2f}%" if c.rel_strength_5d is not None else "?"
                r20 = f"{c.rel_strength_20d:+.2f}%" if c.rel_strength_20d is not None else "?"
                rel_line = f"\n- rel-strength vs {bench}: 5d {r5}, 20d {r20}"

            sections.append(
                f"### {c.ticker}\n"
                f"- close: ${_fmt(c.close, '.2f')} (as of {c.as_of})\n"
                f"- RSI(14): {_fmt(c.rsi_14, '.1f')}\n"
                f"- MACD line/signal/hist: {_fmt(c.macd_line, '+.3f')} / {_fmt(c.macd_signal, '+.3f')} / {_fmt(c.macd_histogram, '+.3f')}\n"
                f"- ATR(14): ${_fmt(c.atr_14, '.2f')}\n"
                f"- SMA20 ${_fmt(c.sma_20, '.2f')} (above: {c.above_sma_20}), SMA50 ${_fmt(c.sma_50, '.2f')} (above: {c.above_sma_50})\n"
                f"- 5-day return: {_fmt(c.return_5d_pct, '+.2f')}%, 20-day: {_fmt(c.return_20d_pct, '+.2f')}%\n"
                f"- volume ratio (today vs 20-day avg): {_fmt(c.volume_ratio, '.2f')}"
                + rel_line
                + cost_line
                + earnings_line
                + insider_line
                + filings_lines
                + news_lines
            )

        sections.append(
            "## Required output\n\n"
            "Return a **JSON object with two keys**: `predictions` (one entry per "
            "candidate above) and `picks` (the subset you want to trade today).\n\n"
            "### `predictions` — score EVERY candidate\n\n"
            "One object per ticker in the candidates section above. The point of "
            "this list is statistical model validation: we measure how well your "
            "ranking predicts realised returns end-of-day. **Score the full set** "
            "including names you're confident will fall or do nothing — getting "
            "those right is just as valuable as getting the rising ones right.\n\n"
            "```json\n"
            "{\n"
            "  \"ticker\": \"...\",\n"
            "  \"predicted_class\": \"strong_up\" | \"mild_up\" | \"flat\" | \"mild_down\" | \"strong_down\",\n"
            "  \"predicted_return_pct\": <float, your point estimate for today's intraday return>,\n"
            "  \"conviction\": <0.0-1.0>,\n"
            "  \"rationale\": \"<1 short sentence — the dominant signal driving the call>\"\n"
            "}\n"
            "```\n\n"
            "### `picks` — top names you want long today (subset of predictions)\n\n"
            "```json\n"
            "{\n"
            "  \"ticker\": \"...\",\n"
            "  \"allocation_pct\": <float>,\n"
            "  \"stop_loss_pct\": <float or null>,\n"
            "  \"take_profit_pct\": <float or null>,\n"
            "  \"hold_days\": 1 | 2 | 3 | 5 | 10,\n"
            "  \"thesis\": \"<1-2 sentence rationale>\"\n"
            "}\n"
            "```\n\n"
            f"Hard rules for picks: up to {cfg.max_positions} entries, allocations "
            f"sum to ≤ 100, each between (min_position_gbp/capital × 100) and "
            f"{cfg.max_position_pct}. Bracket-order constraint: `stop_loss_pct` < 0 "
            "and `take_profit_pct` > 0 if set, or both null. Cash is a valid position — "
            "return `picks: []` if nothing is compelling, but still score every "
            "candidate in `predictions`.\n\n"
            "**`hold_days` — pick the right horizon for the thesis.** "
            "Choose from {1, 2, 3, 5, 10} trading days. The cost gate scales "
            "linearly with horizon, so a 5-day pick must clear a 5× higher "
            "predicted-return bar than a 1-day pick — only ask for the longer "
            "hold if the thesis genuinely needs it. Guidance:\n"
            "- `1` — event-driven, earnings reaction, catalyst-priced-in-today, "
            "gap continuation. Default for momentum scalps.\n"
            "- `2-3` — multi-day momentum continuation, breakout confirmation, "
            "sector rotation that needs a couple of sessions to play out.\n"
            "- `5` — mean reversion to 20d SMA, macro-aligned overshoots, "
            "post-earnings drift, technical setups with stop wider than 1d "
            "noise allows.\n"
            "- `10` — strategic/positional plays where the thesis is structural "
            "(macro regime shift, central-bank pivot, commodity supercycle).\n"
            "If unsure, `1` is always safe. Stops + take-profits remain bracket "
            "orders that fire intra-day regardless of `hold_days`.\n\n"
            "Full required shape:\n\n"
            "```json\n"
            "{ \"predictions\": [...], \"picks\": [...] }\n"
            "```"
        )

        return "\n\n".join(sections)

    def _build_cross_asset_block(self, tools: set[str]) -> str:
        """Compact universe-level snapshot. Fetched once per pipeline run;
        each sub-block is opt-in via the strategy's tools list."""
        parts: list[str] = []

        if "get_yield_curve" in tools:
            try:
                yc = get_yield_curve()
                bits: list[str] = []
                if yc.y3m is not None: bits.append(f"3M={yc.y3m:.2f}%")
                if yc.y5y is not None: bits.append(f"5Y={yc.y5y:.2f}%")
                if yc.y10y is not None: bits.append(f"10Y={yc.y10y:.2f}%")
                if yc.y30y is not None: bits.append(f"30Y={yc.y30y:.2f}%")
                if yc.spread_3m10y is not None: bits.append(f"3m10y spread={yc.spread_3m10y:+.2f}bp")
                if bits:
                    parts.append(f"**Yield curve** (as of {yc.as_of}): " + " · ".join(bits))
            except Exception as e:
                log.warning("get_yield_curve failed: %s", e)

        if "get_credit_spreads" in tools:
            try:
                cs = get_credit_spreads()
                if cs.hyg_5d_return_pct is not None and cs.lqd_5d_return_pct is not None:
                    parts.append(
                        f"**Credit (5d)**: HYG {cs.hyg_5d_return_pct:+.2f}% vs LQD "
                        f"{cs.lqd_5d_return_pct:+.2f}% → HY-IG diff {cs.hy_vs_ig_5d_diff:+.2f}pp "
                        "(positive = risk-on, spreads tightening)"
                    )
            except Exception as e:
                log.warning("get_credit_spreads failed: %s", e)

        if "get_dollar_index" in tools:
            try:
                dxy = get_dollar_index()
                if dxy.level is not None:
                    parts.append(
                        f"**Dollar index** (as of {dxy.as_of}): {dxy.level:.2f} "
                        f"(5d {dxy.return_5d_pct:+.2f}%, 20d {dxy.return_20d_pct:+.2f}%)"
                    )
            except Exception as e:
                log.warning("get_dollar_index failed: %s", e)

        if "get_commodity_prices" in tools:
            try:
                commodities = get_commodity_prices()
                rows = []
                for c in commodities:
                    if c.close is None:
                        continue
                    rows.append(
                        f"  - {c.name} ({c.ticker}): ${c.close:.2f} "
                        f"(5d {c.return_5d_pct:+.2f}% / 20d {c.return_20d_pct:+.2f}%)"
                        if c.return_5d_pct is not None and c.return_20d_pct is not None
                        else f"  - {c.name} ({c.ticker}): ${c.close:.2f}"
                    )
                if rows:
                    parts.append("**Commodity prices**\n" + "\n".join(rows))
            except Exception as e:
                log.warning("get_commodity_prices failed: %s", e)

        if "get_sector_strength" in tools:
            try:
                sectors = get_sector_strength()
                if sectors:
                    rows = []
                    for i, s in enumerate(sectors, 1):
                        r5 = f"{s.return_5d_pct:+.2f}%" if s.return_5d_pct is not None else "—"
                        r20 = f"{s.return_20d_pct:+.2f}%" if s.return_20d_pct is not None else "—"
                        rows.append(f"  {i:2d}. {s.ticker} {s.label:<24s} 5d {r5}  20d {r20}")
                    parts.append("**Sector strength ranking** (best → worst by 5d)\n" + "\n".join(rows))
            except Exception as e:
                log.warning("get_sector_strength failed: %s", e)

        if not parts:
            return ""
        return "## Cross-asset & sector snapshot\n\n" + "\n\n".join(parts)

    def _maybe_fetch_earnings(self, tools: set[str], candidates: list) -> dict:
        if "get_earnings_info" not in tools:
            return {}
        out = {}
        for c in candidates:
            try:
                out[c.ticker] = get_earnings_info(c.ticker)
            except Exception as e:
                log.debug("get_earnings_info(%s) failed: %s", c.ticker, e)
        return out

    def _persist_earnings_gate(self, sid: str, on_date: date, dropped: int, total: int) -> None:
        """Phase 10B — log earnings-gate stats per (date, strategy, region).
        Filename includes region so a strategy running in both US and
        UK-EU on the same day doesn't overwrite the first region's row."""
        import json as _json
        from trading_bot.state.paths import STATE_ROOT
        d = STATE_ROOT / "earnings_gate"
        d.mkdir(parents=True, exist_ok=True)
        region = (self.config.region or "?").replace("/", "-")
        p = d / f"{on_date.isoformat()}.{sid}.{region}.json"
        p.write_text(_json.dumps({
            "strategy_id": sid, "region": region, "date": on_date.isoformat(),
            "candidates_dropped": dropped, "candidates_total": total,
        }))

    def _filter_pre_earnings(self, candidates: list, *, on_date: date, window_days: int) -> list:
        """Drop candidates whose next earnings date falls inside the
        window. Earnings data lookup is best-effort — if yfinance
        returns nothing we keep the candidate (better to over-include
        than over-exclude when data quality is the only obstacle)."""
        from datetime import timedelta
        cutoff = on_date + timedelta(days=window_days)
        kept = []
        dropped = []
        for c in candidates:
            try:
                info = get_earnings_info(c.ticker)
            except Exception as e:
                log.debug("earnings-gate: %s lookup failed (%s) — keeping", c.ticker, e)
                kept.append(c)
                continue
            next_iso = info.next_earnings_date if info else None
            if not next_iso:
                kept.append(c)
                continue
            try:
                next_date = date.fromisoformat(next_iso[:10])
            except ValueError:
                kept.append(c)
                continue
            if on_date <= next_date <= cutoff:
                dropped.append(f"{c.ticker}({next_iso[:10]})")
                continue
            kept.append(c)
        if dropped:
            log.info("earnings-gate dropped: %s", ", ".join(dropped))
        return kept

    def _maybe_fetch_insiders(self, tools: set[str], candidates: list) -> dict:
        if "get_insider_trades" not in tools:
            return {}
        out = {}
        for c in candidates:
            try:
                out[c.ticker] = get_insider_trades(c.ticker, days=60)
            except Exception as e:
                log.debug("get_insider_trades(%s) failed: %s", c.ticker, e)
        return out

    def _maybe_fetch_filings(self, tools: set[str], candidates: list) -> dict:
        if "get_filing_summary" not in tools:
            return {}
        # Only the top-15 candidates get filings — they're the most expensive
        # tool (one EDGAR call per ticker, plus potentially body fetches for
        # 8-Ks). Saves ~half the per-run EDGAR traffic.
        top = candidates[:15]
        out = {}
        for c in top:
            try:
                out[c.ticker] = get_filing_summary(c.ticker, days=30)
            except Exception as e:
                log.debug("get_filing_summary(%s) failed: %s", c.ticker, e)
        return out

    def _log_predictions(
        self,
        candidates: list,
        *,
        llm_predictions: dict[str, dict],
        picked_intents: list[TradeIntent],
        on_date: date,
    ) -> None:
        """Write a PredictionRecord for every pre-filtered candidate.

        Class / predicted_return_pct / conviction / rationale come from the
        LLM's `predictions` block. If a candidate is missing from the LLM
        response (parse failure, truncated output), we fall back to a
        technical-heuristic classification so the row still gets logged.

        actual_return_pct is filled by meta.reflection.grade_predictions at
        the exit step.
        """
        cfg = self.config
        picked_tickers = {i.ticker for i in picked_intents}

        for c in candidates:
            ticker = c.ticker
            llm_pred = llm_predictions.get(ticker)
            if llm_pred is not None:
                predicted_class = str(llm_pred.get("predicted_class", "flat"))
                predicted_pct = _safe_float(llm_pred.get("predicted_return_pct"), default=0.0)
                conviction = _safe_float(llm_pred.get("conviction"), default=0.5)
                rationale = str(llm_pred.get("rationale", "")).strip() or "(no rationale)"
            else:
                # Fallback: candidate not scored by LLM (truncated response etc.)
                predicted_pct = c.return_5d_pct if c.return_5d_pct is not None else 0.0
                predicted_class = _classify(predicted_pct, c.rsi_14)
                conviction = 0.3  # low — LLM didn't engage with this name
                rationale = "Not scored by LLM; fallback technical heuristic."

            was_traded = ticker in picked_tickers
            # Capture which tools were active for this run so the
            # attribution layer can later compute IC by tool-combination.
            # Sorted for stable equality checks downstream.
            tools_used = sorted(set(cfg.tools or []))
            append_prediction(
                PredictionRecord(
                    strategy_id=cfg.id,
                    region=cfg.region,
                    prediction_date=on_date.isoformat(),
                    ticker=ticker,
                    predicted_class=predicted_class,
                    predicted_return_pct=round(float(predicted_pct), 2),
                    conviction=round(float(conviction), 2),
                    rationale=rationale,
                    was_traded=was_traded,
                    tools_used=tools_used,
                    prefilter_mode=(cfg.prefilter_mode or "").lower().strip(),
                )
            )

    def _extract_predictions_and_picks(self, response) -> tuple[dict[str, dict], list]:
        """Pull `predictions` (keyed by ticker) + `picks` (list) out of the LLM
        response. Tolerates the legacy shape (response = a bare list of picks)
        as a fallback so old strategy outputs don't break."""
        if isinstance(response, dict):
            preds_raw = response.get("predictions") or []
            picks_raw = response.get("picks") or []
        elif isinstance(response, list):
            preds_raw = []
            picks_raw = response
        else:
            return {}, []

        predictions: dict[str, dict] = {}
        if isinstance(preds_raw, list):
            for entry in preds_raw:
                if not isinstance(entry, dict):
                    continue
                ticker = entry.get("ticker")
                if not ticker:
                    continue
                predictions[str(ticker).upper()] = entry

        if not isinstance(picks_raw, list):
            picks_raw = []
        return predictions, picks_raw

    def _parse_picks(self, response) -> list[TradeIntent]:
        cfg = self.config
        if not isinstance(response, list):
            log.error("%s: LLM response was not a JSON array — got %s", cfg.id, type(response).__name__)
            return []

        intents: list[TradeIntent] = []
        for item in response:
            if not isinstance(item, dict):
                continue
            ticker = item.get("ticker")
            alloc = item.get("allocation_pct")
            if not ticker or alloc is None:
                log.warning("%s: skipping malformed pick: %s", cfg.id, item)
                continue
            try:
                alloc_f = float(alloc)
            except (TypeError, ValueError):
                continue
            if alloc_f <= 0 or alloc_f > cfg.max_position_pct:
                log.warning(
                    "%s: %s allocation %.1f%% outside [0, %.1f%%] — skipping",
                    cfg.id, ticker, alloc_f, cfg.max_position_pct,
                )
                continue

            stop = item.get("stop_loss_pct")
            tp = item.get("take_profit_pct")

            # Phase 12B — hold_days must be in the whitelist or we fall back
            # to 1 (Wave 1 same-day behaviour). The whitelist is small so the
            # LLM can't pick e.g. 17 days, which would slip past every cost-
            # gate calibration we've done.
            hold_raw = item.get("hold_days")
            hold_days = 1
            if hold_raw is not None:
                try:
                    hold_int = int(float(hold_raw))
                except (TypeError, ValueError):
                    hold_int = 1
                if hold_int in (1, 2, 3, 5, 10):
                    hold_days = hold_int
                else:
                    log.warning(
                        "%s: %s hold_days=%s not in {1,2,3,5,10} — defaulting to 1",
                        cfg.id, ticker, hold_raw,
                    )

            intents.append(
                TradeIntent(
                    ticker=str(ticker).upper(),
                    allocation_pct=alloc_f,
                    stop_loss_pct=float(stop) if stop is not None else None,
                    take_profit_pct=float(tp) if tp is not None else None,
                    thesis=str(item.get("thesis") or ""),
                    hold_days=hold_days,
                )
            )

        # Apply the max_positions cap defensively
        intents = intents[: cfg.max_positions]
        if intents:
            from collections import Counter
            horizons = Counter(i.hold_days for i in intents)
            horizon_summary = ", ".join(
                f"{n}×{d}d" for d, n in sorted(horizons.items())
            )
            log.info(
                "%s: parsed %d picks from LLM response (horizons: %s)",
                cfg.id, len(intents), horizon_summary,
            )
        else:
            log.info("%s: parsed 0 picks from LLM response", cfg.id)
        return intents
