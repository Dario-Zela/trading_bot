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


def _safe_float(val, *, default: float) -> float:
    try:
        f = float(val)
        if f != f:  # NaN
            return default
        return f
    except (TypeError, ValueError):
        return default


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

        # Pre-filter the universe with cheap Python so we don't pay tokens for
        # candidates that obviously don't fit. Same filter style as the stub.
        tickers = get_universe(cfg.universe)
        log.info("%s: scoring %d universe candidates", cfg.id, len(tickers))
        techs = get_technicals(tickers, end_date=on_date)
        candidates = self._prefilter(techs)
        log.info("%s: %d candidates passed pre-filter", cfg.id, len(candidates))
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
            "## Required output\n\n"
            "Score every candidate above with a 5-class prediction. Return JSON only:\n\n"
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
        """Minimal pre-filter that preserves directional diversity.

        Keeps any ticker with usable technicals (rsi_14 + return_5d_pct
        both computable). Ranks by ABS(5-day return) so the LLM sees the
        biggest movers up AND down, letting it score across rising,
        falling, and flat regimes rather than only winners. Liquidity is
        enforced via universe selection upstream, not here — a today-vs-
        avg volume ratio is unreliable mid-session when today's bar only
        carries partial volume.
        """
        keep = []
        for ticker, t in techs.items():
            if t.rsi_14 is None or t.return_5d_pct is None:
                continue
            keep.append(t)
        # Sort by absolute magnitude of recent move — picks up both up- and
        # down-movers symmetrically.
        keep.sort(key=lambda x: abs(x.return_5d_pct), reverse=True)
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
            f"- use_stops: {cfg.use_stops}\n"
            f"- use_take_profits: {cfg.use_take_profits}\n"
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
            estimate_round_trip_cost_pct, yf_ticker_classify,
        )
        typical_position_gbp = cfg.capital_gbp * (cfg.max_position_pct / 100.0)

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
                instrument_type="share",
                notional_gbp=typical_position_gbp,
                quantity=typical_position_gbp / max(c.close, 1.0),
            )
            cost_line = f"\n- round-trip cost: {cost['note']}"

            sections.append(
                f"### {c.ticker}\n"
                f"- close: ${c.close:.2f} (as of {c.as_of})\n"
                f"- RSI(14): {c.rsi_14:.1f}\n"
                f"- MACD line/signal/hist: {c.macd_line:+.3f} / {c.macd_signal:+.3f} / {c.macd_histogram:+.3f}\n"
                f"- ATR(14): ${c.atr_14:.2f}\n"
                f"- SMA20 ${c.sma_20:.2f} (above: {c.above_sma_20}), SMA50 ${c.sma_50:.2f} (above: {c.above_sma_50})\n"
                f"- 5-day return: {c.return_5d_pct:+.2f}%, 20-day: {c.return_20d_pct:+.2f}%\n"
                f"- volume ratio (today vs 20-day avg): {c.volume_ratio:.2f}"
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
            "  \"thesis\": \"<1-2 sentence rationale>\"\n"
            "}\n"
            "```\n\n"
            f"Hard rules for picks: up to {cfg.max_positions} entries, allocations "
            f"sum to ≤ 100, each between (min_position_gbp/capital × 100) and "
            f"{cfg.max_position_pct}. Bracket-order constraint: `stop_loss_pct` < 0 "
            "and `take_profit_pct` > 0 if set, or both null. Cash is a valid position — "
            "return `picks: []` if nothing is compelling, but still score every "
            "candidate in `predictions`.\n\n"
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
            intents.append(
                TradeIntent(
                    ticker=str(ticker).upper(),
                    allocation_pct=alloc_f,
                    stop_loss_pct=float(stop) if stop is not None else None,
                    take_profit_pct=float(tp) if tp is not None else None,
                    thesis=str(item.get("thesis") or ""),
                )
            )

        # Apply the max_positions cap defensively
        intents = intents[: cfg.max_positions]
        log.info("%s: parsed %d picks from LLM response", cfg.id, len(intents))
        return intents
