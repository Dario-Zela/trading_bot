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
from trading_bot.strategy.base import Strategy
from trading_bot.tools import get_recent_news, get_technicals, get_universe


log = logging.getLogger(__name__)


# Pre-filter parameters — chosen to give Claude a meaningful but bounded
# universe to reason over. Adjustable per-strategy in a later refactor.
_PREFILTER_RSI_MIN = 30.0
_PREFILTER_RSI_MAX = 80.0
_PREFILTER_MIN_VOL_RATIO = 0.5
_PREFILTER_TOP_N = 30  # how many candidates to hand to the LLM


class LLMStrategy(Strategy):
    """Strategy backed by Claude Code in single-call mode."""

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

        prompt = self._build_prompt(
            on_date=on_date,
            candidates=candidates,
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
            log.error("%s: LLM call failed: %s", cfg.id, e)
            return []

        return self._parse_picks(response)

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
        keep = []
        for ticker, t in techs.items():
            if t.rsi_14 is None or t.return_5d_pct is None or t.above_sma_20 is None:
                continue
            if not t.above_sma_20:
                continue
            if not (_PREFILTER_RSI_MIN <= t.rsi_14 <= _PREFILTER_RSI_MAX):
                continue
            if t.volume_ratio is not None and t.volume_ratio < _PREFILTER_MIN_VOL_RATIO:
                continue
            keep.append(t)
        keep.sort(key=lambda x: x.return_5d_pct, reverse=True)
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

        sections.append("## Your strategy bias / approach\n" + deep_analysis_prompt.strip())
        sections.append("## Final selection instructions\n" + final_select_prompt.strip())

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
            sections.append(
                f"### {c.ticker}\n"
                f"- close: ${c.close:.2f} (as of {c.as_of})\n"
                f"- RSI(14): {c.rsi_14:.1f}\n"
                f"- MACD line/signal/hist: {c.macd_line:+.3f} / {c.macd_signal:+.3f} / {c.macd_histogram:+.3f}\n"
                f"- ATR(14): ${c.atr_14:.2f}\n"
                f"- SMA20 ${c.sma_20:.2f} (above: {c.above_sma_20}), SMA50 ${c.sma_50:.2f} (above: {c.above_sma_50})\n"
                f"- 5-day return: {c.return_5d_pct:+.2f}%, 20-day: {c.return_20d_pct:+.2f}%\n"
                f"- volume ratio (today vs 20-day avg): {c.volume_ratio:.2f}"
                + news_lines
            )

        sections.append(
            "## Required output format\n\n"
            "Return a JSON array — one object per pick. If nothing is compelling today, "
            "return an empty array `[]`. **Do not** include explanations outside the JSON.\n\n"
            "```json\n"
            "[\n"
            "  {\n"
            "    \"ticker\": \"AAPL\",\n"
            "    \"allocation_pct\": 25.0,\n"
            "    \"stop_loss_pct\": -3.0,\n"
            "    \"take_profit_pct\": 5.0,\n"
            "    \"thesis\": \"1-2 sentence rationale citing the specific technicals or news that justified this pick.\"\n"
            "  }\n"
            "]\n"
            "```\n\n"
            f"Allocations must sum to ≤ 100. Up to {cfg.max_positions} picks. "
            f"Each `allocation_pct` between (min_position_gbp/capital × 100) and {cfg.max_position_pct}. "
            "Bracket-order constraint: `stop_loss_pct` < 0 and `take_profit_pct` > 0 if set, "
            "or both null for a plain market order without protection."
        )

        return "\n\n".join(sections)

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
