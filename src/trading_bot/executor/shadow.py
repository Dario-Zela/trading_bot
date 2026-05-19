from __future__ import annotations

import logging
import uuid
from datetime import date

log = logging.getLogger(__name__)

from trading_bot.executor.base import Executor, TradeIntent
from trading_bot.state import (
    TradeRecord,
    append_trade,
    mark_trade_exited,
    read_open_trades,
)
from trading_bot.state.ledger import filter_due_for_exit
from trading_bot.tools import get_history
from trading_bot.tools.fees import TradeContext, compute_fees, yf_ticker_classify
from trading_bot.tools.fx import to_gbp_multiplier


_TIER = "shadow"


class ShadowExecutor(Executor):
    """Records trades on paper using real market prices. No orders placed anywhere.

    Entry: looks up the most recent close as the assumed fill price (Wave 1 simplification —
    we'll switch to the day's open price once we run intraday and have access to that bar).
    Exit: looks up the close on the exit date, computes P&L.
    Stops / take-profits: simulated at end-of-day by checking the intra-day high/low against
    the trigger prices and recording the trigger fill if hit.
    """

    def enter(
        self,
        intents: list[TradeIntent],
        *,
        strategy_id: str,
        region: str,
        capital_gbp: float,
        on_date: date,
    ) -> None:
        if not intents:
            return

        tickers = [i.ticker for i in intents]
        # Phase 11A — pull a longer history so we can compute a 20-day
        # average and catch anomalous prints. Earlier we only fetched
        # 2 days which gave the sanity check no basis to judge.
        history = get_history(tickers, lookback_days=22, end_date=on_date)

        from trading_bot.tools.price_sanity import is_price_anomalous

        for intent in intents:
            bars = history.get(intent.ticker)
            if not bars:
                # No price data available; skip this intent. Wave 1 doesn't have
                # a richer fallback — we just don't record the trade.
                continue
            entry_price = bars[-1].close

            # Phase 11A — drop on anomalous prices. Catches the
            # SNDK-at-$1407 case where yfinance returned a split-
            # not-adjusted close so the shadow filled at a fictional
            # notional 30× the real one.
            bad, reason = is_price_anomalous(close=entry_price, bars=bars)
            if bad:
                log.warning(
                    "%s: skipping shadow entry on %s — %s",
                    strategy_id, intent.ticker, reason,
                )
                continue
            allocation_gbp = capital_gbp * (intent.allocation_pct / 100.0)
            quantity = allocation_gbp / entry_price if entry_price > 0 else 0.0

            exch, ccy = yf_ticker_classify(intent.ticker)
            # Phase 12A — compute target_exit_date from hold_days.
            from trading_bot.tools.calendar import add_trading_days
            hold_days = max(1, int(intent.hold_days))
            target_exit = add_trading_days(on_date, hold_days, region)
            record = TradeRecord(
                trade_id=str(uuid.uuid4()),
                strategy_id=strategy_id,
                region=region,
                tier=_TIER,
                ticker=intent.ticker,
                side="long",
                entry_date=on_date.isoformat(),
                entry_price=entry_price,
                quantity=quantity,
                allocation_pct=intent.allocation_pct,
                stop_loss_pct=intent.stop_loss_pct,
                take_profit_pct=intent.take_profit_pct,
                thesis=intent.thesis,
                currency=ccy,
                exchange=exch,
                instrument_type="share",
                hold_days=hold_days,
                target_exit_date=target_exit.isoformat(),
            )
            append_trade(record)

    def exit_scheduled(
        self,
        *,
        strategy_id: str,
        region: str,
        on_date: date,
    ) -> list[dict]:
        # Sweep ALL open trades for this strategy+region, not just today's.
        # If a prior session missed its exit (holiday, workflow failure, etc),
        # those trades sit stranded with exit_date=None. We close them here
        # at today's yfinance close — small mark-to-market drift but no
        # silent strand.
        open_trades = read_open_trades(strategy_id=strategy_id, region=region)
        if not open_trades:
            return []

        # Phase 12A — only close trades whose target_exit_date is
        # on-or-before today. Legacy rows with no target exit today
        # (matches Wave 1 same-day round-trip behaviour).
        due_trades = filter_due_for_exit(open_trades, on_date)
        held_over = len(open_trades) - len(due_trades)
        if held_over:
            log.info(
                "%s/%s: %d positions still within hold window — leaving open",
                strategy_id, region, held_over,
            )
        if not due_trades:
            return []
        open_trades = due_trades

        tickers = list({t["ticker"] for t in open_trades})
        history = get_history(tickers, lookback_days=1, end_date=on_date)

        closed: list[dict] = []
        for trade in open_trades:
            bars = history.get(trade["ticker"])
            if not bars:
                continue
            bar = bars[-1]
            entry_price = float(trade["entry_price"])
            quantity = float(trade["quantity"])

            exit_price, exit_reason = _resolve_exit(
                entry_price=entry_price,
                day_high=bar.high,
                day_low=bar.low,
                day_close=bar.close,
                stop_loss_pct=trade.get("stop_loss_pct"),
                take_profit_pct=trade.get("take_profit_pct"),
            )

            # Shadow mirrors whichever live broker we'd use for this
            # region. yfinance prices are in the instrument's native
            # currency — except for LSE listings where yfinance returns
            # pence-corrected pounds (so .L tickers are GBP-priced by the
            # time we see them, no conversion needed). Convert to GBP at
            # spot for everything else.
            native_ccy = (trade.get("currency") or "USD").upper()
            exch = trade.get("exchange") or yf_ticker_classify(trade["ticker"])[0]
            if native_ccy == "GBP":
                native_to_gbp = 1.0
            else:
                native_to_gbp = to_gbp_multiplier(native_ccy) or 1.0
            gross_pnl_gbp = (exit_price - entry_price) * quantity * native_to_gbp
            entry_notional_gbp = abs(entry_price * quantity * native_to_gbp)
            exit_notional_gbp = abs(exit_price * quantity * native_to_gbp)
            fees = compute_fees(TradeContext(
                tier=_TIER, currency=native_ccy, exchange=exch,
                instrument_type=trade.get("instrument_type") or "share",
                entry_notional_gbp=entry_notional_gbp,
                exit_notional_gbp=exit_notional_gbp,
                quantity=abs(quantity),
            ))
            fees_gbp = fees.total_gbp
            fees_breakdown = fees.as_dict()
            pnl_gbp = gross_pnl_gbp - fees_gbp
            # pnl_pct is the raw price-side return; fees do not factor in
            # here because the dashboard already shows fees as a separate
            # line and we want the % to mean "the price moved by this".
            pnl_pct = (exit_price / entry_price - 1.0) * 100.0 if entry_price > 0 else 0.0

            outcome_notes, risks_observed = _templated_reflection(
                pnl_pct=pnl_pct, exit_reason=exit_reason, ticker=trade["ticker"]
            )

            mark_trade_exited(
                trade_id=trade["trade_id"],
                exit_date=on_date,
                exit_price=exit_price,
                pnl_gbp=pnl_gbp,
                pnl_pct=pnl_pct,
                exit_reason=exit_reason,
                outcome_notes=outcome_notes,
                risks_observed=risks_observed,
                fees_gbp=fees_gbp,
                fees_breakdown=fees_breakdown,
            )

            closed_row = {
                **trade,
                "exit_date": on_date.isoformat(),
                "exit_price": exit_price,
                "pnl_gbp": pnl_gbp,
                "pnl_pct": pnl_pct,
                "exit_reason": exit_reason,
                "outcome_notes": outcome_notes,
                "risks_observed": risks_observed,
                "fees_gbp": fees_gbp,
                "fees_breakdown": fees_breakdown,
            }
            closed.append(closed_row)
            # Phase 10A — track stop-driven exits for the cost gate
            try:
                from trading_bot.state.trail_exits import append_trail_exit
                append_trail_exit(closed_row)
            except Exception:
                pass
        return closed


def _templated_reflection(
    *, pnl_pct: float, exit_reason: str, ticker: str
) -> tuple[str, str]:
    """Templated outcome + risks text for Wave 1's rule-based control strategy.

    Wave 6's LLM reflection agent replaces this with real per-trade analysis
    that draws on the strategy's full reasoning chain and the day's market
    context. Keeping the templated path because it makes Wave 1 emails useful
    out of the box.
    """
    if exit_reason == "take_profit":
        outcome = (
            f"Take-profit hit at +{pnl_pct:.2f}%. Momentum thesis held — the prior "
            f"session's buying pressure carried through and {ticker} reached the target."
        )
    elif exit_reason == "stop":
        outcome = (
            f"Stopped out at {pnl_pct:+.2f}%. The setup reversed against us early; "
            f"the bracket stop limited the damage to the configured floor."
        )
    elif pnl_pct > 0.5:
        outcome = (
            f"Closed at {pnl_pct:+.2f}%. Follow-through on the previous-day strength held "
            f"into the close — clean momentum continuation."
        )
    elif pnl_pct < -0.5:
        outcome = (
            f"Closed at {pnl_pct:+.2f}%. Previous-day strength faded — likely profit-taking "
            f"or rotation out of the name. No catalyst either way; pure technical fade."
        )
    else:
        outcome = (
            f"Closed near flat at {pnl_pct:+.2f}%. Drifted sideways through the session — "
            f"no follow-through but no meaningful reversal either."
        )

    if pnl_pct > 0:
        risks = (
            "Rule-based control has no fundamental or news filter. Today's win is "
            "consistent with the strategy's design; watch concentration and sector "
            "balance across the basket."
        )
    else:
        risks = (
            "The rule-based control can't distinguish 'rallying on solid fundamentals' "
            "from 'rallying on a squeeze about to unwind.' This loss validates the case "
            "for news-aware and mean-reverter strategies that would have filtered it."
        )

    return outcome, risks


def _resolve_exit(
    *,
    entry_price: float,
    day_high: float,
    day_low: float,
    day_close: float,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
) -> tuple[float, str]:
    """Decide the exit fill price for a single position given the day's range.

    Conservative simulation:
    - If both stop and take-profit could have triggered intra-day, assume the worst
      (stop fires first). This avoids over-stating shadow P&L.
    - If only one triggered, that's the exit.
    - Otherwise, exit at the day's close.
    """
    stop_price = entry_price * (1 + stop_loss_pct / 100.0) if stop_loss_pct is not None else None
    target_price = (
        entry_price * (1 + take_profit_pct / 100.0) if take_profit_pct is not None else None
    )

    stop_hit = stop_price is not None and day_low <= stop_price
    target_hit = target_price is not None and day_high >= target_price

    if stop_hit and target_hit:
        return stop_price, "stop"  # type: ignore[return-value]
    if stop_hit:
        return stop_price, "stop"  # type: ignore[return-value]
    if target_hit:
        return target_price, "take_profit"  # type: ignore[return-value]
    return day_close, "scheduled"
