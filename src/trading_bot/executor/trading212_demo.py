"""Tier 1.5 executor — places market orders on a Trading212 Practice account.

Mirrors AlpacaPaperExecutor's shape but targets T212's demo API. T212 doesn't
support native bracket orders the same way Alpaca does, and the bot does
same-day round-trips anyway, so this executor simply:
- enter(): submit a market BUY for each intent
- exit_scheduled(): market SELL every position opened today by this strategy
- clear_slot(): cancel all pending orders + sell all positions

Fill prices come exclusively from T212. After submitting an order we poll
`GET /equity/orders/{id}` until the order leaves the active list, then
fall through to `GET /equity/history/orders` for the definitive fill.
yfinance is only ever a sizing seed (qty = allocation / yfinance_close);
recorded entry/exit prices and P&L come from the broker.

Intraday stop / take-profit are not enforced here — the daily exit cron
handles closure. If a stop/take-profit is configured on the strategy, it
acts only as a sizing hint and is logged in the ledger; the actual fill is
whatever T212 returns at market.

Ticker handling: T212 uses its own instrument tickers (`VOD_LON_EQ`) that
differ from yfinance (`VOD.L`). Translation is via
trading_bot.tools.t212_instruments.Translator, which caches the full
instrument list locally.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import date
from typing import Any

import requests

from trading_bot.executor.base import Executor, TradeIntent
from trading_bot.state import TradeRecord, append_trade, mark_trade_exited, read_open_trades
from trading_bot.state.ledger import filter_due_for_exit
from trading_bot.t212_slot import T212_PAPER_BUDGET_GBP, T212Creds, load_slot_creds
from trading_bot.tools.fees import (
    TradeContext,
    compute_fees,
    t212_exchange_from_ticker,
    t212_instrument_type,
)
from trading_bot.tools.fx import convert_from_gbp, to_gbp_multiplier
from trading_bot.tools.t212_instruments import Translator, fetch_instruments


_TIER = "trading212-paper"
log = logging.getLogger(__name__)


def _history_status(item: dict[str, Any]) -> str:
    """T212 history items nest the order under `order` with its own
    `status`. Submit / active-orders endpoints return the order fields
    at top-level. Handle both shapes so callers don't need to care
    which endpoint they came from."""
    if not isinstance(item, dict):
        return ""
    inner = item.get("order") if isinstance(item.get("order"), dict) else item
    return (inner.get("status") or "").lower()


class Trading212DemoExecutor(Executor):
    def __init__(self, slot: int):
        self.slot = slot
        self.creds: T212Creds = load_slot_creds(slot, demo=True)
        # Lazy — only built when we actually need to place/close orders
        self._translator: Translator | None = None

    # ---- core executor API -------------------------------------------------

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

        translator = self._get_translator()
        free_cash = self._get_free_cash()
        if free_cash is None:
            # Don't proceed uncapped — a transient free-cash read failure must
            # not let us blow the shared £50k T212 paper budget. Fall back to a
            # conservative ledger-derived figure: the cap minus the GBP entry
            # cost of every open T212-paper position (across strategies, since
            # they share one budget). Under-counts pending entries (sentinel
            # price 0), which only makes it more conservative on the downside.
            committed = sum(
                float(t.get("entry_price") or 0) * float(t.get("quantity") or 0)
                for t in read_open_trades(tier=_TIER)
            )
            free_cash = max(0.0, T212_PAPER_BUDGET_GBP - committed)
            log.warning(
                "Could not read T212 free cash — using ledger-derived budget "
                "£%.2f (cap £%.0f − committed £%.2f)",
                free_cash, T212_PAPER_BUDGET_GBP, committed,
            )
        else:
            log.info(
                "T212 slot %d free cash: £%.2f (account cap £%.0f)",
                self.slot, free_cash, T212_PAPER_BUDGET_GBP,
            )

        for intent in intents:
            t212_ticker = translator.translate(intent.ticker)
            if t212_ticker is None:
                log.warning(
                    "No T212 instrument match for %s — skipping (yfinance ticker has no T212 equivalent or isn't ISA-eligible)",
                    intent.ticker,
                )
                continue

            inst = translator.get_instrument(t212_ticker) or {}
            min_qty = float(inst.get("minTradeQuantity") or 0.01)

            # Estimate fill price via yfinance (T212 doesn't expose a snapshot
            # endpoint to non-pro retail accounts). The actual fill price comes
            # back from T212 after submission.
            entry_estimate = self._estimate_price(intent.ticker)
            if entry_estimate is None or entry_estimate <= 0:
                log.warning("No usable entry price for %s — skipping", intent.ticker)
                continue

            allocation_gbp = capital_gbp * (intent.allocation_pct / 100.0)
            # T212 quotes each instrument in its listing currency. yfinance
            # returns LSE bars pence-corrected to £ (history.py) but EUR/USD
            # bars unchanged. To size correctly across currencies we convert
            # the £ allocation into the instrument's native unit before
            # dividing — otherwise we under-allocate on EUR stocks by the
            # FX rate (~15% on EURGBP) and bizarre on USD.
            ccy = (inst.get("currencyCode") or "GBP").upper()
            if ccy == "GBX":
                # yfinance entry_estimate is already in £ but T212 wants the
                # quantity, which is just shares; £ / £-per-share works.
                allocation_native = allocation_gbp
            elif ccy == "GBP":
                allocation_native = allocation_gbp
            else:
                allocation_native = convert_from_gbp(allocation_gbp, ccy)
                if allocation_native is None:
                    log.warning(
                        "Skipping %s: FX rate unavailable for %s — can't size %s position safely",
                        intent.ticker, ccy, ccy,
                    )
                    continue
            # T212 rejects fractional quantities for most non-US instruments
            # with `quantity-precision-mismatch` (only some allow fractional,
            # and the instrument metadata endpoint doesn't expose the per-
            # ticker precision). Whole shares are universally accepted, so
            # we floor to int across the board. Small allocations on
            # expensive stocks may round to zero — caught by the min_qty
            # check below.
            quantity = float(int(allocation_native / entry_estimate))
            if quantity < max(min_qty, 1.0):
                log.warning(
                    "Computed qty %.0f below T212 minimum (%.2f shares ~£%.2f) for %s — skipping",
                    quantity, min_qty, allocation_gbp, intent.ticker,
                )
                continue

            # Budget pre-flight: T212 demo accounts cap at £50k. Skip rather
            # than letting T212 reject downstream — we want clean logs and
            # a deterministic in-flight state.
            if free_cash is not None and allocation_gbp > free_cash:
                log.warning(
                    "Skipping %s: would allocate £%.2f but T212 slot %d only has £%.2f free "
                    "(account cap £%.0f). Lower capital_gbp or rotate slots.",
                    intent.ticker, allocation_gbp, self.slot, free_cash, T212_PAPER_BUDGET_GBP,
                )
                continue
            if free_cash is not None:
                free_cash -= allocation_gbp  # local decrement so subsequent intents see updated budget

            # Phase 12A — multi-day positioning
            from trading_bot.tools.calendar import add_trading_days
            hold_days = max(1, int(intent.hold_days))
            # hold_days = total trading days the position is held.
            # hold_days=1 → exit today's close (no overnight).
            # hold_days=2 → one overnight, exit next trading day. Etc.
            target_exit = add_trading_days(on_date, hold_days - 1, region)

            client_order_id = f"{strategy_id}-{uuid.uuid4().hex[:12]}"
            order = self._submit_market_order(
                t212_ticker=t212_ticker, quantity=quantity
            )
            if order is None:
                continue

            broker_order_id = str(order.get("id")) if order.get("id") is not None else None

            # Poll for the definitive fill. T212's POST response is usually
            # an acknowledgment; the real fill price arrives once the order
            # transitions to FILLED, which we read from the active-order
            # endpoint and then from history if it's already moved.
            filled = self._wait_for_fill(order_id=broker_order_id, t212_ticker=t212_ticker)

            if filled is not None:
                raw_fill = self._fill_price_of(filled)
                fill_price = self._to_gbp(t212_ticker, raw_fill) if raw_fill is not None else None
            else:
                fill_price = None

            if fill_price is None:
                # Order was submitted (we have the broker_order_id) but the
                # fill didn't surface within timeout. Record as PENDING so
                # exit_scheduled can reconcile via T212 order history later
                # — better than orphaning the position at T212.
                log.warning(
                    "T212 order %s for %s didn't fill within poll timeout — recording as pending. "
                    "Exit will reconcile entry_price from T212 order history.",
                    broker_order_id, intent.ticker,
                )
                record = TradeRecord(
                    trade_id=client_order_id,
                    strategy_id=strategy_id,
                    region=region,
                    tier=_TIER,
                    ticker=intent.ticker,
                    side="long",
                    entry_date=on_date.isoformat(),
                    entry_price=0.0,  # sentinel — reconciled at exit
                    quantity=quantity,
                    allocation_pct=intent.allocation_pct,
                    stop_loss_pct=intent.stop_loss_pct,
                    take_profit_pct=intent.take_profit_pct,
                    thesis=intent.thesis,
                    broker_order_id=broker_order_id,
                    currency=ccy,
                    exchange=t212_exchange_from_ticker(t212_ticker),
                    instrument_type=t212_instrument_type(inst.get("type", "")),
                    hold_days=hold_days,
                    target_exit_date=target_exit.isoformat(),
                )
                append_trade(record)
                continue

            record = TradeRecord(
                trade_id=client_order_id,
                strategy_id=strategy_id,
                region=region,
                tier=_TIER,
                ticker=intent.ticker,  # yfinance ticker — keep ledger consistent across executors
                side="long",
                entry_date=on_date.isoformat(),
                entry_price=fill_price,
                quantity=quantity,
                allocation_pct=intent.allocation_pct,
                stop_loss_pct=intent.stop_loss_pct,
                take_profit_pct=intent.take_profit_pct,
                thesis=intent.thesis,
                broker_order_id=broker_order_id,
                currency=ccy,
                exchange=t212_exchange_from_ticker(t212_ticker),
                instrument_type=t212_instrument_type(inst.get("type", "")),
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
        # First pass: sweep T212 portfolio for orphans (positions with no
        # matching ledger entry today) and attribute them to this strategy.
        # The first T212-paper strategy whose exit runs claims all orphans —
        # subsequent strategies for the same slot won't see them as orphans
        # because the entries are now in the ledger. Attribution is imperfect
        # when multiple strategies share a slot, but the alternative is
        # leaving positions open at T212 indefinitely.
        try:
            recovered = self.reconcile_orphans(
                attribute_to_strategy=strategy_id,
                region=region,
                on_date=on_date,
            )
            if recovered:
                log.info(
                    "Recovered %d orphan T212 position(s) into %s's ledger: %s",
                    len(recovered), strategy_id, [r["ticker"] for r in recovered],
                )
        except Exception as e:
            log.warning("Orphan reconcile failed for %s (non-fatal): %s", strategy_id, e)

        # Sweep ALL open trades for this strategy+region — but ONLY at
        # this executor's tier. After a promotion (shadow → T212-paper)
        # the strategy may have lingering shadow-tier rows; those must
        # be handled by the shadow executor, not here. Without this
        # filter the T212 executor was logging "T212 position still
        # within hold window" for shadow rows it had no business
        # touching (observed 2026-05-21 after sector-rotator promotion).
        open_trades = read_open_trades(
            strategy_id=strategy_id, region=region, tier=_TIER,
        )
        if not open_trades:
            return []

        # Phase 12A — only close trades whose target_exit_date is
        # on-or-before today. Multi-day positions stay open across
        # sessions. Legacy rows with no target_exit_date are treated
        # as same-day round-trips so Wave 1 behaviour is preserved.
        due_trades = filter_due_for_exit(open_trades, on_date)
        held_over = len(open_trades) - len(due_trades)
        if held_over:
            log.info(
                "%s/%s: %d T212 position(s) still within hold window — leaving open",
                strategy_id, region, held_over,
            )
        if not due_trades:
            return []
        open_trades = due_trades

        translator = self._get_translator()
        closed: list[dict] = []

        # Phase 0 — resolve each trade's T212 ticker and reconcile any
        # pending-entry entry_price (sentinel 0 from a timed-out enter).
        actionable: list[tuple[dict, str]] = []
        for trade in open_trades:
            t212_ticker = translator.translate(trade["ticker"])
            if t212_ticker is None:
                log.warning("Cannot resolve T212 ticker for exit on %s — leaving open", trade["ticker"])
                continue

            if (not trade.get("entry_price") or float(trade.get("entry_price") or 0) == 0.0) and trade.get("broker_order_id"):
                reconciled = self._get_order_from_history(t212_ticker, trade["broker_order_id"])
                if reconciled is not None:
                    recovered = self._fill_price_of(reconciled)
                    if recovered is not None and recovered > 0:
                        converted = self._to_gbp(t212_ticker, recovered)
                        if converted is not None:
                            trade["entry_price"] = converted
                            log.info(
                                "Reconciled pending entry for %s (order %s): entry_price=%.4f from T212 history",
                                trade["ticker"], trade["broker_order_id"], converted,
                            )
                if not trade.get("entry_price") or float(trade["entry_price"]) == 0.0:
                    log.warning(
                        "Could not reconcile entry price for pending order %s on %s — "
                        "closing position but P&L will be unrecoverable",
                        trade.get("broker_order_id"), trade["ticker"],
                    )

            actionable.append((trade, t212_ticker))

        # Phase 12D — when multi-day positions land here, the trail script
        # may have placed a GTC stop on them. Cancel any open STOP orders
        # for tickers we're about to close so the GTC stop can't fire
        # against a future re-entry on the same instrument. Done once
        # up front to amortise the GET against the per-trade close loop.
        stops_to_cancel = self._index_stops_for_tickers(
            {t212_ticker for _, t212_ticker in actionable}
        )

        # Phase 1 — fire-and-forget: submit every close order without
        # waiting for any to fill. Every order goes into T212's queue
        # immediately; we collect fills in phase 2 after they've all had
        # a head start. Per-trade entry: (trade, t212_ticker, close_order_id).
        # close_order_id=None means "no position to close" (use history lookup
        # later) or "close submit failed" (we skip recording).
        submitted: list[tuple[dict, str, str | None, str]] = []
        for i, (trade, t212_ticker) in enumerate(actionable):
            if i > 0:
                # T212 rate-limits /equity/portfolio/{ticker} hard (~1 req/s).
                # Hammering it with 9 lookups in 3s returns 429s, which look
                # like "no position" to the rest of the code. Space them out.
                time.sleep(0.6)
            position = self._get_position(t212_ticker)
            if position is None:
                # No live position — entry never filled, or close already
                # happened externally / in a prior run. Phase 2 will look
                # in history for a matching sell on this date.
                submitted.append((trade, t212_ticker, None, "no_position"))
                continue

            # Phase 12D — cancel any open stop(s) on this ticker before
            # the market close, so a leftover GTC stop can't fire against
            # a re-entry tomorrow. We don't need the close to wait for the
            # cancel to land — if it loses the race, T212 still won't fire
            # a stop on a zero position.
            for stop_id in stops_to_cancel.get(t212_ticker, []):
                self._delete_order_silent(stop_id)

            qty = -float(position.get("quantity") or trade["quantity"])
            close_order = self._submit_market_order(t212_ticker=t212_ticker, quantity=qty)
            if close_order is None:
                log.warning("T212 close-order submit failed for %s — leaving open", trade["ticker"])
                # Don't record at all; trade stays open in the ledger for retry.
                continue
            submitted.append((
                trade,
                t212_ticker,
                str(close_order.get("id")) if close_order.get("id") is not None else None,
                "submitted",
            ))

        n_live_submits = sum(1 for _, _, oid, status in submitted if status == "submitted" and oid is not None)
        if n_live_submits:
            log.info(
                "T212: submitted %d close order(s) for %s, polling for fills",
                n_live_submits, strategy_id,
            )

        # Phase 2 — poll each submitted close. By the time we start, every
        # order has been in T212's queue for at least N submit-latencies.
        closed: list[dict] = []
        for i, (trade, t212_ticker, close_order_id, status) in enumerate(submitted):
            if i > 0 and status == "no_position":
                # T212 history endpoint shares the same rate-limit pool as
                # portfolio fetches; space the calls out.
                time.sleep(0.6)
            close_fill_price: float | None = None
            exit_reason: str

            if status == "no_position":
                # Try to recover the close from T212 order history (a sell
                # that fired today). If found → record P&L; else → cancelled.
                hist = self._find_recent_sell(t212_ticker, on_date)
                if hist is not None:
                    raw = self._fill_price_of(hist)
                    if raw is not None and raw > 0:
                        close_fill_price = self._to_gbp(t212_ticker, raw)
                        # Phase 10 — if the matching history entry is a
                        # STOP order, the trail or initial stop fired
                        # rather than our scheduled-market close. Tag
                        # so trail_exits picks it up.
                        hist_order = (hist.get("order") or {}) if isinstance(hist, dict) else {}
                        hist_otype = (hist_order.get("type") or "").upper()
                        exit_reason = "trail_stop" if hist_otype in ("STOP", "STOP_LIMIT") else "scheduled"
                        if close_fill_price is None:
                            log.warning(
                                "Recovered close for %s from T212 history but FX conversion "
                                "failed — closing with unrecoverable P&L",
                                trade["ticker"],
                            )
                        else:
                            log.info(
                                "Recovered close for %s from T212 history: exit_price=%.4f (type=%s)",
                                trade["ticker"], close_fill_price, hist_otype or "?",
                            )
                    else:
                        exit_reason = "cancelled"
                else:
                    exit_reason = "cancelled"
            elif close_order_id is None:
                # Submitted but no order id came back — odd; mark cancelled.
                exit_reason = "cancelled"
            else:
                filled = self._wait_for_fill(order_id=close_order_id, t212_ticker=t212_ticker)
                if filled is None:
                    log.warning(
                        "T212 close order %s for %s didn't fill in time — leaving open",
                        close_order_id, trade["ticker"],
                    )
                    continue  # leave open in ledger; next exit run will retry
                raw_fill = self._fill_price_of(filled)
                if raw_fill is None:
                    log.warning(
                        "T212 close FILLED for %s but no price — recording cancelled",
                        trade["ticker"],
                    )
                    exit_reason = "cancelled"
                else:
                    close_fill_price = self._to_gbp(t212_ticker, raw_fill)
                    exit_reason = "scheduled"
                    if close_fill_price is None:
                        log.warning(
                            "T212 close FILLED for %s but FX conversion failed — "
                            "closing with unrecoverable P&L",
                            trade["ticker"],
                        )

            entry_price = float(trade.get("entry_price") or 0.0)  # may be None if reconcile failed
            quantity = float(trade["quantity"])
            fees_gbp = 0.0
            fees_breakdown: dict = {}
            if close_fill_price is None:
                pnl_gbp: float | None = None
                pnl_pct: float | None = None
                exit_price_to_record: float | None = None
            else:
                gross_pnl_gbp = (close_fill_price - entry_price) * quantity
                # T212 stores fill prices in GBP via _to_gbp(), so entry +
                # exit are already GBP. compute_fees() treats them as GBP
                # natively when currency=GBP, and converts via FX when not.
                # For T212-paper-recorded prices the currency is always
                # effectively GBP (already converted at fill time), but the
                # FX/stamp-duty rules attach to the underlying instrument's
                # actual currency + exchange — which we stored on the entry
                # record. So we pass that metadata through.
                native_ccy = (trade.get("currency") or "GBP").upper()
                exch = trade.get("exchange") or t212_exchange_from_ticker(t212_ticker)
                inst_type = trade.get("instrument_type") or "share"
                fees = compute_fees(TradeContext(
                    tier=_TIER, currency=native_ccy, exchange=exch,
                    instrument_type=inst_type,
                    # T212 fill prices are already GBP (via _to_gbp at entry
                    # and exit), so notionals are GBP without further work.
                    entry_notional_gbp=abs(entry_price * quantity),
                    exit_notional_gbp=abs(close_fill_price * quantity),
                    quantity=abs(quantity),
                ))
                fees_gbp = fees.total_gbp
                fees_breakdown = fees.as_dict()
                pnl_gbp = gross_pnl_gbp - fees_gbp
                pnl_pct = (close_fill_price / entry_price - 1.0) * 100.0 if entry_price > 0 else 0.0
                exit_price_to_record = close_fill_price

            mark_trade_exited(
                trade_id=trade["trade_id"],
                exit_date=on_date,
                exit_price=exit_price_to_record if exit_price_to_record is not None else 0.0,
                pnl_gbp=pnl_gbp if pnl_gbp is not None else 0.0,
                pnl_pct=pnl_pct if pnl_pct is not None else 0.0,
                exit_reason=exit_reason,
                fees_gbp=fees_gbp,
                fees_breakdown=fees_breakdown,
            )
            closed_row = {
                **trade,
                "exit_date": on_date.isoformat(),
                "exit_price": exit_price_to_record,
                "pnl_gbp": pnl_gbp if pnl_gbp is not None else 0.0,
                "pnl_pct": pnl_pct if pnl_pct is not None else 0.0,
                "exit_reason": exit_reason,
            }
            closed.append(closed_row)
            # Phase 10A — track stop-driven exits for the cost gate
            try:
                from trading_bot.state.trail_exits import append_trail_exit
                append_trail_exit(closed_row)
            except Exception:
                pass
        return closed

    def reconcile_orphans(
        self,
        *,
        attribute_to_strategy: str,
        region: str,
        on_date: date,
    ) -> list[dict]:
        """Find T212 positions that have no matching open ledger trade for
        today and write entries for them, so the daily exit can close them
        normally. Used when a prior entry run dropped the ledger write
        (fill-poll timeout before the pending-record fix). Returns the
        list of recovered trades."""
        # Go through `_request_with_retry` so a 429 during reconcile (T212's
        # free tier rate-limits per-endpoint at ~1 req/s) doesn't silently
        # skip orphan recovery. Bare requests.get returned None on the 429
        # and the exit proceeded without ever seeing positions T212 still
        # held, leaving open trades to drift across the close.
        response = self._request_with_retry(
            "GET", self._url("/equity/portfolio"), headers=self._headers(),
        )
        if response is None:
            log.error("T212 portfolio fetch errored during reconcile (network)")
            return []
        if not response.ok:
            log.error(
                "T212 portfolio fetch returned %s during reconcile: %s",
                response.status_code, response.text[:200],
            )
            return []
        try:
            positions = response.json() or []
        except json.JSONDecodeError:
            log.error("T212 portfolio response wasn't JSON")
            return []

        translator = self._get_translator()
        # Build the set of T212-paper ledger tickers that are CURRENTLY
        # OPEN in the region (any strategy on this shared slot, any
        # entry date). Two prior bugs this fixes:
        #   1. on_date=today filter missed multi-day positions entered
        #      yesterday but still open on T212 — they'd get
        #      "orphan-reconciled" into a second ledger row.
        #   2. Multiple T212-paper strategies share slot 1. Strategy A's
        #      open trade for CURY.L belongs to A; strategy B's reconcile
        #      should see it and skip, not claim it as B's orphan.
        # Solution: read ALL open T212-paper trades in the region across
        # all strategies. Filter by tier so shadow/alpaca-paper rows on
        # the same region don't confuse the lookup.
        existing_open = read_open_trades(region=region, tier=_TIER)
        existing_yf_tickers = {t["ticker"] for t in existing_open}

        recovered: list[dict] = []
        for pos in positions:
            t212_ticker = pos.get("ticker")
            if not t212_ticker:
                continue

            # Reverse-translate T212 → yfinance via the instrument metadata.
            # The instrument record carries the ISIN/shortName we need to
            # reconstruct what our universes call this ticker.
            yf_ticker = self._t212_to_yfinance(t212_ticker, translator)
            if yf_ticker is None:
                log.warning(
                    "Can't reverse-translate %s to yfinance ticker — skipping reconcile",
                    t212_ticker,
                )
                continue

            if yf_ticker in existing_yf_tickers:
                continue  # Already in ledger — nothing to reconcile

            quantity = float(pos.get("quantity") or 0)
            avg_price = pos.get("averagePrice")
            if quantity <= 0 or avg_price is None:
                continue
            try:
                entry_price = float(avg_price)
            except (TypeError, ValueError):
                continue
            entry_price_gbp = self._to_gbp(t212_ticker, entry_price)
            if entry_price_gbp is None:
                log.warning(
                    "Can't convert orphan entry price for %s to GBP — skipping reconcile",
                    yf_ticker,
                )
                continue
            entry_price = entry_price_gbp

            client_order_id = f"{attribute_to_strategy}-reconciled-{uuid.uuid4().hex[:8]}"
            record = TradeRecord(
                trade_id=client_order_id,
                strategy_id=attribute_to_strategy,
                region=region,
                tier=_TIER,
                ticker=yf_ticker,
                side="long",
                entry_date=on_date.isoformat(),
                entry_price=entry_price,
                quantity=quantity,
                allocation_pct=0.0,
                thesis=f"Reconciled from T212 portfolio ({t212_ticker}) — original ledger entry missing",
            )
            append_trade(record)
            log.info(
                "Reconciled orphan: %s qty=%.0f @ %.4f → attributed to %s",
                yf_ticker, quantity, entry_price, attribute_to_strategy,
            )
            recovered.append({"ticker": yf_ticker, "quantity": quantity, "entry_price": entry_price})
        return recovered

    def _t212_to_yfinance(self, t212_ticker: str, translator: Translator) -> str | None:
        """Best-effort reverse translation. We use the instrument's shortName
        and embedded exchange letter / country code to reconstruct the
        yfinance ticker (the format our ledger uses)."""
        inst = translator.get_instrument(t212_ticker)
        if inst is None:
            return None
        short = (inst.get("shortName") or "").upper()
        if not short:
            return None
        stem = t212_ticker[:-3] if t212_ticker.endswith("_EQ") else t212_ticker
        # Single-letter form: VODl, ASMLa, SAPd → last lowercase char identifies exchange
        if stem and stem[-1].islower():
            letter = stem[-1]
            for yf_suffix, t212_letter in {
                ".L": "l", ".DE": "d", ".PA": "p", ".AS": "a",
                ".MC": "e", ".MI": "m", ".ST": "s", ".HE": "h", ".CO": "c",
            }.items():
                if letter == t212_letter:
                    return f"{short}{yf_suffix}"
        # Country-code form: VOD_US, UCB_BE
        if "_" in stem:
            parts = stem.split("_")
            country = parts[-1]
            for yf_suffix, t212_country in {
                ".BR": "BE", ".LS": "PT", ".VI": "AT", ".TO": "CA",
            }.items():
                if country == t212_country:
                    return f"{short}{yf_suffix}"
            if country == "US":
                return short  # No suffix for US
        return None

    # ---- safety controls ---------------------------------------------------

    def clear_slot(self) -> None:
        """Cancel pending orders, sell all positions, mark ledger trades cancelled."""
        try:
            self._cancel_all_orders()
            self._close_all_positions()
            n_marked = self._mark_ledger_trades_cancelled()
        except Exception as e:
            log.error("clear_slot encountered error: %s", e)
            raise
        log.info(
            "T212 slot %d cleared (marked %d ledger trade%s cancelled)",
            self.slot, n_marked, "" if n_marked == 1 else "s",
        )

    def _mark_ledger_trades_cancelled(self) -> int:
        from datetime import date as _date

        import yaml

        from trading_bot.strategy.registry import _strategies_dir

        # Find every strategy / region that's bound to this T212 slot. A
        # strategy can be on T212 in one region but Alpaca in another.
        bound_keys: list[tuple[str, str]] = []
        for path in _strategies_dir().glob("*/config.yaml"):
            raw = yaml.safe_load(path.read_text())
            sid = raw.get("id")
            if not sid:
                continue
            runs_in = raw.get("runs_in")
            if isinstance(runs_in, list):
                for entry in runs_in:
                    if entry.get("tier") == "trading212-paper" and entry.get("t212_slot") == self.slot:
                        bound_keys.append((sid, entry.get("region", "us")))
            elif raw.get("tier") == "trading212-paper" and raw.get("t212_slot") == self.slot:
                bound_keys.append((sid, raw.get("region", "us")))

        if not bound_keys:
            return 0

        today = _date.today()
        n = 0
        for sid, region in bound_keys:
            for trade in read_open_trades(strategy_id=sid, region=region):
                mark_trade_exited(
                    trade_id=trade["trade_id"],
                    exit_date=today,
                    exit_price=0.0,
                    pnl_gbp=0.0,
                    pnl_pct=0.0,
                    exit_reason="cleared",
                    outcome_notes=(
                        "T212 slot was cleared (clear-slot ran) before this trade reached "
                        "a natural exit. Either it was cancelled before any fill, or any "
                        "real position was liquidated by clear-slot."
                    ),
                    risks_observed=(
                        "P&L not recorded because the actual fill/close price wasn't "
                        "captured cleanly by clear-slot."
                    ),
                )
                n += 1
        return n

    # ---- HTTP plumbing -----------------------------------------------------

    def _get_translator(self) -> Translator:
        if self._translator is None:
            self._translator = Translator(fetch_instruments(self.creds))
        return self._translator

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": self.creds.auth_header(),
            "Content-Type": "application/json",
        }

    def _url(self, path: str) -> str:
        return f"{self.creds.base_url}{path}"

    def _request_with_retry(self, method: str, url: str, **kw) -> requests.Response | None:
        """Wrap a T212 HTTP call with 429-aware retry. T212's free-tier
        rate-limit is tight (~1 req/s per endpoint pool) and the per-loop
        sleep we already do can still get caught when phases overlap.
        Up to 3 retries with linear backoff."""
        for attempt in range(4):
            try:
                r = requests.request(method, url, timeout=15, **kw)
            except requests.RequestException as e:
                log.debug("T212 %s %s errored: %s", method, url, e)
                return None
            if r.status_code != 429:
                return r
            wait = 0.8 * (attempt + 1)
            log.debug("T212 429 on %s (attempt %d) — sleeping %.1fs", url, attempt + 1, wait)
            time.sleep(wait)
        return r  # last response, still 429

    def _submit_market_order(
        self,
        *,
        t212_ticker: str,
        quantity: float,
    ) -> dict[str, Any] | None:
        """Submit a T212 market order. Positive quantity = BUY, negative = SELL."""
        payload = {"ticker": t212_ticker, "quantity": quantity}
        try:
            response = requests.post(
                self._url("/equity/orders/market"),
                headers=self._headers(),
                json=payload,
                timeout=15,
            )
        except requests.RequestException as e:
            log.error("Market order submit errored for %s: %s", t212_ticker, e)
            return None
        if not response.ok:
            log.error(
                "Market order failed for %s qty=%s: %s %s",
                t212_ticker, quantity, response.status_code, response.text[:300],
            )
            return None
        return response.json()

    def _fill_price_of(self, order: dict[str, Any]) -> float | None:
        """Pull the fill price from an order record. T212 uses different
        field names + nesting depending on which endpoint returned it:
        - submit-response / active-order: flat dict with fillPrice / etc.
        - history: nested `{order: {...}, fill: {price: ...}}`
        Returns None if no fill price is present."""
        if not isinstance(order, dict):
            return None
        # Nested history shape first
        fill = order.get("fill")
        if isinstance(fill, dict) and fill.get("price") is not None:
            try:
                return float(fill["price"])
            except (TypeError, ValueError):
                pass
        for key in ("fillPrice", "filledPrice", "averagePrice", "averageFillPrice", "price"):
            v = order.get(key)
            if v is None:
                continue
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
        return None

    def _to_gbp(self, t212_ticker: str, raw_price: float) -> float | None:
        """Normalize a T212-reported price to the ledger's base unit (£).

        T212 quotes LSE in GBX (pence — divide by 100), EUR/USD/etc. in
        their native currency (multiply by spot FX rate). FX rates come
        from yfinance via the fx module and are cached per process.

        Returns None when the conversion can't be trusted — the instrument
        record is missing (currency unknown) or the FX rate is unavailable.
        Callers MUST treat None as "price unknown" (record pending / skip /
        unrecoverable P&L) rather than recording native units as if they
        were GBP. GBP/GBX use fixed factors and never fail; only EUR/USD/etc.
        can hit the FX-unavailable path.
        """
        inst = self._get_translator().get_instrument(t212_ticker)
        if inst is None:
            log.warning(
                "No T212 instrument record for %s — cannot convert %s to GBP",
                t212_ticker, raw_price,
            )
            return None
        ccy = (inst.get("currencyCode") or "").upper()
        mult = to_gbp_multiplier(ccy)
        if mult is None:
            log.warning(
                "FX rate unavailable for %s (currency=%s) — cannot convert to GBP",
                t212_ticker, ccy,
            )
            return None
        return raw_price * mult

    _STALE_LOOKBACK_DAYS = 7

    def _find_recent_sell(self, t212_ticker: str, on_date: date) -> dict[str, Any] | None:
        """Find the most recent SELL order for this ticker that filled
        within the last week. Used when exit_scheduled sees no live
        position — the close already happened (externally / in a prior
        run that lost the response / a session strand from a prior day)
        and we need to recover the fill price from T212's order history.
        7-day window covers weekend gaps and most workflow strands.

        T212's history items are shaped `{order: {...}, fill: {...}}`.
        The order metadata (side, ticker, status, timestamp) lives under
        `order`; the executed price under `fill.price`. Returns the
        history item; the caller pulls the price via `_fill_price_of`
        which knows about the same nesting.
        """
        response = self._request_with_retry(
            "GET",
            self._url("/equity/history/orders"),
            headers=self._headers(),
            params={"ticker": t212_ticker, "limit": 50},
        )
        if response is None:
            return None
        if not response.ok:
            log.warning(
                "T212 history fetch returned %s for %s during close recovery",
                response.status_code, t212_ticker,
            )
            return None
        try:
            body = response.json() or {}
        except json.JSONDecodeError:
            return None
        items = body.get("items") if isinstance(body, dict) else body
        if not isinstance(items, list) or not items:
            log.info("T212 history for %s: no items returned", t212_ticker)
            return None

        from datetime import timedelta as _td
        oldest_acceptable = (on_date - _td(days=self._STALE_LOOKBACK_DAYS)).isoformat()
        candidates: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            order = item.get("order") or {}
            fill = item.get("fill") or {}
            if not isinstance(order, dict) or not isinstance(fill, dict):
                continue

            # Date filter — use order.createdAt (T212 returns ISO 8601 like
            # "2026-05-18T15:03:43.000Z"). Fall back to fill.filledAt.
            # Accept anything from the last STALE_LOOKBACK_DAYS so stale
            # strands from prior sessions can be reconciled.
            ts = order.get("createdAt") or fill.get("filledAt") or ""
            if not ts or ts[:10] < oldest_acceptable:
                continue
            if ts[:10] > on_date.isoformat():
                continue

            # Direction filter — order.side is the authoritative field
            side = (order.get("side") or "").upper()
            if side != "SELL":
                # Also accept by sign of filledQuantity as a fallback
                fq = order.get("filledQuantity") or order.get("quantity") or 0
                try:
                    if float(fq) >= 0:
                        continue
                except (TypeError, ValueError):
                    continue

            # Must be filled
            status = (order.get("status") or "").upper()
            if status not in ("FILLED", "EXECUTED", "COMPLETED"):
                continue

            # Must have a fill price
            if fill.get("price") is None:
                continue
            candidates.append(item)

        if not candidates:
            log.info("T212 history for %s: no SELL match on or before %s", t212_ticker, on_date.isoformat())
            return None
        candidates.sort(
            key=lambda i: (i.get("order") or {}).get("createdAt") or "",
            reverse=True,
        )
        return candidates[0]

    def _get_order(self, order_id: int | str) -> dict[str, Any] | None:
        """Fetch an order from the active orders endpoint. Returns None if
        the order is no longer active (most likely because it has filled
        or been cancelled and moved to history)."""
        if order_id is None:
            return None
        try:
            response = requests.get(
                self._url(f"/equity/orders/{order_id}"),
                headers=self._headers(),
                timeout=10,
            )
        except requests.RequestException as e:
            log.debug("T212 active-order fetch errored for %s: %s", order_id, e)
            return None
        if response.status_code == 404:
            return None
        if not response.ok:
            return None
        try:
            return response.json()
        except json.JSONDecodeError:
            return None

    def _get_order_from_history(
        self, t212_ticker: str, order_id: int | str
    ) -> dict[str, Any] | None:
        """Look up a single order in T212's history endpoint. The endpoint
        is paginated and filterable by ticker, so we pull the most recent
        page for this ticker and scan for our id. Recent fills are at the
        top so this is cheap in practice."""
        if order_id is None:
            return None
        try:
            response = requests.get(
                self._url("/equity/history/orders"),
                headers=self._headers(),
                params={"ticker": t212_ticker, "limit": 50},
                timeout=15,
            )
        except requests.RequestException as e:
            log.debug("T212 history fetch errored: %s", e)
            return None
        if not response.ok:
            return None
        try:
            body = response.json() or {}
        except json.JSONDecodeError:
            return None
        items = body.get("items") if isinstance(body, dict) else body
        if not isinstance(items, list):
            return None
        oid_str = str(order_id)
        for item in items:
            if not isinstance(item, dict):
                continue
            # T212 history items are shaped {order: {...}, fill: {...}} —
            # the order id lives inside `order`, not at top-level. Fall
            # back to `item.id` for any flatter response variants.
            inner = item.get("order") if isinstance(item.get("order"), dict) else item
            if str(inner.get("id")) == oid_str:
                return item
        return None

    def _wait_for_fill(
        self,
        *,
        order_id: int | str | None,
        t212_ticker: str,
        timeout_s: float = 300.0,
    ) -> dict[str, Any] | None:
        """Poll until the order reaches a terminal state. Returns the order
        record if it FILLED, None if it was cancelled / rejected / didn't
        resolve within `timeout_s`.

        The poll alternates between the active-orders endpoint (where the
        order lives while pending) and the history endpoint (where it
        lands after fill). On paper accounts market orders usually resolve
        within seconds, but slot 1 has shown 2-3 minute delays during UK
        market hours so the default budget is 5 minutes. The cron has
        plenty of headroom (entry runs ~30 min before close)."""
        if order_id is None:
            return None
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            active = self._get_order(order_id)
            if active is not None:
                status = (active.get("status") or "").lower()
                if status in ("filled", "executed", "completed"):
                    return active
                if status in ("cancelled", "canceled", "rejected", "expired"):
                    return None
                # Still pending — wait and retry
            else:
                # Not in active list → check history
                historical = self._get_order_from_history(t212_ticker, order_id)
                if historical is not None:
                    status = _history_status(historical)
                    if status in ("filled", "executed", "completed"):
                        return historical
                    return None
            time.sleep(0.5)
        # Final check after timeout
        historical = self._get_order_from_history(t212_ticker, order_id)
        if historical is not None:
            status = _history_status(historical)
            if status in ("filled", "executed", "completed"):
                return historical
        return None

    def _get_position(self, t212_ticker: str) -> dict[str, Any] | None:
        response = self._request_with_retry(
            "GET",
            self._url(f"/equity/portfolio/{t212_ticker}"),
            headers=self._headers(),
        )
        if response is None:
            return None
        if response.status_code == 404:
            return None
        if not response.ok:
            log.warning("Position fetch returned %s for %s", response.status_code, t212_ticker)
            return None
        return response.json()

    def _index_stops_for_tickers(self, tickers: set[str]) -> dict[str, list[str]]:
        """Phase 12D — fetch open orders once, return a per-ticker map of
        STOP / STOP_LIMIT order IDs. Used by exit_scheduled to clear out
        leftover GTC stops before closing the underlying positions.
        Returns an empty dict on any error (we'd rather skip cancellation
        than block the close)."""
        if not tickers:
            return {}
        try:
            response = requests.get(
                self._url("/equity/orders"), headers=self._headers(), timeout=10,
            )
        except requests.RequestException as e:
            log.debug("T212 stops index failed: %s", e)
            return {}
        if not response.ok:
            return {}
        try:
            orders = response.json() or []
        except json.JSONDecodeError:
            return {}
        out: dict[str, list[str]] = {}
        for order in orders if isinstance(orders, list) else []:
            if not isinstance(order, dict):
                continue
            otype = (order.get("type") or "").upper()
            if otype not in ("STOP", "STOP_LIMIT"):
                continue
            tkr = order.get("ticker")
            oid = order.get("id")
            if not tkr or tkr not in tickers or oid is None:
                continue
            out.setdefault(tkr, []).append(str(oid))
        return out

    def _delete_order_silent(self, order_id: str) -> None:
        """Best-effort DELETE of a single order. Logs at debug only; we
        never want stop-cleanup to block the close path."""
        try:
            r = requests.delete(
                self._url(f"/equity/orders/{order_id}"),
                headers=self._headers(),
                timeout=10,
            )
            if not (r.ok or r.status_code == 404):
                log.debug(
                    "T212 cancel stop %s returned %d: %s",
                    order_id, r.status_code, r.text[:150],
                )
        except requests.RequestException as e:
            log.debug("T212 cancel stop %s errored: %s", order_id, e)

    def _cancel_all_orders(self) -> None:
        try:
            response = requests.get(
                self._url("/equity/orders"), headers=self._headers(), timeout=10
            )
        except requests.RequestException as e:
            log.warning("List orders failed during clear: %s", e)
            return
        if not response.ok:
            return
        for order in response.json() or []:
            oid = order.get("id")
            if not oid:
                continue
            try:
                requests.delete(
                    self._url(f"/equity/orders/{oid}"), headers=self._headers(), timeout=10
                )
            except requests.RequestException:
                continue
            # T212 docs ask for a brief gap between cancels to avoid limit-order rate cap
            time.sleep(0.1)

    def _close_all_positions(self) -> None:
        try:
            response = requests.get(
                self._url("/equity/portfolio"), headers=self._headers(), timeout=15
            )
        except requests.RequestException as e:
            log.warning("Portfolio fetch failed during clear: %s", e)
            return
        if not response.ok:
            return
        for pos in response.json() or []:
            t212_ticker = pos.get("ticker")
            qty = pos.get("quantity")
            if not t212_ticker or qty is None:
                continue
            self._submit_market_order(t212_ticker=t212_ticker, quantity=-float(qty))
            time.sleep(0.1)

    def _get_free_cash(self) -> float | None:
        """Read account `free` cash from T212. Returns None on error so the
        caller can decide whether to proceed (vs blocking on a transient
        connectivity glitch)."""
        try:
            response = requests.get(
                self._url("/equity/account/cash"),
                headers=self._headers(),
                timeout=10,
            )
        except requests.RequestException as e:
            log.warning("T212 cash fetch errored: %s", e)
            return None
        if not response.ok:
            log.warning("T212 cash fetch returned %s: %s", response.status_code, response.text[:200])
            return None
        try:
            body = response.json() or {}
        except json.JSONDecodeError:
            return None
        # T212 returns {"free": ..., "total": ..., "invested": ..., "ppl": ..., ...}
        free = body.get("free")
        if free is None:
            return None
        try:
            return float(free)
        except (TypeError, ValueError):
            return None

    def _estimate_price(self, yf_ticker: str) -> float | None:
        """Use yfinance's recent close as the sizing seed. T212 doesn't expose
        a free real-time snapshot endpoint, and the actual fill price is what
        we record in the ledger anyway."""
        from trading_bot.tools import get_history
        history = get_history([yf_ticker], lookback_days=2)
        bars = history.get(yf_ticker)
        if not bars:
            return None
        return bars[-1].close
