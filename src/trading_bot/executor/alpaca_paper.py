"""Tier 1 executor — places real bracket orders on an Alpaca paper account.

Each strategy is bound to a specific Alpaca paper account (a "slot") via its
config. The executor takes a slot number at construction; credentials come
from env vars per the convention in trading_bot.alpaca_slot.

Bracket orders are Alpaca's native primitive for entry + stop + take-profit
in a single submission. The broker enforces the stop/target server-side so we
don't need a mid-day cron to watch positions.

Fill prices come exclusively from Alpaca's order endpoints. After submitting
an order we poll `GET /v2/orders/{id}` until it reaches a terminal state
(filled / canceled / rejected) and read `filled_avg_price` from the terminal
record. On exit, if the position is already gone (a bracket child fired
intraday), we re-fetch the entry parent order with `nested=true` and read
the filled leg's price. Snapshot prices from the market data API are only
used as bracket-pricing seeds — never recorded as fills.

Wave 2a scope:
- enter(): place bracket orders for each TradeIntent
- exit_scheduled(): close any remaining positions opened today (market sell)
- suspend() / resume(): toggle suspend_trade flag
- clear_slot(): suspend + cancel orders + liquidate positions (used when
  reassigning a slot to a new strategy)
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import date
from typing import Any

import requests

from trading_bot.alpaca_slot import AlpacaCreds, load_slot_creds
from trading_bot.executor.base import Executor, TradeIntent
from trading_bot.state import TradeRecord, append_trade, mark_trade_exited, read_open_trades


_TIER = "alpaca-paper"
log = logging.getLogger(__name__)


class AlpacaPaperExecutor(Executor):
    def __init__(self, slot: int):
        self.slot = slot
        self.creds: AlpacaCreds = load_slot_creds(slot)

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

        for intent in intents:
            # Pull latest snapshot to seed the entry price estimate
            try:
                snap = self._get_snapshot(intent.ticker)
            except Exception as e:
                log.warning("Snapshot fetch failed for %s: %s — skipping", intent.ticker, e)
                continue
            entry_estimate = snap.get("c")
            if not entry_estimate or entry_estimate <= 0:
                log.warning("No usable entry price for %s — skipping", intent.ticker)
                continue

            allocation_gbp = capital_gbp * (intent.allocation_pct / 100.0)
            # Alpaca paper trades in USD. We approximate 1:1 GBP→USD for sizing
            # (Wave 2 simplification; Wave 3 will introduce real FX conversion).
            allocation_usd = allocation_gbp
            # Bracket orders require whole shares on Alpaca — fractional orders
            # are restricted to "simple" (non-bracketed) orders. Round down so
            # we never over-allocate; if the result rounds to zero the position
            # is skipped.
            is_bracket = (
                intent.stop_loss_pct is not None and intent.take_profit_pct is not None
            )
            if is_bracket:
                quantity = float(int(allocation_usd / entry_estimate))
            else:
                quantity = round(allocation_usd / entry_estimate, 4)
            if quantity <= 0:
                log.warning(
                    "Computed qty 0 for %s at $%.2f (allocation $%.2f) — skipping",
                    intent.ticker, entry_estimate, allocation_usd,
                )
                continue

            client_order_id = f"{strategy_id}-{uuid.uuid4().hex[:12]}"
            order = self._submit_bracket_order(
                ticker=intent.ticker,
                qty=quantity,
                client_order_id=client_order_id,
                stop_loss_pct=intent.stop_loss_pct,
                take_profit_pct=intent.take_profit_pct,
                entry_estimate=entry_estimate,
            )
            if order is None:
                continue

            # Poll until the parent order reaches a terminal state. The
            # submit response is an acknowledgment; the definitive fill
            # price only appears once Alpaca marks the order filled.
            filled = self._wait_for_fill(order.get("id"))
            if filled is None or (filled.get("status") or "").lower() != "filled":
                log.warning(
                    "Alpaca order %s for %s didn't fill within timeout (status=%s) — not recording ledger",
                    order.get("id"), intent.ticker,
                    filled.get("status") if filled else "unknown",
                )
                continue
            fill_price_raw = filled.get("filled_avg_price")
            if fill_price_raw is None:
                log.warning(
                    "Alpaca reported FILLED for %s but filled_avg_price is missing — skipping",
                    intent.ticker,
                )
                continue
            entry_price = float(fill_price_raw)

            record = TradeRecord(
                trade_id=client_order_id,
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
            )
            append_trade(record)

    def exit_scheduled(
        self,
        *,
        strategy_id: str,
        region: str,
        on_date: date,
    ) -> list[dict]:
        open_trades = read_open_trades(strategy_id=strategy_id, region=region, on_date=on_date)
        if not open_trades:
            return []

        closed: list[dict] = []
        for trade in open_trades:
            try:
                position = self._get_position(trade["ticker"])
                close_fill_price: float | None = None
                exit_reason: str

                if position is not None:
                    # Position still open at Alpaca — close at market, then
                    # poll the resulting order until filled.
                    close_order = self._close_position(trade["ticker"])
                    if close_order is None:
                        log.warning("Close-order submit failed for %s — leaving open", trade["ticker"])
                        continue
                    filled = self._wait_for_fill(close_order.get("id"))
                    if filled is not None and (filled.get("status") or "").lower() == "filled":
                        fap = filled.get("filled_avg_price")
                        if fap is not None:
                            close_fill_price = float(fap)
                    if close_fill_price is None:
                        log.warning(
                            "Close order %s for %s didn't yield a fill price — recording cancelled",
                            close_order.get("id"), trade["ticker"],
                        )
                        exit_reason = "cancelled"
                    else:
                        exit_reason = self._infer_exit_reason(trade, close_fill_price)
                else:
                    # No position. Either a bracket leg fired intraday or
                    # the entry never filled. Re-fetch the entry order with
                    # nested=true to see if a child (stop / take-profit)
                    # filled and at what price. If so, that IS the exit fill.
                    parent = self._get_order_by_client_id(trade["trade_id"])
                    leg_fill = self._extract_filled_leg(parent) if parent else None
                    if leg_fill is None:
                        close_fill_price = None
                        exit_reason = "cancelled"
                    else:
                        close_fill_price = float(leg_fill["filled_avg_price"])
                        leg_type = (leg_fill.get("order_type") or "").lower()
                        if "stop" in leg_type:
                            exit_reason = "stop"
                        elif "limit" in leg_type:
                            exit_reason = "take_profit"
                        else:
                            exit_reason = "scheduled"
            except Exception as e:
                log.error("Exit failed for %s: %s", trade["ticker"], e)
                continue

            entry_price = float(trade["entry_price"])
            quantity = float(trade["quantity"])
            if close_fill_price is None:
                pnl_gbp: float | None = None
                pnl_pct: float | None = None
                exit_price_to_record: float | None = None
            else:
                pnl_gbp = (close_fill_price - entry_price) * quantity
                pnl_pct = (
                    (close_fill_price / entry_price - 1.0) * 100.0 if entry_price > 0 else 0.0
                )
                exit_price_to_record = close_fill_price

            mark_trade_exited(
                trade_id=trade["trade_id"],
                exit_date=on_date,
                exit_price=exit_price_to_record if exit_price_to_record is not None else 0.0,
                pnl_gbp=pnl_gbp if pnl_gbp is not None else 0.0,
                pnl_pct=pnl_pct if pnl_pct is not None else 0.0,
                exit_reason=exit_reason,
            )
            closed.append(
                {
                    **trade,
                    "exit_date": on_date.isoformat(),
                    "exit_price": exit_price_to_record,
                    "pnl_gbp": pnl_gbp if pnl_gbp is not None else 0.0,
                    "pnl_pct": pnl_pct if pnl_pct is not None else 0.0,
                    "exit_reason": exit_reason,
                }
            )
        return closed

    # ---- safety controls ---------------------------------------------------

    def suspend(self) -> None:
        """Set suspend_trade=true on this slot. Existing positions keep their
        bracket exits; no new orders can be placed."""
        self._patch_config({"suspend_trade": True})
        log.info("Slot %d suspended", self.slot)

    def resume(self) -> None:
        self._patch_config({"suspend_trade": False})
        log.info("Slot %d resumed", self.slot)

    def clear_slot(self) -> None:
        """Wipe the slot in preparation for a new strategy assignment:
        suspend trading, cancel all open orders, liquidate all positions,
        mark any ledger trades still attached to this slot as cancelled,
        then resume. The suspend window guards against a race where new
        orders land mid-clear; after resume the slot is ready for the next
        strategy to start trading on the next pipeline run."""
        self.suspend()
        try:
            self._cancel_all_orders()
            self._close_all_positions()
            n_marked = self._mark_ledger_trades_cancelled()
        finally:
            self.resume()
        log.info(
            "Slot %d cleared and resumed (marked %d ledger trade%s cancelled)",
            self.slot, n_marked, "" if n_marked == 1 else "s",
        )

    def _mark_ledger_trades_cancelled(self) -> int:
        """For every strategy bound to this slot, mark its open ledger trades
        as exited with reason='cleared'. Preserves audit trail while keeping
        the ledger in sync with Alpaca after a wipe."""
        from datetime import date as _date

        import yaml

        from trading_bot.strategy.registry import _strategies_dir

        bound_strategies: list[str] = []
        for path in _strategies_dir().glob("*/config.yaml"):
            raw = yaml.safe_load(path.read_text())
            if raw.get("alpaca_slot") == self.slot and raw.get("tier") == "alpaca-paper":
                bound_strategies.append(raw["id"])
        if not bound_strategies:
            return 0

        today = _date.today()
        n = 0
        for sid in bound_strategies:
            for trade in read_open_trades(strategy_id=sid):
                mark_trade_exited(
                    trade_id=trade["trade_id"],
                    exit_date=today,
                    exit_price=0.0,
                    pnl_gbp=0.0,
                    pnl_pct=0.0,
                    exit_reason="cleared",
                    outcome_notes=(
                        "Slot was cleared (clear-slot ran) before this trade reached a "
                        "natural exit. Either it was cancelled before any fill, or any "
                        "real position was liquidated by clear-slot."
                    ),
                    risks_observed=(
                        "P&L not recorded because the actual fill/close price wasn't "
                        "captured cleanly by clear-slot. Wave 2c reflection can flag this "
                        "as data missing rather than a real outcome."
                    ),
                )
                n += 1
        return n

    # ---- HTTP plumbing -----------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.creds.api_key,
            "APCA-API-SECRET-KEY": self.creds.api_secret,
            "accept": "application/json",
            "content-type": "application/json",
        }

    def _url(self, path: str) -> str:
        return f"{self.creds.trading_base_url}{path}"

    def _submit_bracket_order(
        self,
        *,
        ticker: str,
        qty: float,
        client_order_id: str,
        stop_loss_pct: float | None,
        take_profit_pct: float | None,
        entry_estimate: float,
    ) -> dict[str, Any] | None:
        # If either stop or take-profit is unset, fall back to a plain market
        # order — Alpaca's bracket order class requires BOTH legs.
        if stop_loss_pct is None or take_profit_pct is None:
            payload = {
                "symbol": ticker,
                "qty": str(qty),
                "side": "buy",
                "type": "market",
                "time_in_force": "day",
                "client_order_id": client_order_id,
            }
        else:
            # 1-cent safety buffer in each direction guards against tiny drift
            # between our seed price and Alpaca's order-validation base_price.
            stop_price = round(entry_estimate * (1 + stop_loss_pct / 100.0) - 0.01, 2)
            target_price = round(entry_estimate * (1 + take_profit_pct / 100.0) + 0.01, 2)
            payload = {
                "symbol": ticker,
                "qty": str(qty),
                "side": "buy",
                "type": "market",
                "time_in_force": "day",
                "order_class": "bracket",
                "client_order_id": client_order_id,
                "stop_loss": {"stop_price": str(stop_price)},
                "take_profit": {"limit_price": str(target_price)},
            }

        response = requests.post(
            self._url("/v2/orders"), headers=self._headers(), json=payload, timeout=15
        )
        if not response.ok:
            log.error(
                "Order placement failed for %s: %s %s",
                ticker, response.status_code, response.text[:300]
            )
            return None
        return response.json()

    def _get_snapshot(self, ticker: str) -> dict[str, Any]:
        """Best available "current price" for sizing & bracket-price computation.

        Prefers latest_trade.p (most aligned with how Alpaca validates bracket
        orders), falls back to quote midpoint, then today's daily close. We
        intentionally avoid the quote ask in isolation because on weekends /
        after-hours the ask can be wide and well above the regular-hours price
        Alpaca uses for its `base_price` order-validation reference.
        """
        url = f"https://data.alpaca.markets/v2/stocks/{ticker}/snapshot"
        response = requests.get(url, headers=self._headers(), timeout=10)
        if not response.ok:
            log.warning(
                "Snapshot fetch failed for %s: %s %s",
                ticker, response.status_code, response.text[:120],
            )
            return {"c": None}
        body = response.json() or {}

        # Try latest trade first (most accurate "current price")
        latest_trade = body.get("latestTrade") or {}
        trade_price = latest_trade.get("p")
        if trade_price and trade_price > 0:
            return {"c": float(trade_price)}

        # Fall back to quote midpoint
        latest_quote = body.get("latestQuote") or {}
        bp = latest_quote.get("bp")
        ap = latest_quote.get("ap")
        if bp and ap and bp > 0 and ap > 0:
            return {"c": (float(bp) + float(ap)) / 2.0}

        # Last resort: today's daily bar close
        daily_bar = body.get("dailyBar") or {}
        close = daily_bar.get("c")
        if close and close > 0:
            return {"c": float(close)}

        return {"c": None}

    def _get_position(self, ticker: str) -> dict[str, Any] | None:
        response = requests.get(
            self._url(f"/v2/positions/{ticker}"), headers=self._headers(), timeout=10
        )
        if response.status_code == 404:
            return None
        if not response.ok:
            log.warning("Position fetch failed for %s: %s", ticker, response.status_code)
            return None
        return response.json()

    def _get_order_by_id(self, order_id: str | None) -> dict[str, Any] | None:
        """Fetch one order by its Alpaca-assigned id with nested legs."""
        if not order_id:
            return None
        try:
            response = requests.get(
                self._url(f"/v2/orders/{order_id}"),
                headers=self._headers(),
                params={"nested": "true"},
                timeout=10,
            )
        except requests.RequestException as e:
            log.debug("Order fetch errored for %s: %s", order_id, e)
            return None
        if response.status_code == 404:
            return None
        if not response.ok:
            return None
        try:
            return response.json()
        except Exception:
            return None

    def _get_order_by_client_id(self, client_order_id: str) -> dict[str, Any] | None:
        """Look up an order by our client-assigned id (we use this to find
        the original entry parent at exit time so we can read filled legs)."""
        try:
            response = requests.get(
                self._url("/v2/orders:by_client_order_id"),
                headers=self._headers(),
                params={"client_order_id": client_order_id, "nested": "true"},
                timeout=10,
            )
        except requests.RequestException as e:
            log.debug("by-client-id fetch errored for %s: %s", client_order_id, e)
            return None
        if response.status_code == 404:
            return None
        if not response.ok:
            return None
        try:
            return response.json()
        except Exception:
            return None

    def _wait_for_fill(self, order_id: str | None, *, timeout_s: float = 10.0) -> dict[str, Any] | None:
        """Poll the order endpoint until the order reaches a terminal
        state. Returns the terminal order record (filled or not) so the
        caller can read `status` and `filled_avg_price`. Returns None only
        if we can't reach the order at all."""
        if not order_id:
            return None
        deadline = time.time() + timeout_s
        order: dict[str, Any] | None = None
        while time.time() < deadline:
            order = self._get_order_by_id(order_id)
            if order is not None:
                status = (order.get("status") or "").lower()
                # Alpaca terminal statuses
                if status in ("filled", "canceled", "rejected", "expired", "done_for_day"):
                    return order
            time.sleep(0.5)
        return order  # last seen; may still be pending

    def _extract_filled_leg(self, parent: dict[str, Any]) -> dict[str, Any] | None:
        """Given a bracket parent order, find the leg (stop or take-profit)
        that filled. Returns the leg dict or None if neither leg filled."""
        legs = parent.get("legs") or []
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            if (leg.get("status") or "").lower() == "filled" and leg.get("filled_avg_price") is not None:
                return leg
        return None

    def _close_position(self, ticker: str) -> dict[str, Any] | None:
        """Liquidate the position at market and return the resulting order
        record (which carries the eventual filled_avg_price once Alpaca fills
        it). Returns None on error or when there's no position to close."""
        response = requests.delete(
            self._url(f"/v2/positions/{ticker}"), headers=self._headers(), timeout=15
        )
        if response.status_code in (404, 422):
            return None
        if not response.ok:
            log.error(
                "Close position failed for %s: %s %s",
                ticker, response.status_code, response.text[:200],
            )
            return None
        try:
            return response.json()
        except Exception:
            return None

    def _infer_exit_reason(self, trade: dict, exit_price: float) -> str:
        entry = float(trade["entry_price"])
        stop_pct = trade.get("stop_loss_pct")
        tp_pct = trade.get("take_profit_pct")
        if stop_pct is not None:
            stop_price = entry * (1 + stop_pct / 100.0)
            if exit_price <= stop_price * 1.005:  # within 0.5% slippage tolerance
                return "stop"
        if tp_pct is not None:
            tp_price = entry * (1 + tp_pct / 100.0)
            if exit_price >= tp_price * 0.995:
                return "take_profit"
        return "scheduled"

    def _cancel_all_orders(self) -> None:
        response = requests.delete(
            self._url("/v2/orders"), headers=self._headers(), timeout=15
        )
        if not response.ok and response.status_code not in (207, 404):
            log.warning("Cancel orders returned %s: %s", response.status_code, response.text[:200])

    def _close_all_positions(self) -> None:
        response = requests.delete(
            self._url("/v2/positions"),
            headers=self._headers(),
            params={"cancel_orders": "true"},
            timeout=20,
        )
        if not response.ok and response.status_code not in (207, 404):
            log.warning("Close all positions returned %s: %s", response.status_code, response.text[:200])

    def _patch_config(self, body: dict) -> None:
        response = requests.patch(
            self._url("/v2/account/configurations"),
            headers=self._headers(),
            json=body,
            timeout=10,
        )
        if not response.ok:
            log.error("Config patch failed: %s %s", response.status_code, response.text[:200])
            response.raise_for_status()
