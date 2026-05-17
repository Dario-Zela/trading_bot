"""Tier 1 executor — places real bracket orders on an Alpaca paper account.

Each strategy is bound to a specific Alpaca paper account (a "slot") via its
config. The executor takes a slot number at construction; credentials come
from env vars per the convention in trading_bot.alpaca_slot.

Bracket orders are Alpaca's native primitive for entry + stop + take-profit
in a single submission. The broker enforces the stop/target server-side so we
don't need a mid-day cron to watch positions.

Wave 2a scope:
- enter(): place bracket orders for each TradeIntent
- exit_scheduled(): close any remaining positions opened today (market sell)
- suspend() / resume(): toggle suspend_trade flag
- clear_slot(): suspend + cancel orders + liquidate positions (used when
  reassigning a slot to a new strategy)
"""
from __future__ import annotations

import logging
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
            quantity = round(allocation_usd / entry_estimate, 4)
            if quantity <= 0:
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

            record = TradeRecord(
                trade_id=client_order_id,
                strategy_id=strategy_id,
                region=region,
                tier=_TIER,
                ticker=intent.ticker,
                side="long",
                entry_date=on_date.isoformat(),
                entry_price=float(order.get("filled_avg_price") or entry_estimate),
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
            # First, get the current position (it may have already been closed
            # by a bracket stop or take-profit firing during the day).
            position = self._get_position(trade["ticker"])
            try:
                if position is not None:
                    # Position still open — close at market
                    self._close_position(trade["ticker"])
                # Look up the most recent fill for the close of this position
                exit_price = self._get_latest_close_price(trade)
                exit_reason = self._infer_exit_reason(trade, exit_price)
            except Exception as e:
                log.error("Exit failed for %s: %s", trade["ticker"], e)
                continue

            entry_price = float(trade["entry_price"])
            quantity = float(trade["quantity"])
            pnl_gbp = (exit_price - entry_price) * quantity
            pnl_pct = (exit_price / entry_price - 1.0) * 100.0 if entry_price > 0 else 0.0

            mark_trade_exited(
                trade_id=trade["trade_id"],
                exit_date=on_date,
                exit_price=exit_price,
                pnl_gbp=pnl_gbp,
                pnl_pct=pnl_pct,
                exit_reason=exit_reason,
            )
            closed.append(
                {
                    **trade,
                    "exit_date": on_date.isoformat(),
                    "exit_price": exit_price,
                    "pnl_gbp": pnl_gbp,
                    "pnl_pct": pnl_pct,
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
        suspend, cancel all open orders, liquidate all positions."""
        self.suspend()
        self._cancel_all_orders()
        self._close_all_positions()
        log.info("Slot %d cleared (positions liquidated, orders cancelled)", self.slot)

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

    def _close_position(self, ticker: str) -> None:
        response = requests.delete(
            self._url(f"/v2/positions/{ticker}"), headers=self._headers(), timeout=15
        )
        if not response.ok and response.status_code not in (404, 422):
            log.error("Close position failed for %s: %s %s", ticker, response.status_code, response.text[:200])

    def _get_latest_close_price(self, trade: dict) -> float:
        """Find the close fill price for this trade. Tries position avg price,
        then latest quote, then entry price as last resort."""
        url = f"https://data.alpaca.markets/v2/stocks/{trade['ticker']}/quotes/latest"
        try:
            response = requests.get(url, headers=self._headers(), timeout=10)
            if response.ok:
                quote = response.json().get("quote") or {}
                bp = quote.get("bp")
                if bp:
                    return float(bp)
        except Exception:
            pass
        return float(trade["entry_price"])

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
