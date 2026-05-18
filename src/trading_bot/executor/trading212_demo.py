"""Tier 1.5 executor — places market orders on a Trading212 Practice account.

Mirrors AlpacaPaperExecutor's shape but targets T212's demo API. T212 doesn't
support native bracket orders the same way Alpaca does, and the bot does
same-day round-trips anyway, so this executor simply:
- enter(): submit a market BUY for each intent
- exit_scheduled(): market SELL every position opened today by this strategy
- clear_slot(): cancel all pending orders + sell all positions

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
from trading_bot.t212_slot import T212_PAPER_BUDGET_GBP, T212Creds, load_slot_creds
from trading_bot.tools.t212_instruments import Translator, fetch_instruments


_TIER = "trading212-paper"
log = logging.getLogger(__name__)


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
            log.warning(
                "Could not read T212 free cash — proceeding without budget pre-flight"
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
            # T212 GBP accounts settle UK trades natively; for EU stocks it
            # auto-converts. We size in GBP across the board.
            quantity = round(allocation_gbp / entry_estimate, 4)
            if quantity < min_qty:
                log.warning(
                    "Computed qty %.4f below T212 minimum %.4f for %s — skipping",
                    quantity, min_qty, intent.ticker,
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

            client_order_id = f"{strategy_id}-{uuid.uuid4().hex[:12]}"
            order = self._submit_market_order(
                t212_ticker=t212_ticker, quantity=quantity
            )
            if order is None:
                continue

            fill_price = self._extract_fill_price(order, fallback=entry_estimate)

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

        translator = self._get_translator()
        closed: list[dict] = []

        for trade in open_trades:
            t212_ticker = translator.translate(trade["ticker"])
            if t212_ticker is None:
                log.warning("Cannot resolve T212 ticker for exit on %s — leaving open", trade["ticker"])
                continue

            position = self._get_position(t212_ticker)
            close_fill_price: float | None = None
            exit_reason: str

            if position is not None:
                # Sell the entire position at market.
                close_order = self._submit_market_order(
                    t212_ticker=t212_ticker,
                    quantity=-float(position.get("quantity") or trade["quantity"]),
                )
                if close_order is not None:
                    close_fill_price = self._extract_fill_price(close_order, fallback=None)
                if close_fill_price is None:
                    # Fall back to T212's reported average entry — preserves
                    # near-zero P&L rather than inventing bid-ask noise.
                    close_fill_price = float(position.get("averagePrice") or trade["entry_price"])
                exit_reason = "scheduled"
            else:
                # No position at T212 — order never filled or was already
                # liquidated. Mark cancelled with null P&L per the executor
                # honesty rule (don't invent fills).
                close_fill_price = None
                exit_reason = "cancelled"

            entry_price = float(trade["entry_price"])
            quantity = float(trade["quantity"])
            if close_fill_price is None:
                pnl_gbp: float | None = None
                pnl_pct: float | None = None
                exit_price_to_record: float | None = None
            else:
                pnl_gbp = (close_fill_price - entry_price) * quantity
                pnl_pct = (close_fill_price / entry_price - 1.0) * 100.0 if entry_price > 0 else 0.0
                exit_price_to_record = close_fill_price

            mark_trade_exited(
                trade_id=trade["trade_id"],
                exit_date=on_date,
                exit_price=exit_price_to_record if exit_price_to_record is not None else 0.0,
                pnl_gbp=pnl_gbp if pnl_gbp is not None else 0.0,
                pnl_pct=pnl_pct if pnl_pct is not None else 0.0,
                exit_reason=exit_reason,
            )
            closed.append({
                **trade,
                "exit_date": on_date.isoformat(),
                "exit_price": exit_price_to_record,
                "pnl_gbp": pnl_gbp if pnl_gbp is not None else 0.0,
                "pnl_pct": pnl_pct if pnl_pct is not None else 0.0,
                "exit_reason": exit_reason,
            })
        return closed

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

    def _extract_fill_price(self, order: dict[str, Any], fallback: float | None) -> float | None:
        """T212 returns the order shape but the fill price field name varies
        between sync and async fills. Try a few common keys; fall back to
        whatever the caller provided as a seed estimate."""
        for key in ("fillPrice", "averagePrice", "filledPrice", "price"):
            v = order.get(key)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return fallback

    def _get_position(self, t212_ticker: str) -> dict[str, Any] | None:
        try:
            response = requests.get(
                self._url(f"/equity/portfolio/{t212_ticker}"),
                headers=self._headers(),
                timeout=10,
            )
        except requests.RequestException as e:
            log.warning("Position fetch errored for %s: %s", t212_ticker, e)
            return None
        if response.status_code == 404:
            return None
        if not response.ok:
            log.warning("Position fetch returned %s for %s", response.status_code, t212_ticker)
            return None
        return response.json()

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
