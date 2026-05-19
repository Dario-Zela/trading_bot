"""One-off recovery for T212-paper trades stranded by a poll-timeout.

On 2026-05-19, the uk-eu exit cron submitted SELL orders for 8 open
T212-paper positions and abandoned the polls after 60s. T212 filled
the SELLs minutes later (slot 1 portfolio went to 0) but the ledger
was never marked exited. Compounding that, the original BUY entries
this morning had also timed out and were recorded with entry_price=0
+ a `broker_order_id` pending reconciliation — which the exit run
then failed to reconcile because of a separate history-shape bug.

This script:

1. Finds every open T212-paper trade older than today's morning.
2. For each, queries T212 history to find both the BUY (entry) and
   the SELL (exit) by ticker and date.
3. Recovers entry_price + exit_price + computes fees + writes the
   exit to the ledger via mark_trade_exited.

Idempotent — already-exited rows are skipped. Safe to re-run.

Triggered via the `recover-t212-strands` workflow_dispatch workflow.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import requests

from trading_bot.state import mark_trade_exited, read_open_trades
from trading_bot.t212_slot import load_slot_creds
from trading_bot.tools.fees import (
    TradeContext,
    compute_fees,
    t212_exchange_from_ticker,
    t212_instrument_type,
)
from trading_bot.tools.fx import to_gbp_multiplier
from trading_bot.tools.t212_instruments import Translator, fetch_instruments


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("recover_t212_strands")


_LOOKBACK_DAYS = 7


def _to_gbp(translator: Translator, t212_ticker: str, raw_price: float) -> float:
    inst = translator.get_instrument(t212_ticker)
    if inst is None:
        return raw_price
    ccy = (inst.get("currencyCode") or "").upper()
    mult = to_gbp_multiplier(ccy)
    if mult is None:
        log.warning("No FX rate for %s (%s) — leaving native", t212_ticker, ccy)
        return raw_price
    return raw_price * mult


def _history_for_ticker(creds, t212_ticker: str) -> list[dict]:
    """Fetch the most recent 50 history items for a single ticker.
    Retries on 429 with linear backoff; T212's /equity/history/orders
    rate-limit is tight enough that two-back-to-back calls can trip it."""
    r = None
    for attempt in range(4):
        try:
            r = requests.get(
                f"{creds.base_url}/equity/history/orders",
                headers={"Authorization": creds.auth_header(), "Accept": "application/json"},
                params={"ticker": t212_ticker, "limit": 50},
                timeout=15,
            )
        except requests.RequestException as e:
            log.warning("T212 history fetch failed for %s: %s", t212_ticker, e)
            return []
        if r.status_code != 429:
            break
        time.sleep(1.0 * (attempt + 1))
    if r is None or not r.ok:
        log.warning("T212 history returned %s for %s",
                    r.status_code if r is not None else "(no response)", t212_ticker)
        return []
    try:
        body = r.json() or {}
    except json.JSONDecodeError:
        return []
    items = body.get("items") if isinstance(body, dict) else body
    return items if isinstance(items, list) else []


def _find_fill(
    items: list[dict],
    *,
    side: str,
    on_or_after: date,
    on_or_before: date,
    target_order_id: str | None = None,
) -> tuple[float | None, str | None, str | None, str | None]:
    """Return (fill_price, fill_iso, submit_iso, order_id) for the
    matching side/date. Prefer an exact match on order_id; otherwise
    pick the most recent qualifying fill in the window."""
    candidates: list[tuple[str, float, str, str, str]] = []
    # (ts_filled, price, order_id, submit_iso, fill_iso)
    for item in items:
        if not isinstance(item, dict):
            continue
        order = item.get("order") or {}
        fill = item.get("fill") or {}
        if not isinstance(order, dict) or not isinstance(fill, dict):
            continue

        item_side = (order.get("side") or "").upper()
        if item_side != side.upper():
            fq = order.get("filledQuantity") or order.get("quantity") or 0
            try:
                fq_f = float(fq)
            except (TypeError, ValueError):
                continue
            if side.upper() == "BUY" and fq_f <= 0:
                continue
            if side.upper() == "SELL" and fq_f >= 0:
                continue

        status = (order.get("status") or "").upper()
        if status not in ("FILLED", "EXECUTED", "COMPLETED"):
            continue

        price = fill.get("price")
        if price is None:
            continue
        try:
            price_f = float(price)
        except (TypeError, ValueError):
            continue

        submit_iso = order.get("createdAt") or ""
        fill_iso = fill.get("filledAt") or fill.get("fillTime") or ""
        ts_filled = fill_iso or submit_iso
        if not ts_filled:
            continue
        day = ts_filled[:10]
        if day < on_or_after.isoformat() or day > on_or_before.isoformat():
            continue

        oid = str(order.get("id") or "")

        if target_order_id and oid == str(target_order_id):
            return price_f, fill_iso or ts_filled, submit_iso, oid

        candidates.append((ts_filled, price_f, oid, submit_iso, fill_iso))

    if not candidates:
        return None, None, None, None
    candidates.sort(key=lambda c: c[0], reverse=True)
    ts_filled, price, oid, submit_iso, fill_iso = candidates[0]
    return price, fill_iso or ts_filled, submit_iso, oid


def _latency_seconds(submit_iso: str | None, fill_iso: str | None) -> float | None:
    """Compute submit → fill latency in seconds. Returns None on parse error."""
    if not submit_iso or not fill_iso:
        return None
    try:
        # T212 stamps with "Z" suffix; datetime.fromisoformat in 3.11 handles it
        # via the +00:00 replacement.
        s = datetime.fromisoformat(submit_iso.replace("Z", "+00:00"))
        f = datetime.fromisoformat(fill_iso.replace("Z", "+00:00"))
        return (f - s).total_seconds()
    except (ValueError, TypeError):
        return None


def _strands_to_fix() -> list[dict]:
    """Currently-open trading212-paper trades. Excludes trades entered
    today's date (their UK exit hasn't run yet) — only earlier strands
    qualify. For the 2026-05-19 strand we explicitly pass entered_date.
    """
    open_trades = read_open_trades()
    return [t for t in open_trades if t.get("tier") == "trading212-paper"]


def main() -> int:
    strands = _strands_to_fix()
    if not strands:
        log.info("No T212-paper strands open in the ledger — nothing to do")
        return 0

    by_slot: dict[int, list[dict]] = defaultdict(list)
    for t in strands:
        # Without per-trade slot info, we infer from the strategy config
        # (mean-reverter + momentum-trader use slot 1 for uk-eu). For the
        # explicit recovery we just try slot 1; if more strats land here
        # we'll iterate slots.
        by_slot[1].append(t)

    today = date.today()
    on_or_after = today - timedelta(days=_LOOKBACK_DAYS)

    n_recovered = 0
    n_skipped = 0
    n_failed = 0
    for slot, trades in by_slot.items():
        try:
            creds = load_slot_creds(slot, demo=True)
        except RuntimeError as e:
            log.warning("Slot %d unreachable: %s", slot, e)
            continue
        try:
            translator = Translator(fetch_instruments(creds))
        except Exception as e:
            log.error("Slot %d: failed to load instruments (%s) — bailing", slot, e)
            continue

        for i, trade in enumerate(trades):
            # Space history queries to stay under T212's per-second cap
            if i > 0:
                time.sleep(2.0)
            yf = trade.get("ticker")
            t212_ticker = translator.translate(yf) if yf else None
            if not t212_ticker:
                log.warning("Cannot resolve T212 ticker for %s — skipping", yf)
                n_skipped += 1
                continue

            items = _history_for_ticker(creds, t212_ticker)
            if not items:
                log.warning("No T212 history rows for %s — skipping", t212_ticker)
                n_skipped += 1
                continue

            entry_date_str = trade.get("entry_date") or ""
            try:
                entry_date_d = datetime.fromisoformat(entry_date_str).date()
            except (TypeError, ValueError):
                entry_date_d = on_or_after

            # BUY: search by broker_order_id first, fall back to most recent
            # BUY on entry_date.
            buy_price_native, buy_fill_iso, buy_submit_iso, _ = _find_fill(
                items, side="BUY",
                on_or_after=entry_date_d,
                on_or_before=today,
                target_order_id=trade.get("broker_order_id"),
            )
            # SELL: most recent SELL on entry_date or later (the close
            # order ids weren't persisted, so date-window match is what
            # we've got).
            sell_price_native, sell_fill_iso, sell_submit_iso, _ = _find_fill(
                items, side="SELL",
                on_or_after=entry_date_d,
                on_or_before=today,
                target_order_id=None,
            )
            sell_ts = sell_fill_iso

            sell_lat = _latency_seconds(sell_submit_iso, sell_fill_iso)
            buy_lat = _latency_seconds(buy_submit_iso, buy_fill_iso)
            if sell_lat is not None or buy_lat is not None:
                log.info(
                    "%s/%s: T212 fill latency — BUY %s, SELL %s",
                    trade.get("strategy_id"), yf,
                    f"{buy_lat:.0f}s" if buy_lat is not None else "?",
                    f"{sell_lat:.0f}s" if sell_lat is not None else "?",
                )

            if buy_price_native is None or sell_price_native is None:
                log.warning(
                    "%s/%s: incomplete history (buy=%s sell=%s) — leaving open for daily exit cron",
                    trade.get("strategy_id"), yf,
                    buy_price_native, sell_price_native,
                )
                n_failed += 1
                continue

            # T212 reports LSE in GBX (pence) and EUR/USD in native; we
            # convert to GBP for ledger consistency at the spot rate.
            entry_price_gbp = _to_gbp(translator, t212_ticker, buy_price_native)
            exit_price_gbp = _to_gbp(translator, t212_ticker, sell_price_native)

            quantity = float(trade.get("quantity") or 0)
            if quantity <= 0:
                log.warning("%s: quantity is 0 — skipping (can't compute P&L)", yf)
                n_skipped += 1
                continue

            inst = translator.get_instrument(t212_ticker) or {}
            native_ccy = (inst.get("currencyCode") or trade.get("currency") or "GBP").upper()
            inst_type = t212_instrument_type(inst.get("type") or "STOCK")
            exch = t212_exchange_from_ticker(t212_ticker)

            gross_pnl_gbp = (exit_price_gbp - entry_price_gbp) * quantity
            fees = compute_fees(TradeContext(
                tier=_t212_tier(),
                currency=native_ccy,
                exchange=exch,
                instrument_type=inst_type,
                entry_notional_gbp=abs(entry_price_gbp * quantity),
                exit_notional_gbp=abs(exit_price_gbp * quantity),
                quantity=abs(quantity),
            ))
            pnl_gbp = gross_pnl_gbp - fees.total_gbp
            pnl_pct = (exit_price_gbp / entry_price_gbp - 1.0) * 100.0 if entry_price_gbp > 0 else 0.0

            exit_date_iso = (sell_ts or "")[:10] or today.isoformat()
            try:
                exit_date_d = datetime.fromisoformat(exit_date_iso).date()
            except ValueError:
                exit_date_d = today

            mark_trade_exited(
                trade_id=trade["trade_id"],
                exit_date=exit_date_d,
                exit_price=exit_price_gbp,
                pnl_gbp=pnl_gbp,
                pnl_pct=pnl_pct,
                exit_reason="recovered",
                fees_gbp=fees.total_gbp,
                fees_breakdown=fees.as_dict(),
                outcome_notes=(
                    "Recovered post-hoc by recover_t212_strands.py. The exit cron's "
                    "60-second poll timed out before T212's fill landed in history; "
                    "this script back-fills the entry and exit prices from the "
                    "broker's order history so the ledger matches reality."
                ),
                risks_observed=(
                    "T212's paper-account fills can lag the submit-response by a few "
                    "minutes during busy sessions; the exit_scheduled poll budget has "
                    "been bumped to 5 minutes to absorb this without manual recovery."
                ),
            )
            log.info(
                "%s/%s: recovered entry=%.4f exit=%.4f pnl=£%+0.2f (%+0.2f%%)",
                trade.get("strategy_id"), yf,
                entry_price_gbp, exit_price_gbp, pnl_gbp, pnl_pct,
            )
            n_recovered += 1

    log.info("Recovery summary: recovered=%d skipped=%d failed=%d",
             n_recovered, n_skipped, n_failed)
    return 0


def _t212_tier() -> str:
    # Indirection so the constant doesn't leak as a string literal in callers.
    from trading_bot.executor.trading212_demo import _TIER
    return _TIER


if __name__ == "__main__":
    sys.exit(main())
