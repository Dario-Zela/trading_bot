"""One-off recovery for Alpaca-paper ledger rows that got marked
`cancelled` with exit_price=0 because the exit_scheduled "no
position" branch couldn't find a filled bracket leg.

This happens specifically when the position was closed by a plain
market sell on a prior session (not by a bracket trigger), and the
ledger update with the real exit_price was dropped by smart-merge
or otherwise didn't reach main. The recovery branch tried to
extract a filled stop/take-profit leg from the parent order, found
none, and gave up.

This script walks the ledger for rows matching that pattern, queries
Alpaca's `/v2/orders` history for a recent SELL on the ticker, and
back-fills exit_price + pnl_gbp + pnl_pct via `mark_trade_exited`.

Idempotent — rows already with a non-zero exit_price are skipped.
"""
from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any

import requests

from trading_bot.alpaca_slot import load_slot_creds
from trading_bot.state import mark_trade_exited
from trading_bot.state.ledger import _iter_records   # noqa: WPS437
from trading_bot.tools.fees import TradeContext, compute_fees
from trading_bot.tools.fx import to_gbp_multiplier


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("recover_alpaca_cancelled")


_TIER = "alpaca-paper"
_LOOKBACK_DAYS = 7


def _strategy_slot(sid: str) -> int:
    """Crude slot mapping — the bot stores per-strategy alpaca_slot in
    the config. For US recovery, slot 1 is the default; if a strategy
    is bound to slot 2 or 3 we read it from the config."""
    import yaml
    from pathlib import Path
    cfg_path = Path("strategies") / sid / "config.yaml"
    if not cfg_path.exists():
        return 1
    try:
        raw = yaml.safe_load(cfg_path.read_text())
    except Exception:
        return 1
    runs_in = raw.get("runs_in") or []
    if isinstance(runs_in, list):
        for entry in runs_in:
            if entry.get("region") == "us" and entry.get("tier") == _TIER:
                return int(entry.get("alpaca_slot") or 1)
    return int(raw.get("alpaca_slot") or 1)


def _find_recent_sell(creds, ticker: str, today: date) -> dict[str, Any] | None:
    after = (today - timedelta(days=_LOOKBACK_DAYS)).isoformat()
    try:
        r = requests.get(
            f"{creds.trading_base_url}/v2/orders",
            headers={
                "APCA-API-KEY-ID": creds.api_key,
                "APCA-API-SECRET-KEY": creds.api_secret,
                "Accept": "application/json",
            },
            params={
                "status": "closed",
                "symbols": ticker,
                "side": "sell",
                "after": after,
                "limit": 50,
                "direction": "desc",
            },
            timeout=15,
        )
    except requests.RequestException as e:
        log.warning("Alpaca order-history fetch failed for %s: %s", ticker, e)
        return None
    if not r.ok:
        log.warning("Alpaca order-history returned %s for %s", r.status_code, ticker)
        return None
    try:
        orders = r.json() or []
    except Exception:
        return None
    if not isinstance(orders, list):
        return None
    for order in orders:
        if not isinstance(order, dict):
            continue
        if (order.get("status") or "").lower() != "filled":
            continue
        if (order.get("side") or "").lower() != "sell":
            continue
        if order.get("filled_avg_price") is None:
            continue
        return order
    return None


def main() -> int:
    # Pull every Alpaca-paper row that's been marked cancelled with
    # exit_price=0 — that's the symptom of the smart-merge race.
    targets: list[dict] = []
    for rec in _iter_records():
        if rec.get("tier") != _TIER:
            continue
        if rec.get("exit_reason") != "cancelled":
            continue
        try:
            exit_price = float(rec.get("exit_price") or 0)
        except (TypeError, ValueError):
            continue
        if exit_price != 0:
            continue
        targets.append(rec)

    if not targets:
        log.info("No Alpaca cancelled-with-£0 rows to recover.")
        return 0

    log.info("Found %d Alpaca rows to recover", len(targets))

    # Group by strategy → slot so we can amortise the Alpaca creds load
    by_slot: dict[int, list[dict]] = {}
    for t in targets:
        slot = _strategy_slot(t.get("strategy_id") or "")
        by_slot.setdefault(slot, []).append(t)

    n_recovered = 0
    n_skipped = 0
    for slot, rows in by_slot.items():
        try:
            creds = load_slot_creds(slot)
        except RuntimeError as e:
            log.warning("Alpaca slot %d unreachable: %s", slot, e)
            continue

        for trade in rows:
            ticker = trade.get("ticker")
            if not ticker:
                continue
            try:
                exit_date_str = trade.get("exit_date") or ""
                ed = datetime.fromisoformat(exit_date_str).date()
            except (TypeError, ValueError):
                ed = date.today()

            order = _find_recent_sell(creds, ticker, ed)
            if order is None:
                log.warning("%s/%s: no recent SELL in Alpaca history — leaving as cancelled",
                            trade.get("strategy_id"), ticker)
                n_skipped += 1
                continue

            try:
                exit_price_usd = float(order["filled_avg_price"])
                entry_price_usd = float(trade.get("entry_price") or 0)
                quantity = float(trade.get("quantity") or 0)
            except (TypeError, ValueError, KeyError):
                n_skipped += 1
                continue

            if quantity <= 0 or entry_price_usd <= 0:
                n_skipped += 1
                continue

            pnl_usd = (exit_price_usd - entry_price_usd) * quantity
            usd_to_gbp = to_gbp_multiplier("USD") or 0.79
            gross_pnl_gbp = pnl_usd * usd_to_gbp
            entry_notional_gbp = abs(entry_price_usd * quantity * usd_to_gbp)
            exit_notional_gbp = abs(exit_price_usd * quantity * usd_to_gbp)
            fees = compute_fees(TradeContext(
                tier=_TIER,
                currency=(trade.get("currency") or "USD").upper(),
                exchange=trade.get("exchange") or "NYSE",
                instrument_type=trade.get("instrument_type") or "share",
                entry_notional_gbp=entry_notional_gbp,
                exit_notional_gbp=exit_notional_gbp,
                quantity=abs(quantity),
            ))
            pnl_gbp = gross_pnl_gbp - fees.total_gbp
            pnl_pct = (
                (exit_price_usd / entry_price_usd - 1.0) * 100.0
                if entry_price_usd > 0 else 0.0
            )

            mark_trade_exited(
                trade_id=trade["trade_id"],
                exit_date=ed,
                exit_price=exit_price_usd,
                pnl_gbp=pnl_gbp,
                pnl_pct=pnl_pct,
                exit_reason="recovered",
                fees_gbp=fees.total_gbp,
                fees_breakdown=fees.as_dict(),
                outcome_notes=(
                    "Recovered post-hoc by recover_alpaca_cancelled.py. The "
                    "exit_scheduled 'no position' branch couldn't extract a "
                    "filled bracket leg because the position was closed via "
                    "plain market sell, not via stop/TP trigger. This script "
                    "back-fills the exit price from Alpaca's order history."
                ),
                risks_observed=(
                    "Bracket-leg-only recovery missed market-sell exits on a "
                    "ledger-update race; the fallback `_find_recent_sell` in "
                    "alpaca_paper.py now covers this on future runs."
                ),
            )
            log.info(
                "%s/%s: recovered entry=%.4f exit=%.4f pnl=£%+0.2f (%+0.2f%%)",
                trade.get("strategy_id"), ticker,
                entry_price_usd, exit_price_usd, pnl_gbp, pnl_pct,
            )
            n_recovered += 1

    log.info("Recovery summary: recovered=%d skipped=%d", n_recovered, n_skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
