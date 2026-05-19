"""Print T212 order history for tickers stranded in today's exit run.

For each currently-open T212-paper trade in the ledger, fetch the last
50 history items from /equity/history/orders?ticker=... and report
BUY / SELL fills with submit + fill timestamps. Lets us see whether
T212's fill latency exceeds the bot's 60-second poll budget.

Run via the `recover-t212-strands` workflow with mode=diag (no writes).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime

import requests

from trading_bot.state import read_open_trades
from trading_bot.t212_slot import load_slot_creds
from trading_bot.tools.t212_instruments import Translator, fetch_instruments


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("t212_diag")


def _latency_seconds(submit_iso: str | None, fill_iso: str | None) -> float | None:
    if not submit_iso or not fill_iso:
        return None
    try:
        s = datetime.fromisoformat(submit_iso.replace("Z", "+00:00"))
        f = datetime.fromisoformat(fill_iso.replace("Z", "+00:00"))
        return (f - s).total_seconds()
    except (ValueError, TypeError):
        return None


def main() -> int:
    strands = [t for t in read_open_trades() if t.get("tier") == "trading212-paper"]
    if not strands:
        print("No open T212-paper trades in ledger.")
        return 0

    try:
        creds = load_slot_creds(1, demo=True)
    except RuntimeError as e:
        print(f"Slot 1 unreachable: {e}")
        return 1

    try:
        translator = Translator(fetch_instruments(creds))
    except Exception as e:
        print(f"instruments fetch failed: {e}")
        return 1

    print(f"\n=== T212 history diagnostic for {len(strands)} stranded trades ===\n")

    for i, trade in enumerate(strands):
        # T212 rate-limits /equity/history/orders aggressively; pace
        # ourselves to stay under their per-second budget. Two-second
        # spacing matched the empirical limit on slot 1.
        if i > 0:
            time.sleep(2.0)
        yf = trade.get("ticker")
        sid = trade.get("strategy_id")
        entry_oid = trade.get("broker_order_id")
        t212_ticker = translator.translate(yf) if yf else None
        if not t212_ticker:
            print(f"--- {yf} ({sid}): cannot resolve T212 ticker ---")
            continue

        # Up to 3 retries with linear backoff on 429
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
                print(f"--- {yf}: history fetch errored: {e} ---")
                r = None
                break
            if r.status_code != 429:
                break
            time.sleep(1.0 * (attempt + 1))
        if r is None:
            continue
        if not r.ok:
            print(f"--- {yf}: history returned {r.status_code}: {r.text[:160]} ---")
            continue

        try:
            body = r.json() or {}
        except json.JSONDecodeError:
            print(f"--- {yf}: non-JSON response ---")
            continue

        items = body.get("items") if isinstance(body, dict) else body
        if not isinstance(items, list) or not items:
            print(f"--- {yf}: 0 history rows returned ---")
            continue

        print(f"--- {yf} / T212 {t212_ticker} / sid={sid} / entry_oid={entry_oid} ---")
        # Sort by createdAt desc
        items.sort(
            key=lambda it: ((it.get("order") or {}).get("createdAt") or ""),
            reverse=True,
        )
        for it in items[:15]:
            order = it.get("order") or {}
            fill = it.get("fill") or {}
            oid = order.get("id")
            side = order.get("side") or "?"
            otype = order.get("type") or "?"
            status = order.get("status") or "?"
            qty = order.get("filledQuantity") or order.get("quantity") or "?"
            submit_iso = order.get("createdAt") or ""
            fill_iso = fill.get("filledAt") or fill.get("fillTime") or ""
            price = fill.get("price")
            lat = _latency_seconds(submit_iso, fill_iso)
            lat_s = f"{lat:.1f}s" if lat is not None else "—"
            mark = "  *" if str(oid) == str(entry_oid) else "   "
            print(
                f"{mark}id={oid}  side={side:4s}  type={otype:6s}  status={status:9s}  "
                f"qty={qty}  price={price}  submit={submit_iso}  fill={fill_iso}  Δ={lat_s}"
            )
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
