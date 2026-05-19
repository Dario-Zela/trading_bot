"""Print live balances for every configured broker slot.

Reads credentials from env, calls each broker's account-info endpoint,
and emits a structured summary. Used by the `account-status` workflow
as an on-demand diagnostic — compare against the bot's tracked P&L on
the dashboard to spot drift (fees not captured, FX mislabelling, etc.).

Exits 0 even when individual endpoints fail (this is a diagnostic, not
a gate).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

import requests

log = logging.getLogger("account_status")


def _t212_cash(slot: int) -> dict[str, Any] | None:
    """Query the T212 demo account /equity/account/cash endpoint."""
    from trading_bot.t212_slot import load_slot_creds
    try:
        creds = load_slot_creds(slot, demo=True)
    except RuntimeError as e:
        log.info("T212 slot %d: %s", slot, e)
        return None
    try:
        r = requests.get(
            f"{creds.base_url}/equity/account/cash",
            headers={"Authorization": creds.auth_header(), "Accept": "application/json"},
            timeout=15,
        )
    except requests.RequestException as e:
        log.warning("T212 slot %d cash GET failed: %s", slot, e)
        return None
    if not r.ok:
        log.warning("T212 slot %d returned %s: %s", slot, r.status_code, r.text[:200])
        return None
    try:
        return r.json() or {}
    except json.JSONDecodeError:
        log.warning("T212 slot %d returned non-JSON: %s", slot, r.text[:200])
        return None


def _t212_portfolio(slot: int) -> list[dict] | None:
    """List open positions on the T212 demo account."""
    from trading_bot.t212_slot import load_slot_creds
    try:
        creds = load_slot_creds(slot, demo=True)
    except RuntimeError:
        return None
    try:
        r = requests.get(
            f"{creds.base_url}/equity/portfolio",
            headers={"Authorization": creds.auth_header(), "Accept": "application/json"},
            timeout=15,
        )
    except requests.RequestException as e:
        log.warning("T212 slot %d portfolio GET failed: %s", slot, e)
        return None
    if not r.ok:
        log.warning("T212 slot %d portfolio returned %s", slot, r.status_code)
        return None
    try:
        return r.json() or []
    except json.JSONDecodeError:
        return None


def _alpaca_account(slot: int) -> dict[str, Any] | None:
    """Query Alpaca's /v2/account for the given paper slot."""
    from trading_bot.alpaca_slot import load_slot_creds
    try:
        creds = load_slot_creds(slot)
    except RuntimeError as e:
        log.info("Alpaca slot %d: %s", slot, e)
        return None
    try:
        r = requests.get(
            f"{creds.trading_base_url}/v2/account",
            headers={
                "APCA-API-KEY-ID": creds.api_key,
                "APCA-API-SECRET-KEY": creds.api_secret,
                "Accept": "application/json",
            },
            timeout=15,
        )
    except requests.RequestException as e:
        log.warning("Alpaca slot %d /v2/account failed: %s", slot, e)
        return None
    if not r.ok:
        log.warning("Alpaca slot %d returned %s: %s", slot, r.status_code, r.text[:200])
        return None
    try:
        return r.json() or {}
    except json.JSONDecodeError:
        return None


def _alpaca_positions(slot: int) -> list[dict] | None:
    from trading_bot.alpaca_slot import load_slot_creds
    try:
        creds = load_slot_creds(slot)
    except RuntimeError:
        return None
    try:
        r = requests.get(
            f"{creds.trading_base_url}/v2/positions",
            headers={
                "APCA-API-KEY-ID": creds.api_key,
                "APCA-API-SECRET-KEY": creds.api_secret,
                "Accept": "application/json",
            },
            timeout=15,
        )
    except requests.RequestException:
        return None
    if not r.ok:
        return None
    try:
        return r.json() or []
    except json.JSONDecodeError:
        return None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"\n=== Broker account status — {now} ===\n")

    # T212 — try slots 1-3 (Practice account, single slot is the common case)
    for slot in (1, 2, 3):
        cash = _t212_cash(slot)
        if cash is None:
            continue
        portfolio = _t212_portfolio(slot) or []
        print(f"--- Trading212 demo · slot {slot} ---")
        # Cash response shape: {"free", "total", "invested", "ppl", "result", ...}
        # `ppl` = open-position P&L; `result` = closed P&L since reset
        print(f"  total equity         £{cash.get('total', '—')}")
        print(f"  free cash            £{cash.get('free', '—')}")
        print(f"  invested             £{cash.get('invested', '—')}")
        print(f"  open P&L (PPL)       £{cash.get('ppl', '—')}")
        print(f"  realised result      £{cash.get('result', '—')}")
        print(f"  open positions       {len(portfolio)}")
        for p in portfolio[:6]:
            ticker = p.get("ticker", "—")
            qty = p.get("quantity", "—")
            ppl = p.get("ppl", "—")
            cp = p.get("currentPrice", p.get("price", "—"))
            print(f"    · {ticker:<16} qty={qty}  current={cp}  PPL=£{ppl}")
        if len(portfolio) > 6:
            print(f"    ... ({len(portfolio) - 6} more)")
        print()

    # Alpaca — slots 1-3
    for slot in (1, 2, 3):
        acct = _alpaca_account(slot)
        if acct is None:
            continue
        positions = _alpaca_positions(slot) or []
        print(f"--- Alpaca paper · slot {slot} ---")
        currency = acct.get("currency", "USD")
        print(f"  account currency     {currency}")
        print(f"  equity               {currency} {acct.get('equity', '—')}")
        print(f"  last_equity          {currency} {acct.get('last_equity', '—')}")
        print(f"  cash                 {currency} {acct.get('cash', '—')}")
        print(f"  portfolio_value      {currency} {acct.get('portfolio_value', '—')}")
        print(f"  buying_power         {currency} {acct.get('buying_power', '—')}")
        print(f"  account_number       {acct.get('account_number', '—')}")
        print(f"  open positions       {len(positions)}")
        for p in positions[:6]:
            sym = p.get("symbol", "—")
            qty = p.get("qty", "—")
            mv = p.get("market_value", "—")
            upl = p.get("unrealized_pl", "—")
            print(f"    · {sym:<8} qty={qty}  MV={mv}  unrealised={upl}")
        if len(positions) > 6:
            print(f"    ... ({len(positions) - 6} more)")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
