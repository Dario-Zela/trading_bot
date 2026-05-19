"""Phase 8D — trailing stops for Alpaca paper bracket orders.

Once a bracket position is in profit by at least `activation_pct`, walk
the stop child up to `current_price - trail_pct%`. Doesn't lower the
stop (a falling price still hits the original stop) and doesn't touch
the take-profit leg.

Scheduled by `.github/workflows/midday-trail.yml` at ~17:00 UK
(mid-US-session) — half-day intraday move is the sweet spot for
catching a "ran +3% in the morning, gave it back at close" pattern.

T212 doesn't support bracket orders the same way and uses scheduled
close-at-market exits, so this is Alpaca-only for now.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from trading_bot.alpaca_slot import AlpacaCreds, load_slot_creds

log = logging.getLogger(__name__)


# Activation: trail only kicks in once we're up this much from entry
DEFAULT_ACTIVATION_PCT = 1.0
# Trail width: new stop sits this far below current price
DEFAULT_TRAIL_PCT = 0.8
# Polls per PATCH — Alpaca usually accepts immediately but sometimes
# returns 422 transiently during heavy order activity.
_PATCH_RETRIES = 2


@dataclass
class TrailAction:
    """One trail adjustment for the run log."""
    symbol: str
    slot: int
    entry_price: float
    current_price: float
    pct_up: float
    old_stop: float
    new_stop: float
    status: str = "applied"        # applied / skipped / failed
    reason: str = ""

    def __str__(self) -> str:
        return (
            f"  {self.status.upper():<8} slot={self.slot} {self.symbol:<6} "
            f"entry={self.entry_price:.2f} now={self.current_price:.2f} "
            f"(+{self.pct_up:.2f}%) stop {self.old_stop:.2f} → {self.new_stop:.2f} "
            f"{self.reason}"
        )


def trail_alpaca_slots(
    slots: list[int] | None = None,
    *,
    activation_pct: float = DEFAULT_ACTIVATION_PCT,
    trail_pct: float = DEFAULT_TRAIL_PCT,
) -> list[TrailAction]:
    """Scan every configured Alpaca slot, modify bracket-child stops
    upward on positions in profit. Returns audit log."""
    slots = slots or [1, 2, 3]
    out: list[TrailAction] = []
    for slot in slots:
        try:
            creds = load_slot_creds(slot)
        except RuntimeError:
            continue
        try:
            out.extend(_trail_one_slot(creds, activation_pct, trail_pct))
        except Exception as e:
            log.warning("trail: slot %d failed: %s", slot, e)
    return out


def _trail_one_slot(creds: AlpacaCreds, activation_pct: float, trail_pct: float) -> list[TrailAction]:
    """Inner loop for one Alpaca slot."""
    base = creds.trading_base_url
    headers = {
        "APCA-API-KEY-ID": creds.api_key,
        "APCA-API-SECRET-KEY": creds.api_secret,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    actions: list[TrailAction] = []

    # 1) List current positions — gives us entry + current price per symbol
    try:
        r = requests.get(f"{base}/v2/positions", headers=headers, timeout=15)
        if not r.ok:
            log.warning("trail: slot %d /v2/positions returned %d", creds.slot, r.status_code)
            return actions
        positions = r.json() or []
    except requests.RequestException as e:
        log.warning("trail: slot %d positions fetch failed: %s", creds.slot, e)
        return actions
    if not positions:
        return actions

    # 2) List open bracket-child stops so we can find each position's stop
    try:
        r = requests.get(
            f"{base}/v2/orders",
            params={"status": "open", "nested": "true", "limit": 100},
            headers=headers, timeout=15,
        )
        open_orders = r.json() if r.ok else []
    except requests.RequestException:
        open_orders = []

    # Index stop orders by symbol so we can pair them with positions
    stop_orders_by_symbol: dict[str, list[dict]] = {}
    for o in open_orders or []:
        if not isinstance(o, dict):
            continue
        # Top-level open orders that are stop-type bracket children show
        # up alongside their parent in nested=true; we look at both the
        # parent's legs and standalone orders for resilience.
        sym = (o.get("symbol") or "").upper()
        otype = (o.get("type") or o.get("order_type") or "").lower()
        if otype in ("stop", "stop_limit"):
            stop_orders_by_symbol.setdefault(sym, []).append(o)
        for leg in o.get("legs") or []:
            if not isinstance(leg, dict):
                continue
            leg_type = (leg.get("type") or leg.get("order_type") or "").lower()
            if leg_type in ("stop", "stop_limit") and (leg.get("status") or "").lower() in ("new", "accepted", "held"):
                stop_orders_by_symbol.setdefault((leg.get("symbol") or sym).upper(), []).append(leg)

    # 3) For each position, decide whether to trail
    for pos in positions:
        symbol = (pos.get("symbol") or "").upper()
        try:
            entry_price = float(pos.get("avg_entry_price") or 0)
            current_price = float(pos.get("current_price") or 0)
        except (TypeError, ValueError):
            continue
        if entry_price <= 0 or current_price <= 0:
            continue
        pct_up = (current_price / entry_price - 1.0) * 100.0

        if pct_up < activation_pct:
            continue   # not far enough into profit yet

        stops = stop_orders_by_symbol.get(symbol) or []
        if not stops:
            actions.append(TrailAction(
                symbol=symbol, slot=creds.slot, entry_price=entry_price,
                current_price=current_price, pct_up=pct_up,
                old_stop=0.0, new_stop=0.0,
                status="skipped",
                reason="no bracket stop found (position may have no stop, or stop already filled)",
            ))
            continue

        for stop in stops:
            try:
                old_stop = float(stop.get("stop_price") or 0)
            except (TypeError, ValueError):
                continue
            target_stop = round(current_price * (1.0 - trail_pct / 100.0), 2)
            if target_stop <= old_stop:
                actions.append(TrailAction(
                    symbol=symbol, slot=creds.slot, entry_price=entry_price,
                    current_price=current_price, pct_up=pct_up,
                    old_stop=old_stop, new_stop=target_stop,
                    status="skipped",
                    reason=f"would not raise stop ({target_stop} <= {old_stop})",
                ))
                continue

            ok, err = _patch_stop(base, headers, str(stop.get("id")), target_stop)
            actions.append(TrailAction(
                symbol=symbol, slot=creds.slot, entry_price=entry_price,
                current_price=current_price, pct_up=pct_up,
                old_stop=old_stop, new_stop=target_stop,
                status="applied" if ok else "failed",
                reason="" if ok else err,
            ))
    return actions


def _patch_stop(base: str, headers: dict, order_id: str, new_stop: float) -> tuple[bool, str]:
    """PATCH the stop_price on an open bracket child. Retries once on
    transient 422 (Alpaca occasionally rejects during heavy load)."""
    body = {"stop_price": str(new_stop)}
    last_err = ""
    for attempt in range(_PATCH_RETRIES + 1):
        try:
            r = requests.patch(f"{base}/v2/orders/{order_id}", json=body, headers=headers, timeout=15)
        except requests.RequestException as e:
            last_err = f"network error: {e}"
            time.sleep(0.5)
            continue
        if r.ok:
            return True, ""
        last_err = f"HTTP {r.status_code}: {r.text[:150]}"
        if r.status_code != 422:
            break
        time.sleep(0.8)
    return False, last_err


def format_log(actions: list[TrailAction]) -> str:
    if not actions:
        return "(no positions in profit beyond activation threshold)"
    return "\n".join(str(a) for a in actions)
