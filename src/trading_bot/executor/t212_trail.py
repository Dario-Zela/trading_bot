"""Phase 8D — broker-side trailing stops on Trading 212 Invest/ISA.

T212's Invest API supports standalone STOP orders
(`POST /api/v0/equity/orders/stop`) but **not** bracket orders and
**no PATCH/PUT** to modify a stop. Trailing on T212 is therefore:

1. Read open positions + open orders for the slot.
2. For each position in profit ≥ `activation_pct`, compute the new
   stop at `current_price × (1 - trail_pct/100)`.
3. If a stop order already exists for that symbol and its
   `stopPrice` is already at-or-above the new target, skip (we don't
   lower stops).
4. Otherwise: cancel any existing stop for that symbol, then place
   a fresh stop at the new target.

The original scheduled close-at-market remains the safety net: even
if the trail never fires, the position exits at the close.

Source: https://docs.trading212.com/api/orders confirms support for
standalone STOP orders + cancel-and-replace as the modify pattern.
Trailing stops are CFD-only on T212; we implement our own.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

from trading_bot.t212_slot import T212Creds, load_slot_creds

log = logging.getLogger(__name__)


DEFAULT_ACTIVATION_PCT = 1.0
DEFAULT_TRAIL_PCT = 0.8
_CANCEL_AFTER_PLACE_RETRIES = 2

# Phase 10A — UK LSE shares pay 0.5% stamp duty on every entry. If the
# trail fires and the strategy re-buys tomorrow we pay it *again*. To
# keep firing economic, only trail UK shares when we're already up
# enough that the post-stamp-duty net is meaningfully positive.
UK_SHARE_ACTIVATION_PCT = 1.8
UK_SHARE_TRAIL_PCT = 0.6


@dataclass
class T212TrailAction:
    ticker: str
    slot: int
    entry_price: float       # from ledger / portfolio average
    current_price: float
    pct_up: float
    old_stop: float | None
    new_stop: float
    status: str              # placed / tightened / skipped / failed
    reason: str = ""

    def __str__(self) -> str:
        old = f"{self.old_stop:.4f}" if self.old_stop else "—"
        return (
            f"  {self.status.upper():<10} slot={self.slot} {self.ticker:<14} "
            f"entry={self.entry_price:.4f} now={self.current_price:.4f} "
            f"(+{self.pct_up:.2f}%) stop {old} → {self.new_stop:.4f} "
            f"{self.reason}"
        )


def trail_t212_slots(
    slots: list[int] | None = None,
    *,
    activation_pct: float = DEFAULT_ACTIVATION_PCT,
    trail_pct: float = DEFAULT_TRAIL_PCT,
) -> list[T212TrailAction]:
    slots = slots or [1, 2, 3]
    out: list[T212TrailAction] = []
    for slot in slots:
        try:
            creds = load_slot_creds(slot, demo=True)
        except RuntimeError:
            continue
        try:
            out.extend(_trail_one_slot(creds, activation_pct, trail_pct))
        except Exception as e:
            log.warning("t212 trail: slot %d failed: %s", slot, e)
    return out


def _trail_one_slot(creds: T212Creds, activation_pct: float, trail_pct: float) -> list[T212TrailAction]:
    base = creds.base_url
    headers = {
        "Authorization": creds.auth_header(),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    actions: list[T212TrailAction] = []

    # Load the cached instrument metadata so we know which positions
    # are UK LSE shares (stamp-duty-charged) vs ETFs / non-UK. The
    # cache file is populated at order time and persists across runs.
    instruments_by_ticker = _load_instruments_cache()

    # Phase 12D — figure out which T212 positions are multi-day so we can
    # place GTC stops on them (instead of DAY stops that would expire each
    # session and leave the position naked overnight). Cross-references
    # the ledger's open trades — same-day round-trips stay on DAY stops
    # so we don't leave orphaned GTC stops if the exit cron drops a beat.
    multi_day_t212_tickers = _build_multi_day_ticker_set(instruments_by_ticker)

    portfolio = _safe_get(f"{base}/equity/portfolio", headers)
    if not isinstance(portfolio, list) or not portfolio:
        return actions

    # Pull open orders so we know if a stop already exists for each symbol.
    open_orders = _safe_get(f"{base}/equity/orders", headers) or []
    stops_by_ticker: dict[str, list[dict]] = {}
    for o in open_orders if isinstance(open_orders, list) else []:
        if not isinstance(o, dict):
            continue
        otype = (o.get("type") or "").upper()
        if otype in ("STOP", "STOP_LIMIT"):
            # T212 internal tickers are CASE-SENSITIVE (e.g. 'TMGl_EQ' —
            # lowercase 'l' for LSE). Don't uppercase or both the cache
            # lookup and the in-memory stops index miss.
            sym = o.get("ticker") or ""
            stops_by_ticker.setdefault(sym, []).append(o)

    # Rate-limit headroom: portfolio + orders are 2 requests; each trail
    # action is up to 2 more (cancel + place). T212 docs say sustained
    # ~1 req/s on most slots.
    for pos in portfolio:
        if not isinstance(pos, dict):
            continue
        # Preserve T212 case for cache lookup; uppercase a display copy
        # only where we need it.
        ticker = pos.get("ticker") or ""
        try:
            qty = float(pos.get("quantity") or 0)
            entry = float(pos.get("averagePrice") or 0)
            cur = float(pos.get("currentPrice") or 0)
        except (TypeError, ValueError):
            continue
        if qty <= 0 or entry <= 0 or cur <= 0:
            continue
        pct_up = (cur / entry - 1.0) * 100.0
        # Phase 10A — instrument-aware thresholds. T212 quotes LSE
        # shares in pence (currency code 'GBX'), not 'GBP'. Stamp duty
        # applies to UK shares regardless of the quote unit.
        inst = instruments_by_ticker.get(ticker, {})
        ccy = (inst.get("currencyCode") or "").upper()
        # T212 (GBP-base account) won't place stop orders on non-base-currency
        # instruments — POST /equity/orders/stop returns 400 "Invalid payload"
        # for them. (Market buys auto-FX across currencies, but stop/limit
        # orders are restricted to the account's main currency.) So EU/US
        # positions can't get an intraday trailing stop; they're still closed
        # by the scheduled EOD exit. Skip rather than fail the run every pass.
        if ccy not in ("GBP", "GBX"):
            actions.append(T212TrailAction(
                ticker=ticker, slot=creds.slot, entry_price=entry, current_price=cur,
                pct_up=pct_up, old_stop=None, new_stop=0.0,
                status="skipped",
                reason=f"non-base-currency ({ccy or '?'}) — T212 rejects stops on it; covered by scheduled exit",
            ))
            continue
        is_uk_share = (
            inst.get("type") == "STOCK"
            and ccy in ("GBP", "GBX")
        )
        eff_activation = UK_SHARE_ACTIVATION_PCT if is_uk_share else activation_pct
        eff_trail = UK_SHARE_TRAIL_PCT if is_uk_share else trail_pct
        if pct_up < eff_activation:
            continue

        # 2dp — T212 reports these instrument prices at 2 decimals and
        # rejects finer-precision stopPrice payloads with 400 "Invalid
        # payload" (a 4dp price, e.g. 1222.2432, was rejected live).
        target_stop = round(cur * (1.0 - eff_trail / 100.0), 2)
        existing_stops = stops_by_ticker.get(ticker, [])
        old_stop_val: float | None = None
        if existing_stops:
            try:
                old_stop_val = max(float(s.get("stopPrice") or 0) for s in existing_stops)
            except (TypeError, ValueError):
                old_stop_val = None

        if old_stop_val is not None and old_stop_val >= target_stop:
            actions.append(T212TrailAction(
                ticker=ticker, slot=creds.slot, entry_price=entry, current_price=cur,
                pct_up=pct_up, old_stop=old_stop_val, new_stop=target_stop,
                status="skipped",
                reason=f"existing stop already at {old_stop_val:.4f}; trail won't lower it",
            ))
            continue

        # Phase 12D — multi-day positions get GTC stops so they survive
        # overnight. Same-day round-trips stay on DAY validity so the
        # stop self-expires at session close.
        time_validity = "GTC" if ticker in multi_day_t212_tickers else "DAY"

        # Safety reorder: place the new (tighter) stop BEFORE cancelling the
        # old one. T212 has no modify-stop endpoint, so trailing is
        # cancel+replace — but cancelling first means a rejected placement
        # leaves the position with NO stop (this happened live: a 4dp price
        # cancelled the old stop, then the new POST and the restore both
        # failed → UNPROTECTED). Placing first guarantees we never strip
        # protection: if the new placement is rejected we simply leave the
        # existing stop untouched, and we only cancel the old stop once the
        # new one is confirmed live.
        placed = _submit_stop(base, headers, ticker, -qty, target_stop, time_validity=time_validity)
        if not placed:
            actions.append(T212TrailAction(
                ticker=ticker, slot=creds.slot, entry_price=entry, current_price=cur,
                pct_up=pct_up, old_stop=old_stop_val, new_stop=target_stop,
                status="failed",
                reason=(
                    "new stop POST failed — left existing stop in place"
                    if existing_stops
                    else "new stop POST failed — position has no stop (UNPROTECTED)"
                ),
            ))
            continue

        # New stop is live — cancel the prior stop(s) so only the new one
        # remains. A failed cancel here leaves a redundant (looser) stop, not
        # an unprotected position, so it's the safe direction to err.
        for stp in existing_stops:
            sid = str(stp.get("id") or "")
            if sid:
                _cancel_order(base, headers, sid)
                time.sleep(0.4)
        actions.append(T212TrailAction(
            ticker=ticker, slot=creds.slot, entry_price=entry, current_price=cur,
            pct_up=pct_up, old_stop=old_stop_val, new_stop=target_stop,
            status="placed" if old_stop_val is None else "tightened",
            reason=f"validity={time_validity}",
        ))
        time.sleep(0.4)

    return actions


def _build_multi_day_ticker_set(instruments_by_ticker: dict[str, dict]) -> set[str]:
    """Phase 12D — return the set of T212 tickers whose matching ledger
    trade has `hold_days > 1`. These get GTC stops in the trail loop so
    they survive overnight; same-day round-trips stay on DAY validity.

    Cross-references all open ledger trades (across all strategies bound
    to T212 slots — the trail script runs against the slot, not per-
    strategy) and translates each yfinance ticker forward to its T212
    equivalent via the cached instruments list.
    """
    if not instruments_by_ticker:
        return set()
    try:
        from trading_bot.state import read_open_trades
        from trading_bot.tools.t212_instruments import Translator
    except Exception as e:
        log.warning("multi-day ticker set: import failed (%s) — defaulting to empty", e)
        return set()

    open_trades = read_open_trades()
    if not open_trades:
        return set()

    multi_day_yf = {
        t["ticker"]
        for t in open_trades
        if int(t.get("hold_days") or 1) > 1 and t.get("tier") == "trading212-paper"
    }
    if not multi_day_yf:
        return set()

    translator = Translator(list(instruments_by_ticker.values()))
    out: set[str] = set()
    for yf in multi_day_yf:
        t212 = translator.translate(yf)
        if t212:
            out.add(t212)
    return out


def _load_instruments_cache() -> dict[str, dict]:
    """Read state/t212_instruments.json and index by ticker. Returns
    an empty dict on miss so callers can default to non-UK thresholds."""
    from trading_bot.state.paths import STATE_ROOT
    p = STATE_ROOT / "t212_instruments.json"
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    items = raw if isinstance(raw, list) else raw.get("instruments", [])
    out: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        tkr = item.get("ticker")
        if tkr:
            out[tkr] = item
    return out


def _safe_get(url: str, headers: dict) -> Any:
    try:
        r = requests.get(url, headers=headers, timeout=15)
    except requests.RequestException as e:
        log.warning("T212 GET %s failed: %s", url, e)
        return None
    if not r.ok:
        log.warning("T212 GET %s returned %d: %s", url, r.status_code, r.text[:200])
        return None
    try:
        return r.json()
    except json.JSONDecodeError:
        return None


def _cancel_order(base: str, headers: dict, order_id: str) -> bool:
    try:
        r = requests.delete(f"{base}/equity/orders/{order_id}", headers=headers, timeout=15)
    except requests.RequestException as e:
        log.warning("T212 cancel %s errored: %s", order_id, e)
        return False
    # T212 returns 200 or 204 on success; 404 is fine too (already cancelled).
    if r.ok or r.status_code == 404:
        return True
    log.warning("T212 cancel %s returned %d: %s", order_id, r.status_code, r.text[:150])
    return False


def _submit_stop(
    base: str,
    headers: dict,
    ticker: str,
    quantity: float,
    stop_price: float,
    *,
    time_validity: str = "DAY",
) -> bool:
    payload = {
        "ticker": ticker,
        "quantity": quantity,
        "stopPrice": stop_price,
        "timeValidity": time_validity,
    }
    for attempt in range(_CANCEL_AFTER_PLACE_RETRIES + 1):
        try:
            r = requests.post(
                f"{base}/equity/orders/stop", headers=headers, json=payload, timeout=15,
            )
        except requests.RequestException as e:
            log.warning("T212 stop submit errored for %s (attempt %d): %s", ticker, attempt, e)
            time.sleep(0.6)
            continue
        if r.ok:
            return True
        log.warning("T212 stop submit %s returned %d: %s", ticker, r.status_code, r.text[:200])
        if r.status_code in (400, 401, 403):
            break    # not retryable
        time.sleep(0.8)
    return False


def format_log(actions: list[T212TrailAction]) -> str:
    if not actions:
        return "(no positions in profit beyond activation threshold)"
    return "\n".join(str(a) for a in actions)
