"""Midday take-profit pass — close positions that have hit their
strategy's take-profit target by midday, before the EOD drift gives it
back.

Empirical observation (2026-05-26 onward): positions opened in the
morning often peak at midday and drift back to flat by the close. The
existing midday-trail walks the broker stop UP on positions in profit
but only fires if the price subsequently falls; it doesn't realise the
gain at the peak. This pass does — for any open position whose
midday pct_up >= strategy.take_profit_pct × strategy.midday_tp_factor,
it market-closes immediately.

Two brokers handled:

- **Alpaca**: list positions, find each one's bracket children (TP +
  stop), cancel the children, then DELETE /v2/positions/{ticker} to
  market-close. Cancelling first avoids an orphan TP firing against
  a future re-entry on the same symbol.

- **T212**: list portfolio positions, find any open STOP / STOP_LIMIT
  for that symbol and delete (same anti-orphan logic), then submit a
  market SELL of the position quantity.

Runs BEFORE the trailing-stop pass in `scripts/midday_trail.py` —
closed positions don't need trailing.

The threshold is per-strategy via `midday_tp_factor` (default 0.7).
Tunable by the evolution agent within (0.3, 1.5). The CLI flag
`--default-tp-factor` provides an ops-time override for strategies
whose config hasn't been written yet.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

import requests

from trading_bot.alpaca_slot import AlpacaCreds, load_slot_creds as load_alpaca_slot
from trading_bot.state import mark_trade_exited, read_open_trades
from trading_bot.strategy.registry import load_strategy_config
from trading_bot.t212_slot import T212Creds, load_slot_creds as load_t212_slot

log = logging.getLogger(__name__)

DEFAULT_TP_FACTOR = 0.7


@dataclass
class TakeProfitAction:
    """One midday take-profit decision for the run log."""
    ticker: str
    slot: int
    broker: str                # "alpaca" | "t212"
    strategy_id: str
    entry_price: float
    current_price: float
    pct_up: float
    threshold_pct: float       # what the position had to clear
    take_profit_pct: float     # strategy's nominal TP
    tp_factor: float           # strategy's midday factor
    status: str                # "closed" | "skipped" | "failed"
    reason: str = ""

    def __str__(self) -> str:
        return (
            f"  {self.status.upper():<7} {self.broker} slot={self.slot} "
            f"{self.ticker:<10} {self.strategy_id} "
            f"entry={self.entry_price:.4f} now={self.current_price:.4f} "
            f"(+{self.pct_up:.2f}%) threshold={self.threshold_pct:.2f}% "
            f"(TP {self.take_profit_pct:.1f}% × {self.tp_factor:.2f}) "
            f"{self.reason}"
        )


def format_log(actions: list[TakeProfitAction]) -> str:
    if not actions:
        return "  (no actions)"
    return "\n".join(str(a) for a in actions)


# ---------------------------------------------------------------------------
# Strategy-config lookup (cached per process)
# ---------------------------------------------------------------------------

_STRATEGY_CONFIG_CACHE: dict[str, Any] = {}


def _strategy_thresholds(strategy_id: str, cli_default_factor: float) -> tuple[float | None, float, float]:
    """Return (take_profit_pct, tp_factor, computed_threshold_pct) for a
    strategy. `take_profit_pct=None` signals "no TP configured — skip
    this position." Otherwise the threshold is TP × factor.

    CLI `default_factor` is only consulted when the strategy's config
    is missing the field (legacy strategies pre-this-change).
    """
    cfg = _STRATEGY_CONFIG_CACHE.get(strategy_id)
    if cfg is None:
        try:
            cfg = load_strategy_config(strategy_id)
        except Exception as e:
            log.warning("midday-tp: cannot load config for %s: %s", strategy_id, e)
            return None, cli_default_factor, 0.0
        _STRATEGY_CONFIG_CACHE[strategy_id] = cfg

    tp = cfg.take_profit_pct
    factor = getattr(cfg, "midday_tp_factor", None)
    if factor is None:
        factor = cli_default_factor
    if tp is None or tp <= 0:
        return None, factor, 0.0
    return tp, factor, tp * factor


# ---------------------------------------------------------------------------
# Alpaca side
# ---------------------------------------------------------------------------

def take_profit_alpaca_slots(
    slots: list[int] | None = None,
    *,
    default_tp_factor: float = DEFAULT_TP_FACTOR,
) -> list[TakeProfitAction]:
    """Scan every configured Alpaca slot, close positions that have hit
    their strategy's midday take-profit threshold."""
    slots = slots or [1, 2, 3]
    out: list[TakeProfitAction] = []
    for slot in slots:
        try:
            creds = load_alpaca_slot(slot)
        except RuntimeError:
            continue
        try:
            out.extend(_take_profit_one_alpaca_slot(creds, default_tp_factor))
        except Exception as e:
            log.warning("midday-tp: alpaca slot %d failed: %s", slot, e)
    return out


def _take_profit_one_alpaca_slot(
    creds: AlpacaCreds, default_factor: float,
) -> list[TakeProfitAction]:
    base = creds.trading_base_url
    headers = {
        "APCA-API-KEY-ID": creds.api_key,
        "APCA-API-SECRET-KEY": creds.api_secret,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    actions: list[TakeProfitAction] = []

    try:
        r = requests.get(f"{base}/v2/positions", headers=headers, timeout=15)
        if not r.ok:
            return actions
        positions = r.json() or []
    except requests.RequestException as e:
        log.warning("midday-tp: alpaca slot %d positions fetch failed: %s", creds.slot, e)
        return actions
    if not positions:
        return actions

    # Index open bracket children by symbol — we cancel these before
    # closing so an orphan TP can't fire against a future re-entry.
    try:
        r = requests.get(
            f"{base}/v2/orders",
            params={"status": "open", "nested": "true", "limit": 100},
            headers=headers, timeout=15,
        )
        open_orders = r.json() if r.ok else []
    except requests.RequestException:
        open_orders = []
    orders_by_symbol: dict[str, list[str]] = {}
    for o in open_orders or []:
        if not isinstance(o, dict):
            continue
        oid = o.get("id")
        sym = (o.get("symbol") or "").upper()
        if oid and sym:
            orders_by_symbol.setdefault(sym, []).append(oid)
        for leg in o.get("legs") or []:
            if not isinstance(leg, dict):
                continue
            lsym = (leg.get("symbol") or sym).upper()
            lid = leg.get("id")
            if lid and lsym:
                orders_by_symbol.setdefault(lsym, []).append(lid)

    # Ledger lookup: ticker → (strategy_id, entry_price). The ledger row
    # carries the strategy_id we need to read take_profit_pct from.
    ledger_by_ticker = _index_open_trades_by_ticker(tier="alpaca-paper")

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

        ledger_row = ledger_by_ticker.get(symbol)
        if ledger_row is None:
            log.debug("midday-tp: alpaca position %s has no open ledger row — skipping", symbol)
            continue
        sid = ledger_row.get("strategy_id") or ""
        tp_pct, factor, threshold = _strategy_thresholds(sid, default_factor)
        if tp_pct is None:
            continue

        if pct_up < threshold:
            continue

        # Cancel bracket children first so an orphan TP/SL doesn't fire
        # on a future re-entry on the same symbol.
        for oid in orders_by_symbol.get(symbol, []):
            try:
                requests.delete(
                    f"{base}/v2/orders/{oid}", headers=headers, timeout=10,
                )
            except requests.RequestException:
                pass

        # DELETE /v2/positions/{ticker} = market close at next opportunity.
        try:
            r = requests.delete(
                f"{base}/v2/positions/{symbol}", headers=headers, timeout=15,
            )
        except requests.RequestException as e:
            actions.append(TakeProfitAction(
                ticker=symbol, slot=creds.slot, broker="alpaca",
                strategy_id=sid, entry_price=entry_price,
                current_price=current_price, pct_up=pct_up,
                threshold_pct=threshold, take_profit_pct=tp_pct, tp_factor=factor,
                status="failed", reason=f"close request errored: {e}",
            ))
            continue
        if not r.ok and r.status_code not in (404, 422):
            actions.append(TakeProfitAction(
                ticker=symbol, slot=creds.slot, broker="alpaca",
                strategy_id=sid, entry_price=entry_price,
                current_price=current_price, pct_up=pct_up,
                threshold_pct=threshold, take_profit_pct=tp_pct, tp_factor=factor,
                status="failed", reason=f"alpaca {r.status_code}: {r.text[:100]}",
            ))
            continue

        # Mark exited in the ledger. Use current_price as the exit price —
        # the close fills at-market and we don't poll for the actual fill
        # here (the next exit-pipeline run will reconcile via order
        # history if there's any drift). Fees are 0 for Alpaca paper.
        _mark_exit(
            ledger_row=ledger_row, exit_price=current_price,
            reason="midday_take_profit",
        )
        actions.append(TakeProfitAction(
            ticker=symbol, slot=creds.slot, broker="alpaca",
            strategy_id=sid, entry_price=entry_price,
            current_price=current_price, pct_up=pct_up,
            threshold_pct=threshold, take_profit_pct=tp_pct, tp_factor=factor,
            status="closed",
            reason=f"cancelled {len(orders_by_symbol.get(symbol, []))} child(ren) + market close",
        ))

    return actions


# ---------------------------------------------------------------------------
# T212 side
# ---------------------------------------------------------------------------

def take_profit_t212_slots(
    slots: list[int] | None = None,
    *,
    default_tp_factor: float = DEFAULT_TP_FACTOR,
) -> list[TakeProfitAction]:
    """Scan every configured T212 slot, close positions that have hit
    their strategy's midday take-profit threshold."""
    slots = slots or [1, 2, 3]
    out: list[TakeProfitAction] = []
    for slot in slots:
        try:
            creds = load_t212_slot(slot, demo=True)
        except RuntimeError:
            continue
        try:
            out.extend(_take_profit_one_t212_slot(creds, default_tp_factor))
        except Exception as e:
            log.warning("midday-tp: t212 slot %d failed: %s", slot, e)
    return out


def _take_profit_one_t212_slot(
    creds: T212Creds, default_factor: float,
) -> list[TakeProfitAction]:
    base = creds.base_url
    headers = {
        "Authorization": creds.auth_header(),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    actions: list[TakeProfitAction] = []

    try:
        r = requests.get(f"{base}/equity/portfolio", headers=headers, timeout=15)
        if not r.ok:
            return actions
        portfolio = r.json() or []
    except requests.RequestException as e:
        log.warning("midday-tp: t212 slot %d portfolio fetch failed: %s", creds.slot, e)
        return actions
    if not portfolio:
        return actions

    # Index any open STOP / STOP_LIMIT orders so we can cancel them
    # before market-closing (same anti-orphan logic as Alpaca).
    try:
        r = requests.get(f"{base}/equity/orders", headers=headers, timeout=15)
        open_orders = r.json() if r.ok else []
    except requests.RequestException:
        open_orders = []
    stops_by_ticker: dict[str, list[str]] = {}
    for o in open_orders if isinstance(open_orders, list) else []:
        if not isinstance(o, dict):
            continue
        otype = (o.get("type") or "").upper()
        if otype not in ("STOP", "STOP_LIMIT"):
            continue
        sym = o.get("ticker") or ""
        oid = o.get("id")
        if sym and oid is not None:
            stops_by_ticker.setdefault(sym, []).append(str(oid))

    # Ledger lookup. T212 portfolio returns T212-internal tickers
    # (e.g. 'LLOYl_EQ' for LBG.L); the ledger uses normalised yfinance
    # tickers (e.g. 'LBG.L'). We need the translator to bridge.
    ledger_by_ticker = _index_open_trades_by_ticker(tier="trading212-paper")
    from trading_bot.executor.trading212_demo import Trading212DemoExecutor  # noqa: PLC0415
    try:
        translator = Trading212DemoExecutor(creds.slot)._get_translator()
    except Exception as e:
        log.warning("midday-tp: t212 translator unavailable for slot %d: %s", creds.slot, e)
        return actions

    yf_to_t212: dict[str, str] = {}
    for yf_ticker in list(ledger_by_ticker.keys()):
        try:
            t212_ticker = translator.translate(yf_ticker)
        except Exception:
            t212_ticker = None
        if t212_ticker:
            yf_to_t212[yf_ticker] = t212_ticker

    t212_to_yf = {v: k for k, v in yf_to_t212.items()}

    for pos in portfolio:
        if not isinstance(pos, dict):
            continue
        t212_ticker = pos.get("ticker") or ""
        if not t212_ticker:
            continue
        yf_ticker = t212_to_yf.get(t212_ticker)
        if yf_ticker is None:
            continue
        ledger_row = ledger_by_ticker.get(yf_ticker)
        if ledger_row is None:
            continue

        try:
            current_price_raw = float(pos.get("currentPrice") or 0)
            avg_price_raw = float(pos.get("averagePrice") or 0)
            quantity = float(pos.get("quantity") or 0)
        except (TypeError, ValueError):
            continue
        if avg_price_raw <= 0 or current_price_raw <= 0 or quantity <= 0:
            continue

        # T212 quotes in native units (GBX for LSE, EUR/USD/etc. elsewhere).
        # `pct_up` is unit-invariant — both prices share the same units.
        pct_up = (current_price_raw / avg_price_raw - 1.0) * 100.0

        sid = ledger_row.get("strategy_id") or ""
        tp_pct, factor, threshold = _strategy_thresholds(sid, default_factor)
        if tp_pct is None:
            continue
        if pct_up < threshold:
            continue

        # Cancel any open stops on this ticker before the market close.
        for oid in stops_by_ticker.get(t212_ticker, []):
            try:
                requests.delete(
                    f"{base}/equity/orders/{oid}", headers=headers, timeout=10,
                )
            except requests.RequestException:
                pass

        # Submit market sell for the full position quantity.
        payload = {"ticker": t212_ticker, "quantity": -quantity}
        try:
            r = requests.post(
                f"{base}/equity/orders/market",
                headers=headers, json=payload, timeout=15,
            )
        except requests.RequestException as e:
            actions.append(TakeProfitAction(
                ticker=yf_ticker, slot=creds.slot, broker="t212",
                strategy_id=sid, entry_price=avg_price_raw,
                current_price=current_price_raw, pct_up=pct_up,
                threshold_pct=threshold, take_profit_pct=tp_pct, tp_factor=factor,
                status="failed", reason=f"close request errored: {e}",
            ))
            continue
        if not r.ok:
            actions.append(TakeProfitAction(
                ticker=yf_ticker, slot=creds.slot, broker="t212",
                strategy_id=sid, entry_price=avg_price_raw,
                current_price=current_price_raw, pct_up=pct_up,
                threshold_pct=threshold, take_profit_pct=tp_pct, tp_factor=factor,
                status="failed", reason=f"t212 {r.status_code}: {r.text[:100]}",
            ))
            continue

        # The submit response usually doesn't carry a fill price yet —
        # T212 fills asynchronously. We record exit_price using the
        # current_price the position record reported (already a GBX/EUR
        # native value); the next scheduled exit-pipeline run will
        # reconcile against actual fill via order history if needed.
        # Convert the raw price to GBP for the ledger's base unit.
        exit_price_gbp = current_price_raw  # placeholder; reconcile below
        try:
            inst = translator.get_instrument(t212_ticker)
            ccy = (inst.get("currencyCode") or "").upper() if isinstance(inst, dict) else ""
            from trading_bot.tools.fx import to_gbp_multiplier  # noqa: PLC0415
            mult = to_gbp_multiplier(ccy)
            if mult is not None:
                exit_price_gbp = current_price_raw * mult
        except Exception:
            pass

        _mark_exit(
            ledger_row=ledger_row, exit_price=exit_price_gbp,
            reason="midday_take_profit",
        )
        actions.append(TakeProfitAction(
            ticker=yf_ticker, slot=creds.slot, broker="t212",
            strategy_id=sid, entry_price=avg_price_raw,
            current_price=current_price_raw, pct_up=pct_up,
            threshold_pct=threshold, take_profit_pct=tp_pct, tp_factor=factor,
            status="closed",
            reason=f"cancelled {len(stops_by_ticker.get(t212_ticker, []))} stop(s) + market sell",
        ))
        # T212 rate-limits POSTs; brief spacing between closes.
        time.sleep(0.3)

    return actions


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _index_open_trades_by_ticker(*, tier: str) -> dict[str, dict]:
    """Build {ticker: ledger_row} for open trades at this tier. If the
    same ticker has multiple open rows (rare — usually a stranded entry
    from a prior session), the most recent entry_date wins so the
    matching position is the one we just opened."""
    out: dict[str, dict] = {}
    for row in read_open_trades(tier=tier):
        ticker = (row.get("ticker") or "").upper()
        if not ticker:
            continue
        prev = out.get(ticker)
        if prev is None or (row.get("entry_date") or "") > (prev.get("entry_date") or ""):
            out[ticker] = row
    return out


def _mark_exit(*, ledger_row: dict, exit_price: float, reason: str) -> None:
    """Write the exit to the ledger. Idempotent at the row level —
    mark_trade_exited raises if the trade_id is already exited, so we
    swallow the KeyError if we hit a race between this pass and the
    EOD exit pipeline."""
    try:
        entry = float(ledger_row.get("entry_price") or 0)
        qty = float(ledger_row.get("quantity") or 0)
    except (TypeError, ValueError):
        return
    if entry <= 0 or qty <= 0:
        return
    pnl_gbp = (exit_price - entry) * qty
    pnl_pct = (exit_price / entry - 1.0) * 100.0
    try:
        mark_trade_exited(
            trade_id=ledger_row["trade_id"],
            exit_date=date.today(),
            exit_price=exit_price,
            pnl_gbp=pnl_gbp,
            pnl_pct=pnl_pct,
            exit_reason=reason,
        )
    except KeyError:
        log.info("midday-tp: trade %s already exited — skipping ledger update",
                 ledger_row.get("trade_id"))
    except Exception as e:
        log.warning("midday-tp: ledger update failed for %s: %s",
                    ledger_row.get("trade_id"), e)
