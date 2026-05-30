from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import yaml

from trading_bot.state.paths import STATE_ROOT, ledger_path, predictions_path
from trading_bot.strategy.registry import _strategies_dir


log = logging.getLogger(__name__)


def _docs_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "docs"


def _iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_strategy_configs() -> dict[str, dict]:
    """Returns {strategy_id: raw_config}. Multi-region configs (runs_in list)
    are kept as-is — the build pipeline expands per-region downstream."""
    configs: dict[str, dict] = {}
    for path in _strategies_dir().glob("*/config.yaml"):
        raw = yaml.safe_load(path.read_text())
        configs[raw["id"]] = raw
    return configs


def _regions_for_config(raw: dict) -> list[dict]:
    """Expand a config into per-region descriptors. Single-region configs
    return one entry; multi-region (runs_in) configs return one per region."""
    if isinstance(raw.get("runs_in"), list):
        out = []
        for entry in raw["runs_in"]:
            if not isinstance(entry, dict) or not entry.get("region"):
                continue
            merged = {k: v for k, v in raw.items() if k != "runs_in"}
            merged.update(entry)
            out.append(merged)
        return out or [{**raw, "region": "us"}]
    return [{**raw, "region": raw.get("region", "us")}]


def _equity_curve(trades: list[dict], starting_capital: float) -> list[dict]:
    """Cumulative P&L over time, one point per exited trade in chronological order."""
    closed = [t for t in trades if t.get("exit_date") and t.get("pnl_gbp") is not None]
    closed.sort(key=lambda t: t["exit_date"])
    points: list[dict] = [{"date": None, "equity": starting_capital}]
    running = starting_capital
    for trade in closed:
        running += float(trade["pnl_gbp"])
        points.append({"date": trade["exit_date"], "equity": round(running, 2)})
    return points


def _summary_stats(trades: list[dict], predictions: list[dict]) -> dict:
    closed = [t for t in trades if t.get("exit_date") and t.get("pnl_gbp") is not None]
    n = len(closed)
    open_trades = [t for t in trades if not t.get("exit_date")]
    # Phase 12F — break the open-positions count into "multi-day"
    # (intentionally held across sessions) vs total. A multi-day
    # position is one with hold_days > 1 OR target_exit_date in the
    # future. Legacy rows (no hold_days / no target) are treated as
    # same-day stranded so the dashboard shows them on the "needs
    # attention" side, not the "intentional carryover" side.
    from datetime import date as _date
    today_iso = _date.today().isoformat()
    n_open_multi_day = 0
    for t in open_trades:
        hd = int(t.get("hold_days") or 1)
        target = t.get("target_exit_date") or ""
        if hd > 1 or (target and target > today_iso):
            n_open_multi_day += 1
    if n == 0:
        return {
            "n_closed": 0,
            "n_open": len(open_trades),
            "n_open_multi_day": n_open_multi_day,
            "total_pnl_gbp": 0.0,
            "avg_pnl_pct": 0.0,
            "hit_rate": 0.0,
            "n_predictions": len(predictions),
        }
    total_pnl = sum(float(t["pnl_gbp"]) for t in closed)
    avg_pnl_pct = sum(float(t["pnl_pct"]) for t in closed) / n
    wins = sum(1 for t in closed if float(t["pnl_gbp"]) > 0)
    return {
        "n_closed": n,
        "n_open": len(open_trades),
        "n_open_multi_day": n_open_multi_day,
        "total_pnl_gbp": round(total_pnl, 2),
        "avg_pnl_pct": round(avg_pnl_pct, 2),
        "hit_rate": round(wins / n, 3),
        "n_predictions": len(predictions),
    }


def build_dashboard_data() -> dict:
    """Assemble the JSON payload the static dashboard renders.

    Strategies that run in multiple regions are expanded into one entry per
    (strategy_id, region) pair so the dashboard can show per-region
    performance independently. The HTML side filters by a top-level region
    selector.
    """
    configs = _load_strategy_configs()
    all_trades = list(_iter_jsonl(ledger_path()))
    all_predictions = list(_iter_jsonl(predictions_path()))

    # Group trades and predictions by (strategy_id, region) — region is the
    # primary axis for the dashboard now that strategies are multi-region.
    trades_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in all_trades:
        key = (t["strategy_id"], t.get("region", "us"))
        trades_by_key[key].append(t)

    preds_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in all_predictions:
        if not p.get("was_traded"):
            key = (p["strategy_id"], p.get("region", "us"))
            preds_by_key[key].append(p)

    active: list[dict] = []
    archived: list[dict] = []
    regions_seen: set[str] = set()

    for sid, raw_config in sorted(configs.items()):
        for region_cfg in _regions_for_config(raw_config):
            region = region_cfg.get("region", "us")
            regions_seen.add(region)
            key = (sid, region)
            trades = trades_by_key.get(key, [])
            preds = preds_by_key.get(key, [])
            starting_capital = float(region_cfg.get("capital_gbp", 10000))

            entry = {
                "id": sid,
                "key": f"{sid}@{region}",  # unique identifier for dashboard state
                "display_name": region_cfg.get("display_name", sid),
                "description": region_cfg.get("description", "").strip(),
                "tier": region_cfg.get("tier", "shadow"),
                "region": region,
                "universe": region_cfg.get("universe", "sp500"),
                "capital_gbp": starting_capital,
                # Tier 2 candidate flag — surfaced by the weekly evolution
                # agent as its prediction of "this (strategy, region)
                # sleeve is worth graduating". Per-region as of
                # 2026-05-30. `region_cfg` already merges runs_in entry
                # values over top-level values via _regions_for_config,
                # so this read transparently supports both per-region
                # (new) and legacy top-level flags.
                "tier2_candidate": bool(region_cfg.get("tier2_candidate", False)),
                "tier2_marked_at": region_cfg.get("tier2_marked_at"),
                "tier2_thesis": str(region_cfg.get("tier2_thesis") or ""),
                "summary": _summary_stats(trades, preds),
                "equity_curve": _equity_curve(trades, starting_capital),
                "executed": sorted(trades, key=lambda t: t.get("entry_date", ""), reverse=True),
                "uncommitted": sorted(preds, key=lambda p: p.get("prediction_date", ""), reverse=True),
            }

            if raw_config.get("active"):
                active.append(entry)
            elif trades or preds:
                archived.append(entry)

    # Also surface trades whose strategy_id no longer matches any config
    # (e.g., strategy was deleted but ledger still has the rows). These
    # appear under Archived with region tagged.
    config_ids = set(configs.keys())
    for (sid, region), trades in trades_by_key.items():
        if sid in config_ids:
            continue
        regions_seen.add(region)
        preds = preds_by_key.get((sid, region), [])
        archived.append({
            "id": sid,
            "key": f"{sid}@{region}",
            "display_name": f"{sid} (archived)",
            "description": "Historical data — strategy no longer in registry.",
            "tier": (trades[0].get("tier") if trades else "shadow"),
            "region": region,
            "universe": "unknown",
            "capital_gbp": 10000,
            "summary": _summary_stats(trades, preds),
            "equity_curve": _equity_curve(trades, 10000),
            "executed": sorted(trades, key=lambda t: t.get("entry_date", ""), reverse=True),
            "uncommitted": sorted(preds, key=lambda p: p.get("prediction_date", ""), reverse=True),
        })

    # Global overview: roll up real-broker P&L (alpaca-paper, trading212-paper)
    # separately from shadow simulation so the user can tell at a glance how
    # much of any headline number reflects actual paper-traded fills vs
    # yfinance-simulated exposure that was never placed at a broker.
    # Phase 10D — pass as_of_iso so "today's P&L" uses the latest exit_date
    # rather than UTC today (loses Friday's exits around midnight UTC).
    as_of_iso = _latest_exit_iso(active + archived) or datetime.now(timezone.utc).date().isoformat()
    global_overview = _build_global_overview(active + archived, as_of_iso=as_of_iso)

    # Phase 9D — sector exposure across every real-broker trade (open +
    # closed) so the panel reads as a trading footprint, not a snapshot
    # of currently-held positions. Computed per region — selecting US
    # or UK/EU at the top of the dashboard scopes the panel — plus an
    # `all` bucket for the combined view.
    sector_exposure = {
        region: _build_sector_exposure(active + archived, region=region)
        for region in (list(regions_seen) + ["all"])
    }

    # Phase 8F — surface kill-switch state to the dashboard
    halt_info = _build_halt_info()

    # Phase 10D — pull the latest missed-movers report per region so
    # the dashboard can surface them inline without a click-through.
    missed_movers = _build_missed_movers_snapshot()

    # External research brief from the weekly scan. Optional; if the
    # scan hasn't run yet the dashboard panel just hides.
    external_research = _build_external_research_snapshot()

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of_iso": as_of_iso,
        "regions": sorted(regions_seen) if regions_seen else ["us"],
        "global_overview": global_overview,
        "sector_exposure": sector_exposure,
        "halt": halt_info,
        "external_research": external_research,
        "missed_movers": missed_movers,
        "active": active,
        "archived": archived,
    }


def _build_missed_movers_snapshot() -> dict:
    """Read the most recent missed-movers reports per region."""
    from datetime import timedelta
    from trading_bot.meta.missed_movers import load_report
    today = datetime.now(timezone.utc).date()
    out: dict[str, dict] = {}
    for offset in range(0, 7):
        d = today - timedelta(days=offset)
        for region in ("us", "uk-eu"):
            if region in out:
                continue
            rep = load_report(d, region)
            if rep:
                out[region] = rep
        if len(out) == 2:
            break
    return out


def _latest_exit_iso(entries: list[dict]) -> str | None:
    """Most recent exit_date across all closed trades in any tier.
    Used as the 'today' boundary for headline-P&L calculation."""
    latest = ""
    for e in entries:
        for t in e.get("executed", []):
            ed = t.get("exit_date") or ""
            if ed and ed > latest:
                latest = ed
    return latest or None


def _build_external_research_snapshot() -> dict | None:
    """Surface the latest weekly external-research brief. Returns
    None if no scan has run yet — the dashboard's panel just hides."""
    try:
        from trading_bot.meta.external_research import latest_brief
    except Exception:
        return None
    try:
        b = latest_brief()
    except Exception:
        return None
    if b is None:
        return None
    return {
        "week_iso": b.week_iso,
        "generated_at": b.generated_at,
        "headline": b.headline,
        "themes": b.themes,
        "body_md": b.body_md,
        "sources": b.sources,
    }


def _build_halt_info() -> dict:
    """Read state/halt.json. Returns a dict surfaced to the dashboard."""
    try:
        from trading_bot.state.halt import is_halted
        halted, rec = is_halted()
    except Exception:
        return {"halted": False}
    if not halted:
        return {"halted": False}
    if rec is None:
        return {"halted": True, "reason": "(no record on disk)"}
    return {
        "halted": True,
        "reason": rec.reason,
        "yesterday_pnl_gbp": rec.yesterday_pnl_gbp,
        "yesterday_pnl_pct": rec.yesterday_pnl_pct,
        "set_at": rec.set_at,
    }


def _build_sector_exposure(entries: list[dict], region: str | None = None) -> dict:
    """Sector-exposure footprint across every real-broker trade
    (open + closed), scoped by region. Returns three time slices —
    today, 1 day ago, 1 week ago — so the dashboard can render a
    per-sector mini-chart showing the trajectory.

    `region` filters the source entries: pass 'us' / 'uk-eu' to scope,
    or 'all' / None for the combined view.

    Returns:
        {
          "by_sector": [
            {"sector": "Tech", "gbp": 4200.0, "pct": 32.0,
             "pct_1d": 30.0, "pct_1w": 28.0,
             "delta_1d_pct": 2.0, "delta_1w_pct": 4.0},
            ...
          ],
          "total_gbp":  12450.0,
          "n_trades":   23,
          "n_open":     5,
          "n_closed":   18,
        }
    """
    from collections import defaultdict
    from datetime import date, datetime, timedelta, timezone

    from trading_bot.tools.sectors import bulk_lookup

    today = datetime.now(timezone.utc).date()
    # End-of-yesterday and end-of-(today − 7) — the snapshots represent
    # "state at close on that day", so a trade ENTERED on the cutoff
    # date itself belongs in the snapshot. Off-by-one earlier here meant
    # yesterday's bucket dropped any trades entered yesterday, producing
    # misleading deltas.
    cutoff_1d = (today - timedelta(days=1)).isoformat()
    cutoff_1w = (today - timedelta(days=7)).isoformat()

    # 1) Collect every real-broker trade in scope.
    trades: list[dict] = []
    for entry in entries:
        if region and region != "all" and entry.get("region") != region:
            continue
        for t in entry.get("executed", []):
            if t.get("tier") not in _REAL_BROKER_TIERS:
                continue
            trades.append(t)
    if not trades:
        return {
            "by_sector": [], "total_gbp": 0.0,
            "n_trades": 0, "n_open": 0, "n_closed": 0,
        }

    tickers = sorted({t.get("ticker") for t in trades if t.get("ticker")})
    sector_map = bulk_lookup(tickers)

    # 2) Build three sector→notional buckets corresponding to which
    # trades had been *executed* (entry_date is in the past) at each
    # cutoff. A trade entered yesterday counts in today's bucket and
    # in yesterday's, but not in last-week's. The "exposure footprint"
    # interpretation: each trade's entry notional is added once and
    # never removed — closed trades still count toward where the bot
    # has been deploying capital recently.
    now: dict[str, float] = defaultdict(float)
    d1:  dict[str, float] = defaultdict(float)
    d7:  dict[str, float] = defaultdict(float)
    n_open = 0
    n_closed = 0
    for t in trades:
        sector = sector_map.get(t.get("ticker")) or "Unknown"
        try:
            entry_price = float(t.get("entry_price") or 0)
            qty = float(t.get("quantity") or 0)
            notional = abs(entry_price * qty)
        except (TypeError, ValueError):
            continue
        if notional <= 0:
            continue
        entry_date = t.get("entry_date") or ""
        now[sector] += notional
        if entry_date and entry_date <= cutoff_1d:
            d1[sector] += notional
        if entry_date and entry_date <= cutoff_1w:
            d7[sector] += notional
        if t.get("exit_date"):
            n_closed += 1
        else:
            n_open += 1

    total_now = sum(now.values())
    total_d1  = sum(d1.values())
    total_d7  = sum(d7.values())

    def _pct(buckets: dict, total: float, sector: str) -> float:
        if total <= 0:
            return 0.0
        return round(buckets.get(sector, 0.0) / total * 100, 1)

    rows = []
    for sector, gbp in sorted(now.items(), key=lambda kv: kv[1], reverse=True):
        pct      = _pct(now, total_now, sector)
        pct_1d   = _pct(d1,  total_d1,  sector)
        pct_1w   = _pct(d7,  total_d7,  sector)
        rows.append({
            "sector": sector,
            "gbp":    round(gbp, 2),
            "pct":    pct,
            "pct_1d": pct_1d,
            "pct_1w": pct_1w,
            "delta_1d_pct": round(pct - pct_1d, 1),
            "delta_1w_pct": round(pct - pct_1w, 1),
        })

    return {
        "by_sector": rows,
        "total_gbp": round(total_now, 2),
        "n_trades":  n_open + n_closed,
        "n_open":    n_open,
        "n_closed":  n_closed,
    }


_REAL_BROKER_TIERS = {"alpaca-paper", "trading212-paper", "t212-live"}


def _build_global_overview(entries: list[dict], as_of_iso: str | None = None) -> dict:
    """Aggregate trades across every strategy + region into two buckets:
    real-broker fills (Alpaca / T212) and shadow simulation. Each bucket
    reports today's P&L, all-time P&L, hit rate, and number of trades.

    `as_of_iso` (Phase 10D) — what "today" means for the headline P&L.
    Defaults to UTC today, but the caller passes the latest exit-date
    in the ledger so the UK/US trading day's exits aren't lost around
    midnight UTC."""
    today = as_of_iso or datetime.now(timezone.utc).date().isoformat()

    def _bucket() -> dict:
        return {
            "tiers": [],
            "today_pnl_gbp": 0.0,
            "today_n_trades": 0,
            "alltime_pnl_gbp": 0.0,
            "alltime_n_trades": 0,
            "alltime_n_wins": 0,
            "alltime_hit_rate": None,
            "regions": {},
        }

    real = _bucket()
    shadow = _bucket()
    seen_real_tiers: set[str] = set()

    for e in entries:
        tier = (e.get("tier") or "").lower()
        target = real if tier in _REAL_BROKER_TIERS else shadow
        if tier in _REAL_BROKER_TIERS:
            seen_real_tiers.add(tier)
        region = e.get("region") or "us"
        region_row = target["regions"].setdefault(region, {
            "today_pnl_gbp": 0.0,
            "today_n_trades": 0,
            "alltime_pnl_gbp": 0.0,
            "alltime_n_trades": 0,
        })
        for trade in e.get("executed", []):
            pnl = trade.get("pnl_gbp")
            if pnl is None or trade.get("exit_date") is None:
                continue  # still open or pnl unrecorded
            pnl_f = float(pnl)
            target["alltime_pnl_gbp"] += pnl_f
            target["alltime_n_trades"] += 1
            region_row["alltime_pnl_gbp"] += pnl_f
            region_row["alltime_n_trades"] += 1
            if pnl_f > 0:
                target["alltime_n_wins"] += 1
            if trade.get("exit_date") == today:
                target["today_pnl_gbp"] += pnl_f
                target["today_n_trades"] += 1
                region_row["today_pnl_gbp"] += pnl_f
                region_row["today_n_trades"] += 1

    for bucket in (real, shadow):
        if bucket["alltime_n_trades"]:
            bucket["alltime_hit_rate"] = round(
                bucket["alltime_n_wins"] / bucket["alltime_n_trades"], 3
            )
        bucket["today_pnl_gbp"] = round(bucket["today_pnl_gbp"], 2)
        bucket["alltime_pnl_gbp"] = round(bucket["alltime_pnl_gbp"], 2)
        for region_row in bucket["regions"].values():
            region_row["today_pnl_gbp"] = round(region_row["today_pnl_gbp"], 2)
            region_row["alltime_pnl_gbp"] = round(region_row["alltime_pnl_gbp"], 2)

    real["tiers"] = sorted(seen_real_tiers)
    shadow["tiers"] = ["shadow"]
    return {"real_broker": real, "shadow_simulation": shadow}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trading_bot.dashboard.build")
    parser.add_argument("--out", default=None, help="Output JSON path (defaults to docs/data.json)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    out_path = Path(args.out) if args.out else _docs_dir() / "data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = build_dashboard_data()
    out_path.write_text(json.dumps(data, indent=2))
    log.info("Wrote %s (%d active, %d archived)", out_path, len(data["active"]), len(data["archived"]))

    # Render the static markdown-driven pages alongside the dashboard JSON.
    # Idempotent: re-renders every existing brief / macro view / evolution
    # log so the site stays consistent after any single workflow runs.
    try:
        from trading_bot.dashboard.pages import rebuild_all_pages
        summary = rebuild_all_pages()
        log.info("Rebuilt static pages: %s", summary)
    except Exception as e:
        # Bump to ERROR (was WARNING — invisible in CI logs that filter
        # to ERROR+). Static-page failures are non-fatal for the data
        # pipeline but they mask real template bugs (e.g. f-string
        # syntax errors inside news/article_writer.py prompts). Log with
        # the full traceback so the next failure surfaces immediately.
        log.error("Static page rebuild failed (non-fatal): %s", e, exc_info=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
