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
    configs: dict[str, dict] = {}
    for path in _strategies_dir().glob("*/config.yaml"):
        raw = yaml.safe_load(path.read_text())
        configs[raw["id"]] = raw
    return configs


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
    if n == 0:
        return {
            "n_closed": 0,
            "n_open": sum(1 for t in trades if not t.get("exit_date")),
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
        "n_open": sum(1 for t in trades if not t.get("exit_date")),
        "total_pnl_gbp": round(total_pnl, 2),
        "avg_pnl_pct": round(avg_pnl_pct, 2),
        "hit_rate": round(wins / n, 3),
        "n_predictions": len(predictions),
    }


def build_dashboard_data() -> dict:
    """Assemble the JSON payload the static dashboard renders."""
    configs = _load_strategy_configs()
    all_trades = list(_iter_jsonl(ledger_path()))
    all_predictions = list(_iter_jsonl(predictions_path()))

    trades_by_strategy: dict[str, list[dict]] = defaultdict(list)
    for t in all_trades:
        trades_by_strategy[t["strategy_id"]].append(t)

    preds_by_strategy: dict[str, list[dict]] = defaultdict(list)
    for p in all_predictions:
        if not p.get("was_traded"):
            preds_by_strategy[p["strategy_id"]].append(p)

    active: list[dict] = []
    archived: list[dict] = []
    for sid, config in sorted(configs.items()):
        trades = trades_by_strategy.get(sid, [])
        preds = preds_by_strategy.get(sid, [])
        starting_capital = float(config.get("capital_gbp", 10000))

        entry = {
            "id": sid,
            "display_name": config.get("display_name", sid),
            "description": config.get("description", "").strip(),
            "tier": config.get("tier", "shadow"),
            "region": config.get("region", "us"),
            "universe": config.get("universe", "sp500"),
            "capital_gbp": starting_capital,
            "summary": _summary_stats(trades, preds),
            "equity_curve": _equity_curve(trades, starting_capital),
            "executed": sorted(trades, key=lambda t: t.get("entry_date", ""), reverse=True),
            "uncommitted": sorted(preds, key=lambda p: p.get("prediction_date", ""), reverse=True),
        }

        if config.get("active"):
            active.append(entry)
        elif trades or preds:
            archived.append(entry)
        # Strategies with no data and not active are excluded from the dashboard
        # (they're "dormant" — seeded for later waves)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "active": active,
        "archived": archived,
    }


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
