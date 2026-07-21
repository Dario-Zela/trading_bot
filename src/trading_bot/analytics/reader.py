"""Read-side helpers for the dbt analytics store.

Callers that used to iterate state/*.jsonl and aggregate in Python
should instead SELECT from the mart tables here. If the DuckDB store
doesn't exist yet (first run, or CI cache miss), `connect()` builds
it on-demand via dbt_runner.build().

The higher-level helpers (`get_fee_drag`, `get_weekly_pnl`, ...) return
plain lists of dicts so callers don't take a hard pandas dependency
just to read one row.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import duckdb

from trading_bot.analytics.dbt_runner import _dbt_dir, _duckdb_path, build

log = logging.getLogger(__name__)


def connect(*, build_if_missing: bool = True) -> duckdb.DuckDBPyConnection:
    """Open a read-only connection to the dbt DuckDB store.

    On first use (or after a clean checkout) the file won't exist —
    setting build_if_missing=True runs dbt to create it. Callers that
    need to guarantee freshness should call `dbt_runner.build()`
    themselves before calling `connect()`.
    """
    path = _duckdb_path()
    if not path.exists():
        if build_if_missing:
            log.info("Analytics DuckDB not found at %s — building", path)
            build()
        else:
            raise FileNotFoundError(
                f"Analytics DuckDB not found at {path} — call "
                f"trading_bot.analytics.dbt_runner.build() first"
            )
    # DuckDB doesn't support concurrent writers, but read_only=True
    # opens a snapshot so a running `dbt run` in another process
    # doesn't fight us.
    return duckdb.connect(str(path), read_only=True)


def _rows(cur: Any) -> list[dict]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Typed accessors — one per mart.
# ---------------------------------------------------------------------------

def get_fee_drag(
    strategy_id: str | None = None,
    region: str | None = None,
) -> list[dict]:
    """Rows from mart_fee_drag. Optional filters."""
    with connect() as con:
        sql = "select * from marts.mart_fee_drag where 1=1"
        params: list = []
        if strategy_id is not None:
            sql += " and strategy_id = ?"
            params.append(strategy_id)
        if region is not None:
            sql += " and region = ?"
            params.append(region)
        return _rows(con.execute(sql, params))


def get_weekly_pnl(
    strategy_id: str | None = None,
    region: str | None = None,
    weeks: int | None = None,
) -> list[dict]:
    """Rows from mart_strategy_weekly. Optional filters + trailing-weeks cap."""
    with connect() as con:
        sql = "select * from marts.mart_strategy_weekly where 1=1"
        params: list = []
        if strategy_id is not None:
            sql += " and strategy_id = ?"
            params.append(strategy_id)
        if region is not None:
            sql += " and region = ?"
            params.append(region)
        sql += " order by iso_year desc, iso_week desc"
        if weeks is not None:
            sql += f" limit {int(weeks)}"
        return _rows(con.execute(sql, params))


def get_fee_drag_windowed(
    strategy_id: str,
    region: str | None = None,
    days: int = 14,
) -> dict:
    """Windowed fee-drag rollup for a single strategy. Queries
    stg_ledger directly (not the mart) since the mart is lifetime.

    Returns the same shape the legacy `fees_pct_of_gross` returned:
    {n_trades, fees_gbp, gross_pnl_gbp, net_pnl_gbp, fees_pct_of_gross}.
    """
    with connect() as con:
        params: list = [strategy_id, days]
        region_filter = ""
        if region is not None:
            region_filter = "and region = ?"
            params.append(region)
        row = con.execute(
            f"""
            select
                count(*)             as n_trades,
                sum(fees_gbp)        as fees_gbp,
                sum(gross_pnl_gbp)   as gross_pnl_gbp,
                sum(pnl_gbp)         as net_pnl_gbp
            from staging.stg_ledger
            where is_closed
              and exit_reason not in ('cancelled', 'cleared')
              and strategy_id = ?
              and exit_date >= current_date - cast(? as integer) * interval 1 day
              {region_filter}
            """,
            params,
        ).fetchone()
    n, fees, gross, net = row
    fees = float(fees or 0.0)
    gross = float(gross or 0.0)
    net = float(net or 0.0)
    pct = (fees / abs(gross) * 100.0) if abs(gross) > 0.01 else 0.0
    return {
        "n_trades": int(n or 0),
        "fees_gbp": round(fees, 2),
        "gross_pnl_gbp": round(gross, 2),
        "net_pnl_gbp": round(net, 2),
        "fees_pct_of_gross": round(pct, 1),
    }


def get_exit_attribution(
    scope: str = "system",
    strategy_id: str | None = None,
    region: str | None = None,
) -> list[dict]:
    """Rows from mart_exit_attribution. scope='system' or 'strategy'."""
    with connect() as con:
        sql = "select * from marts.mart_exit_attribution where scope = ?"
        params: list = [scope]
        if strategy_id is not None:
            sql += " and strategy_id = ?"
            params.append(strategy_id)
        if region is not None:
            sql += " and region = ?"
            params.append(region)
        sql += " order by total_pnl_gbp desc"
        return _rows(con.execute(sql, params))
