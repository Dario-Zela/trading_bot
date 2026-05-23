"""Local OHLCV cache backing `tools.history.get_history`.

A single SQLite file at `state/ohlcv.db` storing the last ~1 year of
daily OHLCV bars for every ticker the bot has ever fetched. Grows
organically as strategies fetch — there's no batch-backfill step;
every yfinance miss writes its result back to the store, so after a
few weeks of running the store covers most of the active universe.

Why local cache:
  - The morning entry pipeline used to fetch ~12 min of yfinance OHLCV
    for the universe. With the per-strategy LLM pre-filter that's
    already reduced to ~40s, but the cache makes EVEN that ~free
    after warm-up.
  - Evolution-time reads (reflection + missed-movers prep) become
    fast and reproducible without hammering yfinance.
  - The data is owned by us, so a yfinance outage doesn't kill the
    morning pipeline.

Why SQLite over DuckDB:
  - Standard library, zero install friction. CI doesn't need a wheel.
  - DuckDB's columnar speed advantage doesn't matter at our scale
    (~50 MB, ~hundreds of queries per morning).

1-year rolling cutoff: `prune_old(cutoff_date)` is called by the daily
maintenance cron. Drops bars older than 365 days to bound size.

Stored schema:
  bars(ticker TEXT, bar_date TEXT ISO, open REAL, high REAL, low REAL,
       close REAL, volume INTEGER, PRIMARY KEY (ticker, bar_date))
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from trading_bot.state.paths import STATE_ROOT


log = logging.getLogger(__name__)


_DB_FILENAME = "ohlcv.db"
_DEFAULT_CUTOFF_DAYS = 365


@dataclass(frozen=True)
class StoredBar:
    """Minimal projection of the bars row used by callers. Matches the
    shape of tools.history.Bar so callers can interchange them."""
    ticker: str
    bar_date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


def _db_path() -> Path:
    return STATE_ROOT / _DB_FILENAME


@contextmanager
def _conn():
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(_db_path()))
    c.row_factory = sqlite3.Row
    try:
        # WAL mode lets concurrent readers and writers coexist — important
        # because the parallel-strategies executor will hit get_history in
        # multiple threads.
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        yield c
        c.commit()
    finally:
        c.close()


def init_store() -> None:
    """Create the schema if missing. Idempotent."""
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS bars (
                ticker TEXT NOT NULL,
                bar_date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, bar_date)
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_bars_date ON bars(bar_date)")


def read_bars(ticker: str, start: date, end: date) -> list[StoredBar]:
    """Return stored bars for `ticker` between `start` and `end`
    inclusive, ordered ascending. Empty list if the store has nothing."""
    init_store()
    with _conn() as c:
        rows = c.execute(
            "SELECT bar_date, open, high, low, close, volume FROM bars "
            "WHERE ticker = ? AND bar_date BETWEEN ? AND ? "
            "ORDER BY bar_date ASC",
            (ticker, start.isoformat(), end.isoformat()),
        ).fetchall()
    return [
        StoredBar(
            ticker=ticker,
            bar_date=date.fromisoformat(r["bar_date"]),
            open=r["open"] if r["open"] is not None else 0.0,
            high=r["high"] if r["high"] is not None else 0.0,
            low=r["low"] if r["low"] is not None else 0.0,
            close=r["close"] if r["close"] is not None else 0.0,
            volume=int(r["volume"]) if r["volume"] is not None else 0,
        )
        for r in rows
    ]


def read_bars_bulk(
    tickers: list[str], start: date, end: date,
) -> dict[str, list[StoredBar]]:
    """Bulk version. Returns {ticker: [StoredBar, ...]}; tickers with no
    rows in the requested range are omitted from the result."""
    if not tickers:
        return {}
    init_store()
    # SQLite has a parameter cap (~999 by default). Chunk for safety on
    # universe-size reads.
    out: dict[str, list[StoredBar]] = {}
    chunk_size = 500
    with _conn() as c:
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i : i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            rows = c.execute(
                f"SELECT ticker, bar_date, open, high, low, close, volume "
                f"FROM bars WHERE ticker IN ({placeholders}) "
                f"AND bar_date BETWEEN ? AND ? "
                f"ORDER BY ticker, bar_date ASC",
                (*chunk, start.isoformat(), end.isoformat()),
            ).fetchall()
            for r in rows:
                out.setdefault(r["ticker"], []).append(StoredBar(
                    ticker=r["ticker"],
                    bar_date=date.fromisoformat(r["bar_date"]),
                    open=r["open"] if r["open"] is not None else 0.0,
                    high=r["high"] if r["high"] is not None else 0.0,
                    low=r["low"] if r["low"] is not None else 0.0,
                    close=r["close"] if r["close"] is not None else 0.0,
                    volume=int(r["volume"]) if r["volume"] is not None else 0,
                ))
    return out


def write_bars(rows: list[StoredBar]) -> int:
    """Upsert bars. Returns count written. Safe to call concurrently —
    WAL mode + INSERT-OR-REPLACE handles it cleanly."""
    if not rows:
        return 0
    init_store()
    payload = [
        (r.ticker, r.bar_date.isoformat(), r.open, r.high, r.low, r.close, r.volume)
        for r in rows
    ]
    with _conn() as c:
        c.executemany(
            "INSERT OR REPLACE INTO bars"
            " (ticker, bar_date, open, high, low, close, volume)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            payload,
        )
    return len(payload)


def coverage(ticker: str) -> tuple[date | None, date | None]:
    """Return (earliest, latest) bar dates we have for this ticker, or
    (None, None) if the ticker isn't in the store."""
    init_store()
    with _conn() as c:
        row = c.execute(
            "SELECT MIN(bar_date) AS lo, MAX(bar_date) AS hi FROM bars "
            "WHERE ticker = ?",
            (ticker,),
        ).fetchone()
    if not row or not row["lo"]:
        return (None, None)
    return (date.fromisoformat(row["lo"]), date.fromisoformat(row["hi"]))


def prune_old(cutoff_days: int = _DEFAULT_CUTOFF_DAYS) -> int:
    """Drop bars older than `cutoff_days` ago. Returns rows deleted.
    Called nightly by the maintenance cron to bound store size."""
    init_store()
    cutoff = (date.today() - timedelta(days=cutoff_days)).isoformat()
    with _conn() as c:
        cursor = c.execute("DELETE FROM bars WHERE bar_date < ?", (cutoff,))
        deleted = cursor.rowcount or 0
        # Reclaim space if we deleted a noticeable amount
        if deleted > 1000:
            c.execute("VACUUM")
    log.info("ohlcv_store: pruned %d bars older than %s", deleted, cutoff)
    return deleted


def store_size_bytes() -> int:
    """File size of the SQLite database, for dashboard/observability."""
    p = _db_path()
    return p.stat().st_size if p.exists() else 0


def row_count() -> int:
    """Total bars stored. Useful for monitoring."""
    init_store()
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
