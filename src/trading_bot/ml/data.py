"""Training data for the ML challenger — audit + separate snapshot.

The runtime cache (state/ohlcv.db) lives in GitHub actions/cache, holds
at most ~250 trading sessions (365-day prune) and only covers tickers
some strategy has fetched. Weekend 1 therefore starts with a data
audit; if the runtime store is too thin for training (median depth
< ~150 sessions over the universe), we backfill a *separate* training
snapshot at `state/ml/train.db` via yfinance — never widening the
runtime cache, whose size/coverage is tuned for the morning pipeline.

The snapshot uses the identical `bars` schema as tools.ohlcv_store so
the feature pipeline reads either interchangeably, and fetches with
auto_adjust=False to match what tools.history writes into the runtime
store — the model must train on the same flavour of prices it will be
served at inference time.

CLI:
  python -m trading_bot.ml.data audit    [--universe t212_isa_us]
  python -m trading_bot.ml.data backfill [--days 550] [--batch 200]
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

from trading_bot.state.paths import STATE_ROOT
from trading_bot.tools.ohlcv_store import StoredBar


log = logging.getLogger(__name__)

# Median universe depth (sessions) below which the runtime store is too
# thin to train on and the separate snapshot is required.
MIN_TRAIN_DEPTH_SESSIONS = 150

# Backfill window. ~550 calendar days ≈ 380 sessions: the 63-session
# feature burn-in still leaves a full year of labelled rows. The
# runtime cache's 365-day prune ceiling doesn't apply here — this
# snapshot is training-only and regenerable.
DEFAULT_BACKFILL_CALENDAR_DAYS = 550


def snapshot_path() -> Path:
    return STATE_ROOT / "ml" / "train.db"


def _snap_conn(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(path))
    c.row_factory = sqlite3.Row
    c.execute("""
        CREATE TABLE IF NOT EXISTS bars (
            ticker TEXT NOT NULL,
            bar_date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, bar_date)
        )
    """)
    return c


def write_snapshot_bars(path: Path, rows: list[StoredBar]) -> int:
    if not rows:
        return 0
    c = _snap_conn(path)
    try:
        c.executemany(
            "INSERT OR REPLACE INTO bars (ticker, bar_date, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (r.ticker, r.bar_date.isoformat(), r.open, r.high, r.low, r.close, r.volume)
                for r in rows
            ],
        )
        c.commit()
    finally:
        c.close()
    return len(rows)


def read_snapshot_bars(
    path: Path, tickers: list[str], start: date, end: date,
) -> dict[str, list[StoredBar]]:
    """Same shape as tools.ohlcv_store.read_bars_bulk, from the snapshot."""
    if not tickers or not path.exists():
        return {}
    out: dict[str, list[StoredBar]] = {}
    c = _snap_conn(path)
    try:
        chunk_size = 500
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i : i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            rows = c.execute(
                f"SELECT ticker, bar_date, open, high, low, close, volume "
                f"FROM bars WHERE ticker IN ({placeholders}) "
                f"AND bar_date BETWEEN ? AND ? ORDER BY ticker, bar_date ASC",
                (*chunk, start.isoformat(), end.isoformat()),
            ).fetchall()
            for r in rows:
                out.setdefault(r["ticker"], []).append(StoredBar(
                    ticker=r["ticker"],
                    bar_date=date.fromisoformat(r["bar_date"]),
                    open=r["open"] or 0.0, high=r["high"] or 0.0,
                    low=r["low"] or 0.0, close=r["close"] or 0.0,
                    volume=int(r["volume"] or 0),
                ))
    finally:
        c.close()
    return out


def snapshot_tickers(path: Path) -> list[str]:
    if not path.exists():
        return []
    c = _snap_conn(path)
    try:
        return [r[0] for r in c.execute("SELECT DISTINCT ticker FROM bars ORDER BY ticker")]
    finally:
        c.close()


# ---------------------------------------------------------------------------
# Audit — how deep is the store we'd train on?
# ---------------------------------------------------------------------------

def depth_report(db_path: Path, tickers: list[str]) -> dict:
    """Per-universe depth stats over one bars db. Returns
    {n_universe, n_covered, median_depth, p10_depth, n_depth_ge_150,
    date_min, date_max, total_rows}."""
    empty = {
        "n_universe": len(tickers), "n_covered": 0, "median_depth": 0,
        "p10_depth": 0, "n_depth_ge_150": 0, "date_min": None,
        "date_max": None, "total_rows": 0,
    }
    if not db_path.exists() or not tickers:
        return empty
    c = sqlite3.connect(str(db_path))
    try:
        try:
            rows = c.execute("SELECT ticker, COUNT(*) FROM bars GROUP BY ticker").fetchall()
            lo, hi, total = c.execute(
                "SELECT MIN(bar_date), MAX(bar_date), COUNT(*) FROM bars"
            ).fetchone()
        except sqlite3.OperationalError:
            return empty
    finally:
        c.close()
    wanted = set(tickers)
    depths = sorted(n for t, n in rows if t in wanted)
    if not depths:
        return {**empty, "date_min": lo, "date_max": hi, "total_rows": total}
    return {
        "n_universe": len(tickers),
        "n_covered": len(depths),
        "median_depth": depths[len(depths) // 2],
        "p10_depth": depths[len(depths) // 10],
        "n_depth_ge_150": sum(1 for d in depths if d >= MIN_TRAIN_DEPTH_SESSIONS),
        "date_min": lo, "date_max": hi, "total_rows": total,
    }


def audit(universe_id: str = "t212_isa_us") -> dict:
    """Depth report for both the runtime store and the training snapshot,
    plus the verdict the design doc's Weekend-1 gate needs: is the
    runtime store deep enough, or is the snapshot required?"""
    from trading_bot.tools.universe import get_universe
    tickers = get_universe(universe_id)
    runtime = depth_report(STATE_ROOT / "ohlcv.db", tickers)
    snapshot = depth_report(snapshot_path(), tickers)
    needs_snapshot = runtime["median_depth"] < MIN_TRAIN_DEPTH_SESSIONS
    return {
        "universe": universe_id,
        "runtime_store": runtime,
        "training_snapshot": snapshot,
        "needs_snapshot": needs_snapshot,
    }


# ---------------------------------------------------------------------------
# Backfill — yfinance batch download into the snapshot
# ---------------------------------------------------------------------------

def default_backfill_tickers() -> list[str]:
    """S&P 1500 ∩ t212_isa_us — liquid US names the bot can actually
    trade, ~1.5k of them (the doc's realistic 1–3k band). Survivorship
    caveat: applying *today's* index membership historically omits
    delisted names and flatters the backtest slightly; documented in
    the model card, and the forward shadow run is immune."""
    from trading_bot.tools.universe import get_universe
    sp1500 = set(get_universe("sp1500"))
    t212 = set(get_universe("t212_isa_us"))
    both = sorted(sp1500 & t212)
    # If the T212 instrument cache is unavailable (fresh clone without
    # state), fall back to plain S&P 1500 rather than dying.
    return both if both else sorted(sp1500)


def backfill(
    tickers: list[str] | None = None,
    *,
    days: int = DEFAULT_BACKFILL_CALENDAR_DAYS,
    batch_size: int = 200,
    db_path: Path | None = None,
    end: date | None = None,
) -> dict:
    """Download `days` of daily OHLCV for `tickers` into the snapshot.
    auto_adjust=False to match tools.history's runtime-store writes.
    Failed tickers are skipped (reported), not retried via Stooq here —
    the snapshot doesn't need 100% coverage, just bulk."""
    import yfinance as yf

    tickers = tickers or default_backfill_tickers()
    db = db_path or snapshot_path()
    end = end or date.today()
    start = end - timedelta(days=days)
    written = 0
    failed: list[str] = []

    for i in range(0, len(tickers), batch_size):
        chunk = tickers[i : i + batch_size]
        log.info("backfill: batch %d–%d of %d", i, i + len(chunk), len(tickers))
        try:
            df = yf.download(
                chunk,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                progress=False,
                threads=True,
            )
        except Exception as e:  # yfinance raises all sorts
            log.warning("backfill: batch failed (%s) — skipping %d tickers", e, len(chunk))
            failed.extend(chunk)
            continue
        if df is None or df.empty:
            failed.extend(chunk)
            continue

        rows: list[StoredBar] = []
        for t in chunk:
            try:
                sub = df[t] if len(chunk) > 1 else df
            except KeyError:
                failed.append(t)
                continue
            sub = sub.dropna(subset=["Open", "High", "Low", "Close"])
            if sub.empty:
                failed.append(t)
                continue
            for idx, r in sub.iterrows():
                rows.append(StoredBar(
                    ticker=t,
                    bar_date=idx.date(),
                    open=float(r["Open"]), high=float(r["High"]),
                    low=float(r["Low"]), close=float(r["Close"]),
                    volume=int(r["Volume"]) if r["Volume"] == r["Volume"] else 0,
                ))
        written += write_snapshot_bars(db, rows)

    return {
        "db": str(db), "n_tickers": len(tickers), "n_failed": len(failed),
        "rows_written": written, "start": start.isoformat(), "end": end.isoformat(),
        "failed": failed[:50],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_audit = sub.add_parser("audit", help="depth report over the universe")
    p_audit.add_argument("--universe", default="t212_isa_us")

    p_back = sub.add_parser("backfill", help="yfinance backfill into state/ml/train.db")
    p_back.add_argument("--days", type=int, default=DEFAULT_BACKFILL_CALENDAR_DAYS)
    p_back.add_argument("--batch", type=int, default=200)

    args = p.parse_args(argv)
    if args.cmd == "audit":
        report = audit(args.universe)
        import json as _json
        print(_json.dumps(report, indent=2))
        if report["needs_snapshot"] and report["training_snapshot"]["median_depth"] < MIN_TRAIN_DEPTH_SESSIONS:
            print(
                f"\nVERDICT: runtime store median depth "
                f"{report['runtime_store']['median_depth']} < {MIN_TRAIN_DEPTH_SESSIONS} "
                f"sessions and no adequate snapshot — run "
                f"`python -m trading_bot.ml.data backfill` first.",
                file=sys.stderr,
            )
        return 0
    if args.cmd == "backfill":
        result = backfill(days=args.days, batch_size=args.batch)
        import json as _json
        print(_json.dumps(result, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
