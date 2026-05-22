#!/usr/bin/env python3
"""Build a clean, complete LSE quote-currency map for every `.L` ticker in
the OHLCV store. Patient + resumable: low concurrency, retry-with-backoff,
and unresolved tickers are recorded as 'UNKNOWN' (never silently assumed
pence) so rate-limiting only slows the run, never corrupts the map.

Writes state/lse_quote_ccy.json incrementally. Re-run to resume: anything
already resolved to a real currency is skipped; only UNKNOWN / missing
tickers are retried.
"""
from __future__ import annotations

import json
import random
import sqlite3
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

import yfinance as yf

from trading_bot.state.paths import STATE_ROOT

_DB = STATE_ROOT / "ohlcv.db"
_MAP = STATE_ROOT / "lse_quote_ccy.json"
_WORKERS = 4
_RETRIES = 5


def _lookup(ticker: str) -> str | None:
    for attempt in range(_RETRIES):
        try:
            c = getattr(yf.Ticker(ticker).fast_info, "currency", None)
            if c:
                return str(c)
            return None  # genuinely no currency field
        except Exception as e:
            if "Too Many Requests" in str(e) or "rate" in str(e).lower():
                time.sleep((attempt + 1) * 3 + random.random() * 2)
                continue
            return None
    return None


def main() -> int:
    with sqlite3.connect(str(_DB)) as c:
        lse = sorted(r[0] for r in c.execute(
            "SELECT DISTINCT ticker FROM bars WHERE ticker LIKE '%.L'"))

    try:
        cur_map: dict[str, str] = json.loads(_MAP.read_text())
    except (OSError, ValueError):
        cur_map = {}

    # Resolve-worthy: never looked up, or previously UNKNOWN, or previously
    # poisoned as GBp by the old failure-default (re-verify all GBp to be safe).
    todo = [t for t in lse if cur_map.get(t) in (None, "UNKNOWN", "GBp")]
    print(f"{len(lse)} .L tickers · {len(cur_map)} cached · {len(todo)} to (re)resolve")

    resolved = 0
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futs = {pool.submit(_lookup, t): t for t in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            ccy = fut.result()
            cur_map[t] = ccy if ccy else "UNKNOWN"
            if ccy:
                resolved += 1
            if i % 100 == 0:
                _MAP.write_text(json.dumps(cur_map, sort_keys=True))
                unknown = sum(1 for v in cur_map.values() if v == "UNKNOWN")
                print(f"  {i}/{len(todo)} done · resolved={resolved} · still UNKNOWN={unknown}")

    _MAP.write_text(json.dumps(cur_map, sort_keys=True))
    from collections import Counter
    dist = Counter(cur_map.values())
    print(f"\nFinal map: {len(cur_map)} tickers")
    for k, v in dist.most_common():
        print(f"  {k:8} {v}")
    remaining = sum(1 for t in lse if cur_map.get(t) in (None, "UNKNOWN"))
    print(f"Still unresolved: {remaining} (re-run to retry)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
