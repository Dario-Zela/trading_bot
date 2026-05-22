#!/usr/bin/env python3
"""Targeted retry of UNKNOWN entries in the LSE currency map (rate-limit
casualties from the main sweep). Low concurrency, long backoff. Resumable."""
from __future__ import annotations

import json
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
import yfinance as yf

from trading_bot.state.paths import STATE_ROOT

_MAP = STATE_ROOT / "lse_quote_ccy.json"


def _lookup(t: str) -> str | None:
    for attempt in range(6):
        try:
            tk = yf.Ticker(t)
            c = getattr(tk.fast_info, "currency", None)
            if not c:
                c = (tk.info or {}).get("currency")  # ETP/leveraged lines
            return str(c) if c else None
        except Exception as e:
            if "Too Many Requests" in str(e) or "rate" in str(e).lower():
                time.sleep((attempt + 1) * 4 + random.random() * 3)
                continue
            return None
    return None


def main() -> int:
    m = json.loads(_MAP.read_text())
    todo = sorted(t for t, v in m.items() if v == "UNKNOWN")
    print(f"retrying {len(todo)} UNKNOWN tickers")
    resolved = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(_lookup, t): t for t in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            ccy = fut.result()
            if ccy:
                m[t] = ccy
                resolved += 1
            if i % 25 == 0:
                _MAP.write_text(json.dumps(m, sort_keys=True))
                print(f"  {i}/{len(todo)} · resolved={resolved}")
    _MAP.write_text(json.dumps(m, sort_keys=True))
    from collections import Counter
    print("now:", dict(Counter(m.values())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
