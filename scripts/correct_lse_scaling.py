#!/usr/bin/env python3
"""Correct the OHLCV store in place: ×100 the bars of non-pence `.L` lines.

The old history code divided EVERY `.L` ticker by 100 (assuming pence). That
is correct for GBp-quoted ordinary shares but wrong for GBP/USD/EUR-quoted
ETF/bond/GDR lines, which got stored 100x too small. Since the old scaling
was a uniform ÷100, the true price is exactly stored × 100 for those lines.

Reads the authoritative quote-currency map (state/lse_quote_ccy.json) and
multiplies OHLC (not volume) by 100 for every ticker whose currency is a real
non-pence currency. GBp lines and UNKNOWN lines are left untouched.
"""
from __future__ import annotations

import json
import sqlite3
import sys

from trading_bot.state.paths import STATE_ROOT

_DB = STATE_ROOT / "ohlcv.db"
_MAP = STATE_ROOT / "lse_quote_ccy.json"


def main() -> int:
    m = json.loads(_MAP.read_text())
    non_pence = sorted(t for t, v in m.items() if v not in ("GBp", "UNKNOWN"))
    print(f"{len(non_pence)} non-pence tickers to ×100 "
          f"({sum(v=='UNKNOWN' for v in m.values())} UNKNOWN left untouched)")

    with sqlite3.connect(str(_DB)) as c:
        # sanity: how many bars will change
        ph = ",".join("?" * len(non_pence))
        n = c.execute(f"SELECT COUNT(*) FROM bars WHERE ticker IN ({ph})", non_pence).fetchone()[0]
        print(f"correcting {n:,} bars")
        c.executemany(
            "UPDATE bars SET open=open*100, high=high*100, low=low*100, close=close*100 "
            "WHERE ticker = ?",
            [(t,) for t in non_pence],
        )
        c.commit()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
