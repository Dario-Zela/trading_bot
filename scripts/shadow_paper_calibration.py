"""Phase 11H — shadow vs paper-broker calibration.

Shadow tier fills at yfinance closes (no slippage, no spread, no
partial fills). Paper tiers (Alpaca, T212) hit real broker simulation
with realistic fills. Comparing the same (strategy, region, ticker, date)
trade across tiers tells us how much shadow over-states live P&L.

Output: per-broker median slippage + distribution + by-ticker outliers.
Low-N until we have ≥50 paired trades — the caveat is loud in the
output.

Run: `python scripts/shadow_paper_calibration.py`
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from trading_bot.state.paths import STATE_ROOT, ledger_path


log = logging.getLogger("shadow_paper")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shadow vs paper-broker calibration.")
    parser.add_argument("--min-pairs", type=int, default=10,
                        help="Below this paired count, surface a low-N caveat (default 10).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    p = ledger_path()
    if not p.exists():
        print("No ledger yet — nothing to calibrate.")
        return 0

    # Index trades by (region, ticker, entry_date)
    by_key: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tier = rec.get("tier")
            ed = rec.get("entry_date")
            tkr = rec.get("ticker")
            region = rec.get("region") or "?"
            if not tier or not ed or not tkr:
                continue
            if rec.get("exit_reason") in ("cancelled", "cleared"):
                continue
            key = (region, tkr, ed)
            by_key[key][tier] = rec

    # Pair shadow with each paper tier
    paper_tiers = ("alpaca-paper", "trading212-paper")
    pairs_by_tier: dict[str, list[dict]] = defaultdict(list)
    for key, tier_map in by_key.items():
        sh = tier_map.get("shadow")
        if sh is None:
            continue
        for pt in paper_tiers:
            paper = tier_map.get(pt)
            if paper is None:
                continue
            try:
                sh_pct = float(sh.get("pnl_pct") or 0)
                pp_pct = float(paper.get("pnl_pct") or 0)
            except (TypeError, ValueError):
                continue
            # Slippage: shadow over-state = shadow_pct - paper_pct
            slip = sh_pct - pp_pct
            pairs_by_tier[pt].append({
                "ticker": key[1], "region": key[0], "date": key[2],
                "shadow_pct": sh_pct, "paper_pct": pp_pct, "slippage_pct": slip,
            })

    if not pairs_by_tier:
        print("No paired (shadow, paper) trades on the same (strategy, region, ticker, date) yet.")
        print("This calibration will become meaningful once both tiers have run the same names.")
        return 0

    out_dir = STATE_ROOT / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "shadow_paper_calibration.md"
    with md.open("w") as f:
        f.write("# Shadow vs paper-broker calibration\n\n")
        f.write("Slippage = shadow_pct − paper_pct. Positive means shadow over-states P&L.\n\n")

        print(f"\n{'Tier':<22} {'N':>5} {'Median slip':>14} {'Mean slip':>12} {'P10':>9} {'P90':>9}")
        print("=" * 75)
        for tier, pairs in sorted(pairs_by_tier.items()):
            n = len(pairs)
            slips = sorted(p["slippage_pct"] for p in pairs)
            median = statistics.median(slips)
            mean = statistics.mean(slips)
            p10 = slips[max(0, int(0.1 * n))] if n else 0
            p90 = slips[min(n - 1, int(0.9 * n))] if n else 0
            tag = "" if n >= args.min_pairs else " ⚠ low-N"
            print(f"{tier:<22} {n:>5} {median:>+13.3f}%{mean:>+11.3f}%{p10:>+8.3f}%{p90:>+8.3f}%{tag}")
            f.write(f"## {tier} ({n} paired trades)\n\n")
            if n < args.min_pairs:
                f.write(f"_⚠ Low N ({n} < {args.min_pairs}) — these stats are not yet reliable._\n\n")
            f.write(f"- Median slippage: {median:+.3f}%\n")
            f.write(f"- Mean slippage: {mean:+.3f}%\n")
            f.write(f"- 10th–90th percentile: {p10:+.3f}% → {p90:+.3f}%\n\n")
            # Top 5 outliers
            outliers = sorted(pairs, key=lambda r: abs(r["slippage_pct"]), reverse=True)[:5]
            if outliers:
                f.write("Top outliers:\n\n")
                for o in outliers:
                    f.write(f"- {o['ticker']} ({o['region']}, {o['date']}): "
                            f"shadow {o['shadow_pct']:+.2f}% vs paper {o['paper_pct']:+.2f}% "
                            f"→ slip {o['slippage_pct']:+.2f}%\n")
                f.write("\n")

    print(f"\nReport written to {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
