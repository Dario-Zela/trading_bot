"""Phase 11D — IC noise-floor Monte Carlo.

Question the diagnostic answers: when the evolution agent's promotion
gate demands `IC > 0.05`, is that meaningfully above what a random
strategy would produce on our sample sizes?

Method:
1. Read state/predictions.jsonl (per-trade predictions with predicted
   + actual return).
2. Pair each strategy's predictions; compute its real IC.
3. For each strategy, shuffle the `actual_return_pct` labels N=1000×
   and recompute IC. The 95th percentile of that distribution IS the
   noise floor — IC values below that line are "could be random".
4. Print a per-strategy report + write `state/diagnostics/ic_noise_floor.md`.

Run: `python scripts/ic_noise_floor.py [--n 1000] [--strategy id]`
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

from trading_bot.state.paths import STATE_ROOT


log = logging.getLogger("ic_noise_floor")


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    den_y = (sum((y - my) ** 2 for y in ys)) ** 0.5
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _read_predictions() -> dict[str, list[tuple[float, float]]]:
    """Returns {strategy_id: [(predicted_pct, actual_pct), ...]}."""
    p = STATE_ROOT / "predictions.jsonl"
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    if not p.exists():
        return out
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("strategy_id") or ""
            pred = rec.get("predicted_return_pct")
            actual = rec.get("actual_return_pct")
            if not sid or pred is None or actual is None:
                continue
            try:
                out[sid].append((float(pred), float(actual)))
            except (TypeError, ValueError):
                continue
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IC noise-floor Monte Carlo.")
    parser.add_argument("--n", type=int, default=1000, help="MC iterations per strategy")
    parser.add_argument("--strategy", help="Limit to one strategy id")
    parser.add_argument("--quantile", type=float, default=0.95,
                        help="Confidence (1 - α). 0.95 = 95th-pct noise floor")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data = _read_predictions()
    if not data:
        print("No predictions on file; nothing to analyse.")
        return 0
    if args.strategy:
        data = {args.strategy: data.get(args.strategy, [])}
        if not data[args.strategy]:
            print(f"Strategy {args.strategy} has no predictions on file.")
            return 0

    rng = random.Random(42)   # reproducible
    rows = []
    print(f"\nIC noise-floor Monte Carlo — {args.n} shuffles × {len(data)} strategies\n")
    print(f"{'Strategy':<28} {'N':>6} {'Real IC':>9} {'Noise q95':>10} {'Verdict':>14}")
    print("=" * 75)
    for sid, pairs in sorted(data.items()):
        n = len(pairs)
        if n < 5:
            print(f"{sid:<28} {n:>6} {'—':>9} {'—':>10} {'too few':>14}")
            continue
        preds = [p for p, _ in pairs]
        actuals = [a for _, a in pairs]
        real_ic = _pearson(preds, actuals)
        if real_ic is None:
            print(f"{sid:<28} {n:>6} {'—':>9} {'—':>10} {'degenerate':>14}")
            continue

        # Shuffle MC
        shuffled_ics: list[float] = []
        actuals_copy = list(actuals)
        for _ in range(args.n):
            rng.shuffle(actuals_copy)
            ic = _pearson(preds, actuals_copy)
            if ic is not None:
                shuffled_ics.append(ic)
        shuffled_ics.sort()
        idx = max(0, min(len(shuffled_ics) - 1, int(args.quantile * len(shuffled_ics))))
        noise_floor = shuffled_ics[idx]

        # Verdict
        if real_ic >= noise_floor + 0.02:
            verdict = "above noise"
        elif real_ic >= noise_floor:
            verdict = "marginal"
        else:
            verdict = "noise"

        rows.append({
            "sid": sid, "n": n, "real_ic": real_ic,
            "noise_floor": noise_floor, "verdict": verdict,
        })
        print(f"{sid:<28} {n:>6} {real_ic:>+9.3f} {noise_floor:>+10.3f} {verdict:>14}")

    # Write report
    out_dir = STATE_ROOT / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "ic_noise_floor.md"
    with md_path.open("w") as f:
        f.write(f"# IC noise floor ({args.n} MC iterations, q={args.quantile})\n\n")
        f.write("Real IC vs the IC you'd get by shuffling actual returns randomly.\n")
        f.write("Verdict: 'above noise' means the strategy clears the noise floor by ≥0.02.\n\n")
        f.write("| Strategy | N | Real IC | Noise q95 | Verdict |\n")
        f.write("|---|---:|---:|---:|---|\n")
        for r in rows:
            f.write(f"| {r['sid']} | {r['n']} | {r['real_ic']:+.3f} | {r['noise_floor']:+.3f} | {r['verdict']} |\n")
    print(f"\nReport written to {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
