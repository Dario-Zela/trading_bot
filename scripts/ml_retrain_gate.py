"""Promotion gate for the monthly ml-challenger retrain.

Mirrors the bot's human-gated promotion philosophy: a fresh model is
committed only if its pooled OOS rank-IC Fisher-z 95% lower bound is at
least the current card's value minus a tolerance. Otherwise the retrain
workflow restores the old artifacts and opens an issue with the diff.

Usage (from .github/workflows/retrain-challenger.yml):
  python scripts/ml_retrain_gate.py --old /tmp/old_metrics.json \
      --new strategies/ml-challenger/model/metrics.json [--tolerance 0.02]

Exit 0 → promote (also appends one line to model/CHANGELOG.md).
Exit 1 → reject (prints a markdown diff summary to stdout for the issue body).
A missing/empty --old file (first ever run) always promotes.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--tolerance", type=float, default=0.02)
    args = p.parse_args(argv)

    new = _load(args.new)
    if not new:
        print(f"FATAL: no new metrics at {args.new} — training failed?")
        return 1
    old = _load(args.old)

    new_lb = new.get("fisher_z_lower_bound")
    old_lb = (old or {}).get("fisher_z_lower_bound")

    def row(m: dict | None, label: str) -> str:
        m = m or {}
        return (f"| {label} | {m.get('pooled_oos_ic', '—')} | {m.get('fisher_z_lower_bound', '—')} | "
                f"{m.get('noise_floor', '—')} | {m.get('n_oos', '—')} | {m.get('logloss', '—')} |")

    summary = "\n".join([
        "| model | pooled OOS IC | Fisher-z LB | noise floor | n | log-loss |",
        "|---|---|---|---|---|---|",
        row(old, "current"),
        row(new, "candidate"),
    ])

    if old_lb is None or new_lb is None:
        # First run, or the old card predates metrics.json — promote,
        # the human gate is the PR/commit review.
        verdict = "PROMOTE (no comparable current metrics — first retrain)"
        promote = True
    elif new_lb >= old_lb - args.tolerance:
        verdict = f"PROMOTE (candidate LB {new_lb} ≥ current {old_lb} − {args.tolerance})"
        promote = True
    else:
        verdict = f"REJECT (candidate LB {new_lb} < current {old_lb} − {args.tolerance})"
        promote = False

    print(f"{verdict}\n\n{summary}")

    if promote:
        changelog = args.new.parent / "CHANGELOG.md"
        line = (f"- {date.today().isoformat()}: pooled OOS IC {new.get('pooled_oos_ic')}, "
                f"Fisher-z LB {new_lb}, noise floor {new.get('noise_floor')}, "
                f"n={new.get('n_oos')}, log-loss {new.get('logloss')}\n")
        if changelog.exists():
            changelog.write_text(changelog.read_text() + line)
        else:
            changelog.write_text("# ml-challenger model changelog\n\n" + line)
    return 0 if promote else 1


if __name__ == "__main__":
    raise SystemExit(main())
