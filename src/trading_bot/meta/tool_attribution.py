"""Tool attribution layer — measure whether the inputs we expose to
each LLM strategy actually shift its predictive accuracy.

Each PredictionRecord carries `tools_used` (the sorted set of tools
the strategy had in its prompt that day). After grading, we can
look at every (strategy, tool) pair and ask:

  - On days when this tool was included in the prompt, what was the
    strategy's IC?
  - On days when this tool was NOT included, what was the IC?

The delta is the empirical contribution of that tool. Over enough
weeks of data, this tells us which tools are pulling weight vs.
which are dead-weight that the evolution agent can deactivate.

For v1 the function works on the existing data — strategies whose
configs are stable, tools always-on, give a single tool-mask per
day and the delta calculation is trivial (no comparison row to
compute against). The signal arrives once configs vary, either
through evolution-driven `tune` actions or through deliberate
ablation schedules. The output stays useful as a "what does this
strategy's IC look like across its current toolset" view in the
meantime.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterator

from trading_bot.state.paths import predictions_path


log = logging.getLogger(__name__)


@dataclass
class ToolMaskIC:
    """IC computed over predictions sharing a single tool-mask."""
    strategy_id: str
    tool_mask: tuple[str, ...]    # sorted tuple — stable identity
    n_predictions: int            # rows graded under this mask
    n_days: int                   # distinct dates under this mask
    ic: float | None              # Pearson between predicted_return_pct and actual_return_pct
    hit_rate: float | None        # share of rows where predicted class matched actual class


@dataclass
class ToolDelta:
    """Per-tool IC delta — IC when present vs IC when absent.

    `n_with` / `n_without` give the row count behind each side. When
    one side has 0 rows (tool was always on or always off in the
    window), `delta_ic` is None — we can't say anything yet.
    """
    strategy_id: str
    tool: str
    ic_with: float | None
    ic_without: float | None
    delta_ic: float | None
    n_with: int
    n_without: int
    verdict: str                  # "positive" | "negative" | "neutral" | "insufficient"


def _iter_graded_predictions(
    strategy_id: str | None = None,
    *,
    window_days: int = 60,
    end_date: date | None = None,
    require_tools_used: bool = False,
) -> Iterator[dict]:
    """Yield prediction rows with actual_return_pct populated, filtered
    by strategy + trailing window. When `require_tools_used=True`, rows
    that pre-date the `tools_used` field (missing or empty list) are
    skipped — necessary for tool-attribution math so legacy rows don't
    pollute the per-tool baselines."""
    p = predictions_path()
    if not p.exists():
        return
    end = end_date or date.today()
    cutoff = (end - timedelta(days=window_days)).isoformat()
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if strategy_id is not None and rec.get("strategy_id") != strategy_id:
                continue
            if rec.get("actual_return_pct") is None:
                continue
            pdate = rec.get("prediction_date") or ""
            if not pdate or pdate < cutoff:
                continue
            if require_tools_used:
                tu = rec.get("tools_used")
                if not isinstance(tu, list) or not tu:
                    # Pre-tools_used row — exclude from attribution
                    # math so legacy rows don't end up in the "without"
                    # baseline for every tool the strategy uses today.
                    continue
            yield rec


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 5:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    dy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _ic_for_rows(rows: list[dict]) -> tuple[float | None, float | None]:
    """Return (pearson_ic, hit_rate) for a list of prediction rows."""
    if not rows:
        return None, None
    preds: list[float] = []
    actuals: list[float] = []
    hits = 0
    for r in rows:
        pr = r.get("predicted_return_pct")
        ac = r.get("actual_return_pct")
        if pr is None or ac is None:
            continue
        try:
            preds.append(float(pr))
            actuals.append(float(ac))
        except (TypeError, ValueError):
            continue
        if r.get("predicted_class") and r.get("predicted_class") == r.get("actual_class"):
            hits += 1
    if not preds:
        return None, None
    ic = _pearson(preds, actuals)
    hit_rate = hits / len(preds)
    return ic, hit_rate


def compute_tool_mask_ic(
    strategy_id: str,
    *,
    window_days: int = 60,
    end_date: date | None = None,
) -> list[ToolMaskIC]:
    """Group this strategy's graded predictions by tool-mask and
    compute IC + hit rate for each. Useful for "this strategy ran
    with N different tool configurations over the last N days; here's
    how each performed"."""
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for rec in _iter_graded_predictions(
        strategy_id, window_days=window_days, end_date=end_date,
        require_tools_used=True,
    ):
        mask = tuple(sorted(rec.get("tools_used") or []))
        grouped[mask].append(rec)

    out: list[ToolMaskIC] = []
    for mask, rows in grouped.items():
        ic, hit = _ic_for_rows(rows)
        dates = {r.get("prediction_date") for r in rows if r.get("prediction_date")}
        out.append(ToolMaskIC(
            strategy_id=strategy_id,
            tool_mask=mask,
            n_predictions=len(rows),
            n_days=len(dates),
            ic=ic,
            hit_rate=hit,
        ))
    out.sort(key=lambda t: t.n_predictions, reverse=True)
    return out


def compute_tool_deltas(
    strategy_id: str,
    *,
    window_days: int = 60,
    end_date: date | None = None,
) -> list[ToolDelta]:
    """For each distinct tool ever used by this strategy in the window,
    compute IC over the rows where it WAS present vs the rows where it
    was ABSENT. Verdict labels the empirical lift.

    Insufficient when one side has fewer than 20 rows — small samples
    swing IC wildly. The threshold is arbitrary; we err on the side
    of "no judgement" until the data is meaningful.
    """
    all_rows = list(_iter_graded_predictions(
        strategy_id, window_days=window_days, end_date=end_date,
        require_tools_used=True,
    ))
    if not all_rows:
        return []

    # Union of all tools observed in this strategy's window
    tools_seen: set[str] = set()
    for r in all_rows:
        for t in r.get("tools_used") or []:
            tools_seen.add(t)

    deltas: list[ToolDelta] = []
    for tool in sorted(tools_seen):
        with_rows = [r for r in all_rows if tool in (r.get("tools_used") or [])]
        without_rows = [r for r in all_rows if tool not in (r.get("tools_used") or [])]
        ic_with, _ = _ic_for_rows(with_rows)
        ic_without, _ = _ic_for_rows(without_rows)
        n_with = len(with_rows)
        n_without = len(without_rows)
        delta = None
        if ic_with is not None and ic_without is not None:
            delta = ic_with - ic_without
        if n_with < 20 or n_without < 20:
            verdict = "insufficient"
        elif delta is None:
            verdict = "insufficient"
        elif delta > 0.05:
            verdict = "positive"
        elif delta < -0.05:
            verdict = "negative"
        else:
            verdict = "neutral"
        deltas.append(ToolDelta(
            strategy_id=strategy_id,
            tool=tool,
            ic_with=ic_with,
            ic_without=ic_without,
            delta_ic=delta,
            n_with=n_with,
            n_without=n_without,
            verdict=verdict,
        ))
    return deltas


def attribution_summary_lines(strategy_id: str, *, window_days: int = 60) -> str:
    """One-block markdown summary the evolution prompt can splice in.
    Returns an empty string if there's no data."""
    deltas = compute_tool_deltas(strategy_id, window_days=window_days)
    if not deltas:
        return ""
    lines: list[str] = []
    for d in deltas:
        if d.verdict == "insufficient":
            lines.append(
                f"- `{d.tool}` — insufficient data "
                f"(with={d.n_with}, without={d.n_without})"
            )
            continue
        ic_with_s = f"{d.ic_with:+.3f}" if d.ic_with is not None else "—"
        ic_without_s = f"{d.ic_without:+.3f}" if d.ic_without is not None else "—"
        delta_s = f"{d.delta_ic:+.3f}" if d.delta_ic is not None else "—"
        lines.append(
            f"- `{d.tool}` — IC {ic_with_s} (n={d.n_with}) with vs "
            f"{ic_without_s} (n={d.n_without}) without → "
            f"Δ {delta_s} ({d.verdict})"
        )
    return "\n".join(lines)


def compute_prefilter_mode_ic(
    strategy_id: str,
    *,
    window_days: int = 60,
    end_date: date | None = None,
) -> dict[str, dict]:
    """Group this strategy's graded predictions by `prefilter_mode` and
    compute IC + hit rate for each. Used by the evolution agent to
    A/B-test the Sonnet pre-filter vs the legacy Python heuristic vs
    no filter at all.

    Returns {mode: {n_predictions, n_days, ic, hit_rate}}. Modes with
    fewer than 5 predictions return ic=None (Pearson is unstable below).
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for rec in _iter_graded_predictions(
        strategy_id, window_days=window_days, end_date=end_date,
        require_tools_used=False,    # don't filter — prefilter_mode is
                                     # independent of the tools_used field
    ):
        mode = (rec.get("prefilter_mode") or "").strip().lower()
        if not mode:
            mode = "(unknown)"        # legacy rows from before the field
        grouped[mode].append(rec)

    out: dict[str, dict] = {}
    for mode, rows in grouped.items():
        ic, hit = _ic_for_rows(rows)
        dates = {r.get("prediction_date") for r in rows if r.get("prediction_date")}
        out[mode] = {
            "n_predictions": len(rows),
            "n_days": len(dates),
            "ic": ic,
            "hit_rate": hit,
        }
    return out


def prefilter_summary_lines(strategy_id: str, *, window_days: int = 60) -> str:
    """One-block markdown summary of prefilter_mode IC for a strategy.
    Returns an empty string if all rows fall in one bucket (no comparison
    possible) or no data exists."""
    buckets = compute_prefilter_mode_ic(strategy_id, window_days=window_days)
    if not buckets:
        return ""
    # Only emit if there's actual comparison to make (≥ 2 modes, each
    # with ≥ 5 predictions — same threshold _pearson uses)
    actionable = {m: b for m, b in buckets.items() if (b["n_predictions"] or 0) >= 5}
    if len(actionable) < 2:
        return ""
    lines: list[str] = []
    for mode, b in sorted(actionable.items(), key=lambda kv: -(kv[1]["n_predictions"] or 0)):
        ic = b["ic"]
        ic_s = f"{ic:+.3f}" if ic is not None else "—"
        hit = b["hit_rate"]
        hit_s = f"{hit * 100:.0f}%" if hit is not None else "—"
        lines.append(
            f"- prefilter_mode=`{mode}` — IC {ic_s} (n={b['n_predictions']}, "
            f"days={b['n_days']}, hit-rate {hit_s})"
        )
    return "\n".join(lines)
