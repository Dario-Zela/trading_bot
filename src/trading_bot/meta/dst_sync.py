"""DST sync — keep cron times honest as UK/US clocks shift.

GitHub Actions cron is UTC-only and has no concept of DST. Twice a year the
UK<->UTC offset flips (last Sunday of March / October) and the US<->UTC
offset flips on a slightly different schedule (second Sunday of March /
first Sunday of November), so the "right" UTC time for "8:35 UK" and
"9:35 ET" drifts.

This module re-derives the correct UTC time for each pipeline's
market-local wall-clock targets via zoneinfo and rewrites the cron lines
in the workflow YAML if they differ from the computed values. Idempotent:
nothing happens if the schedule is already correct.

Targets are intentionally hardcoded to the actual market events they
anchor — 5 min after open for entry, 30 min before close for exit — so
even if exchange hours change in future we update one constant.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Sequence

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python < 3.9 — shouldn't happen on our 3.11 baseline
    from backports.zoneinfo import ZoneInfo  # type: ignore


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CronTarget:
    """One scheduled cron in a workflow file, expressed as a local-time anchor."""

    label: str           # human-readable: "entry", "exit"
    tz: str              # IANA timezone name
    local_hour: int      # local hour (24h)
    local_minute: int


@dataclass(frozen=True)
class WorkflowDstSpec:
    """The set of cron targets for one workflow file. Cron order matters —
    the i-th target binds to the i-th `- cron:` line in the file."""

    path: Path
    targets: Sequence[CronTarget]


def _workflows_root() -> Path:
    # src/trading_bot/meta/dst_sync.py → repo root is 4 parents up
    return Path(__file__).resolve().parents[3] / ".github" / "workflows"


def workflow_specs() -> list[WorkflowDstSpec]:
    """The DST-tracked workflows and what their cron lines should anchor to.

    NYSE: 09:30–16:00 America/New_York. Entry 9:35 (5min after open),
          exit 15:30 (30min before close).
    LSE / Euronext / Xetra: 08:00–16:30 Europe/London. Entry 8:35,
          exit 16:00.
    HKEX / TSE: HKEX 09:30–16:00 Asia/Hong_Kong, TSE 09:00–15:00 Asia/Tokyo.
          Anchor entry to HKEX open + 5min (last open of the day across
          our Asian markets) and exit to TSE close - 30min (first close
          of the day). HKEX and TSE don't observe DST so the UTC offset
          is fixed year-round — we re-derive anyway for safety.
    """
    root = _workflows_root()
    return [
        WorkflowDstSpec(
            path=root / "pipeline-us.yml",
            targets=[
                CronTarget("entry", "America/New_York", 9, 35),
                CronTarget("exit",  "America/New_York", 15, 30),
            ],
        ),
        WorkflowDstSpec(
            path=root / "pipeline-uk-eu.yml",
            targets=[
                CronTarget("entry", "Europe/London", 8, 35),
                CronTarget("exit",  "Europe/London", 16, 0),
            ],
        ),
        WorkflowDstSpec(
            path=root / "pipeline-asia.yml",
            targets=[
                CronTarget("entry", "Asia/Hong_Kong", 9, 35),
                CronTarget("exit",  "Asia/Tokyo", 14, 30),
            ],
        ),
    ]


def local_to_utc_today(tz_name: str, hour: int, minute: int, today: date | None = None) -> tuple[int, int]:
    """Convert a local time on a given date to UTC. Uses today's DST status
    so the conversion respects the current offset (not yesterday's)."""
    today = today or date.today()
    local_dt = datetime.combine(today, time(hour, minute), tzinfo=ZoneInfo(tz_name))
    utc_dt = local_dt.astimezone(timezone.utc)
    return utc_dt.hour, utc_dt.minute


_CRON_LINE_RE = re.compile(r"^(\s*-\s*cron:\s*['\"])(\d+\s+\d+\s+[^'\"]+)(['\"])")


def sync_dst(*, today: date | None = None) -> dict:
    """Walk each tracked workflow, rewrite cron lines if the UTC mapping
    has drifted, and report what changed. Idempotent."""
    today = today or date.today()
    results = {"checked": [], "rewrote": [], "unchanged": []}

    for spec in workflow_specs():
        if not spec.path.exists():
            log.warning("Workflow file missing: %s", spec.path)
            continue

        text = spec.path.read_text()
        new_text, changed = _rewrite_workflow(text, spec.targets, today)
        results["checked"].append(spec.path.name)

        if changed:
            spec.path.write_text(new_text)
            results["rewrote"].append({
                "file": spec.path.name,
                "changes": changed,
            })
            log.info("DST sync: rewrote %s — %s", spec.path.name, changed)
        else:
            results["unchanged"].append(spec.path.name)
            log.info("DST sync: %s already correct", spec.path.name)

    return results


def _rewrite_workflow(
    text: str,
    targets: Sequence[CronTarget],
    today: date,
) -> tuple[str, list[dict]]:
    """Find consecutive cron lines (in order) inside this workflow file and
    rewrite each one to match the corresponding target. Returns (new_text,
    list of change descriptions)."""
    lines = text.splitlines(keepends=True)
    changes: list[dict] = []
    target_idx = 0

    for i, line in enumerate(lines):
        if target_idx >= len(targets):
            break
        m = _CRON_LINE_RE.match(line)
        if not m:
            continue

        target = targets[target_idx]
        prefix, current_cron, suffix = m.group(1), m.group(2), m.group(3)
        new_hour, new_minute = local_to_utc_today(target.tz, target.local_hour, target.local_minute, today)

        # Preserve the day-of-month / month / day-of-week part — only mm/hh change
        parts = current_cron.split()
        if len(parts) < 5:
            continue  # malformed; leave it
        new_cron = f"{new_minute} {new_hour} {' '.join(parts[2:])}"

        if new_cron != current_cron:
            lines[i] = line[: m.start(2)] + new_cron + line[m.end(2):]
            changes.append({
                "label": target.label,
                "tz": target.tz,
                "local_target": f"{target.local_hour:02d}:{target.local_minute:02d}",
                "from": current_cron,
                "to": new_cron,
            })
        target_idx += 1

    return "".join(lines), changes
