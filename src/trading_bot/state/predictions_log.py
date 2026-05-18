"""Falsifiable prediction log — the persistence layer behind every
"marking the homework" section across News, Macro, and Evolution.

Each prediction is a testable claim with:
  - a target_date by which it should resolve
  - a falsification_criteria sentence stating what would prove it wrong
  - a status — open / proven / partial / falsified / still-open
  - a horizon — tomorrow / this-week / this-month / this-quarter / this-half / this-year
  - a source — news / macro / evolution (which agent made the call)
  - a section + slug — where in the publication it lives

Stored as JSONL in state/predictions/{source}.jsonl. The grader job
walks open predictions whose target_date has passed and asks an LLM to
score them against the falsifier using current market data; it then
mutates the row in place.

Use `append_prediction()` from the agent that made the call. Use
`read_predictions()` / `mark_prediction_graded()` from the grader and
the page renderer.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator

from trading_bot.state.paths import STATE_ROOT


log = logging.getLogger(__name__)


_VALID_SOURCES = {"news", "macro", "evolution"}
_VALID_STATUSES = {"open", "proven", "partial", "falsified", "still-open"}
_VALID_HORIZONS = {
    "tomorrow", "this-week", "this-month",
    "this-quarter", "this-half", "this-year", "multi-year",
}


@dataclass
class Prediction:
    """A single falsifiable claim made by one of the bot's agents.

    Created with status='open'. Mutated to 'proven' / 'partial' / 'falsified'
    by the grader once the target_date passes. 'still-open' means the
    grader checked but the criterion can't be resolved yet (e.g., data
    hasn't been published).
    """
    id: str
    source: str           # 'news' | 'macro' | 'evolution'
    made_at: str          # ISO 8601 timestamp
    claim: str            # human-readable headline of the prediction
    horizon: str          # 'this-month' / etc — see _VALID_HORIZONS
    target_date: str      # ISO date — when this should resolve
    falsification_criteria: str
    conviction: str = "medium"   # 'low' | 'medium' | 'high'
    source_section: str = ""     # where in the publication it appeared
    source_slug: str = ""        # the article slug if linked to a piece

    status: str = "open"
    graded_at: str | None = None
    grading_note: str = ""


def _path_for(source: str) -> Path:
    if source not in _VALID_SOURCES:
        raise ValueError(f"Unknown source: {source!r}; expected one of {_VALID_SOURCES}")
    p = STATE_ROOT / "predictions"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{source}.jsonl"


def append_prediction(p: Prediction) -> None:
    """Append a fresh prediction. Validates source / status / horizon."""
    if p.source not in _VALID_SOURCES:
        raise ValueError(f"Unknown source: {p.source!r}")
    if p.status not in _VALID_STATUSES:
        raise ValueError(f"Unknown status: {p.status!r}")
    if p.horizon not in _VALID_HORIZONS:
        raise ValueError(f"Unknown horizon: {p.horizon!r}")
    path = _path_for(p.source)
    with path.open("a") as f:
        f.write(json.dumps(asdict(p)) + "\n")


def _iter_rows(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_predictions(
    source: str | None = None,
    *,
    status: str | None = None,
    on_date: date | None = None,
    horizon: str | None = None,
) -> list[Prediction]:
    """Read predictions from disk with optional filters.

    - source=None reads every source (news + macro + evolution).
    - status filters by current status.
    - on_date returns only predictions made on the given date.
    - horizon filters by horizon class.
    """
    sources = (source,) if source else tuple(_VALID_SOURCES)
    out: list[Prediction] = []
    for src in sources:
        path = _path_for(src)
        for row in _iter_rows(path):
            if status is not None and row.get("status") != status:
                continue
            if horizon is not None and row.get("horizon") != horizon:
                continue
            if on_date is not None:
                made_iso = (row.get("made_at") or "")[:10]
                if made_iso != on_date.isoformat():
                    continue
            try:
                out.append(Prediction(**row))
            except TypeError:
                continue
    out.sort(key=lambda p: p.made_at, reverse=True)
    return out


def mark_prediction_graded(
    *,
    prediction_id: str,
    source: str,
    new_status: str,
    note: str = "",
) -> bool:
    """Mutate a prediction row's status + grading note. Rewrites the
    whole file (small enough that this is fine). Returns True if the row
    was found and updated."""
    if new_status not in _VALID_STATUSES:
        raise ValueError(f"Unknown status: {new_status!r}")
    path = _path_for(source)
    if not path.exists():
        return False
    rows = list(_iter_rows(path))
    updated = False
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        if row.get("id") == prediction_id:
            row["status"] = new_status
            row["graded_at"] = now
            row["grading_note"] = note
            updated = True
            break
    if not updated:
        log.warning("mark_prediction_graded: id=%s not found in %s", prediction_id, path)
        return False
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    tmp.replace(path)
    return True


def open_predictions_due_by(d: date, source: str | None = None) -> list[Prediction]:
    """Open predictions whose target_date is on-or-before `d`. The grader
    uses this to know what's ready to be checked."""
    all_open = read_predictions(source=source, status="open")
    iso = d.isoformat()
    return [p for p in all_open if p.target_date and p.target_date <= iso]
