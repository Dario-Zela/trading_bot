"""Cold storage for the prediction ledger.

`state/predictions.jsonl` is append-only and was growing ~120k rows
(~60 MB) a month and accelerating. It crossed GitHub's hard 100 MB
per-file limit in August 2026, after which *every* pipeline push was
rejected by the pre-receive hook — the entry/exit runs still did their
work, then threw it away at the push step. That's the 08-21+ "pipeline
blackout" the weekly evolution issues kept reporting.

The fix is to keep only a hot window in `predictions.jsonl` and shard
everything older into gzipped monthly files under
`state/predictions_archive/YYYY-MM.jsonl.gz`. JSONL of this shape
compresses ~8x, so a month of predictions lands well under 10 MB.

Nothing loses history: `iter_predictions()` walks the archive shards
(oldest first) and then the hot file, so a caller that used to read the
whole ledger still sees every row in chronological order. Callers that
only need a trailing window pass `since=` and the shard scan is skipped
entirely for months that end before it.

`watermark.json` records the cutoff — every row with
`prediction_date < archived_before` lives in the archive and must NOT
be re-added to the hot file. `smart_merge_state.py` reads it so a
concurrent runner holding a pre-trim copy can't resurrect trimmed rows.
"""
from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Iterable, Iterator

from trading_bot.state import paths

log = logging.getLogger(__name__)

ARCHIVE_DIRNAME = "predictions_archive"
WATERMARK_FILENAME = "watermark.json"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def archive_dir() -> Path:
    """Read paths.STATE_ROOT at call time so the `state_root` test fixture
    (which monkeypatches that attribute) redirects the archive too."""
    return paths.STATE_ROOT / ARCHIVE_DIRNAME


def watermark_path() -> Path:
    return archive_dir() / WATERMARK_FILENAME


def shard_path(month: str) -> Path:
    """`month` is 'YYYY-MM'."""
    return archive_dir() / f"{month}.jsonl.gz"


def month_of(record: dict) -> str | None:
    """Shard key for a row: the 'YYYY-MM' of its prediction_date."""
    d = record.get("prediction_date")
    if not isinstance(d, str) or len(d) < 7:
        return None
    return d[:7]


def shard_months() -> list[str]:
    """Every archived month, oldest first."""
    d = archive_dir()
    if not d.is_dir():
        return []
    return sorted(p.name[: -len(".jsonl.gz")] for p in d.glob("*.jsonl.gz"))


# ---------------------------------------------------------------------------
# Watermark
# ---------------------------------------------------------------------------

def read_watermark() -> str | None:
    """ISO date cutoff: rows strictly older than this are in the archive.
    None when nothing has been trimmed yet."""
    p = watermark_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get("archived_before") or None
    except (json.JSONDecodeError, OSError):
        # A corrupt watermark must not strand the ledger — treat it as
        # "nothing archived" so the trim recomputes it from scratch.
        log.warning("Unreadable watermark at %s; treating as un-trimmed", p)
        return None


def write_watermark(archived_before: str, *, rows_archived: int) -> None:
    d = archive_dir()
    d.mkdir(parents=True, exist_ok=True)
    watermark_path().write_text(json.dumps({
        "archived_before": archived_before,
        "rows_archived": rows_archived,
        "shards": shard_months(),
    }, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _iter_jsonl_lines(lines: Iterable[str]) -> Iterator[dict]:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def iter_shard(month: str) -> Iterator[dict]:
    p = shard_path(month)
    if not p.exists():
        return
    with gzip.open(p, "rt") as f:
        yield from _iter_jsonl_lines(f)


def iter_archived(since: str | None = None) -> Iterator[dict]:
    """Archived rows, oldest month first. `since` is an ISO date; shards
    for months that end before it are skipped without being decompressed."""
    since_month = since[:7] if since else None
    for month in shard_months():
        if since_month and month < since_month:
            continue
        for rec in iter_shard(month):
            if since and (rec.get("prediction_date") or "") < since:
                continue
            yield rec


def iter_hot(since: str | None = None) -> Iterator[dict]:
    """Rows still in state/predictions.jsonl."""
    p = paths.predictions_path()
    if not p.exists():
        return
    with p.open() as f:
        for rec in _iter_jsonl_lines(f):
            if since and (rec.get("prediction_date") or "") < since:
                continue
            yield rec


def iter_predictions(since: str | None = None) -> Iterator[dict]:
    """Every prediction row — archive shards first, then the hot file.

    This is the drop-in replacement for reading `state/predictions.jsonl`
    directly. Pass `since` (ISO date) when you only need a trailing
    window; whole shards are then skipped without decompression.
    """
    yield from iter_archived(since)
    yield from iter_hot(since)
