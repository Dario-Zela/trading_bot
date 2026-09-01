"""Cold-storage sharding for state/predictions.jsonl.

The ledger crossed GitHub's 100 MB per-file limit in Aug 2026 and every
pipeline push was rejected for 11 days. These tests pin the two things
that make the fix safe: the trim is lossless, and archived rows can't be
resurrected by the smart-merge fallback.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import smart_merge_state as sms
import trim_predictions_ledger as trim
from trading_bot.state import predictions_archive as pa


def _row(sid="s1", region="us", ticker="AAA", pdate="2026-01-15", **extra):
    r = {
        "strategy_id": sid, "region": region, "ticker": ticker,
        "prediction_date": pdate, "predicted_return_pct": 1.0,
        "actual_return_pct": 0.5,
    }
    r.update(extra)
    return r


def _write_hot(state_root, rows):
    (state_root / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )


def _key(r):
    return (r["strategy_id"], r["region"], r["ticker"], r["prediction_date"])


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def test_iter_predictions_spans_archive_then_hot(state_root):
    _write_hot(state_root, [_row(ticker="HOT", pdate="2026-03-02")])
    trim._merge_into_shard("2026-01", [_row(ticker="OLD", pdate="2026-01-15")])

    rows = list(pa.iter_predictions())
    assert [r["ticker"] for r in rows] == ["OLD", "HOT"], "archive must come first"


def test_since_skips_whole_shards(state_root):
    trim._merge_into_shard("2026-01", [_row(ticker="JAN", pdate="2026-01-15")])
    trim._merge_into_shard("2026-03", [_row(ticker="MAR", pdate="2026-03-15")])
    _write_hot(state_root, [_row(ticker="HOT", pdate="2026-04-01")])

    assert {r["ticker"] for r in pa.iter_predictions(since="2026-03-01")} == {"MAR", "HOT"}
    # A row inside a kept shard but older than the cutoff is still filtered.
    assert {r["ticker"] for r in pa.iter_predictions(since="2026-03-20")} == {"HOT"}


def test_readers_survive_missing_files(state_root):
    assert list(pa.iter_predictions()) == []
    assert pa.read_watermark() is None


# ---------------------------------------------------------------------------
# Trimming
# ---------------------------------------------------------------------------

def test_trim_is_lossless_and_idempotent(state_root):
    today = date(2026, 3, 1)
    rows = [
        _row(ticker=f"T{i}", pdate=(today - timedelta(days=i)).isoformat())
        for i in range(90)
    ]
    _write_hot(state_root, rows)

    trim.main(["--keep-days", "30", "--max-bytes", "0"])

    got = list(pa.iter_predictions())
    assert len(got) == len(rows)
    assert {_key(r) for r in got} == {_key(r) for r in rows}
    # Rows survive byte-identically, not just by key.
    assert sorted(json.dumps(r, sort_keys=True) for r in got) == \
           sorted(json.dumps(r, sort_keys=True) for r in rows)

    hot_after = (state_root / "predictions.jsonl").read_text()
    trim.main(["--keep-days", "30", "--max-bytes", "0"])
    assert (state_root / "predictions.jsonl").read_text() == hot_after
    assert len(list(pa.iter_predictions())) == len(rows)


def test_max_bytes_ceiling_forces_a_trim_even_inside_the_window(state_root):
    """The size ceiling is the real protection: keep_days alone can't hold the
    line if daily prediction volume keeps climbing."""
    today = date.today()
    rows = [
        _row(ticker=f"T{i}{j}", pdate=(today - timedelta(days=i)).isoformat())
        for i in range(5) for j in range(200)
    ]
    _write_hot(state_root, rows)

    before = (state_root / "predictions.jsonl").stat().st_size
    # Everything is inside a 30-day window, so only the byte ceiling can act.
    trim.main(["--keep-days", "30", "--max-bytes", "20000"])

    hot_dates = {r["prediction_date"] for r in pa.iter_hot()}
    assert (state_root / "predictions.jsonl").stat().st_size < before
    assert len(list(pa.iter_predictions())) == len(rows), "no rows lost"
    # The floor is one day: the ceiling moves whole days off the back, but the
    # most recent day always stays hot so same-day reads and the reflection
    # write-back keep working.
    assert hot_dates == {today.isoformat()}


def test_trim_splits_on_whole_days(state_root):
    """A date is never half hot and half archived — otherwise the watermark
    would drop live rows on the next smart merge."""
    today = date.today()
    rows = [
        _row(ticker=f"T{i}{j}", pdate=(today - timedelta(days=i)).isoformat())
        for i in range(6) for j in range(150)
    ]
    _write_hot(state_root, rows)
    trim.main(["--keep-days", "30", "--max-bytes", "20000"])

    hot_dates = {r["prediction_date"] for r in pa.iter_hot()}
    archived_dates = {r["prediction_date"] for r in pa.iter_archived()}
    assert not (hot_dates & archived_dates)


def test_watermark_never_moves_backwards(state_root):
    _write_hot(state_root, [_row(pdate="2026-01-01"), _row(ticker="B", pdate="2026-06-01")])
    trim.main(["--keep-days", "0", "--max-bytes", "0"])
    pa.write_watermark("2026-09-09", rows_archived=0)

    _write_hot(state_root, [_row(ticker="C", pdate="2026-07-01")])
    trim.main(["--keep-days", "0", "--max-bytes", "0"])
    assert pa.read_watermark() == "2026-09-09"


def test_shard_merge_dedups_on_primary_key(state_root):
    r = _row(pdate="2026-02-02")
    added, total = trim._merge_into_shard("2026-02", [r])
    assert (added, total) == (1, 1)
    added, total = trim._merge_into_shard("2026-02", [r])
    assert (added, total) == (0, 1), "re-archiving the same row must be a no-op"


def test_shards_are_byte_stable(state_root):
    """gzip mtime is pinned to 0 so an unchanged shard doesn't churn a fresh
    ~3 MB blob into git on every run."""
    rows = [_row(ticker=f"T{i}", pdate="2026-02-02") for i in range(20)]
    trim._merge_into_shard("2026-02", rows)
    first = pa.shard_path("2026-02").read_bytes()
    pa.shard_path("2026-02").unlink()
    trim._merge_into_shard("2026-02", rows)
    assert pa.shard_path("2026-02").read_bytes() == first


# ---------------------------------------------------------------------------
# Smart-merge interaction — the resurrection guard
# ---------------------------------------------------------------------------

def test_smart_merge_drops_locally_held_archived_rows(tmp_path):
    """A runner that started before the trim still holds the pre-trim ledger.
    Merging it naively would re-add every archived row and blow past 100 MB
    again on the very next push."""
    repo, save = tmp_path / "repo", tmp_path / "save"
    rel = "state/predictions.jsonl"

    (repo / "state/predictions_archive").mkdir(parents=True)
    (repo / "state/predictions_archive/watermark.json").write_text(
        json.dumps({"archived_before": "2026-03-01"})
    )
    # Remote is post-trim; local still has the old row plus a genuinely new one.
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_text(json.dumps(_row(ticker="NEW", pdate="2026-03-05")) + "\n")
    (save / rel).parent.mkdir(parents=True, exist_ok=True)
    (save / rel).write_text(
        json.dumps(_row(ticker="OLD", pdate="2026-01-01")) + "\n"
        + json.dumps(_row(ticker="NEW", pdate="2026-03-05")) + "\n"
        + json.dumps(_row(ticker="FRESH", pdate="2026-03-06")) + "\n"
    )

    sms.merge_file(
        repo, save, rel,
        ("strategy_id", "region", "ticker", "prediction_date"),
        drop_before=sms.read_archive_watermark(repo),
    )
    tickers = {json.loads(l)["ticker"] for l in (repo / rel).read_text().splitlines() if l.strip()}
    assert tickers == {"NEW", "FRESH"}, "archived row must not be resurrected"


def test_smart_merge_unaffected_when_nothing_archived(tmp_path):
    repo, save = tmp_path / "repo", tmp_path / "save"
    rel = "state/predictions.jsonl"
    (repo / rel).parent.mkdir(parents=True)
    (repo / rel).write_text("")
    (save / rel).parent.mkdir(parents=True)
    (save / rel).write_text(json.dumps(_row(ticker="OLD", pdate="2020-01-01")) + "\n")

    assert sms.read_archive_watermark(repo) is None
    sms.merge_file(repo, save, rel, ("strategy_id", "region", "ticker", "prediction_date"),
                   drop_before=sms.read_archive_watermark(repo))
    assert "OLD" in (repo / rel).read_text()


def test_hot_file_preserves_append_order(state_root):
    """The trim must not re-sort the hot file. Rewriting all 50 MB in a new
    order on every daily run would defeat git's delta compression and bloat
    history by a fresh full-size blob each day."""
    today = date.today()
    rows = [
        _row(ticker=f"T{i}", pdate=(today - timedelta(days=i)).isoformat())
        for i in range(60)
    ]  # deliberately newest-first, i.e. NOT sorted by date
    _write_hot(state_root, rows)

    trim.main(["--keep-days", "30", "--max-bytes", "0"])

    kept = [json.loads(l) for l in (state_root / "predictions.jsonl").read_text().splitlines() if l.strip()]
    expected = [r for r in rows if r["prediction_date"] >= (today - timedelta(days=30)).isoformat()]
    assert [r["ticker"] for r in kept] == [r["ticker"] for r in expected]
