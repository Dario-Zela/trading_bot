"""smart_merge_state.merge_file: line-union dedup that preserves local edits."""
from __future__ import annotations

import json

import smart_merge_state as sms


def _write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_union_dedups_on_key_local_wins(tmp_path):
    repo, save = tmp_path / "repo", tmp_path / "save"
    # remote has trade a (stale, still open); local has a (closed) + new b.
    _write(repo / "state/ledger.jsonl", [{"trade_id": "a", "exit_date": None}])
    _write(save / "state/ledger.jsonl",
           [{"trade_id": "a", "exit_date": "2026-05-22"}, {"trade_id": "b"}])

    added = sms.merge_file(repo, save, "state/ledger.jsonl", "trade_id")
    rows = [json.loads(l) for l in (repo / "state/ledger.jsonl").read_text().splitlines()]
    by_id = {r["trade_id"]: r for r in rows}

    assert added == 1                              # b is new
    assert set(by_id) == {"a", "b"}                # no duplicate a
    assert by_id["a"]["exit_date"] == "2026-05-22" # local (closed) wins over stale remote


def test_composite_key_dedup(tmp_path):
    repo, save = tmp_path / "repo", tmp_path / "save"
    key = ("strategy_id", "region", "ticker", "prediction_date")
    row = {"strategy_id": "s", "region": "us", "ticker": "AAPL", "prediction_date": "2026-05-22"}
    _write(repo / "state/predictions.jsonl", [row])
    _write(save / "state/predictions.jsonl", [row])  # same logical row both sides
    sms.merge_file(repo, save, "state/predictions.jsonl", key)
    lines = (repo / "state/predictions.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1  # deduped, not concatenated
