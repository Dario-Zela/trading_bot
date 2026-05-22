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


def test_trail_exits_composite_key_via_main(tmp_path):
    repo, save = tmp_path / "repo", tmp_path / "save"
    row = {"ticker": "AAA", "region": "us", "strategy_id": "s",
           "exit_date": "2026-05-22", "pnl_pct": 1}
    _write(repo / "state/trail_exits.jsonl", [row])
    _write(save / "state/trail_exits.jsonl", [row, {**row, "ticker": "BBB", "pnl_pct": 2}])
    sms.main(str(repo), str(save))
    rows = [json.loads(l) for l in (repo / "state/trail_exits.jsonl").read_text().splitlines()]
    assert sorted(r["ticker"] for r in rows) == ["AAA", "BBB"]  # union, no dup of AAA


def test_glob_target_merges_dynamic_pick_adjustment_files(tmp_path):
    repo, save = tmp_path / "repo", tmp_path / "save"
    # Only the local run created this per-date/strategy file.
    _write(save / "state/pick_adjustments/2026-05-22.mom.jsonl", [{"ticker": "USO", "x": 1}])
    sms.main(str(repo), str(save))
    out = repo / "state/pick_adjustments/2026-05-22.mom.jsonl"
    assert out.exists()  # glob target created the file in the reset repo
    assert json.loads(out.read_text().splitlines()[0])["ticker"] == "USO"


def test_composite_key_dedup(tmp_path):
    repo, save = tmp_path / "repo", tmp_path / "save"
    key = ("strategy_id", "region", "ticker", "prediction_date")
    row = {"strategy_id": "s", "region": "us", "ticker": "AAPL", "prediction_date": "2026-05-22"}
    _write(repo / "state/predictions.jsonl", [row])
    _write(save / "state/predictions.jsonl", [row])  # same logical row both sides
    sms.merge_file(repo, save, "state/predictions.jsonl", key)
    lines = (repo / "state/predictions.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1  # deduped, not concatenated
