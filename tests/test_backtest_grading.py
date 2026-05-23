"""grade_from_live_predictions: aggregate emitted predictions over a
window with no Claude calls. Drop-in replacement for the LLM replay
path in the weekly backtest pass."""
from __future__ import annotations

import json
from datetime import date

from trading_bot.meta.backtest import grade_from_live_predictions


def _write_preds(state_root, rows: list[dict]) -> None:
    path = state_root / "predictions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_grade_handles_missing_file(state_root):
    rep = grade_from_live_predictions(
        "macro-aligned", "us", date(2026, 5, 1), date(2026, 5, 20),
    )
    assert rep.n_trades == 0
    assert rep.total_pnl_pct == 0.0
    assert rep.trades == []


def test_grade_aggregates_in_window_only(state_root):
    rows = [
        # in-window, graded, matching id+region
        {"strategy_id": "macro-aligned", "region": "us",
         "prediction_date": "2026-05-10", "ticker": "AAPL",
         "actual_return_pct": 1.0},
        {"strategy_id": "macro-aligned", "region": "us",
         "prediction_date": "2026-05-12", "ticker": "MSFT",
         "actual_return_pct": -0.5},
        {"strategy_id": "macro-aligned", "region": "us",
         "prediction_date": "2026-05-14", "ticker": "NVDA",
         "actual_return_pct": 2.0},
        # ungraded (last day) — should be skipped
        {"strategy_id": "macro-aligned", "region": "us",
         "prediction_date": "2026-05-20", "ticker": "GOOG",
         "actual_return_pct": None},
        # out of window
        {"strategy_id": "macro-aligned", "region": "us",
         "prediction_date": "2026-04-30", "ticker": "AMZN",
         "actual_return_pct": 5.0},
        # wrong strategy
        {"strategy_id": "mean-reverter", "region": "us",
         "prediction_date": "2026-05-12", "ticker": "TSLA",
         "actual_return_pct": -10.0},
        # wrong region
        {"strategy_id": "macro-aligned", "region": "uk-eu",
         "prediction_date": "2026-05-12", "ticker": "BARC.L",
         "actual_return_pct": 99.0},
    ]
    _write_preds(state_root, rows)
    rep = grade_from_live_predictions(
        "macro-aligned", "us", date(2026, 5, 1), date(2026, 5, 19),
    )
    assert rep.n_trades == 3
    assert rep.total_pnl_pct == 2.5    # 1.0 - 0.5 + 2.0
    assert rep.avg_pnl_pct == round(2.5 / 3, 3)
    assert rep.hit_rate == round(2 / 3, 3)
    # avg_win = (1+2)/2 = 1.5; avg_loss = -0.5; |1.5/-0.5| = 3.0
    assert rep.win_loss_ratio == 3.0
    tickers = sorted(t.ticker for t in rep.trades)
    assert tickers == ["AAPL", "MSFT", "NVDA"]


def test_grade_skips_corrupt_lines(state_root):
    path = state_root / "predictions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"strategy_id":"x","region":"us","prediction_date":"2026-05-10","ticker":"AAPL","actual_return_pct":1.0}\n'
        'not-json garbage line\n'
        '{"strategy_id":"x","region":"us","prediction_date":"2026-05-11","ticker":"MSFT","actual_return_pct":-2.0}\n'
    )
    rep = grade_from_live_predictions(
        "x", "us", date(2026, 5, 1), date(2026, 5, 20),
    )
    assert rep.n_trades == 2
