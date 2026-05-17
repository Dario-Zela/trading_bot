from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    override = os.environ.get("TRADING_BOT_STATE_ROOT")
    if override:
        return Path(override)
    # src/trading_bot/state/paths.py → repo root is 4 parents up
    return Path(__file__).resolve().parents[3]


STATE_ROOT = _repo_root() / "state"


def ledger_path() -> Path:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    return STATE_ROOT / "ledger.jsonl"


def predictions_path() -> Path:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    return STATE_ROOT / "predictions.jsonl"
