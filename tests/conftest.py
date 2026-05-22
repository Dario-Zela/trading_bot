"""Shared test fixtures. Tests are hermetic — no network, no broker creds."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the workflow helper at .github/scripts importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github" / "scripts"))


@pytest.fixture
def state_root(tmp_path, monkeypatch):
    """Redirect the state dir to a temp path. ledger_path()/predictions_path()
    read trading_bot.state.paths.STATE_ROOT at call time, so patching it here
    isolates every state-file test."""
    root = tmp_path / "state"
    root.mkdir()
    monkeypatch.setattr("trading_bot.state.paths.STATE_ROOT", root)
    return root
