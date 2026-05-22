"""run_claude retry helper: retries on ClaudeCodeError, default is single-shot."""
from __future__ import annotations

import pytest

from trading_bot.llm import claude_code as cc


def _result():
    return cc.ClaudeCodeResult(text="ok", total_cost_usd=None, duration_ms=None, raw={})


def test_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(cc.time, "sleep", lambda s: None)  # no real backoff
    calls = {"n": 0}

    def fake(prompt, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise cc.ClaudeCodeError("transient")
        return _result()

    monkeypatch.setattr(cc, "_run_claude_once", fake)
    assert cc.run_claude("p", retries=2).text == "ok"
    assert calls["n"] == 3


def test_default_is_single_attempt(monkeypatch):
    monkeypatch.setattr(cc.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fake(prompt, **kw):
        calls["n"] += 1
        raise cc.ClaudeCodeError("boom")

    monkeypatch.setattr(cc, "_run_claude_once", fake)
    with pytest.raises(cc.ClaudeCodeError):
        cc.run_claude("p")  # retries defaults to 0
    assert calls["n"] == 1


def test_retries_exhausted_reraises(monkeypatch):
    monkeypatch.setattr(cc.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fake(prompt, **kw):
        calls["n"] += 1
        raise cc.ClaudeCodeError("always")

    monkeypatch.setattr(cc, "_run_claude_once", fake)
    with pytest.raises(cc.ClaudeCodeError):
        cc.run_claude("p", retries=2)
    assert calls["n"] == 3  # 1 + 2 retries
