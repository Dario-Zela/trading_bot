"""_extract_json / _loads_lenient: tolerate trailing commas (small-model defect)
while leaving well-formed JSON untouched."""
from __future__ import annotations

import json

import pytest

from trading_bot.llm.claude_code import ClaudeCodeError, _extract_json, _loads_lenient


def test_lenient_parses_wellformed_unchanged():
    assert _loads_lenient('{"a": 1, "b": [2, 3]}') == {"a": 1, "b": [2, 3]}


def test_lenient_strips_trailing_commas():
    assert _loads_lenient('[1, 2, 3,]') == [1, 2, 3]
    assert _loads_lenient('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}
    assert _loads_lenient('{"x": [1, 2,], "y": 3,}') == {"x": [1, 2], "y": 3}


def test_lenient_still_raises_on_garbage():
    with pytest.raises(json.JSONDecodeError):
        _loads_lenient("not json at all")


def test_extract_fenced_with_trailing_comma():
    text = 'Here are my picks:\n```json\n[{"ticker": "AAPL", "alloc": 50,},]\n```\n'
    assert _extract_json(text) == [{"ticker": "AAPL", "alloc": 50}]


def test_extract_prose_then_array_with_trailing_comma():
    # No fence, model wrapped the array in prose, trailing comma — the Haiku case.
    text = 'Based on the data, my selections are [{"ticker": "MSFT"},] and that is all.'
    assert _extract_json(text) == [{"ticker": "MSFT"}]


def test_extract_wellformed_object_unaffected():
    assert _extract_json('```json\n{"actions": []}\n```') == {"actions": []}


def test_extract_raises_when_no_json():
    with pytest.raises(ClaudeCodeError):
        _extract_json("I could not find any suitable candidates today.")
