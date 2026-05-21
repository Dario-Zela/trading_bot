"""Subprocess wrapper around the Claude Code CLI.

Authenticates via the CLAUDE_CODE_OAUTH_TOKEN env var (generated locally with
`claude setup-token`, stored as a repo secret in CI). For our use case — one-shot
analysis calls from inside a pipeline — we always run with `-p` (print mode)
and `--output-format json`.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass


log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 1200  # 20 min — tool-heavy strategies (macro-aligned, news-reactive) blew through 600s on the UK-EU run


class ClaudeCodeError(RuntimeError):
    """Wrapped error from a Claude Code CLI invocation."""


@dataclass(frozen=True)
class ClaudeCodeResult:
    """What a `claude -p --output-format json` invocation returns to us."""

    text: str           # the final assistant message
    total_cost_usd: float | None
    duration_ms: int | None
    raw: dict           # the full parsed JSON for diagnostics / logging


def run_claude(
    prompt: str,
    *,
    model: str = "sonnet",
    timeout_seconds: int = _DEFAULT_TIMEOUT,
    extra_args: list[str] | None = None,
) -> ClaudeCodeResult:
    """Invoke `claude -p <prompt>` and return the parsed result.

    Raises ClaudeCodeError on non-zero exit code or unparseable output.
    """
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if not token:
        raise ClaudeCodeError(
            "CLAUDE_CODE_OAUTH_TOKEN not set — run `claude setup-token` locally "
            "or add the secret to GitHub Actions."
        )

    cmd = [
        "claude",
        "-p", prompt,
        "--model", model,
        "--output-format", "json",
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = {**os.environ, "CLAUDE_CODE_OAUTH_TOKEN": token}

    log.info("Invoking claude (%d-char prompt, model=%s)", len(prompt), model)
    try:
        completed = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        raise ClaudeCodeError(f"Claude Code timed out after {timeout_seconds}s") from e
    except FileNotFoundError as e:
        raise ClaudeCodeError(
            "claude executable not found on PATH. Install with "
            "`curl -fsSL https://claude.ai/install.sh | bash` "
            "or `npm install -g @anthropic-ai/claude-code`."
        ) from e

    if completed.returncode != 0:
        raise ClaudeCodeError(
            f"Claude Code exited {completed.returncode}: {completed.stderr[:400]}"
        )

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as e:
        raise ClaudeCodeError(
            f"Could not parse Claude Code JSON output: {e}\n"
            f"stdout (first 300 chars): {completed.stdout[:300]}"
        ) from e

    # The CLI's JSON shape is { "result": "<assistant text>", "session_id": ...,
    # "total_cost_usd": float, "duration_ms": int, "num_turns": int, ... }
    # We accept either "result" or "content" as the text field for robustness
    # against minor CLI version differences.
    text = payload.get("result") or payload.get("content") or ""
    if not text:
        raise ClaudeCodeError(
            f"Claude Code returned no result text. Payload keys: {list(payload.keys())}"
        )

    return ClaudeCodeResult(
        text=text,
        total_cost_usd=payload.get("total_cost_usd"),
        duration_ms=payload.get("duration_ms"),
        raw=payload,
    )


def run_claude_for_json(
    prompt: str,
    *,
    model: str = "sonnet",
    timeout_seconds: int = _DEFAULT_TIMEOUT,
    extra_args: list[str] | None = None,
) -> dict | list:
    """Convenience: invoke Claude and extract a JSON block from the response.

    Looks for a JSON code fence first (```json ... ```), falls back to the
    first {...} or [...] in the text. Raises ClaudeCodeError if none is found.
    """
    result = run_claude(prompt, model=model, timeout_seconds=timeout_seconds, extra_args=extra_args)
    return _extract_json(result.text)


def _extract_json(text: str):
    # 1. Look for closed ```json fenced block.
    fenced = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass  # fall through

    # 2. Try the entire stripped text as JSON.
    stripped = text.strip()
    if stripped and stripped[0] in "[{":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # 3. Find first balanced {...} or [...] in the text.
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = text.find(open_ch)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    # 4. Truncation recovery — Sonnet sometimes hits its output token
    #    limit mid-JSON, so the closing fence + trailing braces never
    #    arrive. Salvage what we have:
    #      a. Find the start of an unfenced ```json block (no closing
    #         fence to match in step 1).
    #      b. Within that body, attempt to close any unterminated string
    #         and balance braces/brackets, then re-parse.
    open_fence = re.search(r"```(?:json)?\s*\n", text)
    body_start = open_fence.end() if open_fence else None
    if body_start is None:
        first_brace = text.find("{")
        if first_brace >= 0:
            body_start = first_brace
    if body_start is not None:
        body = text[body_start:]
        # Drop any trailing ``` if it was a sole opening line miscount.
        body = re.sub(r"\s*```\s*$", "", body)
        repaired = _repair_truncated_json(body)
        if repaired is not None:
            try:
                result = json.loads(repaired)
                log.warning(
                    "Claude Code response was truncated; recovered partial JSON "
                    "(%d → %d chars). Output token limit may need attention.",
                    len(text), len(repaired),
                )
                return result
            except json.JSONDecodeError:
                pass

    raise ClaudeCodeError(
        f"No parseable JSON found in Claude Code response.\n"
        f"Response (first 400 chars): {text[:400]}"
    )


def _repair_truncated_json(body: str) -> str | None:
    """Best-effort repair for JSON output that got cut mid-stream.

    Walks the body character-by-character tracking string state and
    bracket depth, then appends whatever closing characters are needed
    to make the result syntactically complete. Inside an unterminated
    string the trailing fragment is discarded (better to lose the last
    line of body_md than to confuse the JSON parser with a stray quote).
    Returns the repaired string, or None if the body has nothing
    JSON-like in it.
    """
    if "{" not in body and "[" not in body:
        return None

    stack: list[str] = []
    in_string = False
    escape = False
    last_safe_end = -1   # furthest index whose prefix is parseable so far

    for i, ch in enumerate(body):
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                last_safe_end = i
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                opener = stack.pop()
                if (opener, ch) not in (("{", "}"), ("[", "]")):
                    return None
                if not stack:
                    last_safe_end = i

    if not stack and not in_string:
        return body          # already well-formed, just no closing fence

    truncated_at = last_safe_end + 1 if last_safe_end >= 0 else len(body)
    prefix = body[:truncated_at]

    # If we stopped inside an unterminated string, walk back to the
    # last structural boundary (comma, opening brace, opening bracket)
    # and drop the incomplete key/value pair. Keep the opening brace
    # itself when that's the boundary, so we end up with an empty
    # object/array rather than nothing at all.
    if in_string:
        last_comma = prefix.rfind(",")
        last_open_obj = prefix.rfind("{")
        last_open_arr = prefix.rfind("[")
        cut = max(last_comma, last_open_obj, last_open_arr)
        if cut < 0:
            return None
        if cut == last_open_obj or cut == last_open_arr:
            prefix = prefix[: cut + 1]   # keep the opening brace
        else:
            prefix = prefix[:cut]        # drop the trailing comma

    # Replay just the prefix to know what brackets remain open.
    pending: list[str] = []
    in_s = False
    esc = False
    for ch in prefix:
        if esc:
            esc = False
            continue
        if in_s:
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_s = False
            continue
        if ch == '"':
            in_s = True
            continue
        if ch in "{[":
            pending.append(ch)
        elif ch in "}]" and pending:
            pending.pop()

    closers = []
    for opener in reversed(pending):
        closers.append("}" if opener == "{" else "]")
    repaired = prefix.rstrip().rstrip(",") + "".join(closers)
    return repaired
