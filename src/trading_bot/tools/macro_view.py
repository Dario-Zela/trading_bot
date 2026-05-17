"""get_macro_view tool — reads the most-recent weekly macro view.

Wave 2 implementation: just reads the newest file under state/macro/views/.
Wave 4 introduces a weekly macro agent that writes these files and self-grades
its own predictions. The tool interface here doesn't change between waves —
the file's just authored by Claude instead of by hand once that ships.
"""
from __future__ import annotations

import logging
from pathlib import Path

from trading_bot.state.paths import STATE_ROOT


log = logging.getLogger(__name__)


def get_macro_view() -> str:
    """Return the most-recent weekly macro view as a markdown string.

    Returns an empty string if no view file exists yet. Strategies that need
    a macro view should fall back gracefully — they're describing context,
    not gating on it.
    """
    views_dir: Path = STATE_ROOT / "macro" / "views"
    if not views_dir.exists():
        log.warning("get_macro_view: no macro/views directory at %s", views_dir)
        return ""
    candidates = sorted(views_dir.glob("*.md"))
    if not candidates:
        log.warning("get_macro_view: no .md files in %s", views_dir)
        return ""
    latest = candidates[-1]
    log.info("get_macro_view: returning %s", latest.name)
    return latest.read_text()
