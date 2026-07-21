"""Invoke dbt from Python.

The pipeline workflows call `build()` after every stage that mutates
state files (entry-run, exit-run, grade-predictions, weekly-evolution)
so that any downstream reader sees fresh marts. Runs are idempotent —
dbt materialises tables in-place.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from trading_bot.state.paths import STATE_ROOT

log = logging.getLogger(__name__)


def _repo_root() -> Path:
    return STATE_ROOT.parent


def _dbt_dir() -> Path:
    return _repo_root() / "dbt"


def _duckdb_path() -> Path:
    return _dbt_dir() / "analytics.duckdb"


def _env() -> dict[str, str]:
    """Environment for the dbt subprocess. Anchors state + duckdb paths
    to absolute locations so dbt works regardless of the caller's cwd."""
    env = os.environ.copy()
    env["TRADING_BOT_STATE_ROOT"] = str(STATE_ROOT.resolve())
    env["TRADING_BOT_DUCKDB_PATH"] = str(_duckdb_path().resolve())
    env["TRADING_BOT_OHLCV_PATH"] = str((STATE_ROOT / "ohlcv.db").resolve())
    env["DBT_PROFILES_DIR"] = str(_dbt_dir().resolve())
    return env


def build(*, select: str | None = None, run_tests: bool = False) -> None:
    """Run `dbt run` (and optionally `dbt test`).

    Args:
        select: dbt selector, e.g. 'marts.mart_fee_drag' to rebuild
                just one model. Default rebuilds everything.
        run_tests: also run `dbt test` after the build.
    """
    dbt_dir = _dbt_dir()
    if not dbt_dir.exists():
        raise FileNotFoundError(f"dbt project directory not found: {dbt_dir}")

    cmd = ["dbt", "run", "--project-dir", str(dbt_dir)]
    if select:
        cmd += ["--select", select]

    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, env=_env(), capture_output=True, text=True)
    if result.returncode != 0:
        log.error("dbt run failed:\nSTDOUT:\n%s\nSTDERR:\n%s", result.stdout, result.stderr)
        raise RuntimeError(f"dbt run failed with exit code {result.returncode}")
    log.info("dbt run complete")

    if run_tests:
        test_cmd = ["dbt", "test", "--project-dir", str(dbt_dir)]
        if select:
            test_cmd += ["--select", select]
        log.info("Running: %s", " ".join(test_cmd))
        result = subprocess.run(test_cmd, env=_env(), capture_output=True, text=True)
        if result.returncode != 0:
            # Test failures are non-fatal — surface but continue.
            log.warning("dbt test reported failures:\n%s", result.stdout)
