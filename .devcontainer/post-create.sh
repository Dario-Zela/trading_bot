#!/usr/bin/env bash
# One-time Codespace setup. Runs after the container is built for the
# first time (or after a rebuild). Restarts do NOT re-run this — see
# post-start.sh for per-boot work.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Installing Python package (editable mode)"
pip install --upgrade pip >/dev/null
pip install -e . >/dev/null

echo "==> Fetching Metabase DuckDB driver"
if ! ./metabase/fetch_duckdb_driver.sh; then
  echo ""
  echo "WARN: driver fetch failed — Metabase will start but can't connect to DuckDB"
  echo "      Fix the download URL in metabase/fetch_duckdb_driver.sh and re-run"
fi

echo "==> Building initial dbt analytics store from state/*.jsonl"
if ! python -c "from trading_bot.analytics.dbt_runner import build; build()"; then
  echo ""
  echo "WARN: dbt build failed. Fix the error above and re-run:"
  echo "      python -c \"from trading_bot.analytics.dbt_runner import build; build()\""
fi

echo ""
echo "===================================================================="
echo "Codespace ready. Metabase auto-starts on every container start."
echo "First open: check the PORTS tab for the forwarded 3000 URL."
echo "===================================================================="
