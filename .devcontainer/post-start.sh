#!/usr/bin/env bash
# Runs on every container start — including wake from Codespace
# hibernation. Idempotent: safe to run repeatedly.

set -euo pipefail

cd "$(dirname "$0")/.."

# Refresh the analytics store so Metabase sees any commits pushed
# while the codespace was asleep. Non-fatal — if the build fails you
# still get whatever's already in dbt/analytics.duckdb.
echo "==> Refreshing dbt analytics store"
if ! python -c "from trading_bot.analytics.dbt_runner import build; build()" 2>&1 | tail -8; then
  echo "WARN: dbt build failed; Metabase will start with the previously-built store"
fi

# Ensure Docker is up (it is, once the DinD feature has initialised).
# Bring Metabase up in the background — the compose config restarts on
# failure so a stale container from a previous session is replaced.
echo "==> Starting Metabase"
docker compose up -d metabase

echo ""
echo "===================================================================="
echo "Metabase starting (cold ~45s, warm ~10s)."
echo "  Follow:  docker compose logs -f metabase"
echo "  Ready when the log shows: 'Metabase Initialization COMPLETE'"
echo ""
echo "  URL:    check the PORTS tab in VS Code, port 3000"
echo "          (or run: gh codespace ports -c \$CODESPACE_NAME)"
echo "===================================================================="
