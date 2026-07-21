#!/usr/bin/env bash
# Fetch the community DuckDB driver JAR into metabase/plugins/.
# Metabase must be stopped (or restarted after) for it to pick up the
# new driver — the plugin dir is scanned at startup only.
#
# Verify the release URL before running: the community driver isn't
# maintained by Metabase Inc., so the exact repo has changed hands
# (AlexR2D2/motherduckdb/… fork trail). If the download 404s, check
# the latest release at:
#   https://github.com/motherduckdb/metabase_duckdb_driver/releases
#   https://github.com/AlexR2D2/metabase_duckdb_driver/releases

set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "$0")" && pwd)/plugins"
DRIVER_JAR="$PLUGIN_DIR/duckdb.metabase-driver.jar"

# Default to the MotherDuck fork — the actively maintained community
# driver. Paired with Metabase v0.55.12 (pinned in metabase/Dockerfile)
# because the driver's 1.5.x releases target Metabase v0.55's plugin
# API. Newer Metabase versions send connection options this driver
# doesn't know (`connection-pool-type` etc.); older Metabase (< 0.55)
# doesn't have the plugin API this driver expects. Keep the pair in
# sync when bumping either side.
#
# The AlexR2D2 fork was an earlier community driver but its releases
# are ~2 years old and no longer register the JDBC driver with
# modern Metabase's classloader.
DRIVER_REPO="${DUCKDB_DRIVER_REPO:-motherduckdb/metabase_duckdb_driver}"
DRIVER_VERSION="${DUCKDB_DRIVER_VERSION:-latest}"

if [[ "$DRIVER_VERSION" == "latest" ]]; then
  DRIVER_URL="https://github.com/${DRIVER_REPO}/releases/latest/download/duckdb.metabase-driver.jar"
else
  DRIVER_URL="https://github.com/${DRIVER_REPO}/releases/download/${DRIVER_VERSION}/duckdb.metabase-driver.jar"
fi

mkdir -p "$PLUGIN_DIR"

echo "Fetching DuckDB driver:"
echo "  repo:    $DRIVER_REPO"
echo "  version: $DRIVER_VERSION"
echo "  url:     $DRIVER_URL"
echo "  target:  $DRIVER_JAR"

if ! curl -fL -o "$DRIVER_JAR" "$DRIVER_URL"; then
  echo ""
  echo "ERROR: download failed. The community DuckDB driver may have moved."
  echo "Check for the current release URL at:"
  echo "  https://github.com/${DRIVER_REPO}/releases"
  echo "Then re-run with a specific version:"
  echo "  DUCKDB_DRIVER_VERSION=v0.3.0 $0"
  exit 1
fi

echo ""
echo "Downloaded: $(du -h "$DRIVER_JAR" | cut -f1) → $DRIVER_JAR"
echo "Restart Metabase to load the driver: docker compose restart metabase"
