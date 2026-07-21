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

# Default to the AlexR2D2 fork — the pre-MotherDuck community driver.
# The MotherDuck fork (motherduckdb/metabase_duckdb_driver) hardcodes
# MotherDuck-specific connection options like `motherduck_token` and
# `connection-pool-type` and passes them unconditionally to DuckDB.
# Plain (local) DuckDB doesn't recognize them and rejects the connection:
#
#   Invalid Input Error: The following options were not recognized:
#     motherduck_token
#     connection-pool-type
#
# The AlexR2D2 fork predates that integration and works with local
# DuckDB out of the box. Override via DUCKDB_DRIVER_REPO if you need
# MotherDuck instead.
DRIVER_REPO="${DUCKDB_DRIVER_REPO:-AlexR2D2/metabase_duckdb_driver}"
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
