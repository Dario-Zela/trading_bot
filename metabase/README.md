# Metabase for trading-bot analytics

Reads the dbt-built DuckDB store (`dbt/analytics.duckdb`) and serves
dashboards. Everything you'd write as an ad-hoc Python rollup can live
here instead as a saved question or dashboard tile.

## Fastest path — GitHub Codespaces (no clone, browser-only)

1. On the repo page: **Code → Codespaces → Create codespace on main**
2. Wait ~2 min for first build. The devcontainer auto-installs deps,
   fetches the DuckDB driver, builds the analytics store, and starts
   Metabase in the background.
3. Open the **PORTS** tab (bottom panel in VS Code Web) — click the
   forwarded URL next to port 3000.
4. First load asks you to create an admin account (stored in the
   Codespace's `metabase/data/`; persists across container restarts
   until the Codespace itself is deleted).
5. Add the DuckDB connection (see [Add the DuckDB connection](#add-the-duckdb-connection) below).

Codespaces auto-suspends after 30 min idle; wakes on next request.
Free tier: 60h/month on personal accounts.

## Alternative — Local Docker

Requires: Docker + `curl`. Run all commands from the repo root.

**1. Build the analytics store** (creates `dbt/analytics.duckdb`):

```bash
python -c "from trading_bot.analytics.dbt_runner import build; build()"
```

**2. Fetch the community DuckDB driver:**

```bash
./metabase/fetch_duckdb_driver.sh
```

Verify the release URL first if you want to pin a version — the driver
is community-maintained. The script pulls the latest release from
`motherduckdb/metabase_duckdb_driver` by default; override via
`DUCKDB_DRIVER_REPO=` and `DUCKDB_DRIVER_VERSION=` env vars.

**3. Start Metabase:**

```bash
docker compose up -d
docker compose logs -f metabase   # wait for "Metabase Initialization COMPLETE"
```

Cold start ~45s. Open http://localhost:3000, create an admin account
(local-only, credentials stored in `metabase/data/`).

## Add the DuckDB connection

Admin → Databases → Add database:

- Database type: **DuckDB**
- Display name: `trading_bot_analytics`
- Database file: `/dbt/analytics.duckdb`
- Read-only: **✓** (important — avoids fighting `dbt run` for the write lock)

Save. Metabase syncs the schema (~5s) — you should see `staging` and
`marts` schemas populate, with `stg_ledger`, `mart_strategy_weekly`,
`mart_fee_drag`, `mart_exit_attribution` visible under them.

## Suggested first dashboards

Once the connection is live, these questions get you 80% of what the
current Python rollups produce:

**Weekly PnL trajectory** — line chart from `mart_strategy_weekly`:
```sql
select iso_year * 100 + iso_week as week,
       sum(net_pnl_gbp)          as net_pnl,
       sum(gross_pnl_gbp)        as gross_pnl,
       sum(fees_gbp)             as fees
from marts.mart_strategy_weekly
where strategy_id != 'control-rule-based'   -- benchmark; excluded
group by week
order by week
```

**Per-strategy fee drag** — bar chart from `mart_fee_drag`:
```sql
select strategy_id || '@' || region as strategy,
       n_trades,
       net_pnl_gbp,
       fees_gbp,
       fees_share_of_gross
from marts.mart_fee_drag
where n_trades >= 20
order by fees_share_of_gross desc
```

**Exit-reason attribution** — table from `mart_exit_attribution`:
```sql
select exit_reason, n_trades, hit_rate, total_pnl_gbp, avg_pnl_gbp
from marts.mart_exit_attribution
where scope = 'system'
order by total_pnl_gbp desc
```

## Alerts worth setting up

Metabase can email/Slack when a saved-question result crosses a
threshold. Useful defaults for this bot:

- **Weekly PnL turns negative for 2+ consecutive weeks** — early
  warning that the evolution loop's paper picks are decaying
- **Any strategy's `fees_share_of_gross > 0.8`** — cost model is
  eating the signal; that strategy needs a `cost_gate_multiplier`
  bump before its next weekly review
- **`mart_exit_attribution` scheduled-exit row shows n_trades × avg
  ≤ -£500 in a rolling week** — the "flat trades held to close, cost
  floor eats it" failure mode this session diagnosed

## Refresh cadence

The `mart_*` tables materialise as tables (not views), so they only
change when `dbt run` executes. Two options for keeping Metabase fresh:

- **Cron-driven** — GitHub Actions already runs the trading pipelines;
  add a `dbt run` step after each stage that mutates state (see
  `dbt/README.md` for the migration list).
- **Metabase-triggered** — set Metabase's cache TTL to match your
  dbt cadence (default is `MB_QUERY_CACHE_MAX_AGE=60`s in
  `docker-compose.yml`, so queries always show ≤1min-old rebuild).

## Troubleshooting

**"Driver not found" on connection add** — driver JAR wasn't loaded.
Verify `metabase/plugins/duckdb.metabase-driver.jar` exists and restart
the container: `docker compose restart metabase`.

**"Database locked" error in Metabase** — you connected without
read-only. Edit the connection, tick read-only, save. Or check
whether `dbt run` is currently holding the write lock.

**Metabase shows old data** — a `dbt run` completed but Metabase's
cache hasn't expired. Force refresh via the dashboard's Actions → Get
info → Clear cache. Or lower `MB_QUERY_CACHE_MAX_AGE` in
`docker-compose.yml`.

**Dashboards vanished after `docker compose down -v`** — the `-v`
flag deletes named volumes AND bind mounts' backing files if you
weren't careful. The bind mount at `metabase/data/` is on your host
FS so `down -v` shouldn't touch it, but restore from backup if it did.
Consider swapping to Postgres for `MB_DB_*` env vars if you value
persistence.

## Stack diagram

```
state/*.jsonl  ─┐
                ├─►  dbt (transforms)  ─►  dbt/analytics.duckdb  ─►  Metabase
state/ohlcv.db ─┘                                                       ▲
                                                                        │
                                                                    Docker
```

- **State writes**: pipelines write JSONL under `state/`.
- **Transform**: `python -c "from trading_bot.analytics.dbt_runner import build; build()"`.
- **Read**: Python code via `trading_bot.analytics.reader`, humans via Metabase.
