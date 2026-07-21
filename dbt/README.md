# dbt analytics for trading-bot

DuckDB-backed analytics layer. Reads the append-only state files under
`../state/` and materialises marts under `dbt/analytics.duckdb`, which
`trading_bot.analytics.reader` opens read-only.

For interactive dashboards on top of the same DuckDB store, see
[`../metabase/README.md`](../metabase/README.md) — `docker compose up`
and you get a browser UI over these marts.

## Layout

```
dbt/
├── dbt_project.yml     # project + var config
├── profiles.yml        # DuckDB target (in-project so no ~/.dbt setup)
├── models/
│   ├── sources.yml     # raw JSONL files as sources
│   ├── staging/        # 1:1 typed views of each raw source
│   └── marts/          # business facts consumed by dashboard + evolution
```

Materialisation policy: staging is `view` (cheap, always fresh), marts
are `table` (queried by the dashboard + LLM prompts).

## Running

From the repo root:

```bash
# Regenerate everything
python -c "from trading_bot.analytics.dbt_runner import build; build()"

# Rebuild one mart
python -c "from trading_bot.analytics.dbt_runner import build; build(select='mart_fee_drag')"

# Run tests
python -c "from trading_bot.analytics.dbt_runner import build; build(run_tests=True)"
```

Or drive `dbt` directly (the Python runner sets env vars, so `dbt`
alone needs them set manually):

```bash
export TRADING_BOT_STATE_ROOT="$(pwd)/state"
export TRADING_BOT_DUCKDB_PATH="$(pwd)/dbt/analytics.duckdb"
export TRADING_BOT_OHLCV_PATH="$(pwd)/state/ohlcv.db"
export DBT_PROFILES_DIR="$(pwd)/dbt"
dbt run --project-dir dbt
```

## Adding a new mart

1. Drop `models/marts/mart_<name>.sql` — SELECT from `{{ ref('stg_...') }}`.
2. Register it in `models/marts/schema.yml` with column tests.
3. Add a typed accessor in `src/trading_bot/analytics/reader.py`.
4. Migrate the caller from JSONL iteration to that accessor.

## When to rebuild

The pipeline workflows should call `dbt_runner.build()` after any stage
that mutates state files:

- Entry-run / exit-run (mutates `ledger.jsonl` + `predictions.jsonl`)
- Midday trail / midday TP (mutates `ledger.jsonl` + `trail_exits.jsonl`)
- Grade predictions (mutates `predictions.jsonl`)
- Weekly evolution (mutates `decision_log.jsonl`)

Wiring the call into each workflow is a follow-up.

## Migration status

Callers currently reading state/*.jsonl directly:

- [x] `evolution_inputs.fees_pct_of_gross` → uses `get_fee_drag_windowed`
- [ ] `evolution_inputs.sector_concentration`
- [ ] `evolution_inputs.cost_gate_drop_rate`
- [ ] `evolution_inputs.trail_activation_rate`
- [ ] `evolution_inputs.divergent_strategies`
- [ ] `evolution_inputs.earnings_gate_hit_rate`
- [ ] `evolution_inputs.verdict_rates_by_source`
- [ ] `meta.metrics.compute_metrics` (the big one — IC + decile spread)
- [ ] `dashboard.build` — currently iterates ledger directly for
      weekly/strategy panels; should SELECT from
      `mart_strategy_weekly` + `mart_fee_drag`.

Each migration: add mart or reader helper → swap the function body →
delete the JSONL iterator when its last caller is gone.
