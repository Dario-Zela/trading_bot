"""dbt-backed analytics layer.

The bot's raw state lives in append-only JSONL files under state/. Any
callers that want *aggregated* views (weekly PnL, per-strategy fee drag,
exit-reason attribution) should read from dbt-built tables via the
helpers in this package instead of iterating the JSONL themselves.

Two entry points:

- `dbt_runner.build()` — invokes `dbt run` (and optionally `dbt test`),
  regenerating the DuckDB analytics store from current state/*.jsonl.
  Called by the pipeline workflows before anything that reads a mart.

- `reader.connect()` — returns a read-only DuckDB connection to the
  analytics store. Prefer the higher-level helpers (`get_fee_drag`,
  `get_weekly_pnl`, ...) unless you need a bespoke query.
"""
