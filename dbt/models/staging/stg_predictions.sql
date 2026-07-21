{{ config(materialized='view') }}

-- One row per (strategy, ticker, prediction_date). Feeds IC + decile
-- spread computation downstream. Predictions may or may not become
-- ledger trades — the join to stg_ledger is on (strategy, ticker,
-- prediction_date == entry_date).
--
-- Source columns (verified against actual JSONL): strategy_id, region,
-- prediction_date, ticker, predicted_class, predicted_return_pct,
-- conviction, rationale, actual_return_pct, actual_class, was_traded,
-- created_at.
--
-- All source refs are qualified with `source.` — DuckDB's binder
-- rejects `try_cast(x as t) as x` as a forward alias reference.

with source as (
    select * from read_json_auto(
        '{{ var("state_root") }}/predictions.jsonl',
        format = 'newline_delimited',
        ignore_errors = true
    )
),

typed as (
    select
        try_cast(source.prediction_date as date)        as prediction_date,
        source.strategy_id                              as strategy_id,
        source.region                                   as region,
        source.ticker                                   as ticker,
        source.predicted_class                          as predicted_class,
        try_cast(source.predicted_return_pct as double) as predicted_return_pct,
        try_cast(source.conviction as double)           as conviction,
        source.rationale                                as rationale,
        try_cast(source.actual_return_pct as double)    as actual_return_pct,
        source.actual_class                             as actual_class,
        try_cast(source.was_traded as boolean)          as was_traded,
        try_cast(source.created_at as timestamp)        as created_at
    from source
    where source.prediction_date is not null
      and source.strategy_id is not null
      and source.ticker is not null
),

flagged as (
    select
        *,
        (actual_return_pct is not null)     as is_graded,
        (predicted_return_pct > 0)          as is_long_prediction
    from typed
)

select * from flagged
