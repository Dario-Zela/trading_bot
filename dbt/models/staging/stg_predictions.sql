{{ config(materialized='view') }}

-- One row per (strategy, ticker, prediction_date). Feeds IC + decile
-- spread computation downstream. Predictions may or may not become
-- ledger trades — the join to stg_ledger is on (strategy, ticker,
-- prediction_date == entry_date).

with source as (
    select * from read_json_auto(
        '{{ var("state_root") }}/predictions.jsonl',
        format = 'newline_delimited',
        ignore_errors = true
    )
),

typed as (
    select
        try_cast(prediction_date as date)          as prediction_date,
        strategy_id,
        region,
        ticker,
        try_cast(predicted_return_pct as double)   as predicted_return_pct,
        try_cast(confidence as double)             as confidence,
        try_cast(horizon_days as integer)          as horizon_days,
        try_cast(actual_return_pct as double)      as actual_return_pct,
        try_cast(graded_at as timestamp)           as graded_at,
        rank                                       as prediction_rank,
        thesis,
        try_cast(created_at as timestamp)          as created_at
    from source
    where prediction_date is not null
      and strategy_id is not null
      and ticker is not null
),

flagged as (
    select
        *,
        (actual_return_pct is not null)            as is_graded,
        (predicted_return_pct > 0)                 as is_long_prediction
    from typed
)

select * from flagged
