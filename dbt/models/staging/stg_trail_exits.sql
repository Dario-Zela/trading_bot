{{ config(materialized='table') }}

-- One row per intraday trail / midday-TP exit event. Reference-only:
-- the authoritative PnL lives on stg_ledger; this is for exit-timing
-- and reason-attribution analysis.
--
-- Source columns (verified against actual JSONL): ticker, region,
-- strategy_id, exit_date, exit_reason, pnl_pct, appended_at.

with source as (
    select * from read_json_auto(
        '{{ var("state_root") }}/trail_exits.jsonl',
        format = 'newline_delimited',
        ignore_errors = true
    )
),

typed as (
    select
        try_cast(source.appended_at as timestamp) as appended_at,
        try_cast(source.exit_date as date)        as exit_date,
        source.strategy_id                        as strategy_id,
        source.region                             as region,
        source.ticker                             as ticker,
        source.exit_reason                        as exit_reason,
        try_cast(source.pnl_pct as double)        as pnl_pct
    from source
    where source.appended_at is not null
)

select * from typed
