{{ config(materialized='view') }}

-- One row per intraday trail / midday-TP exit event. Reference-only:
-- the authoritative PnL lives on stg_ledger; this is for exit-timing
-- and reason-attribution analysis.

with source as (
    select * from {{ source('raw', 'trail_exits') }}
),

typed as (
    select
        try_cast(exited_at as timestamp)       as exited_at,
        try_cast(entry_date as date)           as entry_date,
        try_cast(exit_date as date)            as exit_date,
        strategy_id,
        region,
        tier,
        ticker,
        exit_reason,
        try_cast(exit_price as double)         as exit_price,
        try_cast(pnl_pct as double)            as pnl_pct,
        try_cast(pnl_gbp as double)            as pnl_gbp
    from source
    where exited_at is not null
)

select * from typed
