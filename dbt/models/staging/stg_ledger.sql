{{ config(materialized='view') }}

-- One row per trade. Types coerced, obvious-null trades dropped.
-- The is_closed / is_win flags exist so marts can filter without
-- re-deriving the exit-date logic every time.

with source as (
    select * from read_json_auto(
        '{{ var("state_root") }}/ledger.jsonl',
        format = 'newline_delimited',
        ignore_errors = true
    )
),

typed as (
    select
        trade_id,
        strategy_id,
        region,
        tier,
        ticker,
        side,
        cast(entry_date as date)                    as entry_date,
        try_cast(entry_price as double)             as entry_price,
        try_cast(quantity as double)                as quantity,
        try_cast(allocation_pct as double)          as allocation_pct,
        try_cast(stop_loss_pct as double)           as stop_loss_pct,
        try_cast(take_profit_pct as double)         as take_profit_pct,
        thesis,
        try_cast(exit_date as date)                 as exit_date,
        try_cast(exit_price as double)              as exit_price,
        coalesce(try_cast(pnl_gbp as double), 0.0)  as pnl_gbp,
        try_cast(pnl_pct as double)                 as pnl_pct,
        exit_reason,
        outcome_notes,
        risks_observed,
        broker_order_id,
        currency,
        exchange,
        instrument_type,
        try_cast(entry_fx_rate as double)           as entry_fx_rate,
        coalesce(try_cast(fees_gbp as double), 0.0) as fees_gbp,
        try_cast(target_exit_date as date)          as target_exit_date,
        try_cast(hold_days as integer)              as hold_days,
        try_cast(created_at as timestamp)           as created_at
    from source
    where trade_id is not null
),

flagged as (
    select
        *,
        (exit_date is not null)                                    as is_closed,
        (exit_date is not null and pnl_gbp > 0)                    as is_win,
        (exit_date is not null and pnl_gbp <= 0)                   as is_loss,
        -- Gross PnL = realised + fees paid; useful when fees dominate.
        (pnl_gbp + fees_gbp)                                       as gross_pnl_gbp,
        -- Notional size in GBP, inferred from pnl_gbp / (pnl_pct/100)
        -- when pnl_pct != 0, else from fees / cost-floor. Exact enough
        -- for capital-return analysis where the actual notional isn't
        -- persisted on the row.
        case
            when pnl_pct is not null and pnl_pct != 0
                then pnl_gbp / (pnl_pct / 100.0)
            when fees_gbp > 0
                then fees_gbp / 0.003
            else null
        end                                                        as inferred_notional_gbp
    from typed
)

select * from flagged
