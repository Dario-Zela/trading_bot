{{ config(materialized='table') }}

-- PnL attributed by exit_reason, both system-wide and per strategy.
-- Replaces the exit-reason breakdown I've been computing ad-hoc.
--
-- Grain: two levels stacked (`scope` column distinguishes them):
--   scope='system'   → one row per exit_reason
--   scope='strategy' → one row per (strategy_id, region, exit_reason)

with closed as (
    select *
    from {{ ref('stg_ledger') }}
    where is_closed
      and exit_reason not in ('cancelled', 'cleared')
),

by_reason_system as (
    select
        'system'                                                    as scope,
        'ALL'                                                       as strategy_id,
        'ALL'                                                       as region,
        exit_reason,
        count(*)                                                    as n_trades,
        sum(case when is_win then 1 else 0 end)                     as n_wins,
        sum(pnl_gbp)                                                as total_pnl_gbp,
        avg(pnl_gbp)                                                as avg_pnl_gbp,
        sum(gross_pnl_gbp)                                          as total_gross_gbp,
        sum(fees_gbp)                                               as total_fees_gbp
    from closed
    group by exit_reason
),

by_reason_strategy as (
    select
        'strategy'                                                  as scope,
        strategy_id,
        region,
        exit_reason,
        count(*)                                                    as n_trades,
        sum(case when is_win then 1 else 0 end)                     as n_wins,
        sum(pnl_gbp)                                                as total_pnl_gbp,
        avg(pnl_gbp)                                                as avg_pnl_gbp,
        sum(gross_pnl_gbp)                                          as total_gross_gbp,
        sum(fees_gbp)                                               as total_fees_gbp
    from closed
    group by strategy_id, region, exit_reason
)

select
    *,
    case when n_trades > 0 then n_wins::double / n_trades else 0 end as hit_rate
from by_reason_system

union all

select
    *,
    case when n_trades > 0 then n_wins::double / n_trades else 0 end as hit_rate
from by_reason_strategy
