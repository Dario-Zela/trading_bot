{{ config(materialized='table') }}

-- Per-strategy fee drag: gross PnL, fees paid, net PnL, and the
-- fees-as-share-of-gross ratio that surfaces "signal works, costs
-- eat it" cases to the evolution agent.
--
-- Replaces trading_bot.meta.evolution_inputs.fees_pct_of_gross.
-- Grain: one row per (strategy_id, region). System-wide as scope='ALL'.

with closed as (
    select *
    from {{ ref('stg_ledger') }}
    where is_closed
      and exit_reason not in ('cancelled', 'cleared')
),

strategy_level as (
    select
        strategy_id,
        region,
        count(*)                                    as n_trades,
        sum(pnl_gbp)                                as net_pnl_gbp,
        sum(gross_pnl_gbp)                          as gross_pnl_gbp,
        sum(fees_gbp)                               as fees_gbp,
        avg(pnl_pct)                                as avg_net_pct,
        avg(inferred_notional_gbp)                  as avg_notional_gbp
    from closed
    group by 1, 2
),

with_ratios as (
    select
        strategy_id,
        region,
        n_trades,
        net_pnl_gbp,
        gross_pnl_gbp,
        fees_gbp,
        avg_net_pct,
        avg_notional_gbp,
        case
            when abs(gross_pnl_gbp) + fees_gbp > 0
                then fees_gbp / (abs(gross_pnl_gbp) + fees_gbp)
            else null
        end                                         as fees_share_of_gross,
        case when avg_notional_gbp > 0
            then 100.0 * fees_gbp / (n_trades * avg_notional_gbp)
            else null
        end                                         as avg_fee_pct_of_size
    from strategy_level
)

select * from with_ratios
order by strategy_id, region
