{{ config(materialized='table') }}

-- Weekly PnL per (strategy, region). Replaces the ad-hoc weekly rollup
-- that was being computed in Python each time the dashboard rebuilt.
--
-- Grain: one row per (strategy_id, region, iso_year, iso_week).

with closed as (
    select *
    from {{ ref('stg_ledger') }}
    where is_closed
      and exit_reason not in ('cancelled', 'cleared')
),

weekly as (
    select
        strategy_id,
        region,
        extract(isoyear from exit_date)             as iso_year,
        extract(week    from exit_date)             as iso_week,
        min(exit_date)                              as week_start_date,
        max(exit_date)                              as week_end_date,
        count(*)                                    as n_trades,
        sum(case when is_win then 1 else 0 end)     as n_wins,
        sum(case when is_loss then 1 else 0 end)    as n_losses,
        sum(pnl_gbp)                                as net_pnl_gbp,
        sum(gross_pnl_gbp)                          as gross_pnl_gbp,
        sum(fees_gbp)                               as fees_gbp,
        avg(pnl_pct)                                as avg_pnl_pct,
        -- Rough per-trade Sharpe input; stddev of trade pct returns.
        stddev_samp(pnl_pct)                        as std_pnl_pct
    from closed
    group by 1, 2, 3, 4
)

select
    *,
    case when n_trades > 0 then n_wins::double / n_trades else 0 end as hit_rate,
    case when std_pnl_pct > 0 then avg_pnl_pct / std_pnl_pct else null end as sharpe_per_trade
from weekly
order by iso_year, iso_week, strategy_id, region
