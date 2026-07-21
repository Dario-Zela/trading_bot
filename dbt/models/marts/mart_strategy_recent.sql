{{ config(materialized='table') }}

-- Per-(strategy, region) rollup over trailing windows.
-- Complements mart_fee_drag (lifetime) with the windowed view the
-- 30-day leaderboard needs. Two rows per (strategy, region) — one
-- per window — stacked with a `window_days` column so consumers
-- pick the window they want.
--
-- Windows: 7d, 30d, 90d. Add more here if needed rather than in
-- consumer SQL.

with closed as (
    select *
    from {{ ref('stg_ledger') }}
    where is_closed
      and exit_reason not in ('cancelled', 'cleared')
),

windows(window_days) as (
    values (7), (30), (90)
),

rolled as (
    select
        w.window_days,
        c.strategy_id,
        c.region,
        count(*)                                as n_trades,
        sum(case when c.is_win then 1 else 0 end) as n_wins,
        sum(c.pnl_gbp)                          as net_pnl_gbp,
        sum(c.gross_pnl_gbp)                    as gross_pnl_gbp,
        sum(c.fees_gbp)                         as fees_gbp,
        avg(c.pnl_pct)                          as avg_pnl_pct
    from windows w
    cross join closed c
    where c.exit_date >= current_date - cast(w.window_days as integer) * interval 1 day
    group by w.window_days, c.strategy_id, c.region
)

select
    window_days,
    strategy_id,
    region,
    n_trades,
    n_wins,
    net_pnl_gbp,
    gross_pnl_gbp,
    fees_gbp,
    avg_pnl_pct,
    case when n_trades > 0 then n_wins::double / n_trades else 0 end as hit_rate
from rolled
order by window_days, net_pnl_gbp desc
