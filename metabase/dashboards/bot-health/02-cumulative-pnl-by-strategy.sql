-- Bot Health · Cumulative PnL by strategy (last 12 weeks)
--
-- Question type: SQL query
-- Visualization: Line chart, one series per strategy
-- Axes:          x = week, y = cum_pnl, color = strategy_id
-- Dashboard tile: half-width; sits next to the weekly-total tile.
--                 Instantly shows which strategies are dragging the
--                 total up vs down.

with weekly as (
    select
        strategy_id,
        (iso_year::text || '-W' || lpad(iso_week::text, 2, '0'))  as week,
        iso_year * 100 + iso_week                                 as sort_key,
        sum(net_pnl_gbp)                                          as week_pnl
    from main_marts.mart_strategy_weekly
    where strategy_id != 'control-rule-based'
    group by 1, 2, 3
)

select
    strategy_id,
    week,
    sum(week_pnl) over (
        partition by strategy_id
        order by sort_key
        rows between unbounded preceding and current row
    )               as cum_pnl
from weekly
order by strategy_id, sort_key
