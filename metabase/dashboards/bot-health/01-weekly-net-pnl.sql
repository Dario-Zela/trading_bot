-- Bot Health · Weekly net PnL trajectory
--
-- Question type: SQL query
-- Visualization: Line chart
-- Axes:          x = week, y = net_pnl_gbp (also fees / gross overlaid)
-- Dashboard tile: full-width top row — the single most important
--                 line: are we making or losing money this week vs
--                 last?
--
-- Filter idea:   add a variable {{exclude_control}} default 'yes'
--                and swap the `!= control-rule-based` clause for a
--                conditional.

select
    (iso_year::text || '-W' || lpad(iso_week::text, 2, '0'))  as week,
    sum(net_pnl_gbp)                                          as net_pnl,
    sum(gross_pnl_gbp)                                        as gross_pnl,
    sum(fees_gbp)                                             as fees,
    sum(n_trades)                                             as n_trades
from main_marts.mart_strategy_weekly
where strategy_id != 'control-rule-based'
group by 1
order by 1
