-- Bot Health · Fee drag % by strategy
--
-- Question type: SQL query
-- Visualization: Bar chart (horizontal), sorted desc
-- Axes:          x = fees_share_of_gross (%), y = strategy@region
-- Dashboard tile: full-width bottom section.
--                 Anything above ~50% is a strategy where costs are
--                 eating the signal — direct input to evolution-loop
--                 tuning decisions.
--
-- Alert idea:    fires when any strategy's fees_share_of_gross > 0.80.

select
    strategy_id || '@' || region       as strategy,
    n_trades,
    round(net_pnl_gbp::numeric, 2)     as net_pnl_gbp,
    round(gross_pnl_gbp::numeric, 2)   as gross_pnl_gbp,
    round(fees_gbp::numeric, 2)        as fees_gbp,
    round((fees_share_of_gross * 100)::numeric, 1) as fee_share_pct,
    round(avg_fee_pct_of_size::numeric, 3) as avg_fee_pct_of_position
from main_marts.mart_fee_drag
where n_trades >= 20     -- ignore small-sample strategies
  and strategy_id != 'control-rule-based'
order by fee_share_pct desc
