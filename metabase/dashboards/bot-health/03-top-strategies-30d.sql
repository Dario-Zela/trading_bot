-- Bot Health · Top strategies by last-30-day PnL
--
-- Question type: SQL query
-- Visualization: Bar chart (horizontal)
-- Axes:          x = net_pnl_gbp, y = strategy, color = region
-- Dashboard tile: half-width; leaderboard view.
--                 Positive bars on the right, bleeders on the left.
--
-- Reads mart_strategy_recent (windowed rollup mart). The `window_days`
-- filter picks the 30-day slice.

select
    strategy_id || '@' || region      as strategy,
    strategy_id,
    region,
    n_trades,
    n_wins,
    hit_rate,
    net_pnl_gbp,
    gross_pnl_gbp,
    fees_gbp
from main_marts.mart_strategy_recent
where window_days = 30
  and n_trades >= 5      -- drop small-sample noise
  and strategy_id != 'control-rule-based'
order by net_pnl_gbp desc
