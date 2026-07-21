-- Bot Health · System-wide exit-reason attribution
--
-- Question type: SQL query
-- Visualization: Bar chart, one bar per exit_reason
-- Axes:          x = total_pnl_gbp, y = exit_reason
-- Dashboard tile: half-width; shows *how* trades exit and which
--                 exit modes actually earn.
--
-- Expected pattern given the current bot:
--   midday_take_profit → big positive
--   take_profit        → positive
--   scheduled          → very negative (cost drag on flat trades)
--   stop               → negative (wide-stop losses)
--
-- Excludes cancelled/cleared (no realised PnL) and the control
-- benchmark (intentional loser, distorts scale).

select
    exit_reason,
    n_trades,
    hit_rate,
    total_pnl_gbp,
    avg_pnl_gbp,
    total_fees_gbp
from main_marts.mart_exit_attribution
where scope = 'system'
order by total_pnl_gbp desc
