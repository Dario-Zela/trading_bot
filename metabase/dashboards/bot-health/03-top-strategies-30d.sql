-- Bot Health · Top strategies by last-30-day PnL
--
-- Question type: SQL query
-- Visualization: Bar chart (horizontal)
-- Axes:          x = net_pnl_gbp, y = strategy_id, color = region
-- Dashboard tile: half-width; leaderboard view.
--                 Positive bars on the right, bleeders on the left.
--
-- Note: computes windowed rollups directly from stg_ledger because
--       mart_fee_drag is lifetime, not 30d. Same shape though.

select
    strategy_id || '@' || region      as strategy,
    strategy_id,
    region,
    count(*)                          as n_trades,
    sum(pnl_gbp)                      as net_pnl_gbp,
    sum(gross_pnl_gbp)                as gross_pnl_gbp,
    sum(fees_gbp)                     as fees_gbp,
    sum(case when is_win then 1 else 0 end)::double / count(*)  as hit_rate
from main_staging.stg_ledger
where is_closed
  and exit_reason not in ('cancelled', 'cleared')
  and strategy_id != 'control-rule-based'
  and exit_date >= current_date - interval '30' day
group by 1, 2, 3
having count(*) >= 5      -- drop small-sample noise
order by net_pnl_gbp desc
