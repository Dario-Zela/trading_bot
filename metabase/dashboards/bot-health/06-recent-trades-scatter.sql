-- Bot Health · Recent trades scatter (last 30 days)
--
-- Question type: SQL query
-- Visualization: Scatter chart
-- Axes:          x = entry_date, y = pnl_pct, color = strategy_id
-- Dashboard tile: full-width bottom row.
--                 Shows the distribution of individual trades in
--                 time — clusters of losses on one day, TP hits in
--                 a stretch, etc. Great for spotting bad days vs
--                 bad strategies.
--
-- Filter idea:   add {{strategy}} variable so you can drill into
--                one strategy at a time from the same tile.

select
    entry_date,
    strategy_id || '@' || region  as strategy,
    ticker,
    pnl_pct,
    pnl_gbp,
    exit_reason,
    tier
from main_staging.stg_ledger
where is_closed
  and exit_reason not in ('cancelled', 'cleared')
  and strategy_id != 'control-rule-based'
  and entry_date >= current_date - interval '30' day
order by entry_date
