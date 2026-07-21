-- Bot Health · Recent trades scatter (last 30 days)
--
-- Question type: SQL query
-- Visualization: Scatter chart
-- Axes:          x = entry_date, y = pnl_pct, color = strategy
-- Dashboard tile: full-width bottom row.
--                 Shows the distribution of individual trades in
--                 time — clusters of losses on one day, TP hits in
--                 a stretch, etc.
--
-- Reads mart_trades_recent — a per-trade materialized slice of the
-- ledger for the last 30 days. Bounded by the mart, not the query,
-- so Metabase's dashboard stays snappy even as the ledger grows.

select
    entry_date,
    strategy,
    ticker,
    pnl_pct,
    pnl_gbp,
    exit_reason,
    tier,
    hold_days
from main_marts.mart_trades_recent
order by entry_date
