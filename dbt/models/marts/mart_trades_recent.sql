{{ config(materialized='table') }}

-- Individual closed trades in the last 30 days, denormalized for
-- Metabase's per-trade views (scatter charts, per-trade drilldowns).
--
-- Grain: one row per closed trade in the trailing 30-day window.
-- Excluded: cancelled/cleared, control benchmark (distorts scale
-- on any dashboard tile).
--
-- Bounded to 30 days so this mart stays small and Metabase queries
-- against it are instant. If you need longer windows for a specific
-- analysis, add another mart (mart_trades_90d etc.) rather than
-- unbounding this one — dashboards should stay snappy.

select
    trade_id,
    strategy_id,
    region,
    strategy_id || '@' || region  as strategy,
    tier,
    ticker,
    side,
    entry_date,
    exit_date,
    exit_reason,
    entry_price,
    exit_price,
    quantity,
    pnl_gbp,
    pnl_pct,
    gross_pnl_gbp,
    fees_gbp,
    hold_days,
    inferred_notional_gbp,
    currency,
    exchange
from {{ ref('stg_ledger') }}
where is_closed
  and exit_reason not in ('cancelled', 'cleared')
  and strategy_id != 'control-rule-based'
  and exit_date >= current_date - interval '30' day
order by exit_date desc, strategy_id
