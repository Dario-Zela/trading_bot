{{ config(materialized='table') }}

-- The ML-vs-LLM comparison story (§9 of the ml-challenger design doc):
-- per (strategy, region, day) over graded predictions — sample size,
-- cross-sectional rank IC, non-flat directional hit rate, and the
-- conviction-vs-realised calibration gap. One Metabase question over
-- this table is the head-to-head dashboard.
--
-- Two honest asymmetries to keep in mind when reading it:
--   1. ml-challenger grades the full universe daily (~1-3k rows) vs an
--      LLM strategy's ~100 pre-filtered candidates — its per-day IC has
--      far tighter error bars and it crosses metrics.is_alive()'s
--      >=30-graded threshold on day one.
--   2. Shadow trades fill at the most recent close while predictions
--      are graded open→close — compare on the prediction-grading path
--      primarily, trade P&L secondarily (that lives in
--      mart_strategy_weekly).
--
-- Spearman via rank() + corr(): dense integer ranks rather than
-- average-of-ties, which differs negligibly at cross-section sizes and
-- keeps the SQL readable.

with graded as (

    select *
    from {{ ref('stg_predictions') }}
    where is_graded

),

ranked as (

    select
        strategy_id,
        region,
        prediction_date,
        predicted_class,
        actual_class,
        conviction,
        predicted_return_pct,
        actual_return_pct,
        (predicted_class in ('mild_up', 'strong_up'))     as predicted_up,
        (predicted_class in ('mild_down', 'strong_down')) as predicted_down,
        rank() over (
            partition by strategy_id, region, prediction_date
            order by predicted_return_pct
        ) as r_pred,
        rank() over (
            partition by strategy_id, region, prediction_date
            order by actual_return_pct
        ) as r_actual
    from graded

),

daily as (

    select
        strategy_id,
        region,
        prediction_date,
        count(*)                                          as n_graded,
        corr(r_pred, r_actual)                            as spearman_ic,
        sum(case when predicted_up or predicted_down then 1 else 0 end)
                                                          as n_nonflat,
        avg(case
                when predicted_up   then (actual_return_pct > 0)::int::double
                when predicted_down then (actual_return_pct < 0)::int::double
            end)                                          as nonflat_hit_rate,
        -- Positive gap = overconfident: stated conviction exceeds the
        -- realised class-match rate. The LLM strategies' famous failure
        -- mode; the challenger's headline chance to differ.
        avg(conviction)
          - avg((predicted_class = actual_class)::int::double)
                                                          as calibration_gap,
        avg(predicted_return_pct)                         as avg_predicted_pct,
        avg(actual_return_pct)                            as avg_actual_pct
    from ranked
    group by 1, 2, 3

)

select
    *,
    (strategy_id = 'ml-challenger') as is_ml
from daily
order by prediction_date, strategy_id, region
