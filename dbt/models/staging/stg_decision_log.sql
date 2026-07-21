{{ config(materialized='view') }}

-- One row per evolution-agent decision. Used to trace strategy state
-- transitions (promote / demote / tune / spawn / deactivate) and to
-- window metrics from a strategy's `last_tune_date`.
--
-- Source columns (verified against actual JSONL): week_iso, decided_at,
-- strategy_id, region, action, reason, details (nested), pre_metrics
-- (nested), post_metrics (nested), grade (nested), graded_at.
--
-- All source refs are qualified with `source.` — DuckDB's binder
-- rejects `try_cast(x as t) as x` as a forward alias reference.

with source as (
    select * from read_json_auto(
        '{{ var("state_root") }}/decision_log.jsonl',
        format = 'newline_delimited',
        ignore_errors = true
    )
),

typed as (
    select
        try_cast(source.decided_at as date)  as decided_at,
        source.week_iso                      as week_iso,
        source.strategy_id                   as strategy_id,
        source.region                        as region,
        source.action                        as action,
        source.reason                        as reason,
        try_cast(source.graded_at as date)   as graded_at
    from source
    where source.decided_at is not null
      and source.strategy_id is not null
)

select * from typed
