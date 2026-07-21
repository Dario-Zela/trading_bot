{{ config(materialized='view') }}

-- One row per evolution-agent decision. Used to trace strategy state
-- transitions (promote / demote / tune / spawn / deactivate) and to
-- window metrics from a strategy's `last_tune_date`.

with source as (
    select * from read_json_auto(
        '{{ var("state_root") }}/decision_log.jsonl',
        format = 'newline_delimited',
        ignore_errors = true
    )
),

typed as (
    select
        try_cast(decided_at as timestamp)      as decided_at,
        try_cast(week_ending as date)          as week_ending,
        strategy_id,
        region,
        action,
        rationale,
        details,
        applied,
        try_cast(applied_at as timestamp)      as applied_at
    from source
    where decided_at is not null
      and strategy_id is not null
)

select * from typed
