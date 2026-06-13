# Sector-rotator — final selection prompt

Given the deep-analysis output and candidate data, pick today's final trade list.

Hard constraints from config:
- Up to `max_positions` picks
- Each `allocation_pct` between `min_position_gbp / capital × 100` and `max_position_pct`
- Total allocation ≤ 100% — return `[]` if no sector is showing clear leadership today

Sizing guidance for this strategy: sector rotation tends to be a slower-moving game than single-stock momentum — a position may need 3–10 trading days to play out. Use **wider stops and longer-distance take-profits** than an equity strategy would (the strategy config gives you defaults around -4%/+8% — those are good starting points; deviate down if conviction is high, up if the macro context is uncertain). Concentration into 2–3 high-conviction sectors is usually better than spreading thinly across all 4 slots.

For each pick output:

```json
{
  "ticker": "XLK",
  "allocation_pct": 35.0,
  "stop_loss_pct": -4.0,
  "take_profit_pct": 8.0,
  "thesis": "<1-2 sentences citing the relative-strength rank, the macro alignment, and any confirming technical signal>"
}
```

If no sector is showing genuine leadership (top-3 relative strength + technical confirmation), return `[]`. Sitting in cash on an uncertain rotation day is correct.
