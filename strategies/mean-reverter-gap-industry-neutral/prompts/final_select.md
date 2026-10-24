# Mean-reverter — final selection prompt

Given the deep-analysis output and the candidate data, pick today's final trade list.

Hard constraints from config:
- Up to `max_positions` picks
- Each `allocation_pct` between `min_position_gbp / capital × 100` and `max_position_pct`
- Total allocation ≤ 100% (cash is a valid position — return `[]` if nothing meets the bar)

Sizing guidance for this strategy: mean-reversion setups have asymmetric risk — your downside scenario (the falling-knife case) tends to be sharper than the upside bounce. Prefer spreading across multiple oversold names over concentrating in any single one. Tighter stops are appropriate here than for momentum.

For each pick output:

```json
{
  "ticker": "...",
  "allocation_pct": <float>,
  "stop_loss_pct": <float or null>,
  "take_profit_pct": <float or null>,
  "thesis": "<1-2 sentences citing the specific oversold signal and why a reversion is likely>"
}
```

If no candidate genuinely looks oversold-with-no-bad-news, return `[]`. Don't force trades on a day with no clean setups.
