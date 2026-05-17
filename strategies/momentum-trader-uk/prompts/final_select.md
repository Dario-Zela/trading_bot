# Momentum-trader UK — final selection prompt

Given the deep-analysis output for FTSE 100 candidates, pick today's
final trade list.

Hard constraints from config:
- `max_positions` total picks
- Each allocation between `min_position_gbp/capital × 100` and `max_position_pct`
- Total allocation ≤ 100% (cash is a valid position)

Sizing guidance: FTSE momentum is slower and lower-volatility than US
tech momentum. Tighter stops (-3% default) and modest take-profits (+5%)
are appropriate. Concentrate in 2-3 high-conviction names rather than
spreading thinly — diversification benefit is low across FTSE since the
index is sector-concentrated.

For each pick:

```json
{
  "ticker": "AZN.L",
  "allocation_pct": 30.0,
  "stop_loss_pct": -3.0,
  "take_profit_pct": 5.0,
  "thesis": "<1-2 sentences citing the specific technical setup that justified this pick>"
}
```

If no candidate has a clean momentum setup, return `[]`. Day-by-day patience
beats forced trades on the FTSE.
