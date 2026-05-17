# Momentum-trader EU — final selection prompt

Given the deep-analysis output for DAX/CAC/AEX candidates, pick today's
final trade list.

Hard constraints from config:
- `max_positions` total picks
- Each allocation between `min_position_gbp/capital × 100` and `max_position_pct`
- Total allocation ≤ 100% (cash is a valid position)

Sizing guidance: European blue-chip momentum is slower and lower-volatility
than US tech momentum. Tighter stops (-3% default) and modest take-profits
(+5%) are appropriate. Try to spread across at least two exchanges (don't
go 100% DAX or 100% CAC if the signal supports diversification).

For each pick:

```json
{
  "ticker": "SAP.DE",
  "allocation_pct": 30.0,
  "stop_loss_pct": -3.0,
  "take_profit_pct": 5.0,
  "thesis": "<1-2 sentences citing the technical setup>"
}
```

If no candidate has a clean momentum setup, return `[]`. Forced trades on a
quiet European session underperform.
