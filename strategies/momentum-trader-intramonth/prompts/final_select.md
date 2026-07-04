# Momentum-trader — final selection prompt

Given the deep-analysis output for every candidate, pick the final trade list for today.

Hard constraints (from config):
- `max_positions` total picks
- Each position's allocation between `min_position_gbp / capital_gbp` and `max_position_pct`
- Total allocation ≤ 100% (you may hold cash if nothing is compelling)

Sizing guidance for this strategy: concentrate in your highest-conviction picks. A momentum strategy is better off making fewer, larger bets on clear trends than spreading thinly.

For each pick, output:
```json
{
  "ticker": "...",
  "allocation_pct": <float>,
  "stop_loss_pct": <float or null>,
  "take_profit_pct": <float or null>,
  "thesis": "<1-2 sentences — why this, why now>"
}
```

If no candidate clears your bar today, return an empty list. Cash is a position.
