# News-reactive — final selection prompt

Given the deep-analysis output and candidate data, pick today's final trade list.

Hard constraints from config:
- Up to `max_positions` picks
- Each `allocation_pct` between `min_position_gbp / capital × 100` and `max_position_pct`
- Total allocation ≤ 100% — return `[]` if no catalyst-driven setup clears your bar

Sizing guidance for this strategy: catalyst-driven trades are higher conviction than pure technicals, so concentration is appropriate when the news is clear and the market reaction is still developing. But beware of "already priced in" — if a stock is already up 10% on the news, you're paying for the catalyst, not capturing it.

For each pick output:

```json
{
  "ticker": "...",
  "allocation_pct": <float>,
  "stop_loss_pct": <float or null>,
  "take_profit_pct": <float or null>,
  "thesis": "<1-2 sentences naming the specific catalyst (earnings beat, M&A, guidance, etc.) and why the reaction has further to run>"
}
```

Without a real, identifiable catalyst, return `[]`. This strategy is designed to fire selectively, not daily.
