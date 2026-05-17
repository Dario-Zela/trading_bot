# Macro-aligned — final selection prompt

Given the deep-analysis output, the macro view, and candidate data, pick today's final trade list.

Hard constraints from config:
- Up to `max_positions` picks
- Each `allocation_pct` between `min_position_gbp / capital × 100` and `max_position_pct`
- Total allocation ≤ 100% — return `[]` if the macro view is too uncertain or no candidates align cleanly

Sizing guidance for this strategy: macro-aligned plays are slower than single-stock momentum but more durable when the regime call is right. Bias toward concentrating in 2–3 names within your favoured sectors rather than spreading thinly. **Reject any pick whose sector is rated bearish or mildly bearish in the macro view.** Slightly wider stops are appropriate (the macro thesis takes time to play out).

For each pick output:

```json
{
  "ticker": "...",
  "allocation_pct": <float>,
  "stop_loss_pct": <float or null>,
  "take_profit_pct": <float or null>,
  "thesis": "<1-2 sentences citing (1) the macro alignment for this sector and (2) the stock-level setup that makes it the right expression of that view>"
}
```

If the macro view is explicitly uncertain or if no candidates align with bullish sectors, return `[]`. Forcing trades against the macro view defeats the strategy.
