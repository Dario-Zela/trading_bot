# Commodity-momentum — final selection prompt

Given the dollar index, yield-curve (for real rates), macro view, commodity-price snapshot, and per-ETF technicals, pick today's final commodity-ETF positions.

Hard constraints:
- Up to `max_positions` picks (default 3)
- Allocations between `min_position_gbp / capital × 100` and `max_position_pct`
- Total ≤ 100% — return `[]` when no commodity has a clean momentum or macro setup

Sizing guidance:
- Commodities are volatile — wider stops (-4% default) and longer take-profit targets (+8%)
- The dollar is the single biggest cross-asset driver: strong dollar = headwind for all commodities priced in USD
- Gold (GLD/SLV) responds to real yields, not nominal — falling real rates is the bullish setup. Geopolitical stress is a second-order bullish driver.
- Oil (USO) is supply/demand driven — OPEC moves, geopolitical disruption, demand surprises
- **Contango drag** on ETFs like USO is real — don't hold through sideways/choppy underlying; momentum needs to be active

For each pick:

```json
{
  "ticker": "GLD",
  "allocation_pct": 40.0,
  "stop_loss_pct": -4.0,
  "take_profit_pct": 8.0,
  "thesis": "<1-2 sentences citing (1) the macro/cross-asset signal (dollar, real rates, geopolitics) and (2) the technical confirmation>"
}
```

If commodities are in a chop or the dollar setup is unclear, return `[]`. Forcing a trade against contango drag is value-destructive.
