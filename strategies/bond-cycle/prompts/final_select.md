# Bond-cycle — final selection prompt

Given the yield-curve, credit-spread, macro view, and per-ETF technicals, pick today's final bond-ETF positions.

Hard constraints:
- Up to `max_positions` picks (default 3 — bonds are highly correlated, more positions is redundant)
- Allocations between `min_position_gbp / capital × 100` and `max_position_pct`
- Total ≤ 100% — return `[]` if the rate regime is uncertain or no ETF aligns cleanly

Sizing guidance:
- Bonds move slowly — wider holding horizons, tighter stops (-2.5% default) and modest take-profits (+4%) versus equity strategies
- Duration is the main risk dimension: only go long TLT (20Y+) when the macro thesis explicitly calls for falling long rates / disinflation. Short-end exposure (SHY) is the conservative play when uncertain.
- Credit (HYG / LQD) is a separate bet from rates — HYG is risk-on, LQD is rate-sensitive credit

For each pick:

```json
{
  "ticker": "TLT",
  "allocation_pct": 40.0,
  "stop_loss_pct": -2.5,
  "take_profit_pct": 4.0,
  "thesis": "<1-2 sentences citing (1) the rate / yield-curve signal and (2) why this specific duration/credit ETF expresses it best>"
}
```

If the yield curve is flat / range-bound and no clear directional thesis applies, return `[]`. Bond strategies underperform when forced.
