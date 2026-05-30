# Pair rotator — final selection prompt

You have the per-pair analyses. Choose up to `max_positions / 2`
pairs to enter today.

## Selection rules

1. **One pair per portfolio-level theme.** Don't build two pairs
   that both express the same view (e.g. "long tech / short
   utilities" AND "long communications / short utilities"). The
   diversification is illusory.
2. **Net beta neutrality.** The PORTFOLIO net beta — summed across
   every pair you select — should be ≤ 0.5. This sleeve's job is
   uncorrelated return, not levered market exposure.
3. **Sizing.** Each LONG leg sized at `max_position_pct`. Each
   HEDGE leg sized so the £-notional equals the long (1:1
   notional pairing — NOT 1:1 beta — because some of the hedges
   are leveraged inverse products whose £-notional already carries
   higher beta).
4. **Hold target.** Default 10-15 trading days. Set `hold_days`
   per pair:
   - High-conviction sector-vs-sector spread → 15 days
   - Pure-beta-hedge pair (long + broad inverse) → 10 days
     because the spread edge is smaller and decay matters more
5. **Skip days are fine.** If no pair clears the conviction bar
   today, return an empty list. The slate's other strategies are
   long-only; not adding correlation on top of that is the win.

## Sanity checks before emitting

- For each pair, confirm: long_ticker_ccy == hedge_ticker_ccy.
  Cross-currency pairs introduce FX noise that swamps the spread.
- For each hedge, confirm it's not a leveraged inverse held > 10
  days (decay risk).
- Print the portfolio net beta in `notes` so the reader can audit.

## Output

JSON list of pairs with: long_ticker, long_size_pct, hedge_ticker,
hedge_size_pct, pair_thesis, hold_days, expected_spread_pct, and
notes (incl. portfolio net beta estimate).
