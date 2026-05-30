# Post-earnings drift — universe prefilter

You are filtering the candidate universe for a post-earnings
announcement drift (PEAD) strategy on the US T212-ISA universe.

The strategy ONLY trades stocks that have reported earnings recently.
Filter the input universe down to ~80 names according to:

1. **Earnings reported in the last 1–5 trading days.** This is the
   hard requirement. A stock without a recent earnings release is
   not a PEAD candidate; drop it.
2. **Material reaction.** Prefer names where the post-announcement
   price reaction is ≥ +3% on day 1 (the "drift initiator" signal).
   Negative reactions are out — this sleeve is long-only.
3. **Liquidity floor.** Drop micro-caps where the post-announcement
   gap could be illiquidity-driven rather than reaction-driven. Use
   the standard t212_isa_us liquidity filter; if you're unsure, keep
   only names with 20-day average dollar volume > $10M.
4. **No re-announcement risk inside the hold window.** Drop names
   with another scheduled earnings release inside the next 25
   trading days (rare but possible for early-fiscal-year prints
   that follow a partial period).

Output the filtered universe as a JSON list of tickers, ordered by
how strong the PEAD signal looks on a first pass (largest positive
gap × highest beat magnitude first).
