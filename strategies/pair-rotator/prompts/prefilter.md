# Pair rotator — universe prefilter

You are filtering for a long-short sector pair strategy. Each "pair"
is one LONG sector ETF + one SHORT/HEDGE position that neutralises
broad market beta.

Filter the input universe (eu_etfs_sector + relevant hedges) down to
~50 names according to:

1. **Sector ETFs** (the long-leg candidates) — keep all GBP-
   denominated LSE-listed sector ETFs from `eu_etfs_sector`:
   energy, materials, industrials, financials, consumer staples,
   consumer discretionary, healthcare, technology, utilities,
   communications, real estate (one ETF per sector).
2. **Hedge candidates** (the short-leg / market-neutral leg) — keep:
   - Inverse / short ETFs available in the universe (e.g.
     short-FTSE, short-S&P, short-Euro-Stoxx if available)
   - Defensive hedges: long gilts (e.g. SAGG.L, SEGA.L) and
     physical gold ETFs as a risk-off hedge against a long sector
   - Volatility / VIX-style products if any are present in the
     T212 ISA catalog
3. **Liquidity gate.** Drop any ETF with 20-day average dollar
   volume below £1M — illiquid hedges are useless because slippage
   eats the spread.
4. **Pairs that don't pair.** Drop ETFs whose risk profile makes
   them unsuitable as either leg of a pair (e.g. exotic single-
   country EM ETFs where the long-leg sector doesn't have a clean
   hedge).

Output the filtered universe as a JSON list of tickers, sorted by
liquidity (most liquid first) within each bucket (sector ETFs first,
then hedges).
