# PEAD — final selection prompt

You have the per-ticker analyses. Choose up to `max_positions` names
to enter today, sized within position caps.

## Selection rules

1. **Quality bar.** Only enter on high-quality beats: EPS + revenue
   beat AND raised guidance. Reject anything weaker. A clean book
   of 0-2 high-quality names is better than 4-5 mixed-quality ones —
   PEAD edge is small (~25-50bps/month per the literature) and
   easily eaten by a single misjudged entry.
2. **Day-1 hold confirmation.** Prefer names that gapped +3-10% on
   the print AND held the gain through day 2 close. Skip names
   that gapped > +15% (priced in) or have already retraced > 50%
   of the gap.
3. **Diversify by sector.** No more than 2 names from the same
   GICS sector — earnings reactions cluster sector-wide, and we
   want the drift edge, not a sector beta bet.
4. **Hold target.** Default 15-25 trading days. Set `hold_days`
   per name based on conviction:
   - Highest-conviction (clean beat + raised guidance + strong day-2
     follow-through) → 20-25 days
   - Solid beat but ambiguous guidance → 15 days
5. **Stop placement.** -5% from entry. A clean PEAD should not
   retrace meaningfully; if it does, the thesis was wrong, take the
   loss and don't average down.
6. **Take profit.** +8% target. Most PEAD names that complete the
   drift cycle deliver 5-10% in the hold window.

## When to sit out

If no candidate clears the quality bar today, return an empty list.
This strategy is calendar-driven — it should have ZERO trades on
days where no earnings of substance hit. A "trade everything" stance
destroys the edge.

## Output

JSON list of selections with: ticker, conviction (1-10), hold_days,
thesis (one paragraph), and key risks.
