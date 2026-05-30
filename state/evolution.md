
## Weekly evolution — 2026-05-23

- **news-reactive@us** · `demote` · ✅ applied — System confirms demotion criteria met: IC=0.02 (noise at n=899), hit_rate=31%, total_pnl=-£52; Li et al. 2026 (FINSABER) confirms LLM timing systematically underperforms passives across broad cross-sections — structural failure, not a slump.
  - details: `{"from_tier": "alpaca-paper", "slot_cleared": true, "previous_slot": 3, "slot_kind": "alpaca"}`
- **news-reactive@uk-eu** · `demote` · ✅ applied — System confirms demotion criteria met: worst performer in the slate (total_pnl=-£229, IC=-0.127, hit_rate=25%); demoting also frees the T212 slot and brings T212-paper deployed capital from £50k to £40k, back within the budget ceiling.
  - details: `{"from_tier": "trading212-paper", "slot_cleared": false, "previous_slot": 1, "slot_kind": "t212"}`
- **momentum-trader** · `tune` · ✅ applied — LLM pre-filter collapses momentum-trader IC from +0.221 (unknown/no-LLM, n=2130) to -0.013 (LLM, n=98) — a delta of -0.234 on the largest attribution sample in the dataset; python mode's |5d return| sort is a natural momentum pre-screen and aligns the pre-filter bias with the strategy's own selection logic.
  - details: `{"applied": {"prefilter_mode": "python"}, "rejected": {}}`
- **mean-reverter** · `tune` · ✅ applied — UK-EU shows avg_pnl_pct=-0.16 despite hit_rate=44% (44% winning trades losing money in aggregate points squarely at cost drag); cost_gate_multiplier=1.5 gates out trades where expected edge doesn't clear 1.5× estimated round-trip cost, directly testing the 2026-05-21 thesis that UK-EU losses are cost-driven rather than signal-driven.
  - details: `{"applied": {"cost_gate_multiplier": 1.5}, "rejected": {}}`
- **macro-aligned** · `tune` · ✅ applied — LLM pre-filter materially worsens macro-aligned IC (-0.243 with LLM vs -0.063 without, n=197 vs 2131); switching to python removes a layer that is actively destroying signal, moving toward the less-bad baseline while the strategy's own macro lens (get_macro_view, get_sector_strength) does the real filtering.
  - details: `{"applied": {"prefilter_mode": "python"}, "rejected": {}}`
- **sector-rotator@us** · `keep` · ✅ applied — Shadow with IC=0.352 and decile_spread=0.644 — genuine positive signal — but no Alpaca slots are available and meets_promotion_criteria=false; hold for slot availability.
- **sector-rotator@uk-eu** · `keep` · ✅ applied — IC=-0.307 is concerning with n=159 predictions, but PnL is positive (+£22) and n_trades=9 is below the 20-trade hit_rate demotion threshold; watch for one more week — if IC doesn't improve it becomes a demotion candidate.
- **mean-reverter@us** · `keep` · ✅ applied — T2 candidate mark intact; hit_rate=0.6 (n=10) and IC=0.137 confirm US signal is real, and LLM pre-filter attribution (+0.077 with LLM vs +0.001 without) shows the pre-filter is working — no change needed here.
- **mean-reverter@uk-eu** · `keep` · ✅ applied — IC=-0.124 at n=1530 is a credible negative reading that puts the cost-driven thesis under pressure; cost_gate tune applied this week is the direct test — if avg_pnl_pct doesn't improve next cycle, thesis fails and T2 mark should be reconsidered.
- **macro-aligned@us** · `keep` · ✅ applied — Shadow with poor fundamentals (hit_rate=22%, IC=-0.114, total_pnl=-£89); prefilter tune is strategy-wide; no demotion action available for shadow tier.
- **macro-aligned@uk-eu** · `keep` · ✅ applied — PnL barely positive (+£0.40), hit_rate=25% at n=8 trades (below 20-trade threshold), meets_demotion_criteria=false; prefilter tune being applied — monitor IC trajectory over next two weeks.
- **momentum-trader@us** · `keep` · ✅ applied — IC=0.305 and decile_spread=3.214 show strong underlying signal; negative total_pnl (-£37) with positive avg_pnl_pct (+0.717%) suggests a few large stop-outs obscuring real alpha — prefilter tune to python should improve trade selection quality.
- **momentum-trader@uk-eu** · `keep` · ✅ applied — IC=-0.154 at n=1431 is a genuine negative signal and hit_rate=11.1% is alarming, but n_trades=9 is below the 20-trade demotion threshold; prefilter switch to python is strategy-wide and may shift UK-EU selection as well — reassess next week when trade count should cross the threshold.
- **bond-cycle@us** · `keep` · ✅ applied — Zero trades placed, IC=0.017 at n=45 predictions — far too early to act in either direction.
- **bond-cycle@uk-eu** · `keep` · ✅ applied — IC=-0.235 with only 4 trades and 99 predictions; early negative reading but insufficient sample for a structural verdict.
- **commodity-momentum@us** · `keep` · ✅ applied — Shadow, IC=0.159 positive but n=3 trades and 45 predictions — far too early; avg_pnl_pct=-1.202 with n=3 is not stable enough to act on.
- **commodity-momentum@uk-eu** · `keep` · ✅ applied — Shadow, IC=0.387 and decile_spread=3.274 are among the strongest early readings in the slate; 3 trades and 73 predictions are insufficient for a T2 candidate mark but this is the one to watch — target a mark at n≥200 predictions with IC holding above 0.25.
- **control-rule-based@us** · `keep` · ✅ applied — Rule-based control benchmark; catastrophic performance (-£2027, hit_rate=12.9%, drawdown=-24.9%) serves as the negative baseline that validates why LLM-driven signal adds value.
- **control-rule-based@uk-eu** · `keep` · ✅ applied — Rule-based control benchmark; keep as baseline comparison regardless of performance.

## Weekly evolution — 2026-05-30

- **mean-reverter** · `unmark-tier2-candidate` · ✅ applied — US IC inverted from +0.19 to −0.281 this window across 294 graded predictions (4 trades); the original thesis — real directional signal masked by UK costs — has not materialised. UK IC is 0.064 / 297 predictions with only 3 trades, far below conviction threshold. Retracting from the leaderboard entirely; no region retains the evidence that justified the mark.
- **news-reactive** · `mark-tier2-candidate` · ✅ applied — After mean-reverter's unmark the leaderboard is empty; news-reactive US holds the field's only conviction-level IC with a 299-prediction sample — clear leader; marking to track toward paper promotion and the live slot.
  - details: `{"tier2_marked_at": "2026-05-30", "thesis_present": true}`
- **bond-cycle** · `tune` · ✅ applied — Attribution shows python IC +0.391 vs null/unknown IC −0.269 — a 0.66-point gap across a reasonable sample; currently running null (the inferior config) for the eu_etfs_bond universe; dollar_volume_desc is the correct sort key for a bond ETF strategy biased toward liquid duration names.
  - details: `{"applied": {"prefilter_mode": "python", "prefilter_sort_key": "dollar_volume_desc"}, "rejected": {}}`
- **momentum-trader** · `tune` · ✅ applied — US has produced 0 trades across 297 graded predictions — making sort key explicit eliminates any null-default ambiguity in candidate ranking; cost_gate=1.5 guards against a recurrence of the UK side's single −14.1% trade (well past the −3% stop, suggesting a gap-down the stop did not catch).
  - details: `{"applied": {"prefilter_sort_key": "abs_return_5d", "cost_gate_multiplier": 1.5}, "rejected": {}}`
- **macro-aligned** · `tune` · ✅ applied — All prefilter modes return negative IC (python −0.120, unknown −0.063, llm −0.243); dollar_volume_desc is documented for ETF universes but macro-aligned runs on individual-stock universes — switching to abs_return_20d surfaces 20-day regime-trending names more consistent with the macro-regime edge and the 2026 cross-asset regime findings (Pagliaro, Electronics 15(6) 1334).
  - details: `{"applied": {"prefilter_sort_key": "abs_return_20d"}, "rejected": {}}`
- **control-rule-based** · `tune` · ✅ applied — US shadow shows −32.4% max drawdown and −2770 GBP paper P&L with no stop loss set; adding stop_loss=−8% caps catastrophic single-position losses while TP=5% preserves room for winning trades — minimum intervention to make shadow metrics interpretable and prevent a single position from dominating the drawdown signal.
  - details: `{"applied": {"stop_loss_pct": -8.0, "take_profit_pct": 5.0}, "rejected": {}}`
- **commodity-momentum@uk-eu** · `keep` · ✅ applied — Null/unknown prefilter already achieves IC +0.439 vs python +0.187 — switching to python would reduce prediction quality; with 5 trades the sample is too thin to tune further.
- **news-reactive@us** · `keep` · ✅ applied — Newly marked T2 leader; shadow tier continues accumulating evidence — monitor meets_promotion_criteria to trigger alpaca-paper promotion.
- **news-reactive@uk-eu** · `keep` · ✅ applied — Positive P&L (+144 GBP) and 83% hit rate are encouraging but IC 0.047 / 294 predictions is near-random — high hit rate on n=6 trades is likely sampling noise; shadow monitoring continues.
- **sector-rotator@uk-eu** · `keep` · ✅ applied — IC −0.08 is near-zero and cost_gate=3.0 is correctly limiting entries; Man Group 2026 regime model supports Energy/Materials overweight consistent with this strategy's thesis — hold through thin trade history before intervening.
- **mean-reverter@us** · `keep` · ✅ applied — T2 candidate status retracted; 4 trades on alpaca-paper is below the n≥20 demotion threshold — continue accumulating trades and let IC stabilise before a demotion or re-mark decision.
- **mean-reverter@uk-eu** · `keep` · ✅ applied — T2 candidate status retracted; 3 trades on T212-paper is far too few to assess; UK IC 0.064 / 297 predictions needs more trade volume before any action.
- **macro-aligned@us** · `keep` · ✅ applied — Strategy-wide sort-key tune applied; shadow tier — monitor IC response over the next 14-day window before further intervention.
- **macro-aligned@uk-eu** · `keep` · ✅ applied — Strategy-wide sort-key tune applied; IC 0.154 / 8 trades on T212-paper is borderline — watch for improvement before meeting demotion threshold at n≥20.
- **control-rule-based@us** · `keep` · ✅ applied — Strategy-wide stop/TP bounds applied to cap the −32.4% drawdown; shadow tier — not a T2 contender in current form.
- **control-rule-based@uk-eu** · `keep` · ✅ applied — Strategy-wide stop/TP bounds applied; shadow tier — not a T2 contender in current form.
- **bond-cycle@uk-eu** · `keep` · ✅ applied — Strategy-wide prefilter tune applied; IC 0.195 / 3 trades in shadow is worth watching — too thin to promote yet but the python-prefilter fix should improve signal routing materially.
