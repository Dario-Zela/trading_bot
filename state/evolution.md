
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

- **momentum-trader@us** · `demote` · ✅ applied — 0 trades across 297 graded predictions — dormant execution threshold triggered (n_predictions ≥ 150, n_trades ≤ 5); missing prefilter_sort_key means wrong stocks surface and the entry pipeline never fires. Freeing Alpaca slot 1.
  - details: `{"from_tier": "alpaca-paper", "slot_cleared": true, "previous_slot": 1, "slot_kind": "alpaca"}`
- **momentum-trader@uk-eu** · `demote` · ✅ applied — 1 trade across 296 predictions with 0% hit rate and -162.7 GBP P&L — dormant execution, meets_demotion_criteria=true. Freeing T212 slot.
  - details: `{"from_tier": "trading212-paper", "slot_cleared": false, "previous_slot": 1, "slot_kind": "t212"}`
- **mean-reverter** · `unmark-tier2-candidate` · ✅ applied — T2 thesis (IC 0.19, 60% hit rate) has completely inverted: US arm IC collapsed to -0.281 at n=294 — actively anti-predictive (>4 SE below zero), not noise. The empirical basis for candidacy is gone.
- **mean-reverter@us** · `demote` · ✅ applied — IC -0.281 at n=294 is strongly negative, meets_demotion_criteria=true. T2 thesis has failed; freeing Alpaca slot 2.
  - details: `{"from_tier": "alpaca-paper", "slot_cleared": true, "previous_slot": 2, "slot_kind": "alpaca"}`
- **mean-reverter@uk-eu** · `demote` · ✅ applied — T2 basis removed with US arm collapse; UK-EU IC 0.064 is noise at n=297; meets_demotion_criteria=true. Freeing T212 slot.
  - details: `{"from_tier": "trading212-paper", "slot_cleared": false, "previous_slot": 1, "slot_kind": "t212"}`
- **bond-cycle** · `tune` · ✅ applied — Attribution shows prefilter_mode=python at IC +0.391 vs current null/unknown at IC -0.269 — a 0.66 IC point swing, the single most actionable lever in this week's data. dollar_volume_desc is correct for eu_etfs_bond (liquid bond ETFs, not microcap discovery).
  - details: `{"applied": {"prefilter_mode": "python", "prefilter_sort_key": "dollar_volume_desc", "prefilter_top_n": 100}, "rejected": {}}`
- **news-reactive** · `tune` · ✅ applied — IC 0.303 confirms genuine signal but avg_pnl_pct is only -0.041% across 10 trades — cost bleed is eroding a strong prediction edge. A 2× cost gate filters marginal entries and should convert IC quality to net P&L without disrupting the underlying signal.
  - details: `{"applied": {"cost_gate_multiplier": 2.0}, "rejected": {}}`
- **news-reactive** · `mark-tier2-candidate` · ✅ applied — IC 0.303 at n=299 is the only strategy meeting shadow T2 eligibility (IC ≥ 0.25, n ≥ 200). All prior T2 candidates unmarked this week — the leaderboard needs a sole leader and this is the only evidence-grade signal in the slate.
  - details: `{"tier2_marked_at": "2026-05-30", "thesis_present": true}`
- **momentum-trader** · `spawn-variant` · ✅ applied — Parent was dormant (0 trades / 297 predictions) due to missing prefilter_sort_key; this variant fixes the entry pipeline with abs_return_5d and adds a VIX regime gate (Mozes 2026) that breaks the trigger axis — regime-conditional vs always-on capital deployment.
  - details: `{"variant_id": "momentum-trader-vix-gated", "addendum_applied": true}`
- **sector-rotator** · `spawn-variant` · ✅ applied — Breaks the signal-source axis: scores sector ETFs by momentum-of-momentum (Tai et al. 2026, SSRN 6224058) rather than raw relative strength, with residual-reversal smoothing (Gao et al. 2026, SSRN 6371558). Parent's IC -0.08 is consistent with buying crowded late-cycle sectors that meta-momentum correctly de-ranks.
  - details: `{"variant_id": "sector-rotator-factor-momentum", "addendum_applied": true}`
- **commodity-momentum@uk-eu** · `keep` · ✅ applied — IC 0.174 building at n=110; prefilter attribution samples are too small (≤6 days each) to act on. No lever identified yet.
- **macro-aligned@us** · `keep` · ✅ applied — IC 0.213 approaching but below 0.25; all prefilter modes show negative IC in attribution (likely pooling artefact) — no clear tune lever identified.
- **macro-aligned@uk-eu** · `keep` · ✅ applied — meets_demotion_criteria=false; IC 0.154 borderline, P&L -149.72 GBP but no confirmed second-consecutive-week trigger. Occupying T212 slot on a thin margin — on notice for next week.
- **control-rule-based@us** · `keep` · ✅ applied — Control baseline; shadow tier, no slot to free. Held as benchmark despite 10% hit rate.
- **control-rule-based@uk-eu** · `keep` · ✅ applied — Control baseline; shadow tier, no slot to free.
- **sector-rotator@uk-eu** · `keep` · ✅ applied — n_predictions=88 < 150 dormant threshold; 1 trade (100% hit) is not actionable. Attribution confirms LLM prefilter is the right mode (IC +0.114 vs +0.068); current config is correct. Spawning factor-momentum variant in parallel.
