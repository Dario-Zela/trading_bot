# ml-challenger model changelog — us

- 2026-07-30: initial deployment (v2, rolling-250 after the depth-vs-recency ablation): pooled OOS IC 0.0054, Fisher-z LB 0.0023, noise floor 0.0026, n=406,260, log-loss 1.4629; horizon heads h2/h3/h5 IC 0.0016/-0.0007/0.0281.
- 2026-07-30 (spec v2): feature pipeline fix — rolling windows over each ticker's own sessions (cross-venue holiday NaN cascade resolved); retrained rolling-250. Pooled OOS IC 0.0066, Fisher-z LB 0.0035, noise floor 0.0026, n=406,260, log-loss 1.4633. Ship gate: PASSES (beats momentum, logistic, MLP, uniform).
