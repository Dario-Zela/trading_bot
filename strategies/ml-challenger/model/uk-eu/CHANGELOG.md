# ml-challenger model changelog — uk-eu

- 2026-07-30: initial deployment (v1.1, rolling-250): pooled OOS IC 0.0192, Fisher-z LB 0.0137, noise floor 0.0048, n=126,340, log-loss 1.2972; horizon heads h2/h3/h5 IC 0.0127/0.0173/0.0346.
- 2026-07-30 (spec v2): feature pipeline fix recovered 245 dropped dates (561 holey -> 744 contiguous feature dates, +32% rows); retrained rolling-250. Honest numbers came DOWN: pooled OOS IC 0.0031 (was 0.0192 on holey data), Fisher-z LB -0.0024, below the 0.005 noise floor. Ship gate: FAILS (loses to momentum 0.0157, MLP 0.0151, logistic 0.0062 on pooled IC; wins log-loss 1.2506 vs 1.457-1.486). Ships at shadow with the loss stated in the card; forward record decides whether the sleeve stays.
