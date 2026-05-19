# IC noise floor (500 MC iterations, q=0.95)

Real IC vs the IC you'd get by shuffling actual returns randomly.
Verdict: 'above noise' means the strategy clears the noise floor by ≥0.02.

| Strategy | N | Real IC | Noise q95 | Verdict |
|---|---:|---:|---:|---|
| bond-cycle | 42 | -0.021 | +0.230 | noise |
| commodity-momentum | 34 | +0.930 | +0.272 | above noise |
| macro-aligned | 720 | -0.058 | +0.058 | noise |
| mean-reverter | 720 | +0.015 | +0.057 | noise |
| momentum-trader | 720 | +0.267 | +0.061 | above noise |
| news-reactive | 720 | -0.050 | +0.062 | noise |
| sector-rotator | 74 | +0.150 | +0.193 | noise |
