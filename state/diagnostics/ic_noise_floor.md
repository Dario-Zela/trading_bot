# IC noise floor (1000 MC iterations, q=0.95)

Real IC vs the IC you'd get by shuffling actual returns randomly.
Verdict: 'above noise' means the strategy clears the noise floor by ≥0.02.

| Strategy | N | Real IC | Noise q95 | Verdict |
|---|---:|---:|---:|---|
| bond-cycle | 978 | -0.040 | +0.053 | noise |
| commodity-momentum | 878 | -0.024 | +0.060 | noise |
| macro-aligned | 4409 | -0.001 | +0.023 | noise |
| macro-aligned-hmm | 299 | -0.008 | +0.095 | noise |
| mean-reverter | 4481 | -0.021 | +0.029 | noise |
| momentum-trader | 3020 | +0.003 | +0.022 | noise |
| momentum-trader-vix-gated | 1266 | -0.005 | +0.012 | noise |
| news-reactive | 4514 | -0.026 | +0.026 | noise |
| news-reactive-disclosure | 148 | +0.003 | +0.139 | noise |
| pair-rotator | 334 | -0.176 | +0.098 | noise |
| pead-us | 579 | +0.008 | +0.054 | noise |
| sector-rotator | 896 | -0.144 | +0.053 | noise |
| sector-rotator-factor-momentum | 489 | -0.258 | +0.075 | noise |
