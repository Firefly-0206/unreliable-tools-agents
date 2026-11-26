## Overall by model and presentation

| Model | Presentation | n | Mean abs. error in P(healthy) | Mean signed error | Crosses τ* wrongly | Mean decision loss |
| --- | --- | --- | --- | --- | --- | --- |
| deepseek-chat | plain | 54 | 0.011 | +0.009 | 9% | 0.20 |
| deepseek-chat | warning | 54 | 0.009 | +0.004 | 11% | 0.08 |
| deepseek-chat | dedup | 54 | 0.040 | -0.030 | 30% | 0.37 |
| deepseek-reasoner | plain | 36 | 0.004 | +0.004 | 6% | 0.07 |
| deepseek-reasoner | warning | 36 | 0.001 | +0.000 | 0% | 0.01 |
| deepseek-reasoner | dedup | 36 | 0.037 | -0.036 | 28% | 0.19 |

## By evidence pattern (plain presentation): mean reported P(healthy) vs exact


**deepseek-chat**

| Regime | single | retry2 | retry3 | shared3 | indep3 | retry_mixed |
| --- | --- | --- | --- | --- | --- | --- |
| frequent | 0.74 (exact 0.74) | 0.75 (exact 0.75) | 0.81 (exact 0.74) | 0.74 (exact 0.74) | 0.96 (exact 0.96) | 0.87 (exact 0.87) |
| moderate | 0.81 (exact 0.81) | 0.84 (exact 0.84) | 0.88 (exact 0.83) | 0.83 (exact 0.83) | 0.99 (exact 0.99) | 0.92 (exact 0.92) |
| rare | 0.82 (exact 0.82) | 0.93 (exact 0.90) | 0.90 (exact 0.90) | 0.90 (exact 0.90) | 0.99 (exact 0.99) | 0.90 (exact 0.90) |

**deepseek-reasoner**

| Regime | single | retry2 | retry3 | shared3 | indep3 | retry_mixed |
| --- | --- | --- | --- | --- | --- | --- |
| frequent | 0.74 (exact 0.74) | 0.75 (exact 0.75) | 0.74 (exact 0.74) | 0.74 (exact 0.74) | 0.96 (exact 0.96) | 0.87 (exact 0.87) |
| moderate | 0.87 (exact 0.81) | 0.84 (exact 0.84) | 0.83 (exact 0.83) | 0.83 (exact 0.83) | 0.99 (exact 0.99) | 0.92 (exact 0.92) |
| rare | 0.82 (exact 0.82) | 0.90 (exact 0.90) | 0.90 (exact 0.90) | 0.90 (exact 0.90) | 0.99 (exact 0.99) | 0.90 (exact 0.90) |

## Key contrasts (mean over samples)

| Model | Presentation | Regime | retry3 − single: model / exact | indep3 − shared3: model / exact | Decision loss on retry3+shared3 | on indep3 |
| --- | --- | --- | --- | --- | --- | --- |
| deepseek-chat | plain | frequent | +0.07 / +0.00 | +0.22 / +0.22 | 0.79 | 0.00 |
| deepseek-chat | plain | moderate | +0.07 / +0.02 | +0.16 / +0.16 | 0.37 | 0.00 |
| deepseek-chat | plain | rare | +0.08 / +0.08 | +0.09 / +0.09 | 0.00 | 0.00 |
| deepseek-chat | warning | frequent | +0.00 / +0.00 | +0.22 / +0.22 | 0.00 | 0.00 |
| deepseek-chat | warning | moderate | -0.02 / +0.02 | +0.10 / +0.16 | 0.23 | 0.00 |
| deepseek-chat | warning | rare | +0.08 / +0.08 | +0.09 / +0.09 | 0.00 | 0.00 |
| deepseek-chat | dedup | frequent | +0.07 / +0.00 | +0.22 / +0.22 | 0.63 | 0.00 |
| deepseek-chat | dedup | moderate | +0.00 / +0.02 | +0.18 / +0.16 | 0.28 | 0.00 |
| deepseek-chat | dedup | rare | +0.00 / +0.08 | +0.17 / +0.09 | 0.82 | 0.00 |
| deepseek-reasoner | plain | frequent | +0.00 / +0.00 | +0.22 / +0.22 | 0.00 | 0.00 |
| deepseek-reasoner | plain | moderate | -0.04 / +0.02 | +0.16 / +0.16 | 0.21 | 0.00 |
| deepseek-reasoner | plain | rare | +0.08 / +0.08 | +0.09 / +0.09 | 0.00 | 0.00 |
| deepseek-reasoner | warning | frequent | +0.00 / +0.00 | +0.22 / +0.22 | 0.00 | 0.00 |
| deepseek-reasoner | warning | moderate | +0.02 / +0.02 | +0.16 / +0.16 | 0.00 | 0.00 |
| deepseek-reasoner | warning | rare | +0.08 / +0.08 | +0.09 / +0.09 | 0.00 | 0.00 |
| deepseek-reasoner | dedup | frequent | +0.00 / +0.00 | +0.22 / +0.22 | 0.00 | 0.00 |
| deepseek-reasoner | dedup | moderate | -0.00 / +0.02 | +0.18 / +0.16 | 0.00 | 0.00 |
| deepseek-reasoner | dedup | rare | +0.00 / +0.08 | +0.17 / +0.09 | 0.47 | 0.00 |
