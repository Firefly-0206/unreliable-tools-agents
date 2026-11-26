# D1–D5: planning depth, simple baselines, metrics and calibration cost (simulation)

Results are in `results\D1.json` … `results\D5.json`, plus `results\D3_rows.json`. Figures are in `figs\*.pdf`. Revision 2 re-ran D1–D5 with the same seeds, and every number below is read from those files.

**Setup**
- All numbers are exact expectations, with no episode sampling.
- Each regret is paired against Oracle-DP on the same instance. Brackets are 95% cluster-bootstrap intervals over instances (4,000 resamples).
- Families are RETRY and SHARED from `sim.spec`: K = 3, f = 0.97, q ≤ 0.95, and fault flags are independent of the state. So (A1)–(A4) hold, as the draft appendix states for this generator.
- Default loss: λ_err = 10, λ_sev = 30, c_esc = 4. Budget B = 6 unless stated.
- Instances: 300 per family for D1–D3; 200 per family per loss setting for D4.
- Fitted models use n = 100 labelled episodes in D1, D2 and D4, and n = 30 / 100 / 300 in D3.
- Run times with 4 processes: D1 203 s, D2 216 s, D3 56 s, D4 153 s.
- Seeds 401–404. `sim.py` and `exact.py` were not modified.
- "Source" means the true fault source: tool 0's retries in RETRY, and the hidden backend g shared by tools 0 and 1 in SHARED. Rules that use this grouping get the same structural input as the dependence model, but not its reliabilities. "Per tool" rules use tool identity only.

## D1. When does a misspecified model's regret grow with planning depth?

**Regret vs depth (B = 6)**

| model | RETRY VOI-1 | VOI-2 | VOI-3 | VOI-4 | DP | SHARED VOI-1 | VOI-2 | VOI-3 | VOI-4 | DP |
|---|---|---|---|---|---|---|---|---|---|---|
| Exact | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| MMI | 0.37 [0.28,0.48] | 1.55 | 1.82 | 1.85 | 1.89 [1.66,2.12] | 0.67 [0.54,0.80] | 1.72 | 1.90 | 1.92 | 1.94 [1.75,2.13] |
| Fitted, wrong mechanism | 0.99 [0.86,1.13] | 1.56 | 1.56 | 1.56 | 1.56 [1.39,1.73] | 1.28 [1.14,1.43] | 1.53 | 1.53 | 1.53 | 1.53 [1.37,1.69] |
| Fault-unaware | 2.55 [2.33,2.77] | 2.58 | 2.58 | 2.58 | 2.62 [2.40,2.86] | 1.92 [1.74,2.11] | 1.95 | 1.95 | 1.95 | 1.98 [1.79,2.19] |
| Split shared | same as exact | | | | | 0.28 [0.21,0.35] | 0.52 | 0.62 | 0.65 | 0.68 [0.60,0.77] |

**Regret vs budget, written as DP / VOI-1**

| model | RETRY B=2 | B=4 | B=6 | B=8 | SHARED B=2 | B=4 | B=6 | B=8 |
|---|---|---|---|---|---|---|---|---|
| MMI | 0.60/0.27 | 1.72/0.36 | 1.89/0.37 | 1.92/0.38 | 0.74/0.48 | 1.84/0.66 | 1.94/0.67 | 1.95/0.67 |
| Fitted, wrong mechanism | 1.41/1.03 | 1.55/0.99 | 1.56/0.99 | 1.58/0.99 | 1.44/1.27 | 1.53/1.28 | 1.53/1.28 | 1.53/1.28 |
| Fault-unaware | 2.52/2.52 | 2.58/2.55 | 2.62/2.55 | 2.62/2.55 | 1.91/1.91 | 1.95/1.92 | 1.98/1.92 | 1.98/1.92 |
| Split shared | same as exact | | | | 0.51/0.29 | 0.66/0.28 | 0.68/0.28 | 0.69/0.28 |

**Optimism gap at B = 6**, written as VOI-1 → DP, RETRY / SHARED:
- MMI: 0.42 → 2.06 / 0.81 → 2.24
- Fitted, wrong mechanism: 1.39 → 1.98 / 1.88 → 2.15
- Fault-unaware: 4.10 → 4.21 / 3.41 → 3.50
- Split shared: 0 → 0 / 0.37 → 0.90

**Where the extra regret sits (B = 6).** Amplification is each instance's DP regret minus its VOI-1 regret. "Zero planned repeats" means the model's own DP plan has expected repeat-source calls ≤ 1e-9 under its own model. The old 0.05 threshold gives the same bucket means to 3 decimals.

| model | all instances | plan has repeats | zero planned repeats | share with zero |
|---|---|---|---|---|
| MMI RETRY | 1.52 [1.29,1.74] | 2.11 [1.84,2.38] | 0.000 [0.000,0.000] | 28% |
| MMI SHARED | 1.27 [1.08,1.47] | 1.50 [1.27,1.71] | 0.000 [0.000,0.000] | 15% |
| Wrong mechanism RETRY | 0.57 [0.42,0.72] | 0.79 [0.58,1.01] | 0.004 [0.000,0.013] (one instance 0.37) | 29% |
| Wrong mechanism SHARED | 0.25 [0.15,0.36] | 0.30 [0.18,0.43] | 0.000 | 18% |
| Split shared SHARED | 0.40 [0.32,0.48] | 0.52 [0.43,0.61] | 0.000 | 23% |
| Fault-unaware RETRY / SHARED | 0.08 [0.03,0.16] / 0.06 [0.02,0.13] | | 0.000 / 0.000 | 4% / 4% |

**Spearman correlation with amplification (B = 6; RETRY / SHARED).**
- `ov_tool0` is the revision-1 feature: same-tool retry of tool 0 from the prior.
- `ov_max` is the maximum over all repeat pairs, including cross-tool pairs.
- `ov_plan` is the model-minus-truth value of the repeat probes executed along the DP plan's reachable trajectories.

| model | ov_tool0 | ov_max | ov_plan | expected planned repeats | any planned repeat (0/1) |
|---|---|---|---|---|---|
| MMI | 0.12 / −0.18 | 0.12 / −0.19 | 0.71 / 0.45 | 0.61 / 0.40 | 0.50 / 0.28 |
| Wrong mechanism | 0.38 / 0.23 | 0.39 / 0.17 | 0.57 / 0.37 | 0.40 / 0.13 | 0.22 / 0.00 |
| Split shared | — / undefined (feature is 0) | — / 0.30 | — / 0.46 | — / 0.51 | — / 0.29 |
| Fault-unaware | 0.29 / 0.31 | 0.29 / 0.33 | 0.24 / 0.27 | 0.25 / 0.28 | 0.01 / 0.00 |

What D1 shows:
1. **Amplification sits on trajectories with planned repeats.** With zero planned repeats, amplification is exactly 0 for MMI, split-shared and fault-unaware. For wrong-mechanism RETRY it is 0.004 [0.000, 0.013]. For MMI this follows from the identity in point 7, so it is not independent evidence.
2. **Over-valuation predicts amplification once it is measured along the plan.**
   - The revision-1 "weak predictor" result came from a feature that only looked at a same-tool retry.
   - The plan-weighted `ov_plan` is the best predictor for MMI (0.71 / 0.45) and for the wrong-mechanism model (0.57 / 0.37).
   - For split-shared, expected planned repeats (0.51) is slightly ahead of `ov_plan` (0.46).
   - The 0/1 "any repeat" indicator is weak (0.00–0.50).
3. **Most of the jump is between VOI-1 and VOI-2, and it levels off by VOI-3/4.**
4. **The budget limits it.** MMI amplification is 0.32 / 0.27 at B = 2 and 1.18–1.55 at B ≥ 4.
5. **Fault-unaware amplification is at most 0.08.**
6. **The optimism gap grows with depth**, for example 0.81 → 2.24 for MMI in SHARED.
7. **MMI identity.** With repeats forbidden, MMI-DP equals the exact model's DP (0.031 / 0.043). Under MMI, all extra regret comes from trajectories with repeat-source calls.

Regret saturates with depth. There is no general monotone-growth theorem.

## D2. The dedup operations, separated (DP; RETRY / SHARED regret)

- **Naive discard:** the belief ignores repeat-source reports, but the planner does not know this. This is a strawman.
- **Consistent discard:** the planner knows a repeat is worthless, so it never buys one. It is numerically identical to "both" in every cell.
- **Forbid:** repeat-source probes are never planned or executed.
- **Group+forbid:** all reports are kept but pooled per source into one report's weight, plus forbid.

**Fresh episode**

| model | neither | naive discard | forbid = both = consistent discard = group+forbid |
|---|---|---|---|
| Exact | 0.00 / 0.00 | 0.25 [0.21,0.30] / 0.32 [0.27,0.37] | 0.03 / 0.04 |
| Fitted dependence | 0.09 / 0.065 | 0.36 / 0.38 | 0.11 / 0.11 |
| Fitted independent | 1.64 / 1.79 | 2.08 / 2.33 | 0.11 / 0.11 |
| MMI | 1.99 / 2.01 | 2.15 / 2.73 | 0.03 / 0.04 |
| Fault-unaware | 2.62 / 1.96 | 4.09 / 3.50 | 1.97 / 1.59 |

In SHARED, first report per tool scores: fitted independent 0.40 [0.32, 0.48], MMI 0.35, exact 0.03, fitted dependence 0.09.

**Inherited history** (free calls A,A,A in RETRY; A,B,A in SHARED)

| model | neither | naive discard | forbid | both | group+forbid | SHARED per tool: both / group+forbid |
|---|---|---|---|---|---|---|
| Exact | 0.00 / 0.00 | 1.08 / 1.18 | 0.02 / 0.04 | 0.49 [0.47,0.51] / 0.56 [0.54,0.58] | 0.23 / 0.34 | 0.37 / 0.20 |
| Fitted dependence | 0.14 [0.11,0.17] / 0.10 [0.08,0.12] | 1.19 / 1.24 | 0.15 / 0.13 | 0.58 / 0.61 | 0.32 / 0.39 | 0.43 / 0.28 |
| Fitted independent | 2.19 / 1.92 | 3.37 / 3.74 | 1.68 / 1.48 | 0.57 / 0.61 | 0.34 / 0.42 | 1.03 / 0.87 |
| MMI | 2.50 / 1.97 | 3.63 / 4.00 | 1.73 / 1.55 | 0.49 / 0.56 | 0.25 / 0.36 | 1.01 / 0.86 |
| Fault-unaware | 2.21 / 1.65 | 4.86 / 4.42 | 2.22 / 1.67 | 2.30 / 2.00 | 1.69 / 1.25 | 1.97 / 1.80 |

What D2 shows:
1. **Fresh episodes:** all repeat-avoiding variants are the same policy. Table 1's "Oracle-model Dedup" is purely a planning intervention.
2. **Naive discard hurts only because the planner keeps buying repeats it then ignores.** In fresh RETRY, MMI makes 2.83 calls, 2.05 of them repeats, against 1.60 calls with neither. Fault-unaware makes 3.0 calls, 2.0 of them repeats. A consistent planner never buys repeats.
3. **Inherited history:**
   - Forbid fixes the plan but not the belief. The misspecified models still score 1.48–1.73.
   - Discarding inherited repeats throws away information: even the exact model loses 0.49–0.56.
   - Pooling per source halves that for the misspecified models (0.25–0.42), but needs the true grouping. Per tool it only reaches 0.86–0.87.
   - With a correct model, forbid-only is as good as keeping everything, or better: exact 0.02 / 0.04; fitted dependence 0.15 / 0.13 against 0.14 / 0.10.
   - The fitted dependence model with nothing removed (0.14 / 0.10) beats every variant that discards or pools history, for any model, by about 0.10 / 0.11. It beats every misspecified model under every variant by at least 0.11 / 0.26.
   - The pilot items in `../diag_items.py` (`EVIDENCE`: A,A / A,A,A / A,A,A' / shared A,B,C) present this kind of inherited history. The pilot itself is not re-analysed here.

## D3. Same labelled data (DP, fresh episodes)

| model | RETRY n=30 | 100 | 300 | SHARED n=30 | 100 | 300 |
|---|---|---|---|---|---|---|
| Dependence model | 0.31 [0.24,0.39] | 0.07 [0.05,0.09] | 0.02 [0.02,0.03] | 0.30 [0.24,0.37] | 0.11 [0.08,0.14] | 0.02 [0.02,0.03] |
| Independent (α = 0.5) | 1.18 | 1.47 | 1.84 | 1.56 | 1.97 | 2.14 |
| Independent (α = 0.001) | 1.61 | 1.66 | 1.92 | 1.99 | 2.06 | 2.20 |
| Independent, VOI-1 | 0.62 | 0.41 | 0.39 | 0.84 | 0.76 | 0.75 |
| Independent + first report per source (= forbid) | 0.25 [0.20,0.30] | 0.09 [0.08,0.11] | 0.06 [0.05,0.07] | 0.29 [0.23,0.35] | 0.15 [0.12,0.19] | 0.08 [0.06,0.09] |
| Independent + first report per tool | same as per source | | | 0.58 [0.47,0.69] | 0.41 [0.33,0.49] | 0.33 [0.26,0.40] |
| Independent + naive discard | 1.57 | 1.63 | 1.91 | 2.16 | 2.43 | 2.59 |
| Symmetric-q, no faults | 1.59 | 1.69 | 1.88 | 1.72 | 1.71 | 1.71 |
| Fault-unaware (true q) | 2.63 | | | 2.00 | | |

**Paired gain of the dependence model over first report, same n** (n = 30 / 100 / 300):
- RETRY: −0.06 [−0.13, 0.00] / 0.02 [0.00, 0.04] / 0.04 [0.03, 0.05]
- SHARED, per source: −0.01 [−0.06, 0.03] / 0.05 [0.03, 0.07] / 0.05 [0.04, 0.06]
- SHARED, per tool: 0.27 [0.18, 0.38] / 0.30 [0.23, 0.37] / 0.31 [0.24, 0.38]

What D3 shows:
1. **The independent model gets worse as n grows**, approaching MMI's level (1.89–1.99 in D1/D2). Smoothing explains only part of this: with α = 0.001 the trend remains (1.61 → 1.92). Smoothing and estimation noise both shrink the effective per-call reliability.
2. **First report per source nearly matches the dependence model in fresh episodes**, but only if the agent knows the true source grouping. It does not need the reliabilities. Keyed on tool identity, it double counts the shared backend in SHARED and loses 0.27–0.31 per episode.
3. **The dependence model wins clearly in four places:**
   - inherited histories (D2);
   - hidden shared backends;
   - rare faults: oracle dedup regret 0.24 at φ = 0.02 against 0.02 at φ = 0.4 in `../results/X5.json`;
   - extreme loss asymmetry (D4).

## D4. Several metrics (n = 100, 200 instances)

**Main setting** (λ_sev/λ_err = 3, c_esc = 4; RETRY / SHARED)

| policy | regret | severe-error rate | escalation rate | calls | decision loss |
|---|---|---|---|---|---|
| Oracle-DP | 0 / 0 | .007 / .009 | .66 / .64 | 0.69 / 0.70 | 3.02 / 3.01 |
| VOI-1, exact | 0.01 / 0.00 | .006 / .007 | .70 / .67 | 0.56 / 0.63 | 3.18 / 3.07 |
| DP, MMI | 1.99 / 2.16 | .087 / .100 | .28 / .10 | 1.60 / 1.92 | 4.16 / 3.95 |
| DP, fault-unaware | 2.53 / 1.89 | .150 / .126 | .00 / .00 | 1.54 / 1.53 | 4.76 / 4.06 |
| DP, oracle dedup | 0.03 / 0.04 | .006 / .007 | .69 / .67 | 0.54 / 0.58 | 3.22 / 3.16 |
| DP, fitted dependence | 0.07 / 0.09 | .009 / .013 | .64 / .61 | 0.71 / 0.72 | 3.07 / 3.08 |
| DP, fitted independent | 1.78 / 1.68 | .077 / .078 | .36 / .25 | 1.39 / 1.60 | 4.15 / 3.78 |
| Fitted independent + first report per source | 0.11 / 0.12 | .007 / .010 | .69 / .65 | 0.54 / 0.60 | 3.28 / 3.22 |
| Fitted independent + first report per tool | 0.11 / 0.40 | .007 / .024 | .69 / .58 | 0.54 / 0.75 | 3.28 / 3.35 |
| Fixed-3 | 1.74 / 1.64 | .014 / .018 | .50 / .43 | 3.00 / 3.00 | 2.51 / 2.34 |
| Thr-0.9 | 0.80 / 1.10 | .023 / .020 | .09 / .13 | 2.36 / 2.29 | 1.36 / 1.42 |

**Across all 12 loss settings** (RETRY / SHARED)

| policy | mean regret | max regret | settings within 0.1 of Oracle |
|---|---|---|---|
| VOI-1, exact | 0.06 / 0.03 | 0.27 / 0.17 | 9 / 10 |
| DP, MMI | 3.73 / 3.94 | 27.8 / 29.5 | 4 / 4 |
| DP, fault-unaware | 9.87 / 7.99 | 41.7 / 34.5 | 2 / 2 |
| DP, oracle dedup | 0.07 / 0.03 | 0.27 / 0.11 | 8 / 11 |
| DP, fitted dependence | 0.06 / 0.05 | 0.16 / 0.14 | 9 / 10 |
| DP, fitted independent | 3.42 / 3.58 | 24.5 / 24.1 | 4 / 4 |
| First report per source | 0.12 / 0.09 | 0.46 / 0.19 | 6 / 6 |
| First report per tool | 0.12 / 0.38 | 0.46 / 1.30 | 6 / 6 |
| Fixed-3 | 2.12 / 1.97 | 2.70 / 2.35 | 0 / 0 |
| Thr-0.9 | 2.85 / 2.88 | 8.12 / 7.67 | 0 / 0 |

What D4 shows:
1. **Symmetric loss** (λ_sev = λ_err, c_esc = 4–8): fault-unaware regret is only 0.02–0.09, while MMI and the fitted independent model lose 0.36–0.48.
2. **At c_esc = 2:**
   - Oracle escalates 94–100% of the time.
   - The other model-based DP/VOI policies, except fault-unaware, escalate 91–100%.
   - Fixed-3 escalates 29–67% and Thr-0.9 10–19%.
3. **Misspecified models lose through severe errors and extra calls.**
   - In the main setting and at c_esc = 8 with λ_sev/λ_err ≥ 3, their severe-error rate is 7.7–13%, against 0.1–4.1% for Oracle.
   - In those settings they make 1.3–3.7× Oracle's calls.
   - At c_esc = 4 with λ_sev/λ_err ≥ 10 they mostly escalate (54–80%). Their severe-error rate is ≤3.1% and regret is 0.16–3.18 (MMI 0.16 at sev30_esc4 RETRY).
   - At c_esc = 2 with λ_sev/λ_err ≥ 3, their severe-error rate is 0.
   - At λ_sev/λ_err = 30 and c_esc = 8, regret is 24–30.
4. **First report per source trades accuracy for fewer calls and more escalation.**
   - At RETRY, λ_sev/λ_err = 30, c_esc = 8 it escalates 29% vs 15% for the fitted dependence model, makes 0.67 vs 1.30 calls, and has regret 0.46 vs 0.16.
   - First report per tool, in SHARED at c_esc = 8 with λ_sev/λ_err ≥ 3, has regret 0.98–1.30, against 0.09–0.14 for the fitted dependence model.

## D5. Calibration cost

- Probe cost per labelled episode is 9 in RETRY and 15 in SHARED, plus c_label for labelling.
- The break-even point is calibration cost divided by the paired gain per deployment episode.

| comparison | n | RETRY gain/episode | break-even, c_label = 0 / 4 / 30 | SHARED gain/episode | break-even, c_label = 0 / 4 / 30 |
|---|---|---|---|---|---|
| vs fault-unaware | 30 | 2.32 [2.09,2.55] | 116 / 168 / 504 | 1.70 [1.49,1.90] | 265 / 336 / 796 |
| | 100 | 2.56 | 351 / 507 / 1,522 | 1.89 | 793 / 1,005 / 2,379 |
| | 300 | 2.61 | 1,036 / 1,496 / 4,487 | 1.97 | 2,279 / 2,887 / 6,838 |
| vs first report per source, same n | 30 / 100 / 300 | −0.06 / 0.02 / 0.04 | never / 0 / 0 | −0.01 / 0.05 / 0.05 | never / 0 / 0 |
| vs first report per tool, same n | 30 / 100 / 300 | same as per source | | 0.27 / 0.30 / 0.31 | 0 |
| vs first report per source at n = 30 | 100 | 0.18 | 3,479 / 5,026 / 15,078 | 0.18 | 5,808 / 7,357 / 17,425 |
| | 300 | 0.23 | 10,718 / 15,482 / 46,446 | 0.26 | 15,370 / 19,469 / 46,111 |

What D5 shows:
1. **Against an agent that ignores faults, calibration pays back within about 10² to 10³ deployment episodes.**
2. **Against first report per source, the dependence model barely pays in fresh episodes.** Its case rests on:
   - inherited histories (D2);
   - hidden shared backends (0.27–0.31 per episode at no extra data cost);
   - extreme loss asymmetry (D4).

## Implications for the paper
- **Explain the DP > VOI-1 effect with D1:**
  - no extra regret without planned repeats;
  - plan-weighted over-valuation predicts the amplification;
  - most of the jump is at depth 2;
  - the budget limits it;
  - the optimism gap grows with it.

  Add the MMI identity as a short proposition. Do not claim that regret grows monotonically with depth.
- **Split "dedup" into its operations:**
  - Forbidding future repeats is a planning fix, and it is enough when the history is clean.
  - Discarding existing repeats is a belief fix that throws away information.
  - Pooling per source is a cheaper belief fix, but it needs the source grouping.
  - Evaluate discard only with a planner that knows its own update rule.
- **Add the strongest simple baselines:** first report per source, and group+forbid. State that both need the true source grouping.
- **Report several metrics:** severe-error rate, escalation rate and calls alongside total loss. Say that the effect depends on λ_sev/λ_err and c_esc.
- **Limitations:**
  - two synthetic families, with f = 0.97;
  - the inherited prefix (3 free calls) is chosen by us;
  - the fault-unaware baseline is given the true q;
  - c_label is a free parameter;
  - this directory does not cover the f vs q boundary, faults correlated with state, or partial-sharing stress tests.


