# S1–S5 stress tests: what survives when the theory's assumptions are broken

All five stress tests use 200 instances per cell. After review, S1, S2, S4 and S5 were rerun with the same seeds: the earlier policy numbers were reproduced exactly, and the reruns added within-budget diagnostics and a time-varying MMI control for S4. S3 was unaffected and not rerun. Compute was about 15 minutes for the first run and 12 minutes for the rerun, on 4 workers. No DeepSeek API calls were made.

**Files** (all in `<root>\raea4\stress\`):
- **Code:** `core.py`, `run_stress.py`, `checks.py`, `make_tables.py`
- **Docs:** `README.md`, this `SUMMARY.md`
- **Results:** `results\S1.json` to `results\S5.json` and `results\checks.json`. `checks.json` holds the exclusion checks (unrestricted and within budget, plus the corrected cap for S2), the Prop. 1 aggregates for S1 and the depth profiles.
- **Full tables:** `TABLES.md`. Every cell and policy, with regret and CI, severe-error, escalation, calls, repeat-source calls and probe cost.

**Budget note for the theory diagnostics.** With B=6 and group tools of cost 1, an agent can collect at most **6 group reports**. The Prop. 1 grid (n ≤ 20 for RETRY; n_A, n_B ≤ 10 for SHARED) and the benign-run diagnostics (n ≤ 20) go beyond that. Every such count is therefore given twice: unrestricted, which is the theory statement, and **within budget (n ≤ 6)**, which is what can matter for decisions.

**Setup**
- RETRY and SHARED families from `sim.spec`: K=3, λ_sev=30, λ_err=10, c_esc=4, τ*=0.867, budget B=6.
- Every number is an exact expectation in the true environment, which breaks the assumption being tested.
- Regret is the paired per-instance difference to that environment's Oracle-DP. CIs are 95% cluster-bootstrap over instances.
- All cells of one experiment use the same instances, so cells are paired.
- Table format: **DP regret [CI] / VOI-1 regret**.
- "Oracle dedup" never plans a second call to the same source. In closed loop it is therefore the "forbid future repeats" rule. Discarding existing repeat-source history is never exercised here; testing that needs a forced-history protocol, which I did not run.

## S1 Fault emission boundary (f<q, f=q, f>q)

A faulty tool reports "benign" with probability f. In the "=q" row, f equals each faulty tool's own q (A4 holds with equality).

| f | RETRY fault-unaware | RETRY MMI | RETRY dedup | SHARED fault-unaware | SHARED MMI | SHARED split | SHARED dedup |
|---|---|---|---|---|---|---|---|
| 0.50 | 1.52 [1.41,1.62] / 1.36 | 0.48 [0.40,0.56] / 0.10 | 0.02 | 1.07 [0.98,1.16] / 0.94 | 0.63 [0.57,0.70] / 0.19 | 0.17 [0.14,0.19] / 0.08 | 0.04 |
| 0.70 | 2.03 [1.86,2.20] / 1.87 | 0.76 [0.62,0.89] / 0.13 | 0.03 | 1.48 [1.33,1.63] / 1.35 | 1.01 [0.89,1.13] / 0.27 | 0.38 [0.33,0.44] / 0.15 | 0.04 |
| 0.85 | 2.32 [2.12,2.54] / 2.20 | 1.20 [1.01,1.40] / 0.22 | 0.03 | 1.72 [1.54,1.91] / 1.62 | 1.41 [1.24,1.58] / 0.39 | 0.58 [0.49,0.68] / 0.25 | 0.04 |
| =q | 2.40 [2.18,2.63] / 2.30 | 1.50 [1.28,1.73] / 0.25 | 0.03 | 1.80 [1.60,1.99] / 1.71 | 1.63 [1.43,1.82] / 0.45 | 0.61 [0.51,0.71] / 0.27 | 0.04 |
| 0.95 | 2.47 [2.24,2.71] / 2.39 | 1.68 [1.43,1.92] / 0.30 | 0.03 | 1.85 [1.65,2.06] / 1.78 | 1.80 [1.59,2.01] / 0.49 | 0.70 [0.59,0.80] / 0.24 | 0.04 |
| 0.97 | 2.50 [2.27,2.75] / 2.43 | 1.85 [1.59,2.11] / 0.31 | 0.03 | 1.88 [1.67,2.09] / 1.81 | 1.89 [1.67,2.11] / 0.54 | 0.70 [0.60,0.81] / 0.24 | 0.05 |
| 0.99 | 2.53 [2.30,2.78] / 2.47 | 1.97 [1.69,2.25] / 0.37 | 0.04 | 1.91 [1.70,2.12] / 1.85 | 1.98 [1.75,2.21] / 0.60 | 0.70 [0.60,0.81] / 0.25 | 0.05 |

**MMI regret by planning depth, RETRY** (VOI-1 / VOI-2 / VOI-3 / DP):

| f | VOI-1 | VOI-2 | VOI-3 | DP |
|---|---|---|---|---|
| 0.5 | 0.10 | 0.36 | 0.41 | 0.48 |
| =q | 0.25 | 1.14 | 1.40 | 1.50 |
| 0.97 | 0.31 | 1.45 | 1.79 | 1.85 |

Fault-unaware does not change with depth (f=0.97: 2.43 / 2.50 / 2.50 / 2.50).

**Severe errors, escalation and repeats** (DP, RETRY, f=0.97):

| | Oracle | Fault-unaware | MMI | Dedup |
|---|---|---|---|---|
| Severe-error prob. | 0.008 | 0.149 | 0.081 | 0.006 |
| Escalation prob. | 0.64 | 0.00 | 0.28 | 0.70 |
| Repeat-source calls | 0.07 | 0.56 | 0.81 | 0 |

**Proposition 1, checked numerically** against the model's own likelihood tables. RETRY used n = 1 to 20 reports; SHARED used every (n_A, n_B) in 0..10. Within budget means n ≤ 6 (n_A + n_B ≤ 6).
- **Middle expression:** the exact Bayes factor divided by the middle expression is at most 1.000 in every cell, including f<q. The middle expression is always a valid, tight upper bound.
- **With A4 (f ≥ q):** φ × ratio ≤ 1 everywhere. It reaches 1.000 at f=q, where the bound is attained asymptotically.
- **Without A4:** the 1/φ bound fails in all 200 instances, both on the full grid and within budget. The first failure comes at 2.1 reports on average at f=0.85, 1.4–1.5 at f=0.7 and about 1 at f=0.5.

| f | max φ × ratio, full grid (RETRY / SHARED) | max φ × ratio, within budget (RETRY / SHARED) |
|---|---|---|
| 0.85 | 7.4 / 4.0 | 1.7 / 1.8 |
| 0.7 | 348 / 184 | 5.1 / 5.3 |
| 0.5 | 2.9e5 / 1.5e5 | 37 / 39 |

The very large maxima come from n = 20 (or 10+10) runs that no agent can reach. Within budget the violation is a factor of 1.7–39.

**Exclusion condition** (paper's 1/φ form), checked per instance. It holds in 159/200 RETRY and 137/200 SHARED instances. In those instances, how often does the exact posterior after a benign-only run from the group still reach τ*?

| f | RETRY, any n ≤ 20 | RETRY, n ≤ 6 | SHARED, any n ≤ 20 | SHARED, n ≤ 6 |
|---|---|---|---|---|
| ≥ q | 0/159 | 0/159 | 0/137 | 0/137 |
| 0.85 | 81/159 | 20/159 | 32/137 | 13/137 |
| 0.7 | 159/159 | 117/159 | 137/137 | 98/137 |
| 0.5 | 159/159 | 159/159 | 137/137 | 137/137 |

Near the boundary (f=0.85), most failures need more reports than the budget allows: only 20 of 81 (RETRY) and 13 of 32 (SHARED) happen within 6 calls. The crossing point reaches n=20 (RETRY) and 19 (SHARED) at f=0.85, and 14 / 16 at f=0.7. At f=0.5 every failure happens within 6 calls.

**Interpretation.**
- **The dependence findings do not need A4.** MMI's extra regret, and its growth with planning depth, appear at every f, including f=0.5. The growth from VOI-1 to DP is about 3–6x: 4.7–6.0x in RETRY and 3.3–3.8x in SHARED. The effect grows smoothly with f: faults that look more benign do more damage. Split-shared shows the same pattern at a smaller size.
- **The theory does need A4.**
  - The middle expression of Prop. 1 is always valid.
  - The 1/φ cap fails as soon as f<q, including within budget.
  - The exclusion corollary fails in decision-relevant runs (n ≤ 6) for f ≤ 0.7. Near the boundary (f=0.85) it fails within budget in only 9–13% of the instances where the condition holds (20/159 RETRY, 13/137 SHARED). Most of its failures there need more reports than the budget allows.
  - The corollary should be scoped to faults that lean towards the benign report, with the middle expression given for the general case.

## S2 Fault correlated with the state (A1 broken)

- **Parametrization:** P(z=1 | s≠0) = φ(1+κ) and P(z=1 | s=0) = φ(1 − κ(1−π0)/π0). This keeps the marginal fault rate at φ.
- **Clipping:** κ is clipped per instance so both probabilities stay in [0,1]. The mean clipped κ is −0.50, 0, 0.48 and 0.82 for the four settings. The κ=1 cell is heavily clipped.
- **Agents:** "assume A1" is the paper's exact model with the correct marginal φ. MMI uses the true model's per-call marginal.

| κ | RETRY assume-A1 | RETRY fault-unaware | RETRY MMI | SHARED assume-A1 | SHARED fault-unaware | SHARED MMI | dedup (RETRY / SHARED) |
|---|---|---|---|---|---|---|---|
| −0.5 | 0.18 [0.15,0.21] / 0.22 | 0.85 [0.73,0.98] / 0.84 | 1.15 [1.02,1.29] / 1.06 | 0.16 [0.12,0.20] / 0.19 | 0.55 / 0.55 | 0.80 [0.69,0.91] / 0.77 | 0.06 / 0.07 |
| 0 | 0.00 / 0.01 | 2.61 [2.37,2.86] / 2.59 | 1.92 [1.68,2.18] / 0.47 | 0.00 / 0.01 | 1.95 / 1.94 | 2.10 [1.87,2.34] / 0.64 | 0.03 / 0.04 |
| 0.5 | 0.08 [0.06,0.10] / 0.05 | 4.52 [4.19,4.87] / 4.48 | 0.43 [0.27,0.62] / 0.12 | 0.08 [0.05,0.11] / 0.06 | 3.49 / 3.46 | 1.01 [0.78,1.25] / 0.16 | 0.01 / 0.02 |
| 1.0 | 0.20 [0.15,0.25] / 0.14 | 5.71 [5.40,6.05] / 5.66 | 0.24 [0.11,0.41] / 0.11 | 0.21 [0.15,0.27] / 0.16 | 4.46 / 4.42 | 0.53 [0.34,0.74] / 0.09 | 0.01 / 0.01 |

- **Severe errors:** DP with the fault-unaware model rises from 0.15 (κ=0) to 0.26 (κ=1) in RETRY.
- **Exclusion condition, paper's 1/φ form:** never violated for κ ≥ 0 (0/164 RETRY, 0/145 SHARED). At κ=−0.5 it is violated in 58/164 and 60/145, and every one of these violations happens within the budget (n ≤ 6).
- **Exclusion condition, corrected cap 1/min over s≠0 of P(z=1|s)** (checked in `checks.py`, results in `checks.json`). It has **0 violations in every cell**, unrestricted and within budget.

| κ | RETRY: corrected condition holds / violated | SHARED: corrected condition holds / violated |
|---|---|---|
| −0.5 | 91 / 0 | 70 / 0 |
| 0 | 164 / 0 | 145 / 0 |
| 0.5 | 186 / 0 | 178 / 0 |
| 1.0 | 192 / 0 | 187 / 0 |

**Cost of assuming A1, by planning depth** (VOI-1 / VOI-2 / VOI-3 / DP):

| κ | RETRY | SHARED |
|---|---|---|
| −0.5 | 0.22 / 0.20 / 0.18 / 0.18 | 0.19 / 0.17 / 0.16 / 0.16 |
| 0.5 | 0.055 / 0.066 / 0.078 / 0.080 | 0.058 / 0.067 / 0.078 / 0.078 |
| 1.0 | 0.138 / 0.168 / 0.192 / 0.195 | 0.164 / 0.186 / 0.208 / 0.208 |

**Interpretation.**
- **A1 is a mild approximation for decisions.** Assuming it with the correct marginal φ costs at most 0.21 regret. For κ > 0 that cost grows mildly with planning depth, by about 1.3–1.45x from VOI-1 to DP. At κ=−0.5 it shrinks with depth.
- **The exclusion corollary is direction-dependent.** In the 1/φ form it holds for κ ≥ 0 but fails for κ < 0. Without A1 the right cap is 1/min over s≠0 of P(z=1|s), not 1/φ. That form has no violations in any cell, and the paper should state it that way.
- **Fault-unaware degrades sharply as κ grows.**
- **MMI's depth amplification is not universal.** It is present for κ ≥ 0 but disappears at κ=−0.5, where MMI is already wrong at depth 1.

## S3 Partial sharing (SHARED family)

- **Flags:** tool A is faulty if the shared flag or its own flag is on, likewise tool B.
- **Fault rates:** the shared flag has rate ρφ and each own flag has rate 1 − (1−φ)/(1−ρφ), so each tool's marginal fault rate is φ.
- **Endpoints:** ρ=0 means independent faults per tool; ρ=1 is the paper's shared backend.
- **Dedup:** groups calls by backend.

| ρ | assume shared | assume split | MMI | oracle dedup (by backend) |
|---|---|---|---|---|
| 0 | 0.21 [0.18,0.25] / 0.22 | 0.00 / 0.13 | 2.24 [2.03,2.45] / 0.83 | 0.25 [0.22,0.29] |
| 0.25 | 0.08 [0.06,0.10] / 0.09 | 0.05 [0.03,0.06] / 0.06 | 2.11 [1.90,2.32] / 0.70 | 0.12 [0.10,0.15] |
| 0.5 | 0.01 [0.01,0.02] / 0.02 | 0.18 [0.14,0.22] / 0.06 | 2.04 [1.82,2.26] / 0.63 | 0.06 [0.05,0.07] |
| 0.75 | 0.00 / 0.01 | 0.39 [0.33,0.47] / 0.13 | 2.03 [1.81,2.25] / 0.62 | 0.04 [0.03,0.05] |
| 1 | 0.00 / 0.01 | 0.66 [0.55,0.76] / 0.21 | 2.03 [1.81,2.25] / 0.62 | 0.04 [0.03,0.05] |

Oracle-DP repeat-source calls fall from 0.61 (ρ=0) to 0.09 (ρ=1). The assume-shared agent stays at 0.09 and the assume-split agent at about 0.6.

**Interpretation.**
- **The error is asymmetric.** Wrongly assuming a shared source costs at most 0.21 (at ρ=0). Wrongly assuming independence costs up to 0.66 and grows with depth (VOI-1 0.21 → DP 0.66 at ρ=1).
- **The binary shared/independent structure is adequate once ρ ≥ 0.5.**
- **MMI is about 2.0 at every ρ.**
- **Backend-level dedup** works when sharing is strong. It costs 0.25 when the "shared" tools are actually independent.

## S4 Mid-episode recovery (fault follows a hidden Markov chain; s stays fixed)

After every call, each faulty flag clears with probability r; one variant also adds an onset probability for healthy flags. The "static" agent assumes r=0.

There are two MMI controls:
- **MMI (time-0)** uses the per-call marginal at time 0 (`sim.mmi`). Under recovery or onset the true per-call marginal drifts, so this control is marginal-matched only for the first call.
- **MMI_t** (`core.mmi_time`, added after review) uses the true per-call marginal at the call's index t, computed from prior·T^t. It is marginal-matched at every call. At r=0 it is identical to MMI.

| r, onset | RETRY static | RETRY fault-unaware | RETRY MMI (time-0) | RETRY MMI_t | SHARED static | SHARED fault-unaware | SHARED MMI (time-0) | SHARED MMI_t | dedup (RETRY / SHARED) |
|---|---|---|---|---|---|---|---|---|---|
| 0, 0 | 0.00 / 0.01 | 2.61 [2.35,2.88] / 2.57 | 2.00 [1.71,2.29] / 0.45 | 2.00 [1.71,2.29] / 0.45 | 0.00 / 0.01 | 1.98 / 1.96 | 2.16 [1.93,2.41] / 0.93 | 2.16 [1.93,2.41] / 0.93 | 0.03 / 0.04 |
| 0.1, 0 | 0.00 / 0.01 | 2.33 / 2.36 | 1.49 [1.29,1.71] / 0.35 | 2.04 [1.83,2.28] / 0.59 | 0.00 / 0.01 | 1.75 / 1.78 | 1.63 [1.46,1.82] / 0.71 | 1.87 [1.69,2.05] / 0.79 | 0.03 / 0.05 |
| 0.3, 0 | 0.00 [0.00,0.01] / 0.02 | 1.78 / 1.94 | 0.71 [0.62,0.80] / 0.18 | 1.20 [1.11,1.30] / 0.49 | 0.01 / 0.01 | 1.30 / 1.43 | 0.79 [0.70,0.88] / 0.36 | 0.94 [0.85,1.03] / 0.56 | 0.04 / 0.05 |
| 0.5, 0 | 0.06 [0.05,0.07] / 0.07 | 1.29 / 1.58 | 0.21 [0.19,0.24] / 0.11 | 0.37 [0.33,0.42] / 0.27 | 0.09 [0.08,0.11] / 0.10 | 0.94 / 1.18 | 0.24 [0.22,0.27] / 0.17 | 0.27 [0.24,0.31] / 0.29 | 0.10 / 0.15 |
| 0.1, 0.05 | 0.02 [0.01,0.03] / 0.01 | 2.52 / 2.45 | 1.52 [1.31,1.74] / 0.34 | 1.51 [1.31,1.74] / 0.28 | 0.03 [0.02,0.04] / 0.01 | 1.94 / 1.87 | 1.68 [1.52,1.87] / 0.71 | 1.67 [1.50,1.86] / 0.58 | 0.02 / 0.02 |

Oracle-DP repeat-source calls rise with r: RETRY 0.06 → 0.46 and SHARED 0.08 → 0.68. The static model stays at 0.07 and 0.08. Its severe-error probability stays at the Oracle level, about 0.01. MMI_t over-repeats at every r (0.78–1.29 repeat-source calls).

**Interpretation.**
- **The static-fault model is robust to moderate recovery.** Regret is at most 0.03 for r ≤ 0.3, and onset does not hurt it either.
- **At r=0.5 it loses 0.06–0.09**, mainly by retrying too little. This is exactly the case where repeat calls are worth making. Dedup degrades the same way (0.10–0.15).
- **MMI's penalty shrinks as r grows, and this is not an artefact of the marginal mismatch.** The marginal-matched MMI_t shows the same decline (RETRY 2.00 → 0.37, SHARED 2.16 → 0.27), so the dependence penalty is specifically a property of faults that persist. At r ≥ 0.1 MMI_t is in fact worse than time-0 MMI. Using the time-0 marginal overstates later fault rates, which happens to partly offset the independence error.
- **MMI's depth amplification weakens with r and disappears at r=0.5.** VOI-1 → DP:
  - r ≤ 0.3, including the onset variant: 1.7–5.5x across both controls. The low end is SHARED MMI_t at r=0.3 (0.56 → 0.94).
  - r=0.5, time-0 MMI: RETRY 0.11 → 0.21, SHARED 0.17 → 0.24.
  - r=0.5, MMI_t: RETRY 0.27 → 0.37; SHARED flat (0.29 → 0.27).

## S5 Caches that share a stale state value

- **Model:** the hidden state is (s, z, v). A stale source keeps reporting a stale value v with noise η=0.03.
- **How v is drawn:** once per episode, v ~ ψ·δ_s + (1−ψ)·prior.
  - ψ=0 is the setting you asked for: the stale value is unrelated to the current state.
  - ψ=0.5 is an extra variant I added, where the stale copy is still informative.
- **Consequence:** a stale source can now keep reporting "down" while the service is healthy.
- **Agents:** "benign-stuck" is the paper's model (faulty tool reports benign with f=0.97) with the true φ.

| ψ | family | benign-stuck | fault-unaware | MMI | oracle dedup | benign-stuck + dedup |
|---|---|---|---|---|---|---|
| 0 | RETRY | 0.09 [0.07,0.11] / 0.06 | 1.02 [0.94,1.11] / 0.99 | 0.43 [0.37,0.49] / 0.13 | 0.01 | 0.07 |
| 0 | SHARED | 0.08 [0.06,0.10] / 0.06 | 0.63 [0.55,0.71] / 0.61 | 0.51 [0.45,0.57] / 0.21 | 0.02 | 0.08 |
| 0.5 | RETRY | 0.30 [0.26,0.34] / 0.32 | 0.10 [0.08,0.12] / 0.09 | 0.26 [0.24,0.29] / 0.18 | 0.06 | 0.33 |
| 0.5 | SHARED | 0.33 [0.28,0.37] / 0.36 | 0.03 [0.03,0.04] / 0.03 | 0.17 [0.14,0.19] / 0.10 | 0.10 | 0.39 |

- **"Benign-stuck + dedup"** is the misspecified benign-stuck emission model with dedup on top. **"Oracle dedup"** is the correct model with no repeat calls.
- **Behaviour at ψ=0.5:**
  - The Oracle commits far more often. Its escalation falls from 0.70 to 0.09 in RETRY and from 0.56 to 0.06 in SHARED (ψ=0 → ψ=0.5).
  - It makes 0.19 (RETRY) and 0.29 (SHARED) repeat-source calls.
  - The benign-stuck model escalates in 0.55 (RETRY) and 0.50 (SHARED) of episodes. Oracle dedup escalates in 0.14 and 0.13.
- **Exclusion condition:** violated in 23/158 (RETRY) and 36/145 (SHARED) instances at ψ=0, and in 139/158 and 137/145 at ψ=0.5. All of these violations happen within the budget (n ≤ 6).

**Interpretation.**
- **At ψ=0, most of the paper's conclusions survive:**
  - The benign-stuck model is a good approximation (0.08–0.09).
  - Fault-unaware is the worst agent.
  - MMI is bad and its regret grows with depth (VOI-1 0.13–0.21 → DP 0.43–0.51).
  - Dedup is close to the oracle.
- **The exclusion corollary fails even at ψ=0.** A stale benign copy appears only with probability π0, which weakens the cap to roughly 1/(φ·π0).
- **At ψ=0.5 the benign-stuck emission model breaks:**
  - Fault-unaware (0.03–0.10) beats the paper's benign-stuck model (0.30–0.33).
  - The worst option is benign-stuck + dedup (0.33–0.39). Nearly all of that loss comes from the benign-stuck misspecification, since benign-stuck alone already has 0.30/0.33; dedup adds only 0.03–0.06.
  - Oracle dedup stays near-oracle even when stale values are informative: 0.06 in RETRY, where it is second best, and 0.10 in SHARED. Its only cost is that it gives up the informative repeats the Oracle makes (0.19–0.29 repeat-source calls).
  - DP ranking: RETRY is Oracle 0 < oracle dedup 0.06 < fault-unaware 0.10 < MMI 0.26 < benign-stuck 0.30 < benign-stuck + dedup 0.33. SHARED is Oracle 0 < fault-unaware 0.03 < oracle dedup 0.10 < MMI 0.17 < benign-stuck 0.33 < benign-stuck + dedup 0.39.
- **Scope for the paper:** limit the claims about the benign-stuck emission model to faults whose stale output carries little information about the current state. For value-sharing caches, a stale source should be modelled as one informative report that does not replicate, not as a benign-biased one. Counting such a source once (dedup) with a correct emission model remains a good rule.

## What survives overall

- **Robust: ignoring persistence costs more as planning gets deeper.** This holds for MMI and the split models whenever one-step predictions are right but predictions for repeat calls are wrong:
  - S1 at every f
  - S2 with κ ≥ 0
  - S3
  - S4 for r ≤ 0.3, weakening at r=0.5
  - S5 with ψ=0

  The depth effect is absent when the model is already wrong at depth 1: fault-unaware everywhere, and MMI at κ=−0.5. It also fades when faults recover quickly (S4, r=0.5). That pattern is a candidate condition for a formal "regret grows with depth" result; so far it is an empirical regularity, not a theorem.
- **Robust: three of the paper's simplifications cost little.** The binary shared/independent structure, the static-fault assumption up to r ≤ 0.3, and A1 with the correct marginal φ all keep regret at or below 0.21 in every cell tested. The A1 cost grows mildly with depth for κ > 0.
- **Robust: oracle dedup (count each source once, correct model)** stays within 0.10 of the Oracle in every S1, S2, S4 and S5 cell. The exceptions are S3 at ρ ≤ 0.25 (0.12–0.25) and S4 at r=0.5 (0.10–0.15), where repeat calls are genuinely informative.
- **Fragile: the 1/φ cap and the exclusion corollary need three conditions:**
  - A4 (f ≥ q). Near the boundary (f=0.85), most corollary failures need more reports than the budget allows.
  - A1 in the form P(z | s≠0) ≥ φ. Equivalently, use the cap 1/min over s≠0 of P(z=1|s), which has no violations in any S2 cell.
  - Faults that stick on the benign report rather than copying a stale value.

  The middle expression of Prop. 1 is always valid.
- **Fragile: "fault-unaware is worst" and the benign-stuck emission model** both fail for value-sharing caches whose stale value is informative (S5, ψ=0.5). There dedup on top of a misspecified emission model is the worst policy (0.33–0.39), but oracle dedup is not (0.06–0.10).