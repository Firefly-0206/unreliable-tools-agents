# stress/: theory-applicability stress tests (S1-S5)

Imports `sim.py` / `exact.py` from the parent directory (not modified).

Files
- `SUMMARY.md`: **results summary and interpretation** (every number there is taken from `results/*.json`).
- `TABLES.md`: generated full tables (every cell, every policy: regret [CI], severe, escalation, calls,
  repeat-source calls, probe cost, diagnostics).
- `core.py`: `GModel` (a `sim.Model` with explicit hidden states, prior, likelihoods and optional hidden-state
  transition after every call), marginal-matched independence `mmi_of` (time-0 marginal) and `mmi_time`
  (true per-call marginal at call index t, for S4), exact evaluator `evaluate` (same recursion as
  `exact.evaluate_exact`, also returns probe cost), `dedup_of`, `summarize2`.
- `run_stress.py`: environments `env_S1..env_S5`, Prop. 1 / benign-run diagnostics, parallel driver.
  Diagnostics are reported unrestricted (n <= 20 reports, the theory statement) and within budget
  (`*_inB`: n <= B / group-tool cost = 6, what an agent can reach).
- `checks.py`: per-instance exclusion-condition check vs exact posteriors (paper's 1/phi cap; for S2 also the
  corrected cap 1/min_{s!=0} P(z=1|s)), each unrestricted and within budget; S1 Prop. 1 aggregates; depth
  profiles (VOI-1/2/3/DP) -> `results/checks.json`.
- `make_tables.py`: renders `results/S*.json` into `TABLES.md` and prints compact tables.

Run (about 15 min for everything on 4 workers):

    <root>\.venv\Scripts\python.exe run_stress.py S1 S2 S3 S4 S5 --n 200 --workers 4 --depth 2 3
    <root>\.venv\Scripts\python.exe checks.py
    <root>\.venv\Scripts\python.exe make_tables.py

Experiments (200 instances per cell; seeds 201-205; instances are identical across the cells of one experiment)
- S1 faulty-tool benign probability f in {0.5,0.7,0.85,q,0.95,0.97,0.99}; agents true/fault-unaware/MMI/split/dedup.
- S2 P(z=1|s!=0)=phi(1+k), P(z=1|s=0)=phi(1-k(1-pi0)/pi0), k clipped to valid range; agents true/assume-A1/fault-unaware/MMI/dedup.
- S3 shared flag (rho*phi) plus own flags, per-tool marginal fault rate phi; agents true/assume-shared/assume-split/MMI/dedup.
- S4 fault clears w.p. r after each call (optional onset); agents true/static/fault-unaware/MMI (time-0)/MMI_t/dedup.
- S5 stale cache repeats a stale state value v ~ psi*delta_s + (1-psi)*prior; agents true/benign-stuck/fault-unaware/MMI/
  oracle dedup (correct model)/stuck_dedup (benign-stuck model + dedup).

Policies: Oracle-DP (true model), DP / VOI-1 / VOI-2 / VOI-3 per agent (dedup agents: DP and VOI-1).
Revision note: after review, S1, S2, S4, S5 were rerun (same seeds; all earlier policy numbers reproduced exactly)
to add within-budget diagnostics and the S4 `mmi_t` control; S3 was not affected.
Note: dedup never plans a repeat-source call, so in closed loop it equals "forbid future repeats"; discarding
existing repeat-source history is never exercised here.

Key results (details in `SUMMARY.md`)
- MMI regret grows with planning depth at every f in S1 (VOI-1 -> DP: RETRY 4.7-6.0x, SHARED 3.3-3.8x).
- Prop. 1 middle expression always valid; 1/phi cap fails for every f<q instance, also within budget (max phi*ratio
  in budget 1.7-39; 2.9e5 only at n=20). Exclusion corollary at f=0.85: 81/159 RETRY and 32/137 SHARED failures
  for n<=20, but only 20 and 13 within budget.
- S2: corrected cap 1/min_{s!=0} P(z=1|s) has 0 violations in every cell; A1 cost <= 0.21, grows ~1.3-1.45x with
  depth for kappa>0.
- S4: static-fault model <= 0.03 regret for r<=0.3; MMI's penalty shrinks with r for both time-0 MMI and the
  marginal-matched MMI_t (RETRY 2.00 -> 0.37, SHARED 2.16 -> 0.27).
- S5 psi=0.5: benign-stuck model 0.30/0.33 and benign-stuck+dedup 0.33/0.39 (RETRY/SHARED); oracle dedup stays
  near-oracle (0.06/0.10).
