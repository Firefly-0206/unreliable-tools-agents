# theory/ -- statements, proofs and numerical checks

- `theory.tex`  LaTeX fragments (amsthm) ready to paste: event lemma, two-sided Proposition 1, f-vs-q boundary,
  generalized exclusion condition, state-dependent / partial-sharing / stale-value / recovery extensions (incl. the
  x = 1 recovery boundary), estimated phi; MMI one-step equivalence and covariance gap, s-marginal insufficiency with
  a constructive example and a remark on which mechanism the pilot re-query errors match; Theorem 2 (depth
  amplification under MMI) and the simulation-lemma converse.
- `NOTES.md`    plain explanation, standard vs new, limits, verified-claims list with check outputs, changes after review.
- `common.py`   helpers (imports ../sim.py and ../exact.py read-only; enumerators for stale-value and recovery models;
  disabled terminal actions get loss 1e250).
- `check_prop1.py`      Section A claims (LRs from sim.Model posteriors).                -> check_prop1.log / .json (~30 s)
- `check_phi_robust.py [n_inst=300] [n_boot=1000]`  estimated phi, EM fits, bootstrap certificate, 4 workers
                                                                                         -> check_phi_robust.log / .json (~55 min)
- `check_belief.py`     Lemma 3, Lemma 4, Prop. 3, Examples 1-3, pilot re-query items.   -> check_belief.log / .json (~1 min)
- `check_depth.py [n_random=300] [n_family=40]`  Theorem 2, cost sweep, converse, family depth curves, 4 workers
                                                                                         -> check_depth.log / .json (~1 min)

Run from this folder with `<root>\.venv\Scripts\python.exe <script>` (defaults reproduce every reported number).
No API calls are made. The committed `check_phi_robust.log` prints one Wilson lower limit as "-0.0%"; the script now
clips it to 0.0% (same value; not rerun because the run takes ~55 min).

Key results (all from the logs):
- Every checked identity/bound holds on every instance: Prop. 1 (3000/3000), f vs q boundary (300/300 each),
  state-dependent rates (2274/2274; marginal-rate bound fails 530/2274), partial sharing 896/896, stale value
  2265/2265, recovery x<1 1236/1236, x>1 diverges 572/572, x=1 with rho>0 grows linearly (400/400; e.g. 3.48 -> 42.75
  at n = 5 -> 400 vs 1/phi = 3.33), so with recovery the condition is the strict (1-rho) f > q.
- Theorem 2: 300 random qualifying instances (q up to .99, N up to 8), 2440 claims, all hold. Worked instance: MMI
  regret 0 at depth 1-2, 2.736 from depth 3 (exact 0). Families (40 instances, 95% instance bootstrap): RETRY
  0.29 [0.07, 0.55] -> 1.51 -> 1.99 -> 2.07 [1.50, 2.67] (k = 1, 2, 3, DP); SHARED 0.71 [0.37, 1.10] -> 1.93 -> 2.17
  -> 2.41 [1.90, 2.93]; exact <= 0.005 throughout.
- Pilot re-query errors: not explained by a wrong current joint belief (frequent A x3: every matched belief gives
  escalate). A planner with the exact current Pi but independent future same-source reports re-queries A in all 4
  single-source items (Q(query A) 3.71/2.87/2.81/3.24 vs exact 4.97/4.42/4.64/4.18), i.e. Lemma 4's two-step error.
- Estimated phi: plug-in agreement vs majority baseline at n = 25: RETRY 81.7% vs 79.0%, SHARED 76.7% vs 72.7%;
  at n = 1000: 97.3% / 98.0%. Bootstrap certificate (1000 resamples, n = 100, alpha = .05): false certification
  3/63 = 4.8% [1.6, 13.1] RETRY, 1/82 = 1.2% [0.2, 6.6] SHARED; at n = 400: 1/63 and 0/82. Approximate, not a
  proven level-alpha guarantee.
