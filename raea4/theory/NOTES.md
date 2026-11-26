# Theory notes for the revision

All statements are in `theory.tex` (amsthm fragments). Every numbered claim was checked against `sim.py` / `exact.py`
on random instances by `check_*.py`; outputs are in the matching `.log` and `.json` files. Numbers below are copied
from those logs.

## 1. What the revised theory says

The central question is: *with persistent dependence between tool reports, why is a correct estimate of the current
state not enough for correct information-gathering decisions?* The theory answers it in three parts.

**(A) What same-source evidence can justify (Section 4, revised).**
- *Lemma 1 (event bound).* If some latent event E makes every observed report come from a fault channel that does not
  depend on s, and the reports are no more likely in the benign state than under that channel, then the benign
  likelihood ratio against s is at most 1/P(E | s). Every bound below is a special case.
- *Proposition 1 (two-sided).* For any reports from one group (not only benign ones):
  min{1, H_s/H_s'} <= LR <= min{max{1, H_s/H_s'}, 1 + (1-phi)/phi * H_s/F}. The paper's 1/phi bound is the case
  "benign reports and f >= q".
- *Corollary 1 (the f vs q boundary).* f > q: the bound 1 + (1-phi)/phi (q/f)^n *decreases* with n, and long runs
  carry no information about s (LR -> 1). f = q: LR rises monotonically to 1/phi and never reaches it, so 1/phi is
  tight. f < q: LR -> infinity; repeats are genuine but discounted evidence (about log(q / max(f, r)) per repeat
  instead of log(q/r)), and **no exclusion condition exists**. Worked numbers (phi = .3, q = .9, K = 3): f = .97 peaks
  at 2.99 after 2 reports, f = .90 saturates at 3.33 = 1/phi, f = .80 reaches 10.6 after 12 reports.
  This is the prediction for the requested f<q / f=q / f>q boundary experiment.
- *Corollary 2 (exclusion condition, generalized).* Posterior benign odds <= w0 * max_s LR_{0,s}; the condition is
  w0 * Lambda < Omega with Lambda from the case at hand.
- *Proposition 2 (beyond A1/A2/one flag).*
  (a) fault rate depending on the state, phi_s = P(z=1 | s): bound 1/phi_s, so the exclusion condition must use
  the smallest bad-state rate. Long runs tend to phi_0/phi_s (below 1 if faults are more common when the system is
  unhealthy: benign runs then point *away* from the benign state). The bound with the marginal rate is wrong in
  general (violated in 530/2274 random cases; explicit counterexample LR -> 12 > 1/phibar = 3.08).
  (b) partial sharing (tool B hits the shared backend with probability pi_B per call): bound 1/(phi pi_B^{n_B});
  the shared part stays bounded, the rest is fresh evidence (long B-runs exceed 1/phi in 587/600 cases).
  (c) **stale value** (the cache holds a copy v of a past state, v ~ rho(. | s)): bound 1/(phi_s rho(0 | s)).
  If (z, v) is independent of s and the cache is read with healthy accuracy, the bound is
  1 + (1-phi)/(phi rho_0) and it is *the limit*: unlike stuck-at-benign faults, long runs do not wash out.
  (d) **recovery** (a fault clears before each later call w.p. rho, never returns): with x = q/((1-rho) f), x < 1
  gives a uniform bound 1 + (1-phi)(1-rho)/phi x^n + rho x/(1-x); x > 1 gives LR -> infinity; at the boundary x = 1
  with rho > 0 the LR has a closed form that grows *linearly* in n (phi=.3, q=.9, rho=.1, f=1: 3.48, 12.9, 42.7 at
  n = 5, 100, 400, against 1/phi = 3.33). So with recovery the A4 condition is the strict (1-rho) f > q; the
  non-strict f >= q is enough only when rho = 0 (where f = q saturates at 1/phi).
- *Proposition 4 (estimated phi).* The exclusion condition is a threshold in log phi - log w0; estimates are
  classified correctly whenever the log-error is below the margin |log(phi Omega / w0)| (a deterministic consequence
  of the triangle inequality, not something the check can test). A valid one-sided confidence bound gives a
  certificate. Empirically: the plug-in condition barely beats "always holds" at small calibration sizes, and plug-in
  false exclusions are common (RETRY n = 25: 27 of 63 false-condition instances); the percentile-bootstrap
  certificate is approximately, not provably, at level alpha (numbers in section 4).

**(B) A correct current belief is not enough (new section).**
- *Lemma 3.* MMI equals the exact model at the root to one step: same s-marginal, same one-step predictive for every
  tool, same s-posterior after any single report, hence identical VOI-1 action values at the root
  (0 mismatches in 200 RETRY/SHARED instances). After one group report the next same-group predictive already differs
  (median max-abs difference 0.17).
- *Lemma 4.* What MMI misses is exactly a covariance term: P_exact(o,o'|s) - P_MMI(o,o'|s) =
  phi(1-phi)(h(o)-e(o))(h'(o')-e'(o')). MMI expects repeated reports to disagree more than they do (this is an identity for the two-report joint, not for action values; more dispersion is not by itself more informative) (32400/32400 identities hold to 1e-12).
- *Proposition 3.* Terminal risk depends only on the s-marginal Pi(b); if no fault-prone tool is affordable, so does
  the whole value function (standard belief-MDP reduction). Otherwise it does not, **even when the one-step
  predictive distributions of every tool also agree**.
- *Example 1 (constructive, pilot parameters).* Frequent regime, A reports healthy, healthy, degraded (a Table 7
  item). Exact joint: P(s) = (.866, .077, .057), P(stale | s) = (.130, .729, .980); optimal: query V (3.58; query A
  4.42). Same P(s) and the same one-step predictive for A and V, but P(stale | s) = (1.0, .358, .481): query A is
  optimal (2.94). Treating staleness as independent of s with the correct P(stale) = .225 also makes query A optimal
  (2.41). Same pattern in the moderate regime after A x 2 (exact: V, 3.74; matched belief: A, 3.38). In the frequent
  A x 3 item (also a Table 7 re-query error), every matched belief and the s-independent staleness belief give
  escalate, as does the exact belief; only "stale only if healthy", which does NOT match A's predictive, gives
  query A. So Example 1 is an existence result that holds for OK-OK-degraded (frequent) and A x 2 (moderate) only.
- *Remark (which mechanism the pilot errors match).* A wrong current joint belief does NOT explain the pilot
  re-query errors. A step-by-step check of the pilot answers (not packaged) places the first deviation of the 7 chat answers after the current
  belief: 5 had the correct posterior, P(stale) (when given) and next-report probability, but overestimated the
  posterior after one more same-source report; 2 used a next-report model that drops persistence. Both are the
  two-step error of Lemma 4. A "hybrid" planner with the EXACT current Pi(b) but conditionally independent future
  reports of A (MMI update from the correct marginal) re-queries A in all four single-source pilot items:
  post-repeat P(healthy) hybrid vs exact / Q(query A) hybrid vs exact =
  .890 vs .732 / 3.71 vs 4.97 (frequent A x3; exact best escalate);
  .948 vs .877 / 2.87 vs 4.42 (frequent OK-OK-degraded; exact best V);
  .955 vs .828 / 2.81 vs 4.64 (moderate A x2; exact best V);
  .954 vs .904 / 3.24 vs 4.18 (rare A x1; exact best V).
  With the exact next-report predictive but hybrid posteriors ("mixed", the pattern of the 5 cases) Q(query A) =
  3.66 / 2.76 / 2.57 / 3.15, close to the values the pilot answers wrote (about 3.5; 2.75-3.30; 3.05; 3.46). The
  hybrid next-report probability in frequent OK-OK-degraded (.841) is exactly the value one pilot answer wrote.
  Recommendation for L1: supply P(healthy), P(stale | s) and the posterior after a repeated same-source report;
  supplying the current joint belief alone is not shown to be enough.

**(C) When deeper planning amplifies a wrong dependence model (main new result).**
- *Theorem 2 (minimal certify-or-escalate model, one stuck-prone source).* Let m* be the number of benign repeats
  after which MMI's odds pass the certification threshold, m* = ceil(log(Omega/w0)/log l) with
  l = a0/a1 the MMI per-report ratio, and m_D >= m* the first plan length for which "retry m_D times, then certify"
  beats escalation in MMI's own value (explicit inequality D(m)). Under the exclusion condition:
  (i) for every depth k < m*, MMI k-step lookahead escalates at once (it acts like the exact model; excess regret 0
  whenever the exact model cannot justify certification within budget);
  (ii) for every k >= m_D, including DP, it retries, never escalates on the all-benign path, and certifies after
  between m* and N reports, so its regret exceeds the shallow planner's by at least c + Gamma - c_e P(S), with
  Gamma >= phi f^N (lambda phi/(phi + w0) - c_e) > 0, and its severe-error probability is at least (1-pi0) phi f^N.
  Mechanism: MMI's predicted posterior after m more repeats -> 1 geometrically, so a depth >= m* planner sees a
  retry-then-certify plan that VOI-1 cannot see; truly that plan certifies a stale report with probability bounded
  away from zero.
- *Proposition 5 (converse, standard simulation lemma).* MMI-DP regret <= 2 L_max TV <= 2 L_max sum_g (N_g+1) phi_g:
  when faults are rare, depth does not hurt (and repeats are genuinely valuable, so depth helps).

## 2. What is standard and what is new

| Result | Status |
|---|---|
| Lemma 1, Prop. 1, Cor. 1-2 | Elementary mixture-likelihood bounds. The two-sided form, the f=q tightness and the f<q divergence are new *as statements in this paper*, not as mathematics. |
| Prop. 2 (a)-(d) | Elementary consequences of Lemma 1; useful because they say precisely which fault types the 1/phi picture covers (stale *value*, state-dependent faults, partial sharing, recovery). Not deep. |
| Prop. 4 | Trivial; included because reviewers asked about estimated phi. |
| Lemma 3, Lemma 4 | Simple; Lemma 4 is the formal content of "MMI keeps marginals but drops persistence". |
| Prop. 3 (a)(b) | Standard: the belief over the full hidden state is the sufficient statistic of a POMDP (Astrom 1965; Smallwood and Sondik 1973; Kaelbling, Littman and Cassandra 1998). |
| Prop. 3 (c), Example 1 | New: an explicit counterexample with matched marginal *and* matched one-step predictives, in the pilot's own parameters. |
| Theorem 2 | New to our knowledge as a stated result with explicit thresholds. Related ideas: shorter horizons can be better with inaccurate models (Jiang, Kulesza, Singh and Lewis, AAMAS 2015, "The dependence of effective planning horizon on model accuracy"); lookahead pathology in game-tree search (Nau 1982; Pearl 1983). Our contribution is a specific mechanism (marginal-matched independence of persistent faults) with a provable step at m*. |
| Prop. 5 | Standard simulation lemma (e.g. Kearns and Singh 2002). |

All references above are from memory and **must be checked** before submission.

Recommendation for the paper: keep Prop. 1 + Cor. 2 in the main text (shortened), move Lemma 1 / Prop. 2 / Prop. 4
to an appendix, and give the main-text theory weight to Lemma 3-4, Prop. 3 with Example 1, and Theorem 2. That ties
static inference (A), prediction of future observations (Lemma 4) and action choice (Theorem 2, Example 1) to the
central question.

## 3. Limits (state these honestly)

- Theorem 2 is proved for the minimal model: two states, one source, terminal set {certify, escalate}, no verifier.
  With a verifier or a "declare bad" action the regret decomposition in the proof still holds, but whether the deep
  MMI planner chooses the retry plan over the verifier is checked numerically, not proved.
- The step needs condition D(m): calls must be cheap relative to escalation. The cost threshold depends on the
  prior and on f. With pi0 = .5, phi = .3, q = .9, f = .97, lambda = 30, c_e = 4 and N = 6 calls, it holds for
  c <= 0.4 (regret 0 / 0 / 3.13 for k = 1 / 2 / 3) and fails for c >= 0.5 (MMI regret 0 at every depth in the
  single-source model). With pi0 = .6 (same other parameters) it holds up to c = 0.6 (c = .5: m_D = 3, regret
  0 / 0 / 2.157; c = .6: 2.366 from k = 3) and fails from c = 0.8. So the theorem does *not* cover the pilot's cost
  structure (c = 1, c_e = 4) in the one-source model; the amplification in the RETRY/SHARED families (below) arises
  with three states, a verifier and declare actions, which the theorem does not cover.
- The explicit gap bound c + Gamma - c_e P(S) is positive in 94% of random qualifying instances; in the rest it is
  vacuous although the exact regret still jumps (100% of instances). Regret is non-decreasing in k over 1..N in 96% of
  instances, not always (worked instance 2: 1.088 at k = 2, 1.073 at k >= 3); monotonicity is not claimed. Depths
  m* <= k < m_D are not characterized.
- Proposition 5 is loose (observed regret / bound = 0.008 on average) and covers DP only, not k-step lookahead.
- All results are for the static-state model; Prop. 2(d) is the only dynamic element (fault recovery).
- The "stale value" result needs rho(0 | s) > 0; if a stale copy is never benign when the state is bad, or tracks the
  current state with a short lag, benign runs are genuine evidence and no exclusion condition exists.

## 4. Verified claims (check scripts and their outputs)

All numbers are copied from the current `.log` files (rerun after the review fixes, see section 5).

`check_prop1.py` (Section A; LRs read from `sim.Model` posteriors; enumerators for stale value / recovery are
cross-checked against sim to 4e-16 and 0):
```
[P1-gen] min(1,Hs/Hs') <= LR <= min(max(1,Hs/Hs'), 1+(1-phi)/phi*Hs/F): 3000/3000 hold
[P1a] f>q: claims hold in 300/300 ; f=q: 300/300 ; f<q: 300/300
[P1b] LR(0 vs s) <= 1/phi_s: 2274/2274 hold; the marginal-phi bound 1/phibar fails in 530/2274
      long run (phi_0=.1, phi_s=.4): LR 0.2500 = phi_0/phi_s; counterexample LR 12.000 > 1/phibar = 3.077
[P1c partial sharing] LR <= 1/(phi pi^nB): 896/896 hold; long B-runs exceed 1/phi in 587/600
[P1d stale value] LR <= 1/(phi_s rho(0|s)): 2265/2265; independent-value bound: 0 violations;
      long-run limit equals 1+(1-phi)/(phi rho0) (max rel. err 3.9e-15, 369 cases)
[P1e recovery] x<1 bound holds 1236/1236; x>1: LR exceeds 10/phi after 200 reports in 572/572
      x=1, rho>0, 400 instances: closed form matches the enumerator (max rel. err 2.3e-14); lower bound 3600/3600;
      LR strictly increasing over n=1..400 in 400/400; LR(400) > 1/phi in 400/400
      phi=.3 q=.9 r=.05 rho=.1 f=1: LR at n=5,20,50,100,200,400 = 3.48, 4.97, 7.95, 12.92, 22.87, 42.75 (1/phi = 3.33)
```
`check_phi_robust.py` (Prop. 4, EM fits from `sim.fit` on nested calibration logs, 300 instances per family,
1000 bootstrap resamples; about 55 min with 4 workers):
```
RETRY : condition holds in 79.0%  (= agreement of the majority-class rule "always holds")
  plug-in agreement n=25/50/100/400/1000: 81.7 / 85.7 / 89.0 / 91.3 / 97.3%  (gain over baseline +2.7 / +6.7 / +10.0 / +12.3 / +18.3)
  balanced accuracy                       72.7 / 75.2 / 79.1 / 84.6 / 96.0%
  plug-in false exclusions (of 63 false-condition instances): 27 / 27 / 24 / 17 / 4
SHARED: condition holds in 72.7%  (baseline 72.7%)
  plug-in agreement n=25/50/100/400/1000: 76.7 / 81.7 / 87.7 / 95.7 / 98.0%  (gain +4.0 / +9.0 / +15.0 / +23.0 / +25.3)
  balanced accuracy                       69.1 / 74.1 / 83.9 / 94.0 / 97.5%
  plug-in false exclusions (of 82): 39 / 35 / 20 / 8 / 3
"error < margin" (where the deterministic guarantee applies): RETRY 54/68/74/84/92%, SHARED 46/57/68/87/92%
percentile-bootstrap certificate, false certification among false-condition instances [Wilson 95%], power:
  RETRY  n=100 a=.05 : 3/63 = 4.8% [1.6, 13.1]  power 70.5%  | a=.025: 2/63 = 3.2% [0.9, 10.9]  power 65.4%
  RETRY  n=400 a=.05 : 1/63 = 1.6% [0.3,  8.5]  power 89.0%  | a=.025: 0/63 = 0.0% [0.0,  5.7]  power 86.1%
  SHARED n=100 a=.05 : 1/82 = 1.2% [0.2,  6.6]  power 64.2%  | a=.025: 1/82 = 1.2% [0.2,  6.6]  power 55.5%
  SHARED n=400 a=.05 : 0/82 = 0.0% [0.0,  4.5]  power 83.9%  | a=.025: 0/82 = 0.0% [0.0,  4.5]  power 79.4%
```
The observed conditional rates are at or below alpha, but the Wilson upper limits (up to 13%) do not rule out
rates above alpha; the percentile bootstrap of an EM fit has no finite-sample guarantee. Proposition 4's
"wrong with probability at most alpha" holds for a valid lower bound and is not demonstrated by this procedure.

`check_belief.py` (Section B):
```
[Lemma 3] 200 instances: 0 mismatches (s-marginal, one-step predictive, posterior after one report, VOI-1 values)
[Lemma 4] covariance identity: 32400/32400
[Prop 3b] only fault-free tools affordable: V and action depend only on Pi(b): 300/300
[Example 1] OK-OK-degraded, frequent: exact best query_V 3.580; matched Pi+predictives -> query_A 2.936
[Example 3] A x2, moderate: exact best query_V 3.739; matched -> query_A 3.382
[Example 2] A x3, frequent: every matched belief -> escalate (no change)
[Pilot items] exact current Pi + independent future reports of A ("hybrid"), and "mixed" (exact next-report
  predictive, hybrid posteriors):           P(h|+healthy) exact/hybrid   Q(query A) exact/hybrid/mixed   best exact -> hybrid
  frequent A x3                              .732 / .890                  4.97 / 3.71 / 3.66              escalate -> query_A
  frequent OK-OK-degraded                    .877 / .948                  4.42 / 2.87 / 2.76              query_V  -> query_A
  moderate A x2                              .828 / .955                  4.64 / 2.81 / 2.57              query_V  -> query_A
  rare A x1                                  .904 / .954                  4.18 / 3.24 / 3.15              query_V  -> query_A
```
`check_depth.py` (Section C; `sim.mmi`, `Model.Vd`/`V`, `exact.evaluate_exact`; defaults 300 random, 40 family):
```
worked: pi0=.5 phi=.3 q=.9 f=.97 c=.25 lambda=30 c_e=4 N=6: m*=2, m_D=3, Gamma=2.000, P(S)=.186, bound 1.508
        MMI regret k=1,2: 0.000 ; k=3..6 and DP: 2.736 (severe errors 14.1%); exact: 0 at every depth
cost sweep (phi=.3 q=.9 f=.97 lambda=30 c_e=4 N=6): pi0=.5: amplification for c<=.4, none for c>=.5;
        pi0=.6: amplification for c<=.6 (c=.5: 0/0/2.157 at k=1/2/3), none for c>=.8
random: 300 qualifying instances (q in [.55,.99], N in [2,8]; q>.95 in 75, N>=7 in 105), 2440 claims (root action,
        all-benign path, gap bound, severe bound, Gamma closed form, c_e - J* <= c_e P(S), m* formula, declare-bad
        never optimal): all hold. Gap bound > 0 in 94%; regret non-decreasing in k in 96%; P(S)=0 in 163.
converse: 60 rare-fault instances: TV and regret bounds hold in all; max regret 0.91, mean regret/bound 0.008
paper families, true parameters, 40 instances, MMI regret by depth k=1/2/3/(4)/DP [95% instance bootstrap]:
  RETRY  0.29 [0.07,0.55] / 1.51 / 1.99 / 1.99 / 2.07 [1.50,2.67]   exact <= 0.002 at every depth
  SHARED 0.71 [0.37,1.10] / 1.93 / 2.17 / 2.41 [1.90,2.93]           exact <= 0.005 at every depth
```
The family numbers reproduce the draft's Table 1 pattern (MMI VOI-1 0.37/0.63 vs DP 1.89/2.18, 300 instances) on
40 fresh instances and add the intermediate depths: most of the amplification appears at depth 2. The intervals
resample instances (one exact evaluation per independently drawn instance); they are not cluster-bootstrap intervals.

## 5. Changes after review

1. Example 1 no longer claims to explain the pilot re-query errors; the frequent A x3 counter-case is stated, and a
   new Remark ties the pilot errors to Lemma 4 (overestimated post-repeat posterior), with the "hybrid" computation.
2. Prop. 2(d): the recovery condition is the strict (1-rho) f > q; x = 1 case (linear growth) added with closed form
   and checked in `check_prop1.py`.
3. `common.BIG` raised from 1e12 to 1e250 (declare-bad was becoming optimal at high q), `random_par` widened, and a
   "declare-bad never optimal" check added. With 1e12, 6 of 300 wide-range instances had failed (reviewer's run).
4. `check_depth.py` defaults are now 300 random / 40 family instances; intervals relabelled "instance bootstrap";
   D(m) cost threshold stated with pi0 and f and backed by a cost sweep.
5. `check_phi_robust.py`: majority-class baseline and balanced accuracy reported; the tautological check removed
   (only the coverage of the sufficient condition is reported); 1000 resamples on all 300 instances, at n = 100 and
   400, alpha = .05 and .025, with Wilson intervals for the conditional false-certification rate.