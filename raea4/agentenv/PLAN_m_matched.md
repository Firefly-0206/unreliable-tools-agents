# Difficulty-matched fault / fault-free experiment: analysis plan fixed before the runs

Local timestamps and hashes only (`frozen_m_matched.sha256`); not a third-party preregistration.
Written 2026-11-15, after a post hoc comparison on the stored sweeps (`difficulty_matched.py`) found that, at equal
first-step value margin, suboptimal first actions are far more frequent where the fault-capable replica tool is the
best first query than where another tool is. That comparison was chosen after the data were seen, matched on one
measure, and compared different tools. This experiment asks the same question with tasks built for it.

## Question

When the first decision is equally hard by exact measures, do the models choose a suboptimal first action more often
when the best first query is a tool that may be persistently faulty than when it is a tool with no fault at all?

## Episodes (`make_episodes_m.py`, seed 2031 -> `episodes_m.jsonl`, `ground_truth_m.jsonl`, `m_matching.json`)

50 pairs on new parameter draws that no model has seen (first-domain structure with one replica tool; budget, costs
and losses as before). Within a pair the prior, costs, the other tools' accuracies, the scenario and the uniforms are
identical.

- **MF**: tool 0 (the replica tool) has a persistent fault with rate 0.30; accuracies as drawn.
- **MZ**: fault rate 0, so the documentation states no fault. The accuracies of tool 0 and of the log tool are taken
  from a grid (0.60..0.92 and 0.55..0.85, step 0.01) such that, under the exact oracle at the root: the best first
  action is tool 0 in both episodes; the second-best first action is the same; the one-step value-of-information rule
  finds the best first action in both or in neither; the margin Q(second-best) - Q(best) differs by at most 0.02; and
  the oracle's expected total loss differs by at most 0.30. Among admissible grid points the one with the smallest
  margin difference is taken. Draws with no admissible point are skipped and counted.

The matching statistics are in `m_matching.json` and are reported. The match holds at the root only; later steps are
not matched. In MZ tool 0 is necessarily less accurate than in MF (otherwise its margin would be larger), and the
tool is still named and described as reading a replica; only the fault sentence is absent, as in the fault-free twins.

## Runs

Condition K2e (the model writes its observation model as JSON, then acts), one sample per episode, 100 episodes per
model: `deepseek-chat` (fc, `offpeak_runner.py`), `codex:gpt-6.1-terra@medium`, `claude:claude-sonnet-5-5@high`;
settings of the X run (`run_m.sh`). `env.py`, `run.py` and the backends are unchanged. Rows with `api_error` are
rerun by reissuing the command; nothing else is removed. All cells and runs are reported.

## Analysis (`m_analysis.py`, run on mock logs before freezing)

Scored under the true model of each episode. Contrasts are paired by draw; nested bootstrap (scenarios, then draws),
95 %.

- **M1 (primary).** MF minus MZ: share of episodes whose first action is not the oracle's; first-step loss.
- **M2.** MF minus MZ: regret; regret at steps where the oracle would query tool 0 and the model does not.
- **M3.** K2r - K2e within each arm.
- **M4.** M1 on the pairs whose two JSONs were used as written with every parameter within 0.02.

Readings fixed now, per model (an interval that includes 0 is "not detected", never equivalence or absence):

- *Mistakes are tied to the fault-capable tool at matched difficulty*: the M1 interval for the suboptimal share lies
  above 0.
- *No difference detected at matched difficulty*: the M1 interval includes 0. If in addition the MZ suboptimal share
  has an interval lying above 25 %, we say that first-step mistakes of this size occur without any fault and that
  the under-use is not separated from decision difficulty for this model; the corresponding sentences of the paper
  are then weakened.
- *Both contribute*: the M1 interval lies above 0 and the MZ suboptimal share has an interval lying above 10 %.
- *Opposite*: the M1 interval lies below 0; reported as such.
- *Executor gap without a fault*: the M3 interval for MZ lies below 0.
- *Not carried by misread parameters*: the M4 interval lies above 0; reported without a reading if fewer than 20
  pairs qualify.

Whatever the outcome, the result is reported in the main text. It is a behavioural comparison: it does not identify
an internal mechanism, the two arms differ in tool 0's accuracy as well as in the presence of a fault, and the match
is on four root-level measures.

## Realised matching (recorded before the runs)

85 draws were needed for 50 pairs; 35 were skipped for lack of an admissible grid point, mostly draws whose MF first
decision the one-step rule gets wrong (a property that fault-free tasks of this structure rarely have). Over the 50
pairs: mean margin 0.183 (MF) and 0.181 (MZ), largest pairwise difference 0.019; mean oracle loss 4.93 and 4.92; the
one-step rule finds the best first action in 54 % of pairs (both arms, by construction); the second-best first action
is the log tool in every pair; tool 0's accuracy averages 0.90 in MF and 0.74 in MZ, the log tool's 0.71 in both.
The accepted pairs are therefore not a random sample of the first-domain draws, and this is stated with the result.
Mock checks: `run.py --mock docreader` and `--mock random` on all 100 episodes, then `m_analysis.py --mock`.
