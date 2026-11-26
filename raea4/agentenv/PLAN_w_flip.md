# Accuracy sweep in which the optimal first query flips: analysis plan fixed before the runs

Local timestamps and hashes only (`frozen_w_flip.sha256`); not a third-party preregistration.

## Question

In the earlier sweeps the replica tool is the best first query at low fault rates in every episode, so "the models
under-use it" cannot be told apart from "the models avoid a source that is documented as sometimes stale, whatever the
alternatives are". Here the best first query flips between the replica tool and the log tool as an accuracy changes.
Do the models' first queries follow the flip, and where the replica tool is clearly best do they use it?

## Episodes

`make_episodes_w.py` -> `episodes_w.jsonl` (175 episodes): the 25 parameter draws, scenarios and uniforms of the X sweep;
persistent fault; original tool names and order; one accuracy changed, tool outputs re-drawn with the same uniforms.

- Axis A (125): fault rate 0.30; accuracy of the log tool set to 0.55 / 0.65 / 0.75 / 0.85 / 0.95. Exact oracle, share
  of episodes opening with the replica tool: 100 / 100 / 84 / 0 / 0 %; with the log tool: 0 / 0 / 16 / 100 / 100 %.
- Axis B (50): fault rate 0.10; fresh accuracy of the replica tool set to 0.75 / 0.65 (drawn values are 0.85-0.95; the
  X run at fault rate 0.10 is the reference). Oracle opens with the replica tool in 72 / 0 %.

Only accuracies change; costs, losses, budget, structure and documentation wording do not.

## Runs

Condition K2e, one sample, protocols and settings of the X run: `deepseek-chat` (fc, through `offpeak_runner.py`),
`codex:gpt-6.1-terra@medium`, `claude:claude-sonnet-5-5@high`. Rows with `api_error` are rerun by reissuing the
command; nothing else is removed. All cells and all three runs are reported.

## Analysis (`w_analysis.py`, run on a mock log before freezing)

Regret is against the true model. Nested bootstrap (scenarios, then draws), 95%.

- **W1.** Axis A, log accuracy 0.55 and 0.65: model minus oracle share of episodes opening with the replica tool, and
  replica under-use regret.
- **W2.** Axis A: the model's replica first-call share at 0.55 minus at 0.95 (oracle: 100 points).
- **W3.** Axis A, log accuracy 0.85 and 0.95: model minus oracle share opening with the log tool; the model's share
  opening with the replica tool; log under-use regret.
- **W4.** Axis B: model minus oracle replica share at 0.75 and at 0.65; the model's replica share at the drawn accuracy
  (X run) minus at 0.65.
- **W5.** K2r - K2e over all episodes.
- **W6 (prediction).** The post hoc as-if description of `z_asif.py` (an exact planner whose replica fault odds are
  multiplied by g) is tested out of sample with g fixed now: terra 4, Sonnet 4, flash 24. Its first actions and the
  oracle's are in `w_predictions.json`. Estimate: share of episodes whose first action (five options) equals the as-if
  prediction, minus the same for the exact oracle.

Readings fixed now (an interval that includes 0 is never read as equivalence or absence of an effect):

- *Under-use where the replica tool is clearly best*: the W1 share interval lies below 0.
- *Follows the flip*: the W2 interval lies above 0. *No detected response to the alternatives*: it includes 0.
- W1 below 0 with W2 above 0 is read as "responds to the alternatives but values the replica tool too low"; W1 below 0
  with W2 including 0 as "avoidance not detectably responsive to the alternatives"; W1 including 0 as "no under-use
  detected in this region".
- *As-if description supported out of sample*: the W6 difference interval lies above 0; *not supported*: it includes 0
  or lies below.

## Code

No change to `env.py`, `k2decomp.py`, `oracle.py` or `run.py` since the Y runs. Hashes of this plan, `w_analysis.py`,
`make_episodes_w.py`, `x_analysis.py`, `y_analysis.py`, `z_asif.py`, the environment and scoring code, the episode and
truth files and `w_predictions.json` are in `frozen_w_flip.sha256`, written before the first model call.
