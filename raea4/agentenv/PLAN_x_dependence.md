# Matched-marginal dependence experiment (closed loop): analysis plan fixed before the runs

Local timestamps and hashes only (`frozen_x_dependence.sha256`); not a third-party preregistration.

## Questions

1. **Is dependence the active ingredient?** The fault-free twins change reliability, information value and difficulty
   together. Here two arms have identical single-call report laws and differ only in whether repeated replica reports
   are dependent: persistent fault (stale for the session with probability phi) against independent fault (each call
   stale with probability phi). Is the under-use of the replica tool tied to persistence, or does it appear equally when
   the same unreliability is independent across calls?
2. **Do the models track the value of the fault-capable tool, or avoid it by default?** The exact oracle opens with the
   replica tool at low fault rates and avoids it at high ones (computed before any run: share of episodes whose optimal
   first call is the replica tool, phi = 0.10 / 0.30 / 0.50 / 0.70: persistent 100 / 100 / 44 / 0 %, independent
   100 / 92 / 24 / 0 %). Does the models' use follow this turn?

## Episodes

`make_episodes_x.py --n 25 --seed 2027` -> `episodes_x.jsonl`: 25 parameter draws of the PERSIST structure (one replica
tool, a primary-database tool, a log tool, the live probe; new seed, new draws) x 4 fault rates x 2 arms = 200 episodes.
Within a draw all eight episodes share prior, accuracies, costs, scenario, hidden status and uniforms. The only prompt
difference between arms is the replica tool's staleness sentence ("about p% of sessions; a stale replica keeps
returning ..." against "each call independently hits a stale snapshot with probability about p% (per call, not per
session) ..."). Tool order and the six scenarios (names) are as in the earlier runs; they are not randomised further.

## Runs

Condition K2e, one sample per episode, same prompts otherwise, protocols and settings as the earlier runs:
`deepseek-chat` (fc), `codex:gpt-6.1-terra@medium`, `claude:claude-sonnet-5-5@high`. Rows with `api_error` are rerun by
reissuing the command; nothing else is removed.

## Analysis (`x_analysis.py`, written and run on a mock log before freezing)

All regret is against the true model of the episode's arm. Nested bootstrap (scenarios, then draws), 95%.

- **D1 (dependence at matched marginals; fault rates 0.10 and 0.30, where the oracle opens with the replica tool).**
  Per arm: model minus oracle share of episodes whose first call is the replica tool, and the regret at steps where the
  oracle would query the replica tool and the model does not (under-use). Between arms, paired by draw: persistent
  minus independent, for the model's first-call share and for under-use.
  Readings fixed now: under-use is *tied to persistence* if the model's share is below the oracle's in the persistent
  arm (interval below 0) and the persistent-minus-independent interval of the share is below 0; it is *aversion to an
  unreliable source, not specific to dependence* if the share is below the oracle's in both arms and the between-arm
  interval includes 0; there is *no under-use here* if neither arm's interval is below 0.
- **D2 (tracking).** Per arm: the model's first-call share at 0.10 minus at 0.70 (the oracle's difference is 100
  points), the share at 0.70 (oracle 0), and the regret at 0.70 from querying the replica tool where the oracle would
  not (over-use). Reading: the model *tracks* if the 0.10-minus-0.70 interval is above 0.
- **D3 (executor gap).** K2r - K2e per arm, pooled over fault rates. Because the elicitation schema was written for
  persistent faults, we also report K2r*, which counts a group as faulty whenever its written phi is positive.

These are estimates with readings stated in advance, not pass/fail criteria; all cells and all three runs are reported.
Terra at effort high, repeated generation and randomised tool order are not part of this run.

## Code

`env.py` and `k2decomp.py` gained a per-call fault type (additive; the earlier analyses re-run byte-identically:
`results/anatomy_confirm.json`, `results/k2e_anatomy.json`; the versions frozen for the first confirmation are kept in
`frozen_anatomy_confirm/`). Hashes of the plan, `x_analysis.py`, `make_episodes_x.py`, `env.py`, `k2decomp.py`,
`oracle.py`, `nestboot.py`, `run.py`, `k2e_replay.py`, `episodes_x.jsonl`, `ground_truth_x.jsonl` are in
`frozen_x_dependence.sha256`, written before the first model call.
