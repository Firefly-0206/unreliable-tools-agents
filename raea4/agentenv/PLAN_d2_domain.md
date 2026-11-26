# A second, structurally different task domain: analysis plan fixed before the runs

Local timestamps and hashes only (`frozen_d2_domain.sha256`); not a third-party preregistration.

## Question

All earlier closed-loop results come from one task structure (three statuses, four tools, one fault source, budget 6).
Do the two main findings hold in a structurally different, still exactly solvable domain?
(a) An exact planner acting on the model's own written description does better than the model (executor gap).
(b) The stronger models open with the fault-capable but most informative tool less often than the exact oracle at low
fault rates, and stop using it earlier than the oracle as the fault rate rises.

## Domain (`env2_data.py`, `RAEA_DOMAIN=2`)

Five statuses (one benign), five tools, budget 7, costs 1 / 1 / 1 / 2 / 3, the same loss values (60 / 20 / 8):
A a status tool on a read replica (accuracy 0.82-0.94; the replica is stale for the session with probability phi and
then reports the benign status in 97% of calls); B a status tool on an aggregation feed (0.72-0.86; the feed is down for
the session with probability 0.15-0.35 and then reports an arbitrary status); a log sampler (0.50-0.62); a primary-store
tool (0.70-0.80, cost 2); a live probe (0.88-0.95, cost 3). The two fault sources are independent and of different kinds.
Six scenarios with new five-status wording. Same elicitation schema and protocols as before.

## Episodes (`make_episodes_d2.py`, seed 2029) -> `episodes_d2.jsonl`, 200 episodes

- DN (100): 100 parameter draws, replica fault rate drawn from 0.15-0.40.
- DX (100): 25 further draws x replica fault rate 0.10 / 0.30 / 0.50 / 0.70, same draw and uniforms within a draw.

Exact oracle, computed before any model call (`d2_oracle.json`): share of episodes opening with the replica tool,
DN 94 %; DX 100 / 100 / 72 / 0 %. Oracle expected loss: DN 4.26; DX 3.64 / 4.40 / 4.95 / 5.14.

## Runs

Condition K2e, one sample: `deepseek-chat` (fc, through `offpeak_runner.py`), `codex:gpt-6.1-terra@medium`,
`claude:claude-sonnet-5-5@high`. A pilot of four flash episodes checks that the prompts and the parser work; pilot rows
stay in the log and are part of the run. Rows with `api_error` are rerun by reissuing the command; nothing else is
removed. All cells and all three runs are reported, whatever the outcome.

## Analysis (`d2_analysis.py`, run on a mock log before freezing)

Regret is against the true model. Nested bootstrap (scenarios, then draws), 95%.

- **R1 (executor gap).** K2r - K2e on the 100 DN episodes; also on all 200.
- **R2 (under-use).** DX, fault rates 0.10 and 0.30 (the oracle opens with the replica tool in every episode): model
  minus oracle share of episodes opening with the replica tool; regret at steps where the oracle would query the
  replica tool and the model does not.
- **R3 (tracking).** DX: the model's replica first-call share at 0.10 minus at 0.70 (oracle: 100 points); regret at
  0.70 from querying the replica tool where the oracle would not.
- **R4 (descriptive).** DN: share of regret at steps where the replica tool is passed over; verdict regret; first
  actions by tool; share of written models that are structurally correct (two separate persistent fault groups).

Readings fixed now (an interval that includes 0 is never read as equivalence or absence of an effect):

- *The executor gap replicates* for a run if its R1 interval on DN lies below 0; *not detected* if it includes 0.
- *The under-use replicates* if the R2 share interval lies below 0; *not detected* if it includes 0; if the interval
  lies above 0 the run over-uses the replica tool in this domain.
- *Tracks* if the R3 interval lies above 0.
- The paper's wording for the second domain follows these readings per run; a finding that does not replicate is
  reported in the main text.

## Code

`env.py` and `k2decomp.py` gained a domain switch (`RAEA_DOMAIN=2`; additive: all prompts and function schemas of
`episodes.jsonl` and `episodes_x.jsonl` are byte-identical before and after, `results/x_analysis.json`,
`w_analysis.json` and `y_analysis.json` are reproduced byte-identically, and `selftest.py` passes; the versions frozen
for the W run are in `frozen_w_flip/`). In the new domain a reader of the documentation recovers the true model exactly
and the exact planner on the true model has zero regret (checked). Hashes in `frozen_d2_domain.sha256`, written before
the first model call.
