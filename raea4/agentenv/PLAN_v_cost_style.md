# Cost sweep, documentation style and an open-weight model: analysis plan fixed before the runs

Local timestamps and hashes only (`frozen_v_cost_style.sha256`); not a third-party preregistration.

## Questions

1. **Costs.** Earlier sweeps vary the fault rate, the dependence and two accuracies, never a cost. Do the models' first
   queries respond to the cost of the replica tool, and does making an alternative costlier bring them back to it?
2. **Documentation style.** Every earlier prompt documents the tools in the same sentences. Does the under-use at low
   fault rates remain when the same facts are given as a terse field list?
3. **Open-weight model.** Do the executor gap and the under-use appear in a model with public weights?

## Episodes (`make_episodes_v.py` -> `episodes_v.jsonl`, 125 episodes)

The 25 parameter draws, scenarios and uniforms of the X sweep; persistent fault; original names and order.

- VC2, VC3 (50): fault rate 0.10, cost of the replica tool 2 / 3 (drawn: 1; reference: the X run, XP10). Exact oracle
  opens with the replica tool in 100 / 60 / 8 % of episodes at cost 1 / 2 / 3.
- VL3 (25): fault rate 0.30, cost of the log tool 3 (drawn: 1; reference XP30). Oracle opens with the replica tool in
  100 % at both costs.
- VS10, VS30 (50): parameters and outputs of XP10 / XP30 unchanged; tool documentation as a field list
  (`env.tool_doc`, `doc_style = "spec"`: `... | cost: 1 | backend: read replica <name> | fault: replica stale for the
  whole session, p = 0.30; while stale, each call returns '<benign>' with p = 0.97 | accuracy while fresh: 0.90 ...`).
  Checked before the runs: a reader of the field list recovers the true model exactly (planner regret 0).

## Runs

Condition K2e, one sample. `deepseek-chat` (fc, `offpeak_runner.py`), `codex:gpt-6.1-terra@medium`,
`claude:claude-sonnet-5-5@high`: the 125 V episodes, settings of the X run.

Open-weight model: `kimi-k3.1-pro` through the Moonshot API, fc protocol, thinking disabled
(`{"thinking": {"type": "disabled"}}`), max 2000 output tokens per call, one worker. It runs the persistent-arm
episodes of the X sweep in the fixed order XP10, XP30, XP70, XP50 (25 each) until the account credit (15 CNY in total)
is exhausted; only cells with all 25 episodes are analysed; it runs no V episodes. Pilot, before this plan: one
episode with thinking enabled (about 9,400 output tokens and six minutes; set aside in
`logs/pilot_kimi_thinking.jsonl`, not analysed, the credit does not cover this mode) and one with thinking disabled
(XP30-001, same settings as the run; it is part of the run). The model is therefore run in a weaker configuration
than its default, and is reported as such.

**Amendment, 2026-11-13, before any Kimi result was analysed** (the non-thinking run had 2 completed episodes; further
account credit became available). A second Kimi run with thinking enabled (the model's default mode; default request
settings, max 32768 output tokens): the persistent-arm X episodes in the fixed order XP10, XP30, XP70 (25 each), split
into consecutive blocks of 19 ids, one block per account key, until the credit of the available keys is exhausted; a
block is resumed with another key if one runs out; only cells with all 25 episodes are analysed. The thinking pilot
episode (XP10-000) is part of this run. Analyses K1-K4 and their readings apply to each Kimi run separately; both runs
are reported. `run.py` gained request pacing (3 requests per minute) and a key-line switch; no prompt changed.

Rows with `api_error` are rerun by reissuing the command; nothing else is removed. All cells and runs are reported.

## Analysis (`v_analysis.py`, run on a mock log before freezing)

Regret is against the true model. Nested bootstrap (scenarios, then draws), 95 %.

- **V1.** Replica first-call share at cost 1 (X run) minus at cost 3; model minus oracle share at costs 2 and 3.
- **V2.** Replica first-call share at log cost 3 minus at log cost 1 (X run); model minus oracle at log cost 3.
- **V3.** Field list, fault rates 0.10 and 0.30: model minus oracle replica first-call share; field list minus
  sentences (paired with the X run); regret difference; K2r - K2e.
- **V4.** K2r - K2e on the cost episodes; first-action match of the as-if planner (`v_predictions.json`, replica fault
  odds x g with g as in `PLAN_w_flip.md`: terra 4, Sonnet 4, flash 24) minus that of the exact oracle, cost episodes.
- **K1-K4 (open-weight run).** K2r - K2e; model minus oracle replica first-call share at 0.10 and 0.30; share at 0.10
  minus at 0.70; share of regret at steps where the replica tool is passed over.

Readings fixed now (an interval that includes 0 is "not detected", never equivalence or absence of an effect):

- *Responds to the replica tool's cost*: the V1 cost-1-minus-cost-3 interval lies above 0.
- *Returns to the replica tool when the alternative is costlier*: the V2 difference interval lies above 0. Under-use
  *remains* at log cost 3 if the V2 model-minus-oracle interval lies below 0.
- *Under-use remains under the field list*: the V3 model-minus-oracle interval lies below 0. *Style effect detected*:
  the V3 paired difference interval excludes 0.
- *Executor gap*: the K2r - K2e interval lies below 0 (V3, V4, K1). *Under-use in the open-weight run*: the K2 interval
  lies below 0. *Tracks the fault rate*: the K3 interval lies above 0.
- *As-if description supported on the cost axis*: the V4 difference interval lies above 0; otherwise not supported.
- Findings that do not hold are reported in the main text.

## Code

`env.py` (field-list branch of `tool_doc`), `k2decomp.py` (parser of the field list) and `run.py` (second API key and
endpoint) were extended; system prompts, tool schemas and parsed documentation of every earlier episode file
(main, X, Y, W, second domain) were hashed before and after the change and are identical. Hashes of this plan, the
scripts, the environment and scoring code, the episode, truth and prediction files are in
`frozen_v_cost_style.sha256`, written before the first model call of the run.
