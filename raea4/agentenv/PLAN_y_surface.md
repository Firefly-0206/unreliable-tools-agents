# Surface randomisation and repeated generation at low fault rates: analysis plan fixed before the runs

Local timestamps and hashes only (`frozen_y_surface.sha256`); not a third-party preregistration.

## Question

In the matched-marginal sweep (`PLAN_x_dependence.md`) the models open with the replica tool much less often than the
exact oracle at fault rates 0.10 and 0.30. That run used fixed tool names, one fixed tool order and one sample per
episode. Is the under-use stable (a) when names and order are randomised and (b) over repeated generations?

## Part A: randomised surface

`make_episodes_y.py --seed 2028` -> `episodes_y.jsonl`: the 100 X episodes at fault rates 0.10 and 0.30 (25 draws x 2
rates x 2 arms) with identical parameters, hidden status and pre-drawn tool outputs. Two changes: the four tools are
listed in a random order (prompt, cost line, function schemas) and renamed `tool_a`..`tool_d` by a second, independent
permutation. One permutation pair per draw; the list position of the replica tool is balanced (7/6/6/6 draws).
Descriptions, output fields and values, costs and documentation sentences are unchanged, so the scenario wording and the
word "replica" in the documentation remain. Condition K2e, one sample, the X run's protocols and settings.

Estimates (fault rates 0.10 and 0.30 pooled; per arm and both arms together; nested bootstrap over scenarios, then
draws, 95%):

- **A1.** Model minus oracle share of episodes whose first call is the replica tool, and under-use regret (regret at
  steps where the oracle would query the replica tool and the model does not), on the randomised episodes.
  Reading fixed now: the under-use *is present under a randomised surface* if the interval of the share difference,
  both arms together, lies below 0.
- **A2.** Randomised minus original surface, paired by episode with the X run's sample: first-call share, under-use
  regret, regret. Reported as estimates with intervals. An interval that includes 0 is NOT read as "no surface effect";
  an interval that excludes 0 is read as a surface effect on the size of the under-use.
- **A3.** K2r - K2e on the randomised episodes (exact planner on the model's own JSON).
- Descriptive: first-call replica share by list position of the replica tool; list position of the first call; share
  of written models whose replica group is correct.

## Part B: repeated generation

The 20 instances of the persistent arm with draws 000-009 and fault rates 0.10 and 0.30 (`episodes_x.jsonl`, original
surface), four further samples each (`--samples 5 --sample-start 1`, default sampling settings, no temperature sent).
With the X run's sample this gives five samples per instance.

- **B1.** Model minus oracle first-call share over all 100 runs (bootstrap over scenarios, then draws; the samples of a
  draw stay together). Reading fixed now: the under-use *is not an artefact of a single sample* if the interval lies
  below 0.
- Descriptive: share in the X run's sample against the new samples; number of samples (0-5) per instance that open
  with the replica tool; share of unanimous instances; intraclass correlation of the first-call choice; within- and
  between-instance SD of regret; under-use, regret and verdict regret.

If a channel turns out to be deterministic (all instances unanimous and identical trajectories), this is reported and
part B is uninformative for that channel.

## Runs

`deepseek-chat` (fc, through `offpeak_runner.py`), `codex:gpt-6.1-terra@medium`, `claude:claude-sonnet-5-5@high`;
180 episode runs per model. Rows with `api_error` are rerun by reissuing the command; nothing else is removed. All
cells and all three runs are reported.

## Code

`env.py` gained `tool_order` / `register_surface` and `k2decomp.parse_docs` reorders a permuted tool list (additive:
all prompts and function schemas of `episodes.jsonl` and `episodes_x.jsonl` are byte-identical before and after, and
`results/x_analysis.json` is reproduced byte-identically; the versions frozen for the X run are in
`frozen_x_dependence/`). `y_analysis.py` was written and run on mock logs before freezing. Hashes in
`frozen_y_surface.sha256`, written before the first model call.

Not part of this run: cost or accuracy sweeps, other structures, layered information supply in the closed loop.
