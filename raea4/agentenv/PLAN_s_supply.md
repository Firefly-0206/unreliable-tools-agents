# Layered supply of exact quantities inside the closed loop: analysis plan fixed before the runs

Local timestamps and hashes only (`frozen_s_supply.sha256`); not a third-party preregistration.
Written 2026-11-14, after a review asked for a finer localisation of the executor gap than the same-JSON replay,
which replaces updating, prediction, valuation and stopping together.

## Question

When the model is handed, inside the loop and before every action, the exact posterior, then the exact next-report
probabilities, then one-step-ahead scores, then exact multi-step scores, and still chooses every action itself: at
which layer does the under-use of the replica tool, and the regret, go away?

## Episodes

The 50 persistent-arm episodes of the X sweep at fault rates 0.10 and 0.30 (`episodes_x.jsonl`, XP10 and XP30; exact
oracle opens with the replica tool in every one). No new episodes. These episodes were run before under K2e (X sweep,
repeated generation), never under K1 or any condition below.

## Conditions (`env.py`, `SUPPLY`; six per episode, 300 episode runs per model)

All conditions state the observation model exactly (K1), so the supplied quantities agree with what the model was
told. The supply conditions add one sentence to the system prompt and a "Decision aid" block before every action (in
the first user message and after every tool result), computed from the true model and the results so far:

- **K1**: nothing added.
- **KC**: a checklist without numbers (consider the status probabilities and the probability that the replica is
  stale; the next-report probabilities of each affordable tool; the expected final score of each action).
- **KS1**: exact status probabilities, probability that the replica is stale, and the joint (status x replica state).
- **KS2**: KS1 + exact probabilities of the next report of each affordable tool.
- **KS2v**: KS2 + expected final score of each action one step ahead (a call followed by the best immediate verdict).
- **KS3**: KS2 + expected final score of each action under optimal continuation (exact Q).

Checked before the runs (`s_analysis.py --selftest`): a reader that always takes the lowest supplied score has regret
0.000 in all 50 episodes under KS3, and 0.635 on average under KS2v (the one-step rule). System prompts and tool
schemas of every earlier first-domain episode file hash identically before and after the change of `env.py`.

## Runs

One sample per episode and condition. `deepseek-chat` (fc, `offpeak_runner.py`), `codex:gpt-6.1-terra@medium`,
`claude:claude-sonnet-5-5@high`; settings of the X run. Rows with `api_error` are rerun by reissuing the command;
nothing else is removed. All conditions and runs are reported.

## Analysis (`s_analysis.py`, run on a mock log before freezing)

Regret is against the true model. Paired by episode. Nested bootstrap (scenarios, then draws), 95 %.

- **G.** Per condition: model minus oracle share of episodes whose first call is the replica tool; under-use regret.
- **L.** Adjacent layers, regret and first-call share: KC - K1, KS1 - KC, KS2 - KS1, KS2v - KS2, KS3 - KS2v;
  also KS3 - KS2 and KS3 - K1.
- Descriptive: share of steps whose action is exactly optimal, calls, verdict regret, per condition.

Readings fixed now, per model (an interval that includes 0 is "not detected", never equivalence or absence):

- A layer *helps* if its L regret interval lies below 0, and *hurts* if it lies above 0.
- *Reminder effect*: KC - K1 regret below 0.
- *Updating is a bottleneck*: KS1 - KC regret below 0. *Prediction*: KS2 - KS1 below 0. *One-step valuation*:
  KS2v - KS2 below 0. *Multi-step planning*: KS3 - KS2v below 0.
- *Under-use remains at a layer*: the G first-call interval of that condition lies below 0.
- *Supplied exact scores are not fully used*: KS3 regret interval (per-condition mean) excludes 0 from above, i.e.
  its lower end is above 0.
- If no adjacent layer helps but KS3 - K1 lies below 0, the gain is not localised to one layer; we say so.
- A gain under KS2v or KS3 shows that the model adopts supplied scores, not that it can compute them: the
  single-decision placebo result (wrong numbers are adopted too) applies. No placebo condition is run here.
- Findings that do not hold are reported in the main text.

Not covered: independent faults, the second domain, repeated generation, a placebo with wrong numbers, K2-based
supply (documentation only).

## Code

`env.py` gained the five supply conditions (additive) and `supply_text`; `run.py` is unchanged. Hashes of this plan,
the scripts, the environment and scoring code, and the episode and truth files are in `frozen_s_supply.sha256`,
written before the first model call of the run.
