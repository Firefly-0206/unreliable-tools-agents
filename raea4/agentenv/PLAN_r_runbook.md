# Runbook case study (a more realistic surface on the same exact task): analysis plan fixed before the runs

Local timestamps and hashes only (`frozen_r_runbook.sha256`); not a third-party preregistration.
Written 2026-11-15. Motivation: every earlier prompt documents the tools in a few clean sentences with rates given as
probabilities, and returns one tidy JSON line per call. Do the executor gap and the under-use remain when the same
task is presented the way an on-call agent would meet it?

## What changes and what does not

Unchanged: the generative process, the exact oracle and all scoring. The episodes are the 75 persistent-arm episodes
of the X sweep at fault rates 0.10, 0.30 and 0.70 (25 parameter draws; same hidden statuses and tool outputs), so the
exact oracle opens with the replica tool in 100 / 100 / 0 % of episodes. This is a change of surface, not evidence
about deployed systems.

Changed (`make_episodes_r.py` -> `episodes_r.jsonl`; `env.runbook_doc`, `env.raw_output`):

1. **Documentation as runbook entries.** Each tool's entry is about three times longer and mixes the relevant facts
   with operational detail that does not bear on the decision (owner team, authentication, rate limit, latency,
   changelog).
2. **Rates as counts.** Fault rate: "in 12 of the last 40 on-call sessions the replica was not refreshing".
   Persistence is described operationally (the refresh job halts and is never restarted within a session). Stale
   emission and accuracies: counts out of 200 sampled answers. The text says past sessions and checks are
   representative, so the stated frequencies are the true rates and the oracle stays exact.
3. **Tool results as service responses**: the status value sits among request id, response time, HTTP status, latency,
   schema version, region and an empty warnings list. None of these fields depends on the hidden status or the fault.

Checked before the runs: the X episode is reproduced; the counts read back from the text by regular expressions give
exactly the true fault rate, stale emission and accuracies; the planner on the documented values has regret 0 in all
75 episodes; system prompts and tool schemas of every earlier first-domain episode file hash identically before and
after the change of `env.py`. System prompt length: about 1,000 words against 620.

## Runs

Condition K2e (the model writes its observation model as JSON, then acts), one sample per episode.
`deepseek-chat` (fc, `offpeak_runner.py`), `codex:gpt-6.1-terra@medium`, `claude:claude-sonnet-5-5@high`; settings of
the X run. Rows with `api_error` are rerun by reissuing the command; nothing else is removed. All cells and runs are
reported.

## Analysis (`r_analysis.py`, run on a mock log before freezing)

Regret is against the true model. Nested bootstrap (scenarios, then draws), 95 %.

- **Recovery.** Share of episodes whose JSON was used as written and has every parameter within 0.02
  (`k2r_strict_subset.flags`); mean of the largest fault-rate and accuracy errors.
- **R1.** K2r - K2e over all 75 episodes, and on the episodes whose JSON is within tolerance.
- **R2.** Fault rates 0.10 and 0.30: model minus oracle replica first-call share; under-use regret.
- **R3.** First-call share at 0.10 minus at 0.70; share and over-use regret at 0.70.
- **R4.** Runbook minus original surface (the same run's X log, paired by draw and fault rate): first-call share at
  0.10 and 0.30, under-use regret, regret over all rates, share of JSONs within tolerance.

Readings fixed now, per model (an interval that includes 0 is "not detected", never equivalence or absence):

- *Executor gap under the runbook surface*: the R1 interval over all episodes lies below 0. *Not carried by misread
  parameters*: the R1 interval on the within-tolerance episodes lies below 0; if fewer than 20 episodes are within
  tolerance for a run, that subset is reported without a reading.
- *The written description is sufficient*: K2r regret within tolerance at most 0.05; otherwise we report that the
  planner on the written JSON is not at the oracle and say which parameters were misread.
- *Under-use remains*: the R2 first-call interval lies below 0. *Tracks the fault rate*: the R3 interval lies above 0.
- *Surface effect detected*: an R4 interval excludes 0 (reported with its direction).
- Findings that do not hold are reported in the main text. Wording in the paper: "a more realistic surface on the
  same exact task", never deployment evidence.

Not covered: the second domain, independent faults, repeated generation, live services, documentation that omits
facts or gives uncertain rates (here the counts determine the rates exactly).

## Code

`env.py` gained `runbook_doc`, `raw_output` and the two episode flags; `k2decomp.parse_docs` returns the documented
(= true) values for runbook episodes; `run.py` is unchanged. Hashes of this plan, the scripts, the environment and
scoring code, and the episode and truth files are in `frozen_r_runbook.sha256`, written before the first model call.
