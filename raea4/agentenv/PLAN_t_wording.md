# Fault wording against an equivalent probability table: analysis plan fixed before the runs

Local timestamps and hashes only (`frozen_t_wording.sha256`); not a third-party preregistration.
Written 2026-11-14, after a review asked whether the under-use of the unreliable source is a general planning
shortfall or is induced by the tool being described as possibly faulty.

## Question

Every earlier prompt describes the replica tool as one that can be stale. With faults independent across calls, the
tool is an ordinary noisy channel: its per-call report law is one fixed table of probabilities by true status. Does the
under-use of this tool at low fault rates remain when exactly that table is given and staleness is never mentioned?
And is the tool used more under the table than under the same law stated as "fresh snapshot / stale snapshot"?

## Episodes (`make_episodes_t.py` -> `episodes_t.jsonl`, 150 episodes)

The 25 parameter draws, scenarios, hidden statuses and output uniforms of the independent arm of the X sweep, at fault
rates 0.10 and 0.30 (exact oracle opens with the replica tool in 100 % / 92 % of draws). Three cells per draw and fault
rate; parameters, tool outputs and the exact oracle are identical within a triple, only the tool documentation differs:

- **TS**: the sentences of the X sweep, unchanged ("each call independently hits a stale snapshot with probability
  about p % ... a stale answer is '<benign>' about 97 % of the time. Accuracy when the replica is fresh: q ...").
- **TD**: every tool's law as report probabilities ("the value matching the true status q, each other value (1-q)/2");
  the replica tool's law decomposed: "Each call independently hits a stale snapshot with probability p ... on a fresh
  snapshot, ...; on a stale snapshot, '<benign>' 0.97, each other value 0.015, whatever the true status."
- **TT**: as TD, except that the replica tool's law is one table by true status, the exact mixture
  (1-p) fresh + p stale, e.g. "if the true status is healthy: 'ok' 0.921, 'degraded' 0.0395, 'down' 0.0395; ...".
  The prompt contains none of the words stale, fault, fresh, snapshot, incident (asserted). The backend line
  "Served from read replica <name>" is kept in all cells, and the other tools' entries are identical in TD and TT, so
  TD and TT differ only in how the replica tool's law is stated.

Checked before the runs: the TS system prompt equals the X prompt of the same episode; the TT table equals the
per-call law of the true model to 1e-9; system prompts and tool schemas of every earlier first-domain episode file
(main, X, Y, W, V) hash identically before and after the change of `env.py`.

## Runs

Condition **K2** (documentation only, no written observation model: the elicitation schema of K2e speaks of faults and
would reintroduce the wording into TT, and a mixed table is not representable in it). One sample per episode.
`deepseek-chat` (fc, `offpeak_runner.py`), `codex:gpt-6.1-terra@medium`, `claude:claude-sonnet-5-5@high`; settings of
the X run. Rows with `api_error` are rerun by reissuing the command; nothing else is removed. All cells and runs are
reported. There is no same-JSON replay here (no JSON), so this experiment says nothing about the executor gap.

## Analysis (`t_analysis.py`, run on a mock log before freezing)

Regret is against the true model. Fault rates pooled. Nested bootstrap (scenarios, then draws), 95 %.

- **T1.** Per cell: model minus oracle share of episodes whose first call is the replica tool; under-use regret
  (regret at steps where the oracle would query the replica tool and the model probes another tool or stops).
- **T2 (primary).** TT minus TD, paired by draw and fault rate: first-call share, under-use regret, regret.
- **T3.** TD minus TS, paired: the same quantities (effect of stating laws as probabilities).

Readings fixed now (an interval that includes 0 is "not detected", never equivalence or absence of an effect):

- *Under-use remains without fault wording*: the T1 model-minus-oracle interval of TT lies below 0.
  *No under-use detected without fault wording*: that interval includes 0.
- *Fault wording adds to the under-use*: the T2 first-call interval lies above 0 (the tool is opened with more often
  under the table). *Fault wording reduces it*: the interval lies below 0. Otherwise: not detected.
- *Fault wording accounts for the under-use in this setting*: T1 of TD below 0, T1 of TT includes 0, and T2 first-call
  above 0. Anything else is not that reading.
- *Format effect detected*: the T3 first-call interval excludes 0.
- Readings are per model. Findings that do not hold are reported in the main text.

Not covered: persistent faults (a table by true status cannot state a persistent fault without naming a latent
condition), the second domain, repeated generation, other wordings of the table.

## Code

`env.py` gained the two documentation styles (`doc_style` = `tdecomp`, `tmixed`; additive). Hashes of this plan, the
scripts, the environment and scoring code, and the episode and truth files are in `frozen_t_wording.sha256`, written
before the first model call of the run.
