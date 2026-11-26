# A third, structurally different task domain (D3, fact-checking): analysis plan fixed before the runs

Local timestamps and hashes only (`frozen_d3_domain.sha256`); not a third-party preregistration.

Status 2026-11-16: environment, episodes and exact oracle built offline (`env3_data.py`, `RAEA_DOMAIN=3`,
`make_episodes_d3.py` -> `episodes_d3.jsonl`, `ground_truth_d3.jsonl`, `d3_oracle.json`; checks in `d3_selftest.py`).
Frozen 2026-11-16 on the author's go-ahead, before any model call of the D3 runs.

## Change from the first draft

The first draft proposed an approximate reference planner. Dropped: the paper's attribution needs the exact per-step
advantage Q*(b,a) - V*(b), and the replay needs a planner whose own error is zero. D3 instead stays exactly solvable
(about 8-15 s per instance) and changes the *structure* and the *task source*. The approximate-planner question stays a
stated limitation.

## Question

Do the executor gap and the under-use of an informative but fault-capable source hold when (a) the task is not
operations diagnosis, (b) no single cheap tool identifies the state, and (c) the shared fault spans two different
interfaces (two outlets on one wire item)?

## Domain (`env3_data.py`)

Fact-check a reported claim with two checkable parts (figure, attribution): four statuses accurate / wrong_figure /
wrong_source / both_wrong. Six tools, budget 7, losses 60 / 20 / 8 (severe error: publishing "accurate" when wrong).
- outlet A, outlet B (cost 1; 0.82-0.90 each): republish the same wire item; the item is wrong for the session with
  probability phi and then both report "confirmed" (97%).
- figure check, source check (cost 1; 0.85-0.93 each): each answers ONE yes/no sub-question; never faulty.
- fact-check aggregator (cost 1; 0.70-0.80): outage with probability 0.15-0.30, then arbitrary reports.
- press-office call (cost 3; 0.86-0.94): never faulty.
Ranges were tuned offline (no model calls) so that the exact oracle uses every tool: expected oracle calls per episode
over 12 draws: outlets 0.85 / 0.62, figure / source check 0.61 / 0.78, aggregator 0.09, office 0.15. With the first
ranges the office dominated and the sub-question tools and outlet B were almost unused, so the structure did not matter.
Six claim scenarios (layoffs, trial result, turnout, earthquake, merger, transfer fee). Same JSON schema as before; the
tool's sub-question (which bit it answers) is taken from the documentation, as costs are (completion rule).

## Episodes (seed 2031): 200, as in D2

- DN (100): wire fault rate drawn from 0.15-0.40. Oracle first action: outlet A 50 %, outlet B 48 %, other 2 %.
- DX (100): 25 draws x phi 0.10 / 0.30 / 0.50 / 0.70. Oracle opens with an outlet in 100 / 96 / 44 / 0 %.
- Oracle expected loss: DN 4.36; DX 3.67 / 4.42 / 4.87 / 4.95.

## Readings (fixed now; as in PLAN_d2_domain.md, with "replica tool" = either outlet)

An interval that includes 0 is never read as equivalence or absence of an effect. Nested bootstrap (scenarios, then draws), 95%.

- R1 executor gap: K2r - K2e on DN; interval below 0 -> replicates.
- R2 under-use: DX at 0.10 and 0.30, model minus oracle share of episodes opening with an outlet; below 0 -> replicates;
  above 0 -> over-use (reported as such).
- R3 tracking: outlet first-call share at 0.10 minus at 0.70; above 0 -> tracks.
- R4 descriptive: share of DN regret where an outlet is passed over; share where a sub-question tool is passed over;
  structural JSON correctness (outlets grouped together).
- New, D3-specific R5: share of regret from verdicts (the compound state makes verdict errors possible even with
  good evidence). Every reading reported whatever the outcome; a non-replication goes in the main text.

## Runs

Condition K2e, one sample, all 200 episodes, `bash run_d3.sh terra|sonnet` (`codex:gpt-6.1-terra@medium`,
`claude:claude-sonnet-5-5@high`, text protocol, as in D2). A pilot of two episodes per model checks prompts and parser;
pilot rows stay in the log and are part of the run. Rows with `api_error` are rerun by reissuing the command; nothing else
is removed. All cells and both runs are reported, whatever the outcome. flash (DeepSeek, off-peak) is a separate
decision of the author (spend), run later under this same plan if approved. Analysis: `python d3_analysis.py`.

## Code

`env.py`, `k2decomp.py` and `k2d_summary.py` gained a domain-3 branch (`RAEA_DOMAIN=3`). Additive: every system prompt,
first user message and function schema of the domain-1 episode files (episodes, x, y, w, v, t, r, m; 12,250 renderings
over ten conditions and both protocols) and of `episodes_d2.jsonl` (600 renderings, K1 / K2 / K2e) is byte-identical
before and after (checked by hashing against a reconstruction of the previous `env.py`). In D3 a reader of the documentation recovers the true model exactly
and the exact planner on it has zero replay regret (`d3_selftest.py`). Hashes in `frozen_d3_domain.sha256`, written
before the first model call.

## Paper impact

One paragraph in Sec. 5.4 + an appendix; Table 1 row "One domain, closed models?" -> "Our task family only?".
