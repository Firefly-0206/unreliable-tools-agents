# Bench v2: single-decision LLM benchmark with exact targets

Expands the 18-task / 54-item pilot (`diag_items.py`) to 140 base tasks in 3 surface scenarios and 15 prompt
conditions, including a calculator tool condition and a hybrid in which the LLM writes the dependence model and
the exact DP decides. Exact posteriors and action values come from `sim.Model`; the code in the parent folder is
not modified.

## Files
- `core.py`: scenarios, parameter grid, exact targets (`solve_task`), reference policies with both one-step and
  closed-loop evaluation (`make_policy`, `closed_loop`), prompt builders for every condition, the safe calculator
  (`safe_eval`), the answer parser, and the H1 model-JSON -> exact planner step (`parse_h1`).
- `build_items.py`: builds `grid_all.json` (all 432 math tasks), `items.jsonl` (140 selected base tasks),
  `prompts.jsonl` (140 x 15 = 2,100 prompts) and `items_strata.json` (counts).
- `reference.py`: reference-policy tables on the selected items and the full grid, with task and
  pattern-cluster bootstrap CIs -> `results/reference_policies.md` / `.json`.
- `run.py`: runner. Providers are `deepseek` and `mock` (`--dry-run`). Also `--export` / `--ingest` for an external
  runner, `--estimate-cost`, a resumable JSONL output, and a `--max-api-calls` hard cap.
- `score.py`: scorer -> `results/score_<name>.md` and `.json`.
- `results/mock_mock.jsonl`, `results/score_dryrun.*`: dry run with the mock model (4,497 records, 4,480 scored answers).
- `results/smoke_deepseek-chat.jsonl`, `results/score_smoke.*`: the 5-call real smoke test.

## Base tasks
The grid is 12 evidence patterns x phi {0.05, 0.15, 0.3} x q {0.75, 0.85, 0.9} x prior {0.5/0.25/0.25,
0.7/0.15/0.15} x false-benign loss {30, 60}, giving 432 math tasks. Other costs: a wrong non-benign declaration
10, escalation 4, a source call 1, and the independent verifier V 2 (q = 0.8, never stale). Budget 4.

The patterns are:
- single
- retry2, retry3
- retry_mixed (OK, OK, bad)
- retry_bad (bad, bad)
- shared3, shared_mixed
- shared_then_indep (A, B shared, then V)
- indep3, indep_bad
- transient3, transient_mixed: per-call staleness that does not persist. Each transient call is its own source,
  so the calls are never grouped, discarded or forbidden as one source.

The scenarios share the same math and differ in story and state names:
- service: {healthy, degraded, down}
- replication: {replicated, partial, missing}
- payment: {settled, pending, failed}

Selection is deterministic (`build_items.py`):
1. Drop near-ties, where the gap between the best and second-best action is below 0.05 (22 of 432 tasks).
2. Take 28 tasks per selection stratum, going round-robin over evidence patterns and alternating "repeat-wrong"
   tasks with other tasks.
3. Use each math task once and assign scenarios round-robin within each stratum.

The selection strata use the original definition, in which a re-call of the transient tool counts as
`query_same_source`. Keeping that definition leaves the 140-item set unchanged; this was checked id by id.

**The selection is category-balanced and over-samples cases where naive models fail.** Numbers on the 140 items
are not population estimates. `results/reference_policies.md` gives the full-grid numbers. The 140 tasks come from
only 12 evidence patterns, so `score.py` and `reference.py` also report a pattern-cluster bootstrap.

| Stratum | Count |
| --- | --- |
| Selection strata (28 each): declare_benign / declare_nonbenign / escalate / re-query a used tool / query_new_source | 28 / 28 / 28 / 28 / 28 |
| Exact best, reporting categories: declare_benign / declare_nonbenign / escalate / query_same_source (persistent) / query_repeat_transient / query_new_source | 28 / 28 / 28 / 10 / 18 / 28 |
| Full grid (432), same reporting categories | 182 / 66 / 59 / 23 / 24 / 78 |
| Structure: retry / shared / indep / transient | 69 / 24 / 18 / 29 |
| Scenario: service / replication / payment | 45 / 45 / 50 |
| phi 0.05 / 0.15 / 0.3 | 47 / 38 / 55 |
| q 0.75 / 0.85 / 0.9 | 49 / 48 / 43 |
| prior p50 / p70; severity 30 / 60 | 74 / 66; 62 / 78 |
| Repeat is right, persistent (best = re-query a used persistent source): retry_mixed 7, single 3 | 10 |
| Repeat is right, transient (best = re-call the transient tool): transient_mixed 16, transient3 2 | 18 |
| Repeat is wrong (fault-unaware or MMI picks a suboptimal re-query of a persistent source) | 38 |
| Several reports from one persistent source ("dependent"; includes retry_bad) | 78 (64 benign-direction, 14 non-benign) |

Patterns: transient_mixed 23, single 15, retry_mixed 15, retry_bad 14, indep_bad 14, retry2 13, shared3 13,
retry3 12, shared_mixed 9, transient3 6, indep3 4, shared_then_indep 2. Under persistent faults at budget 4,
re-querying a used source is rarely optimal: 10 selected items, and 23 of 432 on the grid.

Each item stores:
- the exact joint posterior over (state, stale flags) and P(stale) for each group;
- P(benign) for the full history and for the first-report-per-source history;
- Q of every action (exact DP within the budget), the best action, its category and the gap;
- the placebo donor;
- for each reference policy: its first action, the one-step regret, the closed-loop regret, closed-loop outcome
  components (call cost, P(escalate), P(false benign), P(other wrong), expected calls) and its P(benign).

## Reference policies: one-step vs closed loop
- **One-step regret (optimal continuation)** = Q(first action) - Q(best). The policy picks only the first action
  and the true-model optimum plays the rest. The LLM score measures this same quantity, because the benchmark
  asks for a single decision.
- **Closed-loop regret**: the policy picks every action from its own belief or constraint until it stops. It is
  the expected total loss under the true model minus the optimum (`core.closed_loop`). Closed-loop optimal play
  reproduces the exact best value on all 432 grid tasks (max error 9e-16). On persistent-structure items, this
  evaluator matches an independent reviewer implementation exactly.

The policies:
- `fault_unaware`, `mmi`, `split`: plan with their own model at every step.
- `dedup_history`: discards every later report from an already-used persistent source, **including reports
  obtained during the episode**.
- `dedup_history_initial`: discards repeat-source reports in the initial history only and keeps every new report.
- `forbid_repeat`: keeps all reports and never re-queries a used persistent source.
- `dedup_both`: dedup_history plus forbid_repeat.

Means are shown with 95% task-bootstrap CIs. The full tables, including the pattern-cluster CIs, are in
`results/reference_policies.md`.

| Policy | Selected 140: one-step | Selected 140: closed loop | Full grid 432: one-step | Full grid 432: closed loop | Grid: CL P(false benign) / P(escalate) / call cost |
| --- | --- | --- | --- | --- | --- |
| optimal | 0 | 0 | 0 | 0 | 1.8% / 19.3% / 0.52 |
| fault_unaware | 1.64 [1.15, 2.22] | 2.13 [1.56, 2.78] | 1.02 [0.79, 1.27] | 1.38 [1.10, 1.66] | 7.0% / 0.5% / 0.29 |
| mmi | 0.45 [0.30, 0.64] | 1.11 [0.70, 1.58] | 0.47 [0.36, 0.59] | 0.87 [0.68, 1.05] | 4.4% / 12.3% / 0.43 |
| split (shared only; 24 / 108) | 1.47 [0.59, 2.62] | 1.85 [0.94, 2.97] | 0.87 [0.54, 1.23] | 1.06 [0.72, 1.44] | 5.9% / 2.6% / 0.51 |
| dedup_history | 0.21 [0.15, 0.27] | 0.73 [0.48, 1.00] | 0.18 [0.14, 0.21] | 0.59 [0.46, 0.72] | 1.4% / 27.7% / 0.94 |
| dedup_history_initial | 0.21 [0.15, 0.27] | 0.24 [0.17, 0.32] | 0.18 [0.14, 0.21] | 0.20 [0.16, 0.24] | 1.3% / 26.8% / 0.65 |
| forbid_repeat | 0.02 [0.01, 0.04] | 0.02 [0.01, 0.04] | 0.02 [0.01, 0.03] | 0.02 [0.01, 0.03] | 1.9% / 19.6% / 0.47 |
| dedup_both | 0.20 [0.13, 0.29] | 0.20 [0.13, 0.29] | 0.17 [0.13, 0.21] | 0.17 [0.13, 0.21] | 1.2% / 28.3% / 0.55 |

What the closed-loop numbers show (full grid, paired; pattern-cluster CI in brackets after the task CI):
- **Discarding history is costly; forbidding repeats is almost free.** Closed loop, dedup_history - forbid_repeat
  = 0.57 [0.45, 0.70] (cluster [0.21, 1.12]), and dedup_history_initial - forbid_repeat = 0.18 [0.14, 0.22]
  (cluster [0.05, 0.35]).
- **Whether new reports are discarded too matters.** dedup_history - dedup_history_initial = 0.39 [0.29, 0.50].
- **The ordering of mmi and dedup depends on the variant.** mmi has higher closed-loop regret than both dedup
  variants. Against dedup_history_initial the gap is 0.67 [0.47, 0.86] (cluster [0.10, 1.27]). Against the strict
  dedup_history it is 0.28 [0.04, 0.52], and the pattern-cluster CI [-0.55, 1.02] includes 0. On the selected 140,
  mmi - dedup_history = 0.38 [-0.15, 0.95].
- **The misspecified models lose more in closed loop than one step suggests.** mmi 0.47 -> 0.87 and fault_unaware
  1.02 -> 1.38 on the grid, because they keep planning from a wrong belief.
- **On transient tasks the source definition decides the result.** With per-call sources, the dedup and forbid
  policies have zero regret on transient tasks. A tool-level version that treats every transient call as one
  source (the definition before this revision) has closed-loop regret of 3.72 for dedup_history, 0.85 for
  dedup_history_initial, 0.27 for forbid_repeat and 1.27 for dedup_both (72 transient grid tasks). Most of the
  previously reported one-step forbid_repeat regret of 0.12 came from these transient tasks.

## Conditions (one prompt each per base task)
| Cond | Content |
| --- | --- |
| P0 (= L0) | plain |
| P1 | + one-sentence warning (pilot wording) |
| P2 | all reports kept, grouped by data source with counts and arrival order (each transient call is its own source) |
| P3 | first report per source only (history discarded, as in the pilot); identical to P0 on transient tasks |
| L1 / L2 / L3 | + exact joint posterior / + predictive distributions and branch posteriors / + exact action values (build_interv.py wording) |
| C1 / C2 / C3 | **scaffold** controls: same blocks with every number replaced by "?" and the instruction to fill them in |
| X1 / X2 / X3 | **placebo** controls: L1-L3 wording ("computed exactly ...") and block shape, numbers from a different task with the same evidence pattern (seeded donor; donor best action = item best action on 61 of 140) |
| T1 | plain + function-calling tool `calculate(expression)`, loop up to 20 tool calls |
| H1 | LLM writes the dependence model as JSON (sources, persistence, fault probability, accuracies, stale-report distribution); the exact DP with the task's prior, losses, costs and budget chooses the action |

Each L-condition can be read against two controls. Lk - Ck = value of the numbers plus removal of the
fill-in-the-scaffold instruction, so C1 - P0 measures a scaffolding effect, not a null. Lk - Xk = value of correct
numbers at fixed wording and block shape. The answer template is now neutral:
`{"p_benign": <probability>, "action": "<action>"}`. The old example line `{"p_benign": 0.00, "action": "escalate"}`
pointed at escalate.
Median prompt length (chars): P0 1,943; L1 2,304; L2 3,235; L3 3,510; C3 3,367; X3 3,510; T1 2,177; H1 2,577.
H1 with the true structure reproduces the exact optimum on all 140 tasks (checked). With all parameters stated
in the prompt, H1 is close to transcription. It becomes informative in a variant where the dependence must be
read from tool documentation.

## Running
    <root>\.venv\Scripts\python.exe bench\build_items.py --workers 4
    <root>\.venv\Scripts\python.exe bench\reference.py --workers 4
    <root>\.venv\Scripts\python.exe bench\run.py --dry-run --conds all --samples 2 --workers 4
    <root>\.venv\Scripts\python.exe bench\run.py --model deepseek-chat --conds all --samples 3 --max-tokens 8192
    <root>\.venv\Scripts\python.exe bench\run.py --model deepseek-reasoner --conds core --samples 2 --max-tokens 65536
    # fixed-budget conditions: same prompts, separate caps; the tag is part of the run key and of the default file name
    <root>\.venv\Scripts\python.exe bench\run.py --model deepseek-reasoner --conds P0,P1,P2,P3 --max-tokens 4096 --tag cap4k
    <root>\.venv\Scripts\python.exe bench\run.py --export bench\export\haiku.jsonl --conds all --samples 2
    <root>\.venv\Scripts\python.exe bench\run.py --ingest answers.jsonl --model claude-haiku-... --out bench\results\haiku.jsonl
    <root>\.venv\Scripts\python.exe bench\score.py bench\results\deepseek_deepseek-chat.jsonl --name chat

The run key is (prompt, model, sample, cap, tag, temperature as sent, hash of `--extra-body`). Variants that differ
only in tag, temperature or extra body therefore never skip each other. `--conds core` means P0-P3, L1-L3, C1-C3
and X1-X3. Thinking models are sent no temperature.

Each record keeps:
- the returned model id;
- usage, including reasoning tokens and cache hits, summed over T1 turns, plus per-turn usage;
- finish_reason, latency, the full content and reasoning_content, and the T1 tool trace;
- `attempt`, and `call_errors` (failed HTTP calls before the recorded one);
- the parse status: ok / no_json / template_copy / bad_action / bad_p / truncated / tool_markup_in_content, or
  for H1: bad_json / inconsistent / incomplete_model.

## Scoring (`score.py`)
**Parsing.** An answer with finish_reason = length is always `truncated` and never parsed. Otherwise only a JSON
object that ends the answer is accepted (fences and emphasis are stripped), and an exact copy of the old template
line is rejected as `template_copy`. H1 is `incomplete_model` when any tool's accuracy, the stale-report
distribution (or its sum within 0.02 of 1), a fault probability or a persistence flag is missing, or a value is
out of range. Each default applied is recorded in `h1_fixes`, and the planner's choice is kept for diagnostics
only.

**Record selection.** The first record of each run key that is ok, or a model-side timeout, is scored. ok=False
records count as failed answers (`api_error` / `timeout`). Later attempts (`--retry-unparsed`, re-runs) never
replace the first; they are counted separately ("later attempts").

**Metrics:**
- Regret is the one-step regret (optimal continuation). Unparsed answers fall back to escalate; the regret on
  parsed answers only is also reported.
- optimal: unparsed counts as not optimal.
- severe: declared benign while the exact P(benign) < tau* = 1 - esc/sev.
- escalation rate and query cost.
- Over-count, on the 78 "dependent" items: p_benign is more than 0.10 beyond the exact value in the direction of
  the repeated reports. Also reported: two-sided over-confidence and the signed error.
- Correct belief, wrong action: |p - exact| <= 0.02 with a suboptimal action, by chosen-action category. For P3 it
  is also computed against the shown-evidence posterior.

**Contrasts.** Answers are averaged within each base task, and CIs are percentile bootstrap over base tasks.
Contrasts are paired by task, with a pattern-cluster bootstrap CI added. The contrasts are:
- P1-P0;
- P2-P0 and P3-P0 on persistent-structure tasks, and separately on transient tasks;
- L1-P0, L2-L1, L3-L2;
- Lk-Ck and Lk-Xk;
- C1-P0, X1-P0, T1-P0, H1-P0.

**Other outputs:**
- regret by exact-best category, with "repeat right" split into persistent and transient, and "repeat wrong";
- trade-offs by severity variant;
- a fixed-budget table: success (finished and parsed), regret, median output and reasoning tokens, and latency per
  model x cap x condition;
- a cost table;
- counts of unparsed and failed answers;
- counts of later attempts.

Answers are re-parsed at scoring time from the stored raw content.

## Smoke test (5 real calls, deepseek-chat, max_tokens 1500, item payment|retry2|phi0.3|q0.85|p50|sev6, best = escalate)
These calls were made before this revision, with the old answer template; no new API calls were made. Re-parsed
with the current rules, the results are unchanged.

| Cond | API calls | Returned model | finish | Parse | Action | p_benign (exact 0.734) | in / out tokens | Latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L3 | 1 | deepseek-flash | stop | ok | escalate | 0.734 | 948 / 130 | 1.9 s |
| P2 | 1 | deepseek-flash | length | truncated | - | - | 474 / 1500 | 7.4 s |
| H1 | 1 | deepseek-flash | stop | ok (correct structure, no defaults) | escalate (via DP) | 0.734 | 627 / 165 | 2.0 s |
| T1 | 2 | deepseek-flash | tool_calls, stop | tool_markup_in_content | - | - | 1607 / 458 (2 turns) | 4.9 s |

The test used 3,656 input and 2,253 output tokens in total, about USD 0.002, with no reasoning tokens
(non-thinking). Findings:
1. A 1,500-token cap truncates plain prompts (the pilot's median answer was 2,807 output tokens), so use at least
   8k for deepseek-chat.
2. The tool loop works: there were 2 parallel calculate calls and the results were returned. On the forced final
   turn (`tool_choice="none"`), however, the model wrote its raw tool-call markup into the content. `run.py` now
   adds an explicit final-answer user message on that turn and labels such outputs `tool_markup_in_content`. This
   fix has not been tested live, because no calls were left.
3. In that T1 answer, the model computed the per-call marginal 0.7*0.85 + 0.3*0.97 = 0.886 and squared it for two
   reports from the same persistently stale source. This is the MMI error, in one answer only.

## Projected cost (USD, list price, no cache discount; x2 at peak)
Assumptions:
- input = chars/3.8 (the smoke test gave 3.6-4.0), plus 300 tokens of tool schema for T1;
- output per answer = pilot medians: chat 2,800, reasoner 31,000. v4-pro is assumed at 25,000 (no data);
- T1 takes 6 API rounds, with the history re-sent each round.

| Run | Answers | Input M tok | Output M tok | Off-peak | Peak |
| --- | --- | --- | --- | --- | --- |
| deepseek-chat, all 15 conditions x 3 | 6,300 | 9.9 | 17.9 | 12.2 | 24.4 |
| deepseek-reasoner, all 15 conditions x 1 | 2,100 | 13.2 | 66.0 | 41.6 | 83.1 |
| deepseek-reasoner, core (P0-P3, L, C, X) x 2 | 3,640 | 3.0 | 112.8 | 68.2 | 136.3 |
| deepseek-v4-pro, all 15 conditions x 1 | 2,100 | 11.1 | 53.2 | 112.6 | 225.3 |
| deepseek-v4-pro, core x 2 | 3,640 | 3.0 | 91.0 | 182.2 | 364.3 |
| deepseek-reasoner, P0-P3 x 1, cap 16k (upper bound) | 560 | 0.3 | 9.2 | 5.6 | 11.1 |
| deepseek-reasoner, P0-P3 x 1, cap 4k (upper bound) | 560 | 0.3 | 2.3 | 1.4 | 2.8 |

The current balance (about 6 CNY) covers none of the full runs. The reasoner and v4-pro output-token assumptions
dominate, so re-estimate them from a 20-answer pilot before committing.

## Revision log (reviewer issues)
1. One-step vs closed-loop regret: renamed and added closed-loop evaluation (`core.closed_loop`), with two
   dedup_history variants.
2. Transient tasks: each call is its own source in P2/P3, in the categories and in the dedup and forbid policies.
   P2/P3 contrasts are restricted to persistent tasks, and "repeat right" is split into persistent and transient.
3. Parser: truncated answers are never parsed, only a JSON object at the end of the answer is accepted, the
   template copy is rejected, and the template is neutral.
4. Retries never replace the first attempt, and ok=False records are scored as failures.
5. The run key includes tag, temperature and extra-body hash, and the tag is in the default file name.
6. The selection bias is disclosed, full-grid numbers are added, and a pattern-cluster bootstrap is added.
7. The "dependent" stratum now includes retry_bad, with a directional and a two-sided over-confidence metric.
8. C1-C3 are relabelled as scaffold controls, and X1-X3 placebo controls are added.
9. H1 has an `incomplete_model` status and records the defaults applied.
