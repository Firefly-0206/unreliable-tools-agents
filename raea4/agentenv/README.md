# agentenv: closed-loop tool-use environment with injected persistent faults (v2)

In this environment a real LLM makes every decision itself: which tool to call, whether to retry, when to switch to
another source and when to stop. The environment is a finite POMDP by construction, so every decision along every
LLM trajectory can be scored against an exact Bayes-optimal oracle.

v2 (this version) addresses a review of v1: opaque ticket ids, a non-dominant probe, paired phi = 0 twins as
controls, six surface scenarios, nested-bootstrap statistics with paired contrasts, a coherent discard baseline,
a split of terminal regret into stop-timing and verdict parts, a K4c control and a fixed tool-doc block. All v1
logs are in `logs/legacy_v1/` and are NOT comparable with v2 episodes (different parameters and worlds).

## Design

**Task.** Diagnose a status s in {benign, mid, bad}. Six surface scenarios, assigned by episode index k mod 6
(8-9 episodes per scenario per family): `service` (checkout-api), `pipeline` (ETL job), `backup` (crm-db backup),
`dns` (api.shop.example), `replication` (orders-replica-2), `queue` (order-events). Each has four tools with JSON
outputs and short docs, always in this order:

| idx | role | backend | noise |
|---|---|---|---|
| 0 (A) | status tool (e.g. `health_check`) | cache replica | persistent stale-replica fault + per-call noise |
| 1 (B) | dashboard (e.g. `status_dashboard`) | the same replica, or a primary DB (PERSIST) | as above / i.i.d. only |
| 2 (L) | log tool (e.g. `log_search`) | independent log store | i.i.d. per call (repeats help) |
| 3 (P) | live probe (e.g. `synthetic_probe`) | live system | i.i.d., accuracy .85-.92, cost 4 |
| - | `submit(verdict)` | | verdict in {3 states, escalate}; ends the episode |

**Losses**: correct 0, benign verdict when the status is not benign 60 (severe), other wrong verdict 20, escalate 8.
Budget 6 cost units. Exclusion threshold tau* = 1 - 8/60 = 0.867, as in the paper.

**Families** (50 episodes each; every number rounded to 2 decimals, so the numbers in the prompt *are* the model):

| family | tools (q range, cost) | phi | design intent |
|---|---|---|---|
| PERSIST | A cache (.85-.95, 1); B primary DB (.70-.80, 2); L (.65-.75, 1); P (.85-.92, 4) | U(.25,.40) | retries of one tool share a fault |
| SHARED | A, B cache (.85-.95 / .80-.90, 1); L (.65-.75, 2); P | U(.15,.35) | two interfaces, one backend |
| MIXED | A, B cache; L (.70-.78, **1**); P | U(.20,.35) | shared fault + cheap log whose repeats help |
| VALUE | as SHARED, **value-stuck** stale replica (frozen value ~ (.8,.1,.1)) | U(.15,.35) | cache shares the specific stale value |
| TRANSIENT | **twin of SHARED-k**: identical parameters, phi = 0 | 0 | structure-matched control |
| TRANSIENT_P | **twin of PERSIST-k**: identical parameters, phi = 0 | 0 | structure-matched control |

Core families for LLM runs: PERSIST, TRANSIENT, SHARED, MIXED. The twin of episode k also shares the scenario and
all uniform draws (inverse-CDF sampling), so it has the same status and the same tool outputs whenever the
original replica is fresh. Pair id: `episode["pair"]`.

**What the agent sees.** The first user message is `Ticket <5-digit opaque number>: determine the current status of
<noun> <subject>...`; it no longer contains the family name (v1 bug). Episode ids, families and pair ids are never
in model context.

**Information conditions.** The system prompt always has the same header (task, prior, losses, costs, budget) and the
same tool-doc block (backend name, per-tool staleness incidence, accuracy) in every condition and both protocols;
in fc the same docs are also in the tool schemas. Conditions differ only by:
- **K1**: + exact observation-model block (which tools share the replica, phi, persistence, stale emission).
- **K2**: nothing added. The agent must infer the sharing from the identical backend names.
- **K3**: K1 + the one-sentence warning.
- **K4**: K1 + one sentence + a recap appended to each tool result, grouped by data source (nothing deleted).
- **K4c**: K1 + one sentence + the same recap in call order, ungrouped (control for K4: isolates the grouping).

Measured base prompt (system + schemas + ticket): K1 1,656, K2 1,404, K3 1,686, K4 1,679, K4c 1,680 tokens.

## Why the oracle is exact, and what it does not cover

The hidden state (s, z) is static (3 x 2 states, 3 x 4 for VALUE). Given (s, z), tool outputs are i.i.d. from
`Model.lik[j]`. The environment draws h and every output from exactly the objects the DP solves (`env.build_model`;
`ValueStuckModel` for VALUE), so `sim.Model.V` is the exact Bayes-optimal value.

`selftest.py` (result PASSED, `results/selftest.txt`) checks on 6 episodes (one per family):
1. `exact_eval(Oracle-DP) == V(b0, B) == exact.evaluate_exact` to 1e-8.
2. 4,000 Monte Carlo episodes through `Env` per policy: mean realised loss and mean advantage-sum match the exact
   expected loss and regret for MMI-DP, Fixed2-A, FaultUnaware-DP (all |z| < 4; observed max |z| 1.14).
3. The oracle has zero advantage at every step.
4. Coherent discard == forbid repeat cache calls, exactly (see below).
5. info + verdict parts sum to the regret on every trajectory.

Variance of the per-step regret estimate: over the 14 policy/episode pairs with nonzero regret, the advantage-sum
sd is 1.19-11.14 (median 2.29) vs realised-loss sd 3.26-24.77 (median 15.53); the ratio is 1.7-13.3. (v1 README's
"about 2.5 vs about 15" was a median-like summary and hid this range.)

Not covered: faults outside this model (partial or mid-episode staleness, faults correlated with the state, free-text
tools, rate limits); the oracle also takes the stated numbers as the truth. In K2 the oracle still uses the full true
model, so K2 regret includes the agent's failure to infer the sharing.

**Regret decomposition.** At each decision, `score.py` computes the exact posterior, P(stale), V*(b_t, B_t) and
Q*(b_t, B_t, a_t); A_t = Q* - V* >= 0 and E[sum A_t] = E[loss] - V*(b0, B). The terminal advantage is split:
- stop-timing = min_a Q*(stop, a) - V* (stopping while the oracle would still gather evidence),
- verdict = Q*(stop, a_chosen) - min_a Q*(stop, a) (wrong verdict given the belief).

**info** = probe-step advantages + stop-timing (information-gathering error); **verdict** = commitment error. Example:
submitting `escalate` immediately in SHARED-000 has regret 2.80, all of it stop-timing (oracle: call tool A).

**Invalid actions and non-compliance.** Invalid actions leave the state unchanged and are counted. Rows with status
`api_error` are excluded from all means (counted per cell in `api_errors_excluded`). A run that ends without a
submit (turn cap) stays in the means, is flagged `no_submit`, and is scored with the WORST verdict at the final belief
(not as a cheap escalation); `regret_submitted_only` is reported alongside.

## Files

| file | purpose |
|---|---|
| `env.py` | scenarios, `FAMILY_SPEC`, twins, sampling (inverse-CDF CRN), `build_model`, prompts for K1-K4c and both protocols, tool schemas, `Env` |
| `oracle.py` | `qvalues`, `score_trajectory` (per-step posterior, P(stale), advantage, stop-timing/verdict split), `exact_eval`, baseline policies |
| `nestboot.py` | nested bootstrap (scenario, then episode within scenario; samples of an episode kept together) and paired differences |
| `make_episodes.py` | writes `episodes.jsonl` and `ground_truth.jsonl` (HIDDEN), `export_prompts.json` (agent-visible), `export_hidden_outputs.json` (HIDDEN) |
| `baselines.py` | exact metrics of every non-LLM baseline, nested CIs, twin contrasts -> `results/baselines.json`, `results/baselines.txt` |
| `run.py` | LLM runner (DeepSeek, fc or text protocol; `--mock random/naive`; resumable logs; `--max-api-calls` cap) |
| `step.py` | stepping CLI for an external orchestrator |
| `score.py` | exact scoring of logs -> `results/score.json` (cells, reference policies, paired condition contrasts) |
| `selftest.py` | exactness checks |
| `project_cost.py` | token and cost projection -> `results/cost_projection.json` |

**Hidden files.** `episodes.jsonl` (contains the pre-sampled output tables), `ground_truth.jsonl` and
`export_hidden_outputs.json` reveal the hidden state (the probe's first output equals s in 80-92% of episodes) and
must only be read by code, never placed in any model's context. `export_prompts.json` holds only agent-visible
material plus `id`/`family`/`pair` join keys, which the orchestrator must not show to the model.

## How to run

```powershell
$py = "<root>\.venv\Scripts\python.exe"
& $py make_episodes.py                         # deterministic (seed 2026); 300 episodes
& $py selftest.py; & $py baselines.py --workers 4
& $py run.py --mock random --families PERSIST TRANSIENT SHARED MIXED VALUE TRANSIENT_P --conditions K1 K2 K3 K4 K4c --episodes 6 --out logs\mock_fc.jsonl
& $py run.py --models deepseek-chat --families PERSIST TRANSIENT SHARED MIXED --conditions K1 K2 K3 K4 K4c --episodes 50 --protocol fc --out logs\api_fc.jsonl
& $py score.py logs\api_fc.jsonl
```

## Results (v2 episodes)

### Exact baselines

Exact expected total loss, with regret vs Oracle-DP and a 95% nested-bootstrap CI (scenarios, then episodes) in
`results/baselines.txt`. Regret only here; 50 episodes per family.

| policy | PERSIST | TRANSIENT | SHARED | MIXED | VALUE | TRANSIENT_P |
|---|---|---|---|---|---|---|
| Oracle-DP (loss) | 5.04 | 2.48 | 5.46 | 4.40 | 5.95 | 2.55 |
| Oracle-VOI1 | +0.98 | +0.15 | +0.19 | +0.77 | +0.20 | +0.17 |
| FaultUnaware-DP | +7.39 | 0 | +5.10 | +6.39 | +3.48 | 0 |
| MMI-DP | +5.40 | 0 | +6.12 | +3.88 | +4.12 | 0 |
| MMI-VOI1 | +1.97 | +0.15 | +2.35 | +2.37 | +1.48 | +0.17 |
| Forbid any repeat (Oracle or MMI; = Oracle+Dedup) | +0.81 | +1.33 | +0.35 | +1.04 | +0.26 | +1.17 |
| Forbid repeat cache calls (Oracle or MMI) | +0.28 | +1.32 | +0.34 | +0.19 | +0.24 | +1.03 |
| Discard repeat cache reports, planner aware (coherent) | +0.28 | +1.32 | +0.34 | +0.19 | +0.24 | +1.03 |
| Discard repeat cache reports, Oracle planner unaware | +1.68 | +4.77 | +1.39 | +1.52 | +1.45 | +5.14 |
| Discard repeat cache reports, MMI planner unaware | +3.17 | +4.77 | +4.42 | +3.14 | +4.82 | +5.14 |
| Fixed2-A / Fixed3-A | +2.52 / +3.02 | +1.02 / +1.11 | +1.54 / +2.03 | +2.91 / +3.41 | +1.55 / +2.00 | +1.08 / +1.08 |
| Probe-once | +2.28 | +4.94 | +1.96 | +2.96 | +1.51 | +4.78 |
| Prior-only | +2.93 | +5.47 | +2.49 | +3.59 | +2.04 | +5.42 |

Oracle-DP behaviour (calls / repeat cache / repeat log / P(used cache) / P(probe) / severe / escalate):

| family | calls | rep cache | rep log | used cache | probe | severe | esc |
|---|---|---|---|---|---|---|---|
| PERSIST | 3.02 | 0.38 | 0.94 | 0.99 | 0.04 | 0.014 | 0.047 |
| TRANSIENT | 1.98 | 0.98 | 0.00 | 1.00 | 0.00 | 0.003 | 0.001 |
| SHARED | 1.86 | 0.33 | 0.02 | 1.00 | 0.39 | 0.013 | 0.084 |
| MIXED | 2.94 | 0.35 | 0.99 | 0.95 | 0.00 | 0.011 | 0.030 |
| VALUE | 1.92 | 0.38 | 0.02 | 0.98 | 0.42 | 0.011 | 0.068 |
| TRANSIENT_P | 2.11 | 1.11 | 0.00 | 1.00 | 0.00 | 0.003 | 0.001 |

The oracle never calls the probe first (0/50 in every family except VALUE, 1/50); mean Q(probe first) - V at the
prior is 0.93-3.22. The probe's first pre-sampled output equals s in 80-92% of episodes (its accuracy).

Twin contrasts (regret in the fault family minus regret in its phi = 0 twin, same parameters, paired; 95% CI):

| policy | SHARED - TRANSIENT | PERSIST - TRANSIENT_P |
|---|---|---|
| MMI-DP | +6.12 [+5.18, +7.11] | +5.40 [+4.09, +6.77] |
| FaultUnaware-DP | +5.10 [+4.23, +6.02] | +7.39 [+6.07, +8.75] |
| Oracle-VOI1 | +0.04 [-0.06, +0.15] | +0.80 [+0.60, +1.03] |
| Forbid any repeat | -0.98 [-1.11, -0.83] | -0.36 [-0.51, -0.21] |
| Forbid repeat cache | -0.98 [-1.11, -0.84] | -0.76 [-0.86, -0.65] |
| Fixed2-A | +0.52 [+0.38, +0.67] | +1.45 [+1.19, +1.73] |
| Probe-once | -2.98 [-3.17, -2.81] | -2.49 [-2.63, -2.37] |

Readings:
1. **A trivial probe-once strategy is no longer near-optimal.** Its regret is 1.5-3.0 in the fault families and
   4.8-4.9 in the twins, against 0.15-0.98 for one-step VOI. The oracle uses a cache tool in 95-100% of episodes,
   so the cache-tool trap is exercised. `score.py` prints the Probe-once reference and the paired difference
   (model - Probe-once) next to every LLM cell, plus regret conditional on using a cache tool.
2. **Depth amplifies the dependence error.** MMI keeps the correct single-call marginal. With one-step VOI it loses
   1.48-2.37 in the fault families; with the full DP it loses 3.88-6.12. In the twins MMI equals the oracle *by
   construction* (with phi = 0, MMI is the true model); this is a sanity check, not evidence.
3. **Forbidding repeats: cost depends on the dependence, with structure held fixed.** Forbidding repeat cache calls
   costs 0.19-0.34 in the fault families and 1.03-1.32 in the phi = 0 twins; the paired difference (same parameters)
   is -0.98 (SHARED) and -0.76 (PERSIST). Forbidding any repeat also forbids the beneficial log repeats and costs
   0.81 in PERSIST and 1.04 in MIXED (vs 0.28 / 0.19 for cache-only).
4. **MIXED now tests beneficial repeats**: the oracle repeats the log 0.99 times per episode (PERSIST 0.94).
5. **Deleting history vs forbidding future calls.** A planner that knows it will discard repeat cache reports never
   buys them (a discarded observation leaves the belief unchanged and costs >= 1, and V is non-increasing in the
   budget), so coherent discard is *exactly* forbid-cache (checked per episode in all 300 episodes and in
   selftest). Deletion differs from forbidding only when the planner does not know its reports will be deleted:
   then it keeps buying unusable repeats (Oracle planner +1.39-5.14; MMI planner +3.14-4.82, with escalation rates
   of 0.22-0.48 in the fault families). v1 reported only the incoherent variant as "history deletion"; the
   coherent baseline is now the reference.
6. With forbid on, MMI is exact (MMI+Forbid == Oracle+Forbid in every episode): once no fault-capable source is
   observed twice, the single-call marginal is all that matters.

### Mock dry run (v2)

`logs/mock_fc.jsonl` (180 runs: 6 families x 5 conditions x 6 episodes, random mock with invalid actions) and
`logs/mock_text.jsonl` (120 runs: naive "retry tool A three times" mock, 4 core families x 5 conditions x 6
episodes) ran end to end; resume reports 0 runs to do; scoring writes `results/score_mock.json`. An injected
api_error row was excluded and an injected turn-cap row was scored as `no_submit` with the worst-verdict penalty.

### Real API calls

None were made for v2 (0 of the 5 allowed). `logs/legacy_v1/` holds smoke runs made earlier against the v1 episode
set (not by this builder): `smoke_chat.jsonl`, 3 K1 episodes with `deepseek-chat` (returned model `deepseek-flash`,
3 API calls each, 5.5-5.9k prompt / 275-905 completion tokens, 0 reasoning tokens, 4.4-7.9 s, all `tool_calls`
finishes); `smoke_reasoner.jsonl`, 1 episode with `deepseek-reasoner` (returned `deepseek-flash`, 2 calls, 15.8k
prompt / 12.5k completion of which 12.1k reasoning, 62 s). These v1 episode ids now point to different worlds, so
these logs must not be scored against v2 episodes.

### Projected cost (v2 prompts)

4 core families x 5 conditions x 50 episodes = 1,000 episodes per model, fc. Prices per 1M tokens: flash in 0.15 /
out 0.60; v4-pro in 0.66 / out 1.98; peak x2. No cache discount (input is an upper bound).

| model | scenario | in / out per episode | total in / out | cost (peak) |
|---|---|---|---|---|
| deepseek-chat | 4 turns x 300 out | 8.8k / 1.2k | 8.8M / 1.2M | 2.04 (4.08) |
| deepseek-chat | 6 turns x 2,800 out | 53.0k / 16.8k | 53.0M / 16.8M | 18.03 (36.07) |
| deepseek-reasoner | 4 turns x 5k | 37.0k / 20k | 37.0M / 20.0M | 17.55 (35.10) |
| deepseek-reasoner | 6 turns x 31k | 476k / 186k | 476M / 186M | 183.00 (366.01) |
| deepseek-v4-pro | 4 turns x 5k | 37.0k / 20k | 37.0M / 20.0M | 64.02 (128.05) |
| deepseek-v4-pro | 6 turns x 31k | 476k / 186k | 476M / 186M | 682.46 (1364.92) |

The legacy smoke runs (about 5.7k in / 0.5k out per `deepseek-chat` episode; 15.8k / 12.5k for one
`deepseek-reasoner` episode) sit inside the low scenarios.
