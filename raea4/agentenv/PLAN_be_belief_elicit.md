# In-loop belief elicitation by forking the logged runs: analysis plan fixed before the runs

Status: FROZEN (2026-11-22), hashes in `frozen_be_belief_elicit.sha256`, written before the first model call.
Decisions (user, 2026-11-22): forks A, B and C for all three models; Sonnet on the fixed 150-point subset.

**Amendment after the smoke test (2026-11-22, before any further call).** The one-row flash smoke test of fork B
(tools=None, accepted by the provider) returned a belief JSON missing its final closing brace (finish_reason stop), which
the frozen parser drops. `be_run.find_belief_r` now retries the text from the last `{"belief"` with 1–3 closing braces
appended and flags the row `belief_repaired`; the analysis reports the number of repaired rows per model and fork. The
smoke row is kept in `logs/be_ds_chat_fc_smoke.jsonl` and is not analysed; its point is run afresh. The original hashes
are kept at the end of `frozen_be_belief_elicit.sha256`; nothing else changed.

Before freezing, no model call had been made. The code exists and was run on
mock fork logs (exact, noisy and random mocks for every model). An independent code review (3 lenses, 2 refuters per
finding) confirmed 7 defects, all fixed:
- flash turns with several tool calls were silently dropped;
- a parallel-call fork reply counted as no action;
- P(stale) was undefined instead of 0 for the fault-free family;
- unusable JSONs were used as a JSON reference;
- a crash on malformed numbers in a reply;
- rank ties in the Spearman correlation;
- B5 pairing was asymmetric, and the reported means were weighted differently from their intervals.

The hashes are in `frozen_be_belief_elicit.sha256`, written before the first model call. These are local timestamps and hashes, not a third-party preregistration. Wording in the paper:
"analysis plan fixed before the runs", never "preregistered".

## Why

The S experiment supplied the exact posterior and the next-report probabilities. Neither aid detectably closed the
gap; only exact multi-step values did. That shows what the models do with correct beliefs handed to them, not what
they believe on their own when they pass over the replica. Two readings of the under-use remain open:

- (a) the model's own belief is wrong at those steps: it overstates the replica's fault, or it has already settled on
  the status;
- (b) the belief is close to exact, but the query is still valued too low. Proposition 2(c) allows a correct state
  posterior together with a wrong probe choice.

Asking the model for its belief at the points where it acted in the logged runs separates (a) from (b), without
rerunning any episode.

## Why forking is exact here

The terra and Sonnet backends are stateless: each turn re-renders the full visible transcript, and nothing hidden is
carried across turns (`codex_backend.render`; `claude_backend` uses the same renderer). For flash in fc mode, any
`reasoning_content` is stored in the transcript and sent back, as in the original run. The logs store each episode's
`transcript`, so the prompt of decision t is `transcript[:i]`, exactly what the model received. The fork differs from
the original turn only through sampling randomness, which fork A measures.

## Decision points (`be_points.py` -> `results/be_points_<model>.jsonl`)

**Source:** the K2e CORE episodes of the decomposition runs (first domain, 100 per model), i.e. the logs of
`k2e_anatomy.py` and `u_units.py`. Every model turn after the observation-model JSON whose logged action is a valid
call or submit is one point.

- Invalid turns (13 for flash) are not points.
- A flash turn with several tool calls is one point, and its first call is the logged action (73 such turns). The
  turn's later calls update the belief but are not points.
- Beliefs reconstructed at every point match `oracle.score_trajectory` exactly (posterior and advantage, all points).

**Labels**, computed offline before any fork:

- **O:** the logged action is optimal (advantage ≤ 1e-9).
- **U (under-use step):** otherwise, the oracle would query a replica (cache-backed) tool and the model does something
  else. This is `k2e_anatomy` pass_cache.
- **R:** every other step.

**References at each point**, from the logged reports. Each is computed under the TRUE model, and separately under the
model's OWN written JSON (only for usable JSONs; terra has 3 unusable episodes, 12 points):

- the status posterior;
- P(replica stale), which is 0 where the model has no fault group for the replica;
- the next-report law of every affordable tool;
- the exact Q* of every action;
- the multi-step value of calling tool j, VOI_j = R − Q*(probe j), where R is the best stop value. Cost is included,
  and VOI_j is negative when calling is not worth it;
- the one-step value.

| model  | points | U   | O   | R   | run                                                   |
|--------|--------|-----|-----|-----|-------------------------------------------------------|
| flash  | 364    | 103 | 157 | 104 | all                                                   |
| terra  | 347    | 105 | 227 | 15  | all                                                   |
| Sonnet | 309    | 113 | 180 | 16  | 150 (75 U and 75 O at random, seed 2036), see decisions |

## Forks (`be_run.py`; each is one extra model turn; nothing is executed in the environment)

- **A (reproducibility):** the logged prompt, unchanged. The action is recorded and compared with the logged action.
- **B (belief):** the logged prompt plus one user message (`be_run.belief_request`). It asks for one JSON object
  `{"belief": {"status": {...}, "replica_stale": p, "next_report": {tool: {value: p}}, "value_of_calling": {tool: x}}}`
  and no action. `value_of_calling` is defined in the message as how much lower the expected score (tool costs plus
  verdict loss) would be if the tool were called next and then the model acted as well as possible, compared with
  submitting the best verdict now, and it is negative if not worth its cost. This is VOI_j. The message lists the
  statuses and each affordable tool with its output values.
  - In fc mode no tool schemas are sent with B.
  - If the provider rejects such a request (untested; the first flash call is a one-row smoke test), B is reissued with
    the schemas, and any tool call in the reply is recorded as `acted_in_B`.
- **C (stated, then act):** B's message, followed by "then take your next action as usual". The action is compared
  with fork A. This is a self-elicited version of the S aids: does writing down one's own belief change the action?

Settings are those of the logged runs: flash at max_tokens 1500 for A and 3000 for B and C; terra and Sonnet as in
`run.py`. Rows with `api_error` are rerun by reissuing the command; nothing else is removed.

## Analysis (`be_analysis.py`; run on mock logs before freezing)

Nested bootstrap over scenarios, then episodes, with all points of an episode kept together; 95%. Means are
episode-weighted, like their intervals. Stated probability vectors are normalised, and their raw sums are reported.

- **B1, belief accuracy:**
  - total-variation distance (TV) between the stated and the exact status posterior, under the true model and under
    the own JSON;
  - absolute error of the stated stale probability;
  - log score of the stated posterior at the true status (clipped at 0.001);
  - TV of the stated next-report law against the JSON's.
- **B2:** each B1 quantity at U points minus at O points (cluster bootstrap).
- **B3, valuation:**
  - Spearman correlation, with average ranks, between the stated and the exact multi-step value of the affordable
    tools (points with ≥ 3 stated tools);
  - at U points, the share where the stated value of the replica tool the oracle would query is below the stated
    value of the logged action (stop = 0);
  - the stated minus the exact value of that tool.
- **B4, reproducibility:** fork A's agreement with the logged action (all, U, O; and on single-call replies only).
- **B5:** fork C minus fork A at U points, in the share that chooses a replica tool, paired by point where both
  parsed.

Readings fixed now, per model. An interval that includes 0 is "not detected", never equivalence or absence:

- *Belief tracks the exact posterior at under-use steps*: mean TV at U under the own JSON ≤ 0.10, and the B2 interval
  for that TV does not lie above 0.05. Then the under-use is read as (b): it occurs with close-to-exact beliefs.
- *Belief error larger at under-use steps*: the B2 interval (status TV, or stale error, own JSON) lies above 0. Then
  (a) contributes, and we report which part is wrong.
- *The replica is valued too low at U*: the interval of the stated minus the exact (own JSON) replica value lies
  below 0.
- *Forks stand in for the original*: B4 agreement ≥ 70%. Below that, every reading is reported as being about the
  forked turn, not the logged one.
- *Stating beliefs changes the replica choice*: the B5 interval excludes 0.
- Stated values are self-reports. Agreement between a stated value and an action is not evidence of how the action
  was computed, and the paper says so.

## Placement in the paper

A new appendix section, and at most one sentence in the "Which aids" subsection, with a matching cut elsewhere (the
main text has no slack).

## Cost and quota (prompt sizes measured on the points)

Each fork is a single turn of about 1.3–2.4k prompt tokens.

- **flash:** 364 points × 3 forks ≈ 1.6M prompt tokens, well under 1 CNY off-peak.
- **terra:** 347 × 3 single turns on the coding plan.
- **Sonnet:** 150 × 3 = 450 single turns. That is roughly the quota of 100–120 K2e episodes, about one 5-hour window.
  All 309 points would take about two windows.

## Decided (user, 2026-11-22)

1. **Fork C:** run for all three models.
2. **Sonnet points:** the 150-point subset (75 U + 75 O, seed 2036).
3. **Go-ahead:** given. The first flash call is a one-row smoke test of fork B without tool schemas (see Forks).

## Not covered

Hidden reasoning in the original turns (none was carried across turns, so none is lost); the second and third domains;
any causal claim about how actions are computed; R points for Sonnet in the subset.
