# K2 decomposition: conditions K2a, K1e, K2e, K2p

Goal: separate what K1 -> K2 changes at once (explicit sharing mapping; persistence / emission / independence statements;
organisation of the information) and separate identification error from planning error. Audit: `K2_AUDIT.md`.
No LLM call was made to build or test any of this (mock backends only).

## Conditions

All conditions share the header, the tool-doc block, the episodes and the common random numbers; K1-K4c prompts, episodes and logs are
unchanged (verified byte-identical to `export_prompts.json`, 300 episodes x 5 conditions x 2 protocols; existing logs re-score
identically). `CONDITIONS` is unchanged, the new ones are opt-in (`--conditions`).

| cond | system prompt | what the agent does | what is scored |
|---|---|---|---|
| K1 / K2 | existing | act | existing |
| **K2a** | K2 + an "Observation model (exact)" block with every K1 number (phi, whole-session persistence, stale emission .97 / .015, fresh accuracy, independence of fresh calls and of sources), stated per REPLICA: "the read replica `X` is stale for the whole session w.p. phi; it affects every call to every tool served from that replica". Which tools are served from `X` is NOT listed; it is read off the "Served from read replica" lines of the docs, as in K2 | act | existing metrics |
| **K2e** | K2 + elicitation paragraph | first reply = observation-model JSON (no action), then act as usual | existing metrics + recovery accuracy |
| **K1e** | K1 + the same paragraph | as K2e | control for the elicitation itself |
| **K2p** | identical to K2e (same prompt, same first user message) | one LLM call for the JSON; then the EXACT DP planner of the recovered model chooses every call and the verdict (`k2decomp.plan_and_play`, same `sim.Model` machinery as the oracle, true costs / prior / loss / budget which are all documented) | regret vs the TRUE oracle, recovery accuracy |

In the twins (phi = 0) the K2a block says "the read replica `X` is verified fresh ... every call's error is independent of every
other call" = the K1 block without tool names. Hence K2a - K1 is ~0 there by construction and K2a - K2 isolates the effect of the
explicit independence statement / block. In PERSIST there is one replica tool, so K2a is K1 without the tool name.

### JSON schema (K1e, K2e, K2p; text from `env.elicit_addendum`)

```
{"observation_model": {"tool_groups": [{"tools": ["<tool>", ...], "shared_fault": true|false, "phi": 0..1,
                                         "persistent": true|false, "stale_emission": {"<status>": p, ...}}, ...],
                       "accuracy": {"<tool>": p, ...}}}
```
The groups must partition the four tools (tools believed to read the same backend go together); `shared_fault` = the group's backend can
become stale in a session and then affects every call of every tool of the group, including repeats of one tool; `persistent` = a
fault lasts the whole session (false = each call faulty independently); `stale_emission` over the three statuses; `accuracy` when
not faulty. Two attempts are allowed (a nudge after a missing / invalid JSON). Tool schemas are not offered in these turns (fc).

### Parsing, documented defaults (all recorded in `elicit.spec.defaults / notes / clipped`)

`k2decomp.normalize_model`. Missing or invalid fields are filled with values that are STATED in the tool docs (parsed from the prompt
text, not from hidden parameters): phi <- the documented incidence of the group's replica tool, persistent <- true ("keeps
returning"), stale emission <- 0.97 on the tool's first value and the rest evenly (VALUE docs: 80 / 10 / 10 mix), accuracy <- the
documented accuracy; `shared_fault` <- phi > 0 if phi is given, else true for a documented replica tool. Unknown or duplicate tools
are ignored (first group wins), unassigned tools become independent never-faulty singletons, phi is clipped to [0, .95], accuracy to
[.02, .98]. If no usable JSON exists after two attempts (`parse_ok = false`, `fallback = true`) the model is the literal per-tool
reading of the docs (every replica tool its own persistent fault group with its documented incidence; NO sharing inferred).
Parse failures are reported as a rate and are included in K2p regret (also look at `regret_if_struct_*`).

### Recovery scores (`score.py`, from the stored spec, so they can be recomputed)

per cell and pooled over CORE: `parse_ok`, `struct_exact` (all 10 bits right: for each tool "in a persistent session-fault group", for
each of the 6 tool pairs "in the same such group"), `pair_ok` / `single_ok` (the two halves), `fault_present_ok` (replica faulty or not),
`persistent_ok` (truly faulty episodes: replica group flagged faulty AND persistent), `phi_abs_err` (mean over replica tools of
|phi_hat - phi|, 0 when no fault is claimed), `phantom_fault` (a fault claimed for a primary / log / probe tool), `emission_tv`
(total-variation distance of the stale emission, truly faulty episodes where a fault is claimed), `acc_mae`, `n_defaults`,
`fallback`, and the regret split by whether the structure was recovered exactly (`regret_if_struct_exact / _wrong`).
Contrasts (paired by episode, nested bootstrap, per family and CORE, printed and in `contrasts`): K2a-K1, K2a-K2, K2e-K2, K2p-K2,
K2p-K1, K1e-K1, K2e-K1e, K2p-K2e (only when both conditions are in the logs). `twin_contrasts_k2decomp` (fault family minus its phi = 0
twin, regret and repeat-cache, SHARED / PERSIST / both) is added for every condition of a run that contains a new condition.
Output for logs without new conditions is unchanged.

### Reading the outcomes

| pattern | reading |
|---|---|
| K2a ~ K1 << K2 (also in the twins) | the loss comes from missing explicit statements / organisation, not from inferring the mapping |
| K2a ~ K2 > K1 in SHARED / MIXED, K2a ~ K1 in twins | the loss is in reading the sharing off backend names |
| K2e: high `struct_exact`, regret ~ K2 | recovery is fine; the failure is in using the model (planning) |
| K2p ~ 0 and K2e ~ K2 | the failure is in acting on a correct-enough model, i.e. planning, not identification |
| K2p ~ K2, low `struct_exact` | identification error is the cause (cost bounded by `K2_AUDIT.md` readings) |
| K1e - K1 != 0 | eliciting the JSON itself changes behaviour: K2e - K2 and K2p are then confounded by that amount |
| TRANSIENT / TRANSIENT_P: K2p ~ 0, K2e ~ K2 | the twin gap is not identification (nothing to identify) |

## Commands

```powershell
$py = "<root>\.venv\Scripts\python.exe"; cd <root>\raea4\agentenv
$fam = "PERSIST","SHARED","MIXED","TRANSIENT","TRANSIENT_P"

# codex terra, text protocol, 25 episodes per family (the same first 25 episodes as the existing K1 / K2 logs)
& $py run.py --models "codex:gpt-6.1-terra@medium" --protocol text --families $fam --conditions K2a K2e K2p K1e --episodes 25 --workers 4 --out logs\codex_terra_medium_k2decomp.jsonl

# deepseek-chat, function-calling protocol
& $py run.py --models deepseek-chat --protocol fc --families $fam --conditions K2a K2e K2p K1e --episodes 25 --workers 4 --out logs\ds_chat_fc_k2decomp.jsonl

# score together with the existing K1 / K2 logs (new output files; contrasts need K1 / K2 in the same call)
& $py score.py logs\codex_terra_medium.jsonl logs\codex_terra_medium_k2decomp.jsonl --out results\score_codex_terra_k2decomp.json
& $py score.py logs\ds_chat_fc.jsonl logs\ds_chat_fc_k2decomp.jsonl --out results\score_ds_chat_fc_k2decomp.json

# optional, zero extra LLM calls: K2p from the JSON of the K2e runs (identical prompt => same distribution as a live K2p run)
& $py k2p_replay.py logs\codex_terra_medium_k2decomp.jsonl --out logs\codex_terra_k2p_replay.jsonl   # then score it as another log
```
Runs are resumable (same `--out`, finished episodes are skipped). `--max-api-calls N` caps real calls. VALUE is not recommended for
K2p: the spec class holds one stale-emission vector per group, so the planner cannot represent the frozen-value structure even with a
perfect recovery. The mocks `--mock docreader|nolink` exist for dry runs; `--mock random|naive` exercise the failure paths.

## LLM-call estimate (125 episodes per condition = 5 families x 25)

Measured on the existing logs, same 5 families, calls per episode (`n_api`): terra text K1 3.30, K2 3.21 (54-58 s and ~55-57k
prompt tokens per episode because Codex re-sends its own prompt on every call); deepseek-chat fc K1 3.59, K2 3.53 (6-7 s, ~6-7k
prompt tokens, 0.6-0.7k completion). Elicitation adds exactly one call per episode (a second only after an unparsable reply).

| condition | calls / episode | calls for 125 episodes (terra / deepseek-chat) |
|---|---|---|
| K2a | ~ K1: 3.3 / 3.6 | ~415 / ~450 |
| K2e | K2 + 1: ~4.3 / ~4.6 | ~540 / ~575 |
| K1e | K1 + 1: ~4.35 / ~4.7 | ~545 / ~590 |
| K2p | 1 (up to 2): ~1.05 | ~130 / ~130 |
| total, all four | | ~1,630 / ~1,750 |
| total without K1e | | ~1,085 / ~1,155 |
| total without K2p (use `k2p_replay.py`) | | ~1,500 / ~1,615 |

Wall time for terra with 4 workers: about 2 h for all four (K2p episodes are ~15-20 s). deepseek-chat tokens: about 4 x 125 x (7k in +
0.7k out) = 3.5M in / 0.35M out, i.e. well under 1 USD at the README prices (0.15 / 0.60 per 1M). Nothing was run.

## Tests done (no LLM)

- `selftest_k2decomp.py` (output in `results/selftest_k2decomp.txt`, PASSED): existing prompts identical to `export_prompts.json`; tool-doc
  block identical across all 9 conditions and equal to the parameters; K2a names no tool in the replica bullet and contains every K1
  number; K2e == K2p prompt; a perfect reader's JSON gives zero recovery error; **K2p with the true model has regret exactly 0.00 on all
  250 non-VALUE episodes** (exact planner on the true model == oracle); parser / default edge cases (6/6). `selftest.py` still PASSES with
  output identical to `results/selftest.txt`.
- `logs/mock_k2decomp.jsonl` (330 runs: 3 episodes x PERSIST SHARED MIXED TRANSIENT TRANSIENT_P x K1 K2 K2a K1e K2e K2p, mocks docreader
  (text and fc), nolink and random (text)), scored to `results/score_mock_k2decomp.{txt,json}`. docreader: parse 1.00, struct_exact 1.00,
  K2p regret 0.00 in every cell; nolink: struct_exact 0.50 in CORE (fails SHARED / MIXED), K2p regret 0.59 (MIXED) and 1.89 (SHARED);
  random: parse 0.00, fallback path exercised. Replaying K2p offline from K2e records reproduces the live K2p regrets in all 20 cells.
- Old logs: `score.py logs\codex_terra_medium.jsonl` into `results\score_codex_terra_rerun_k2decomp.json` is identical to
  `results\score_codex_terra.json` (30 cells, 35 contrasts, no extra keys; text output identical); same for `ds_chat_fc.jsonl`.

## Caveats

- K2e - K2 contains the effect of being asked to write the model; K1e - K1 measures it (so run K1e if K2e/K2p are interpreted).
- The K2p planner knows the documented prior, costs, loss and budget (all in every prompt) and nothing else about the truth.
- `export_prompts.json`, `make_episodes.py` and `step.py` are unchanged and do not cover the new conditions (they use `CONDITIONS`).
- The Codex text protocol re-renders the whole transcript on every call; the elicitation turn is part of that transcript for K2e / K1e.

## Files

Changed (additive): `env.py` (EXTRA_CONDITIONS / ELICIT / BASE_COND, `dependence_text_k2a`, `elicit_addendum`, `first_user_cond`, a
branch in `system_prompt`), `run.py` (elicitation branch in `run_episode`, mocks `docreader` / `nolink`, `--conditions` choices),
`score.py` (new contrasts, recovery section, twin contrasts for new conditions). New: `k2decomp.py`, `selftest_k2decomp.py`,
`k2p_replay.py`, `k2_audit_readings.py`, `K2_AUDIT.md`, `K2_DECOMP.md`, `logs/mock_k2decomp.jsonl`,
`results/{score_mock_k2decomp,selftest_k2decomp,k2_audit_readings,score_codex_terra_rerun_k2decomp}.*`.

## Caveats on interpretation
- K2a is not a numbers-only block: besides the K1 numbers it states the sharing rule (a stale replica affects every tool served from it), whole-session persistence and independence across sources. K2a - K2 therefore bundles numbers + persistence + independence + sharing rule; only the tool-to-replica mapping is left to be read off the docs.
- The elicitation schema (shared_fault, persistent; group tools 'exactly when they read the same backend') primes the dependence concept. K2e recovery rates are therefore upward-biased relative to an unprompted K2, and K2e - K2 carries this hint. K1e controls for the act of writing a model, not for this hint, since K1 already contains the information.
- Parser: a tool_groups list with no valid tool names falls back to the documented-default spec; cache tools omitted from the groups get their own group with documented phi; numeric phi in [5,100] is read as percent; a partial stale_emission has its remainder filled from the documentation (recorded in defaults).
