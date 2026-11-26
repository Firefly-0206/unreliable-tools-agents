# Preregistration: held-out confirmation run (written 2026-11-02, before any model was run on these tasks)

Purpose. The 140 tasks in `items.jsonl` were used repeatedly while the protocol was being developed. This is a small
independent confirmation on new parameter combinations, with the protocol fixed below. No result of any model on
the new tasks exists at the time of writing; the only numbers computed on them are exact targets and exact
reference policies (no LLM), and a mock-provider dry run. This file makes no claim about results.

## 1. Task set (frozen)
- Built by `bench/build_items_confirm.py` (fixed seed salt `confirm-v1`; imports the original `core.solve_task` and
  `build_items.select`, so the generator, the near-tie rule (gap < 0.05 dropped), the five selection strata,
  round-robin over evidence patterns and round-robin scenario assignment are the original ones).
- Same 12 evidence patterns, 3 scenarios, costs (wrong non-benign declaration 10, escalation 4, source call 1,
  verifier V 2 with accuracy 0.8, budget 4) and answer format as the main benchmark.
- Parameter grid, with NO value shared with the original grid:

| | original | confirmation |
| --- | --- | --- |
| phi (stale probability) | 0.05, 0.15, 0.3 | 0.08, 0.22, 0.38 |
| q (source accuracy) | 0.75, 0.85, 0.9 | 0.72, 0.82, 0.92 |
| prior (benign / other / other) | p50 = .5/.25/.25, p70 = .7/.15/.15 | p60 = .6/.2/.2, p80 = .8/.1/.1 |
| false-benign loss | 30, 60 | 40, 80 |

- 12 x 3 x 3 x 2 x 2 = 432 math tasks, 28 near-ties dropped, **134 base tasks** selected (original: 140). The
  "query a used tool again" stratum has only 22 eligible tasks on this grid (28 in the original), the other four
  strata have 28. Files: `items_confirm.jsonl`, `prompts_confirm.jsonl` (134 x conditions P0, L1, L2, H1 = 536
  prompts), `items_strata_confirm.json` (all counts). Prompt text is produced by the unchanged `core.build_prompt`.
- Overlap check (computed in the builder): 0 of 134 (scenario, pattern, phi, q, prior, severity) tuples occur in the
  original 140; 0 shared with the 5-coordinate tuple without scenario; 0 shared item ids; the sets of phi, q, prior
  vectors and severities are disjoint from the original ones.
- Exact-target distribution (best action), confirmation vs original 140:

| best-action category | confirm | original |
| --- | --- | --- |
| declare_benign | 28 | 28 |
| declare_nonbenign (state 1 / state 2) | 28 (14 / 14) | 28 (14 / 14) |
| escalate | 28 | 28 |
| query_new_source | 28 | 28 |
| query_same_source (persistent re-query) | 3 | 10 |
| query_repeat_transient | 19 | 18 |
| total | 134 | 140 |

  Scenario service / replication / payment 43 / 43 / 48 (45 / 45 / 50). Structure retry / shared / indep / transient
  57 / 23 / 22 / 32 (69 / 24 / 18 / 29). "Dependent" tasks (several reports from one persistent source) 69 (78);
  repeat-wrong tasks 38 (38). Mean exact P(benign) 0.714 (0.706). Median best-vs-second gap 0.49 (0.51), minimum
  0.052 (0.054). Full-grid best-action counts (432): declare_benign 172, query_new_source 95, escalate 70,
  declare_nonbenign 62, repeat transient 19, persistent re-query 14.
- Reference policies on the 134 tasks (exact, no LLM), one-step regret: fault_unaware 2.81, mmi 0.49, split 1.31
  (n = 23), dedup_history 0.16, forbid_repeat 0.01 (original 140: 1.64, 0.45, 1.47, 0.21, 0.02).
- Consequence of the set's composition: persistent "repeat is right" tasks are rare (3), so conclusions about that
  stratum cannot be drawn from this set.

## 2. Models and decoding (identical to the main runs)
| | flash | terra |
| --- | --- | --- |
| provider / model id | `deepseek` / `deepseek-chat` (the API returned `deepseek-flash` in the main run) | `codex` / `codex:gpt-6.1-terra@medium` (text protocol: no function calling, read-only sandbox, told not to use tools) |
| decoding | `--temperature 0.7 --max-tokens 8192`, non-thinking | no sampling parameters are sent (`--max-tokens 0`; the Codex CLI sets them) |
| samples per prompt | 1 | 1 |
| conditions | P0, L1, L2, H1 | P0, L1, L2, H1 |
| run tag | `confirm` | `confirm` |
| answers | 536 | 536 |

Prompts are never edited after this registration. H1 is "model writes the observation-model JSON, the exact
planner decides"; its prompt, parser (`core.parse_h1`) and `incomplete_model` rule are the main ones. flash is the
primary model (it is the model on which the layered interventions showed an effect in the original set); terra is
a secondary replication.

## 3. Hypotheses (all on the one-step regret of `score.py`, paired by base task)
Regret = Q(chosen first action) - Q(best), the rest of the episode played optimally; unparsed, truncated, failed
or `incomplete_model` answers fall back to `escalate`, exactly as in the main scorer.
- **H_a**: regret(L2) < regret(P0). Supplying the predictive distributions and branch posteriors lowers regret.
- **H_b**: regret(H1) < regret(P0). Letting the exact planner act on the model's own JSON lowers regret.
- **H_c**: in P0, the share of parsed answers whose reported benign-state marginal is accurate
  (|p_benign - exact P(benign)| <= 0.02) and whose action is not optimal is greater than 0. This concerns the
  benign-state marginal only, not the full state posterior and not the joint posterior P(s, z | H); a positive
  result is not evidence that the model holds a correct belief.
- Secondary (reported, no confirmatory claim): L1 vs P0 (two-sided, no direction registered); L2 vs L1; H1 vs L2.
- Not tested here: C and X controls, T1, P1-P3, L3, closed-loop regret. No claim about look-ahead valuation in
  isolation is registered: the H1 executor swap changes belief updating, branch prediction, action values,
  stopping and the final action together.
- Floor warning (from the original data, stated in advance): terra's P0 regret on the original 140 was already
  0.02, so H_a and H_b may be uninformative for terra whatever the truth. This is not grounds to drop it from the
  report.

## 4. Analysis plan (script: `bench/analyze_confirm.py`, built on the main scorer)
- Same scorer functions as the main results (`score.load_records`, `score_answer`, `run.parse_into`; the first
  record of each run key is scored; answers re-parsed from stored raw content).
- One answer per prompt, averaged within base task. Intervals are two-sided 95% percentile bootstrap, 2000
  replicates, seed 0: (i) over base tasks, (ii) over the 12 evidence patterns (pattern-cluster). No one-sided tests.
- **Success criterion** (per model, per hypothesis): the conservative interval, the union of the task interval and
  the pattern-cluster interval (min of the two lower ends, max of the two upper ends), lies strictly on the
  predicted side of 0 (H_a, H_b: upper end < 0; H_c: lower end > 0). A hypothesis that fails this is reported as
  not confirmed; there is no post-hoc change of metric, subset or interval.
- A model's result "confirms the main pattern" only if both H_a and H_b meet the criterion for it. No multiplicity
  correction is applied; the four primary tests are four separate replication checks, each reported on its own.
- Always reported alongside: parsed share and statuses per condition; optimal-action share; for H1 the number of
  `incomplete_model` answers, the number of answers where the model JSON needed any default or clip (`h1_fixes`),
  other failures, and, as a sensitivity only, the regret obtained if the planner action were used even for
  incomplete models. Primary H_b uses the main rule (incomplete model = unparsed = escalate fallback). For terra,
  the number of answers with a non-empty tool trace is reported (4 of 1960 in the main run; they are kept).
- Exclusions: none by outcome. Infrastructure failures (HTTP error, timeout, usage-limit pause) are handled by the
  runner's retries and by resuming the same command; a key whose first recorded call failed is scored as in the main
  scorer (failed answer). No prompt, task or model setting may change after the first call.
- The report states which hypotheses were met and which were not; no statement beyond the registered contrasts.

## 5. Cost (no API call was made to obtain it)
- `run.py --estimate-cost` (deepseek-chat, 536 prompts, pilot-median 2,800 output tokens): input 0.40M, output
  1.50M tokens, **USD 0.96 off-peak list price, USD 1.92 at peak** (the script's list-price convention).
- Projection from the measured per-condition mean token use of the main flash run (P0 478 in / 3,052 out, L1 688 /
  2,378, L2 1,100 / 1,426, H1 650 / 125 tokens per answer): 0.39M input (0.17M were cache hits in the main run) and
  0.94M output tokens. At DeepSeek off-peak prices (CNY 1 / M input miss, 0.02 / M hit, 4 / M output) this is
  about **CNY 4.0 (about USD 0.55)**; at peak (Beijing time Mon-Fri 09-12 and 14-18, double price) about CNY 8.
- terra: 536 Codex calls, no per-token charge here; main-run median latency 12-27 s per answer, so roughly 45
  minutes with 4 workers. The plan's usage limit is handled by `codex_backend` (it sleeps until the reset).

## 6. Exact commands (run from `<root>\raea4`; PowerShell)
Frozen-file check (SHA-256, first 16 hex): `items_confirm.jsonl` 557C9DCEBB8EBD0B, `prompts_confirm.jsonl`
FE91268B0374336B, `core.py` 9359186B00C28E25, `run.py` 3686FB768A703CF1, `score.py` D07915706F69DA79,
`analyze_confirm.py` 638FE8C9AFFF8959.

```
# 0. rebuild check (deterministic; must reproduce the two hashes above)
<root>\.venv\Scripts\python.exe bench\build_items_confirm.py --workers 4

# 1. flash, off-peak only (resumable; the wrapper pauses for peak hours)
<root>\.venv\Scripts\python.exe <root>\raea4\offpeak_runner.py --out <root>\raea4\bench\results\confirm_flash.jsonl -- <root>\.venv\Scripts\python.exe <root>\raea4\bench\run.py --provider deepseek --model deepseek-chat --items-file bench\items_confirm.jsonl --prompts-file bench\prompts_confirm.jsonl --conds P0,L1,L2,H1 --samples 1 --temperature 0.7 --max-tokens 8192 --tag confirm --workers 8 --max-api-calls 1500 --out <root>\raea4\bench\results\confirm_flash.jsonl

# 2. terra
<root>\.venv\Scripts\python.exe bench\run.py --provider codex --model codex:gpt-6.1-terra@medium --items-file bench\items_confirm.jsonl --prompts-file bench\prompts_confirm.jsonl --conds P0,L1,L2,H1 --samples 1 --max-tokens 0 --tag confirm --workers 4 --out bench\results\confirm_terra.jsonl

# 3. standard score tables (same scorer as the main results) and the preregistered analysis
<root>\.venv\Scripts\python.exe bench\score.py bench\results\confirm_flash.jsonl --items-file bench\items_confirm.jsonl --name confirm_flash
<root>\.venv\Scripts\python.exe bench\score.py bench\results\confirm_terra.jsonl --items-file bench\items_confirm.jsonl --name confirm_terra
<root>\.venv\Scripts\python.exe bench\analyze_confirm.py --run flash=bench\results\confirm_flash.jsonl --run terra=bench\results\confirm_terra.jsonl --items-file bench\items_confirm.jsonl --out bench\results\confirm_analysis
```
Progress or resume: re-issue the same command; keys with an `ok` record are skipped. Dry run (no API):
`bench\run.py --provider mock --model mock --items-file bench\items_confirm.jsonl --prompts-file bench\prompts_confirm.jsonl --conds P0,L1,L2,H1 --out <scratch.jsonl>`
(536 records, scored by `score.py` and `analyze_confirm.py` without errors).

## 7. Tooling changes made for this run (defaults unchanged)
`run.py` gained `--items-file` and `--prompts-file`; `score.py` gained `--items-file` (and prints the severity
trade-off table by the severity names present in an alternate items file). With the flags absent, re-scoring the
existing flash (`ds_chat_xfam.jsonl`) and terra (`codex_terra_medium.jsonl`) records gives byte-identical
`.json` and `.md` score files compared with the pre-change scorer, and these equal the stored
`score_ds_chat_xfam_full.*` and `score_codex_terra.*`.
