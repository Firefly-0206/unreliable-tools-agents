# Round 5 run plan (exact commands, call counts, cost / time estimates)

Nothing in this file has been run except the Claude smoke test (2 live Claude bench calls and 3 for one closed-loop episode; no
DeepSeek and no Codex calls were made). All commands are PowerShell, run from `<root>\raea4`. `$py` is
`<root>\.venv\Scripts\python.exe`. All outputs are resumable (re-issue the same command after any interruption).

```powershell
cd <root>\raea4
$py  = "<root>\.venv\Scripts\python.exe"
$fam = "PERSIST","SHARED","MIXED","TRANSIENT","TRANSIENT_P"
```

## 0. What was added

* `claude_backend.py`: provider `claude:<model>@<effort>` (Claude Code CLI, `claude -p ... --tools "" --strict-mcp-config
  --setting-sources "" --no-session-persistence --output-format json`, fresh empty temp dir, prompt on stdin).
  System prompt (identical in every condition): `Answer as instructed in the user message.` This is NOT a bare API call: Claude Code still adds its SDK identity line, an environment block (temp working directory, platform, shell, OS, model name, date) and a system-reminder with account metadata (~1k extra input tokens per call; tools, MCP, settings, hooks, skills and plugins are off). Disclose it in the paper next to the Codex disclosure. The user prompt is rendered by
  `codex_backend.render` (same preamble and `[ROLE]` transcript as terra). Limit/rate/overload errors (`is_error`, text
  matching `hit your limit` / `usage limit` / rate limit / overloaded / 429 / 529 / `resets`, or `api_error_status` in 429/503/529, `|epoch` or "resets (time)" reset times) are waited out: sleep to the stated
  reset +5 min, else 1, 2, 4 ... 30 min; `--max-waits` (default 12) per call; the run never crashes on a limit.
  Per call it records model, effort, thinking tokens, `total_cost_usd` (notional on a plan), `duration_api_ms`,
  `stop_reason` (bench: `turns[i].claude_meta`; agentenv: `api[i].claude_meta` and per-episode `claude` totals).
* `bench/run.py`: `--provider claude --model claude:claude-sonnet-5-5@high --max-tokens 0` (text protocol, T1 not supported,
  like codex), `--max-waits`. Default output file names are now sanitised (`:` and `@` -> `_`); names for existing models are unchanged.
* `agentenv/run.py`: models named `claude:...` are routed next to `codex:...` (text protocol only), `--max-waits`;
  `--sample-start N` (with `--samples 2`, runs only sample index 1); `--reasoning-passback {auto,never}` and `--extra-body JSON`
  for thinking DeepSeek models; no temperature is ever sent to `deepseek-reasoner` / `deepseek-v4-pro` (it was already
  not sent in `bench/run.py`).
* Codex effort: `codex:gpt-6.1-terra@high` needs no code change (`codex_backend.parse_spec` passes the effort as
  `-c model_reasoning_effort="high"`; the run key contains the full model string, so it never collides with `@medium`).
  Not run live (no Codex calls allowed in this step).

## 1. Verification done (no experiments)

* Byte-identical re-scoring after the edits (md5 before = after): `bench/results/score_ds_chat_xfam_full.{json,md}`,
  `bench/results/score_codex_terra.{json,md}`, `agentenv/results/score_codex_terra_k2decomp.json` and
  `score_ds_chat_fc_k2decomp.json` (the `.txt` files are redirected stdout in another encoding; their decoded text is identical).
* Mock dry runs: bench claude P0/H1 (fake `claude` executable, parse OK, usage/meta recorded); agentenv claude K2e text episode
  (elicitation + turns); bench `--dry-run --model deepseek-reasoner` (P0, T1, H1); agentenv fc with a fake OpenAI client
  and model `deepseek-reasoner` (no temperature in the request; `reasoning_content` returned in the next request with
  `auto`, absent with `never`; reasoning tokens summed); `deepseek_call` request body has no `temperature`.
* Live Claude smoke (scratch, `C:\Users\anon\AppData\Local\Temp\claude\scr\live\`, not in `results/`):
  `bench_smoke.jsonl` (1 P0 + 1 H1) and `ae_smoke.jsonl` (1 closed-loop episode SHARED-000, K2e).

## 2. Smoke-test measurements for `claude:claude-sonnet-5-5@high`

| call | input tok | output tok (thinking) | API ms | wall s | notional USD |
|---|---|---|---|---|---|
| bench P0 | 1,603 | 871 (390) | 6,554 | 11.6 | 0.0151 |
| bench H1 | 1,835 | 331 (0) | 2,838 | 8.6 | 0.0106 |
| K2e turn 1 (model JSON) | 2,454 | 253 | - | 7.7 | 0.0123 |
| K2e turn 2 | 2,733 | 516 | - | 9.3 | 0.0161 |
| K2e turn 3 (submit) | 2,884 | 1,584 | - | 17.5 | 0.0274 |

One K2e episode: 3 API calls, 8.1k input, 2.4k output (1.7k thinking), 34.5 s, USD 0.056. Prompts are small (terra's prompts are 50-90k tokens per episode because Codex adds its own agent prompt). The
elicitation JSON parsed (`parse_ok=True`). Prompt caching was not hit (0 cache reads) in these single calls.

## 3. Plan, call counts, estimates

Per-call averages used: terra/codex logs (`agentenv/logs/codex_terra_medium*.jsonl`, 25 episodes per family, API calls per
episode: K1 3.25, K2 3.16, K2a 3.4, K1e 4.4, K2e 4.4) and `ds_chat_fc*.jsonl` (K1 3.6, K2e 4.7).

| block | model | calls | est. tokens | est. wall time (4 workers) |
|---|---|---|---|---|
| A bench 140: P0,P3,L1,L2,L3,C2,C3,X1,H1 (tag s55) | sonnet-5.5@high | 1,260 | 2.2M in / 0.9M out | ~1 h |
| B bench confirm 134: P0,L1,L2,H1 (tag confirm) | sonnet-5.5@high | 536 | 0.9M in / 0.4M out | ~25 min |
| C closed loop 5 fam x 25 ep x K1 K2 K2e K2a K1e (625 episodes) | sonnet-5.5@high | ~2,330 | ~6M in / ~1.9M out | ~2-2.5 h |
| D bench 140: P0,L1,L2,H1 (tag think) | deepseek-reasoner | 560 | 0.41M in / 2.2-13.7M out | ~1.5-3 h (8 workers; capped answers take ~6 min) |
| E closed loop K1,K2e (250 episodes, fc) | deepseek-reasoner | ~1,040 | 5.3-14M in / 3-14M out | ~2.5 h |
| F bench 140: P0,L2,H1 (tag high) | terra@high | 420 | - | ~1.5 h |
| G closed loop K1,K2e (250 episodes) | terra@high | ~960 | - | ~3-4 h |
| H extra sample 1: K2e and K1, 5 fam x 25 | terra@medium | ~960 (550 K2e + 410 K1) | - | ~3 h |

Claude: notional cost about USD 23 (A+B, 1,796 calls at ~0.013) plus about USD 45 (C, ~0.07 per episode) = ~USD 70 of plan-equivalent
usage; the real limit is the plan quota (the backend waits through limit windows). Thinking length varies by item (0-3k tokens), so expect +-50%.

DeepSeek flash thinking, CNY, off-peak (1 / 0.02 / 4 per M for input-miss / hit / output; peak is Beijing Mon-Fri 9-12 and 14-18, double).
Thinking length is the main uncertainty, from `bench/results/smoke_reasoner.jsonl`: 3 P0 answers with 3,452 / 4,375 / 65,535 (cap hit,
358 s) output tokens, 492 prompt tokens, ~97% of output is reasoning. The legacy fc smoke was 6.2k output per call.

| block | scenario | CNY off-peak | CNY at peak |
|---|---|---|---|
| D bench think (560 calls) | 3.9k out per answer (the two uncapped smoke answers) | 9 | 18 |
| D bench think | 24.4k per answer (smoke mean including the capped one) | 55 | 110 |
| E closed loop (250 ep) | 3k thinking per call | 20 | 41 |
| E closed loop | 6k per call (legacy smoke) | 38 | 76 |
| E closed loop | 12k per call | 73 | 146 |

E inputs include the resent `reasoning_content` within each tool-call chain (assumed all cache-miss; with 70% hits the totals are ~20% lower).
Expected D+E off-peak total: roughly CNY 30-130, central about CNY 50-90. Run both through `offpeak_runner.py`
(it pauses the job before peak and resumes after). The 65,536 cap should be kept for the bench; a handful of capped answers is itself an outcome
(`truncated`), and the first attempt is what `score.py` scores.

## 4. Commands

### Sonnet 5.5 @high

```powershell
$M = "claude:claude-sonnet-5-5@high"
# A. bench, 140 base tasks (1 sample)
& $py bench\run.py --provider claude --model $M --conds P0,P3,L1,L2,L3,C2,C3,X1,H1 --samples 1 --max-tokens 0 --tag s55 --workers 4 --out bench\results\s55_main.jsonl *> bench\results\s55_main.log
& $py bench\score.py bench\results\s55_main.jsonl --name s55_main
# B. held-out confirmation set
& $py bench\run.py --provider claude --model $M --items-file bench\items_confirm.jsonl --prompts-file bench\prompts_confirm.jsonl --conds P0,L1,L2,H1 --samples 1 --max-tokens 0 --tag confirm --workers 4 --out bench\results\confirm_s55.jsonl *> bench\results\confirm_s55.log
& $py bench\score.py bench\results\confirm_s55.jsonl --items-file bench\items_confirm.jsonl --name confirm_s55
# C. closed loop, text protocol (like terra), then the offline K2r replay
cd agentenv
& $py run.py --models $M --protocol text --families $fam --conditions K1 K2 K2e K2a K1e --episodes 25 --workers 4 --out logs\claude_s55.jsonl *> logs\claude_s55.log
& $py k2e_replay.py logs\claude_s55.jsonl --out logs\claude_s55_k2e_replay.jsonl
& $py score.py logs\claude_s55.jsonl logs\claude_s55_k2e_replay.jsonl --out results\score_claude_s55.json > results\score_claude_s55.txt
cd ..
```

If the plan's usage limit is hit the backend sleeps and continues by itself; `--max-waits` bounds the number of sleeps per call. Use
`--workers 2` if limit errors are frequent. Smoke before launching: `--limit 2` for bench, `--episodes 1 --families SHARED --conditions K2e` for agentenv.

### DeepSeek flash thinking (`deepseek-reasoner`), off-peak only

```powershell
# D. bench, thinking
& $py offpeak_runner.py --out bench\results\ds_reasoner_think.jsonl -- $py bench\run.py --provider deepseek --model deepseek-reasoner --conds P0,L1,L2,H1 --samples 1 --max-tokens 65536 --tag think --workers 8 --max-api-calls 1200 --out bench\results\ds_reasoner_think.jsonl
& $py bench\score.py bench\results\ds_reasoner_think.jsonl --name ds_reasoner_think
# E. closed loop, fc protocol (reasoning_content is passed back within tool-call chains: --reasoning-passback auto)
cd agentenv
& $py offpeak_runner.py --out logs\ds_reasoner_fc.jsonl -- $py run.py --models deepseek-reasoner --protocol fc --families $fam --conditions K1 K2e --episodes 25 --workers 4 --max-tokens 32768 --max-api-calls 3000 --out logs\ds_reasoner_fc.jsonl
& $py k2e_replay.py logs\ds_reasoner_fc.jsonl --out logs\ds_reasoner_fc_k2e_replay.jsonl
& $py score.py logs\ds_reasoner_fc.jsonl logs\ds_reasoner_fc_k2e_replay.jsonl --out results\score_ds_reasoner_fc.json > results\score_ds_reasoner_fc.txt
cd ..
```

Before E, check the first completed episode (`logs\ds_reasoner_fc.jsonl`): `status`, `finish_reasons`, `tokens.reasoning`, and that no row has
`api_error: ... 400`. If the API rejects the request with a message about `reasoning_content`, `--reasoning-passback never` will not help (tools are present); report it instead of re-running blindly.

### Terra @high

```powershell
$H = "codex:gpt-6.1-terra@high"
& $py bench\run.py --provider codex --model $H --conds P0,L2,H1 --samples 1 --max-tokens 0 --tag high --workers 4 --out bench\results\codex_terra_high.jsonl *> bench\results\codex_terra_high.log
& $py bench\score.py bench\results\codex_terra_high.jsonl --name codex_terra_high
cd agentenv
& $py run.py --models $H --protocol text --families $fam --conditions K1 K2e --episodes 25 --workers 4 --out logs\codex_terra_high.jsonl *> logs\codex_terra_high.log
& $py k2e_replay.py logs\codex_terra_high.jsonl --out logs\codex_terra_high_k2e_replay.jsonl
& $py score.py logs\codex_terra_high.jsonl logs\codex_terra_high_k2e_replay.jsonl --out results\score_codex_terra_high.json > results\score_codex_terra_high.txt
cd ..
```

### Extra samples: terra @medium, second sample (sample index 1)

Written to NEW files so the existing medium logs and their scores stay untouched.

```powershell
cd agentenv
$Med = "codex:gpt-6.1-terra@medium"
& $py run.py --models $Med --protocol text --families $fam --conditions K2e K1 --episodes 25 --samples 2 --sample-start 1 --workers 4 --out logs\codex_terra_medium_s1.jsonl *> logs\codex_terra_medium_s1.log
& $py k2e_replay.py logs\codex_terra_medium_s1.jsonl --out logs\codex_terra_medium_s1_k2e_replay.jsonl
cd ..
```

(The sample index is part of the episode run key; a Codex "sample" is an independent replicate because the CLI exposes no seed.
`k2e_replay.py` asserts one source row per (episode, sample), so replay the sample-1 log on its own as above. Do not score
sample 0 and sample 1 in one `score.py` call without checking how it averages samples.)

## 5. DeepSeek thinking + tool calls: what is implemented and what is uncertain

Implemented (per the API behaviour as I understand it and as encoded in the existing code): in thinking mode the assistant
message carries `reasoning_content`; within one tool-call chain (assistant tool_calls -> tool results -> next request) the
`reasoning_content` of those assistant messages must be sent back, otherwise the API returns HTTP 400; across a new user
message the earlier `reasoning_content` is ignored/dropped by the API. `agentenv/run.py` appends `reasoning_content` to every
assistant message that has one (`--reasoning-passback auto`, default), which is correct in both situations (extra
reasoning from earlier user turns is harmless). In the fc loop, elicitation turns (K2e, no tools) now also store `reasoning_content`: per the DeepSeek thinking-mode docs, when the request includes `tools`, the `reasoning_content` of ALL earlier turns (also turns without a tool call) must be sent back, otherwise HTTP 400. `--reasoning-passback never` strips it only from requests that carry no tools (it is ignored when tools are sent, since those would 400); usable in practice only for tool-free runs; `--extra-body '{"thinking": {"type": "enabled"}}'` is available if a model needs an explicit thinking switch.
`bench/run.py` T1 loop already passed `reasoning_content` back; T1 is not in this plan.

Uncertain / not verified (no DeepSeek calls were allowed): (1) The passback rule matches the documented thinking-mode behaviour (reviewed against api-docs.deepseek.com/guides/thinking_mode), but nothing was run live; the docs also describe thinking as a request switch (`{"thinking":{"type":"enabled"}}`, effort low/high/max) and use the model name `deepseek-flash`; it is unconfirmed that `deepseek-reasoner` is still an accepted alias, so run ONE 1-call test of it before block D, and run 1-2 K2e episodes before the full E run (look for HTTP 400 mentioning `reasoning_content`; if the alias or the switch is the problem use `--extra-body`). (2) The
legacy smoke `deepseek-reasoner` fc episode (2 calls, 12.1k reasoning tokens) was run with an earlier code version, so the
tool-call turn passed once but with only one chained call. (3) The alias `deepseek-reasoner` currently returns model `deepseek-flash`;
the cost table assumes flash prices. (4) Thinking length is heavy-tailed (1 of 3 smoke answers hit the 65,536 cap), so the cost range
above is wide; `finish_reason=length` rows are truncated outcomes, not errors. (5) A capped fc turn leaves `content=""` and no tool call, which the
loop records as `truncated` and nudges; with 32,768 max tokens this may happen more often than with 65,536, raise
`--max-tokens` to 65536 if more than a few percent of turns truncate. (6) `reasoning_tokens` is read from
`completion_tokens_details`; the smoke shows it is present for the reasoner.
