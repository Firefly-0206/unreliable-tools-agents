# Held-out confirmation of the executor gap and the regret attribution (closed loop)

Analysis plan fixed before any confirmation run. Local timestamps and file hashes only (`frozen_anatomy_confirm.sha256`);
no third-party registration, so this is "an analysis plan fixed before the runs", not a preregistration.

## Why

The step-level attribution of K2e regret (`k2e_anatomy.py`, paper appendix "where the regret arises") was run after the
original runs, and its second partition ("replica tool passed over") was chosen after looking at those logs. This run asks
whether the executor gap and that attribution hold on episodes no model has seen.

## Episodes

`episodes.jsonl` holds 50 pre-drawn episodes per family (seed 2026). Every LLM run so far used indices 0-24 only
(checked: no log row has an index >= 25). Indices 25-49 were used only by the exact, non-LLM baselines (`baselines.json`).
Confirmation set: indices 25-49 of PERSIST, SHARED, MIXED and TRANSIENT, i.e. CORE, n = 100 episodes, same generator,
same parameter ranges, new draws. Selected with `run.py --ids`.

## Runs (one sample per episode, condition K2e only)

| run | model string | protocol | settings |
|---|---|---|---|
| flash | `deepseek-chat` (non-thinking) | fc | defaults of `run.py` (1,500 output tokens per call, 12 turns), as in `ds_chat_fc_k2decomp` |
| terra | `codex:gpt-6.1-terra@medium` | text | as in `codex_terra_medium_k2decomp`; Codex CLI version is whatever is installed (recorded; it has changed since the original runs) |
| Sonnet | `claude:claude-sonnet-5-5@high` | text | as in `claude_s55`; Claude Code CLI version recorded |

Terra at effort high is not rerun (quota). K2r (exact planner on the stored JSON) is produced offline by `k2e_replay.py`.
Rows with status `api_error` are rerun by reissuing the same command (the runner resumes); episodes that end without a
submit stay in and are scored as in `score.py`; episodes without usable JSON stay in (documented fallback).

## Criteria (per run, on the 100 confirmation episodes; nested bootstrap over scenarios then episodes, 95%)

- **C1 executor gap** (flash, terra, Sonnet): the paired interval of K2r - K2e regret lies below 0.
- **C2 attribution** (terra, Sonnet): the interval of the per-episode difference
  (regret at steps where the oracle would query a replica tool and the model probes another source or stops)
  - (all other regret) lies above 0, i.e. passed-over replica tools carry more than half of the regret.
- **C3 use of the source** (terra, Sonnet): in the 75 fault-family episodes, the share of episodes whose first call is a
  replica tool, model (K2e) minus exact planner on the same JSON (K2r), has a paired interval below 0.

A run confirms if all of its criteria are met. For flash C2 and C3 are reported but are not criteria (its exploratory
share was 32%). The twin (25 TRANSIENT episodes) is reported descriptively. No other test decides the outcome, nothing is
excluded after seeing results, and all three runs are reported whatever the result. Pooled 200-episode numbers
(original + confirmation) may be reported as secondary, labelled as such.

## Code

`anatomy_confirm.py` (analysis; `--selftest` reproduces the exploratory numbers on the original logs and was run before
freezing), `k2e_anatomy.py` (`split`), `oracle.py`, `nestboot.py`, `env.py`, `k2decomp.py`, `k2e_replay.py`, `run.py`,
`episodes.jsonl`, `ground_truth.jsonl`: SHA-256 in `frozen_anatomy_confirm.sha256`, written before the first run.
