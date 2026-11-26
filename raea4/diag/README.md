# Paired diagnostic: does an LLM over-count dependent evidence?

54 items = 3 fault regimes (frequent / moderate / rare) x 6 evidence patterns
(single, same-tool x2, same-tool x3, shared backend x3, independent backends x3, OK-OK-degraded)
x 3 presentations (plain / one-sentence warning / source-deduplicated).
Every item carries its exact posterior P(healthy) and the exact expected cost of every action.

## Run (on your own computer; the cloud sandbox cannot reach api.deepseek.com)

Windows PowerShell, in this folder:

    $env:DEEPSEEK_API_KEY = "sk-..."
    python run_deepseek.py

macOS / Linux:

    export DEEPSEEK_API_KEY=sk-...
    python3 run_deepseek.py

- About 270 calls: deepseek-chat x 3 samples, deepseek-reasoner x 2 samples per item.
- Resumable: rerun the same command after an interruption.
- The key is read from the environment only and is never written to any file.
- Output: results.jsonl in this folder. Upload that file back to the chat for analysis.

## Analyse (optional, locally)

    python analyze_diag.py            # needs numpy

Reports, per model and presentation: error in P(healthy), how often the answer lands on the
wrong side of the action threshold 0.867, and the decision loss Q(chosen) - Q(best) under the
exact model; plus the key contrasts (retry3 - single, indep3 - shared3) against their exact values.

## Rerun after the first round (fixes)

- Output cap raised: deepseek-chat 8,192 tokens, deepseek-reasoner 32,768 (the old 4,000 cap
  included the reasoner's hidden reasoning, so its answers were cut off before any text).
- Truncated or unparsed answers count as not done and are retried automatically.
- Put the new run_deepseek.py and analyze_diag.py into the SAME folder as your existing
  results.jsonl and run `python run_deepseek.py` again (117 calls: 106 reasoner, 11 chat).
  To rerun only the reasoner: `python run_deepseek.py deepseek-reasoner`.
- Each record now also stores finish_reason, so truncation is visible.

## Second rerun

- deepseek-reasoner cap raised again to 65,536 tokens: at 32,768, 32 of 59 answers were still cut off
  before writing anything (mostly the single and OK-OK-degraded items), so the missing answers were not random.
  If the API rejects 65,536, the script falls back to 32,768 automatically.
- After topping up: run `python run_deepseek.py deepseek-reasoner` in the same folder. Only the 79 missing
  reasoner answers are requested; the 162 deepseek-chat answers are complete and untouched.
