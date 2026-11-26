# RAEA: code, data and results (snapshot 2026-10-31)

Requirements: Python 3.8+, numpy (simulation and analysis). The DeepSeek runners use the standard library only.
No API key is stored anywhere in this folder: runners read DEEPSEEK_API_KEY from the environment.

## Simulation (Sections 4-6 of the draft)
- sim.py        model (world state x latent fault groups), exact belief update, budgeted DP / VOI-k,
                baselines (fault-unaware, MMI, split shared, dedup), EM-fitted statistical model
- exact.py      exact expected-loss evaluator, dedup wrapper for any model, separate random streams
                (SeedSequence spawn), shared raw calibration log with nested prefixes, cluster-bootstrap CIs
- runs.py       python3 runs.py X1 X2 X3 X4 X5 X6
                X1 dependence controls (+ source dedup on each model)   X2 model sources / structural errors
                X3 calibration size (nested prefixes)                     X4 shift   X5 fault-rate x severity grid
                X6 planning-depth control
- contrasts.py  paired per-instance contrasts
- mechanism.py  exact posterior curves (Figure 2)
- results/      X1-X6 and contrasts outputs (JSON); logA.txt / logB.txt run logs

## Real-LLM pilot (Section 7.1)
- diag_items.py           builds diag/diag_items.json: 54 items with exact posteriors and action values
- diag/run_deepseek.py    local runner (resumable)
- diag/results.jsonl      all 480 call records of the pilot (270 valid; truncated and no-balance records kept)
- diag/analyze_diag.py    Table 6 numbers -> summary.json, final_report.md, error_types.md
- diag/paired_pres.py     task-level paired intervals (warning vs plain, dedup vs plain), dedup scored
                          against the posterior of the shown evidence -> paired_pres.json
- requery_cases.py        exact references for the 9 re-query cases -> diag/requery_cases.json
- diag/requery_outputs/   the 9 original answers

## Next experiments (not yet run)
- build_interv.py + diag/interv/   oracle-information intervention L0-L3 with structure-matched controls
                                   C1-C3
- build_elicit.py + diag/elicit/   structured elicitation of every intermediate quantity (optional)
