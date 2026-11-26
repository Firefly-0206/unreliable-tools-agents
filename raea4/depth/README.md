# depth/: planning depth, simple baselines, metrics and calibration cost (D1-D5)

These experiments build on `../sim.py` and `../exact.py` and leave both files unchanged. Every number is an exact expectation. The evaluator sums over all observation sequences, so there is no episode sampling. Each policy's regret is paired against Oracle-DP on the same instance, and the intervals are cluster-bootstrap 95% CIs over instances (4,000 resamples). Instances use the same SeedSequence scheme as `exact.run()`, with new seeds 401-404. The families are RETRY and SHARED from `sim.spec` (f = 0.97, so A1-A4 hold).

| file | content |
|---|---|
| `core.py` | memoised planner `Q()` (VOI-k / budgeted DP; optional repeat ban, and optional discard dynamics so the planner knows a repeat will be ignored); exact evaluator `evaluate()` with **discard** (belief ignores repeat-source reports; `plan_discard=True` = consistent, otherwise naive), **forbid** (repeat-source probes never planned or executed), **pool** ("keep all reports, group by source": per-source geometric mean of likelihoods), `asrc` (agent's source map: true grouping by default, `tool_sources()` = tool identity only), and an inherited forced prefix; repeat over-valuation features `overvalue_max()` (all repeat pairs incl. cross-tool) and `overvalue_plan()` (along the DP plan's reachable trajectories); data-driven independent model `indep_fit()` (smoothing `alpha`, model name `fit_indep_a<alpha>@n`); instance builder; 4-process pool; paired cluster bootstrap |
| `experiments.py` | `python experiments.py D1 D2 D3 D4 D5` (D5 reads `results/D3_rows.json`) |
| `figs.py` | `python figs.py` -> `figs/*.pdf` (Okabe-Ito palette) |
| `results/D*.json` | summaries (mean of every metric, regret, lo, hi, n); `D3_rows.json` per-instance rows |

Run with `<root>\.venv\Scripts\python.exe` from this directory. Wall-clock times with 4 workers (revision 2): D1 203 s, D2 216 s, D3 56 s, D4 153 s, D5 under 1 s.

Instance counts: D1, D2 and D3 use 300 instances per family; D4 uses 200 per family for each of 12 loss settings. Calibration logs come from `exact.raw_calibration` (3 calls per tool per labelled episode, nested prefixes). D1, D2 and D4 use n = 100; D3 uses n = 30, 100 and 300.

Logs of the revision-2 run: `results/run_log_v2.txt` (the older `run_log*.txt` are from revision 1).

## Results (regret vs Oracle-DP, RETRY / SHARED, B = 6, default loss)

- **D1 depth.** MMI: VOI-1 0.37 / 0.67, DP 1.89 / 1.94; fitted wrong-mechanism 0.99 / 1.28 -> 1.56 / 1.53; fault-unaware 2.55 / 1.92 -> 2.62 / 1.98; split-shared (SHARED) 0.28 -> 0.68. With exactly zero planned repeats, DP-minus-VOI-1 amplification is 0.000 (MMI, split, fault-unaware) and 0.004 [0.000, 0.013] (wrong mechanism, RETRY); for MMI this follows from the MMI identity. Best predictor of amplification: plan-weighted over-valuation of repeats (Spearman MMI 0.71 / 0.45, wrong mechanism 0.57 / 0.37); the revision-1 tool-0-only feature gave 0.12 / -0.18.
- **D2 dedup operations.** Fresh: forbid = both = consistent discard = group+forbid (MMI 0.03 / 0.04, fitted indep 0.11 / 0.11); naive discard is a strawman (MMI 2.15 / 2.73). Inherited (A,A,A / A,B,A): fitted dependence model keeping everything 0.14 / 0.10; with the correct model forbid-only is as good (exact 0.02 / 0.04); discarding inherited repeats costs 0.49 / 0.56 even for the exact model; group+forbid 0.23-0.42 (true grouping); misspecified models under any variant >= 0.25 / 0.36.
- **D3 same labelled data (fresh).** Dependence model 0.31 / 0.30 (n=30), 0.07 / 0.11 (100), 0.02 / 0.02 (300); independent + first report per source 0.25 / 0.29, 0.09 / 0.15, 0.06 / 0.08 (needs the true grouping); per tool (SHARED) 0.58 / 0.41 / 0.33; plain independent DP 1.18 -> 1.84 (RETRY), and still 1.61 -> 1.92 with smoothing alpha = 0.001.
- **D4 metrics.** Main setting: MMI severe 8.7% / 10.0% and 1.60 / 1.92 calls vs Oracle 0.7% / 0.9% and 0.69 / 0.70; across 12 loss settings, mean regret fitted dependence 0.06 / 0.05, first report per source 0.12 / 0.09, per tool 0.12 / 0.38, MMI 3.73 / 3.94.
- **D5 calibration cost.** Vs fault-unaware DP, break-even after 116-6,838 deployment episodes depending on n and c_label; vs first report per source the same-data gain is -0.06 to 0.05 per episode; vs first report per tool (SHARED) 0.27-0.31 per episode.

Metrics per policy: `total`, `cost` (probe cost), `dloss` (decision loss), `severe` (benign commitment when the state is not benign), `wrong` (other wrong commitments), `esc`, `calls`, `repeat` (repeat-source calls), and `regret`/`lo`/`hi` against Oracle-DP. D1 adds `pred_total`, the agent's own model's expected loss for its own policy; `optimism` = true - predicted; `pred_repeat`; and `root`, the planner's root value.
