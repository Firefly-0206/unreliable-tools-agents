# Additional results retained outside the compact paper

This directory preserves secondary tables and expanded explanations removed during editorial compression. No experiment was rerun and no reported numeric value was changed. Core proofs, decision contrasts, JSON completion audits, negative findings, held-out confirmations and X/Y checks remain in main.pdf. The alternative readings of the K2 documentation (formerly here as tab_k2-audit.tex) are now in Appendix D.3 of main.pdf (table "Description loss of alternative readings of the K2 documentation") and are no longer kept in this folder.

The .tex files below retain the original captions and labels. Their `tables/...` and `figs/...` inputs are relative to the manuscript root and are included in this source package. These are readable LaTeX source records, not a second mandatory manuscript. Cross-references in archived blocks retain their original labels; use this index and the compact manuscript source to locate them.

| Preserved block | Original labels | Data inputs | Reason |
|---|---|---|---|
| [tab_app-x3.tex](tab_app-x3.tex) | tab:app-x3 | tables/tab_x3 | Secondary simulation grid or resource analysis; retained verbatim outside the compact paper. |
| [tab_app-grid.tex](tab_app-grid.tex) | tab:app-grid | tables/tab_x5 | Secondary simulation grid or resource analysis; retained verbatim outside the compact paper. |
| [tab_app-x6.tex](tab_app-x6.tex) | tab:app-x6 | tables/tab_x6 | Secondary simulation grid or resource analysis; retained verbatim outside the compact paper. |
| [tab_app-d2f.tex](tab_app-d2f.tex) | tab:app-d2f | tables/tab_d2_fresh | Secondary simulation grid or resource analysis; retained verbatim outside the compact paper. |
| [tab_app-d6.tex](tab_app-d6.tex) | tab:app-d6 | tables/d6_logged | Secondary simulation grid or resource analysis; retained verbatim outside the compact paper. |
| [tab_app-metrics.tex](tab_app-metrics.tex) | tab:app-metrics | tables/tab_metrics | Secondary simulation grid or resource analysis; retained verbatim outside the compact paper. |
| [tab_app-calib.tex](tab_app-calib.tex) | tab:app-calib | tables/tab_calib | Secondary simulation grid or resource analysis; retained verbatim outside the compact paper. |
| [fig_depth.tex](fig_depth.tex) | fig:depth-ab | figs/depth_ab.pdf | Depth-only panels duplicate the retained depth-and-budget figure. |
| [app_mech-regimes.tex](app_mech-regimes.tex) | app:mech-regimes | figs/mechanism_app.pdf | Second-regime illustration; the main mechanism figure and all theoretical boundary results remain. |
| [item_10.tex](item_10.tex) |  |  | Pilot numeric table duplicates the hybrid-versus-exact comparison in the retained mechanism figure. |
| [tab_llm-budget.tex](tab_llm-budget.tex) | tab:llm-budget | tables/llm_budget | Full token/latency ledger; model caps and failure handling remain in the methods. |
| [tab_newruns.tex](tab_newruns.tex) | tab:newruns |  | Per-run resource ledger; model versions and mixed-cap resampling remain in the methods. |
| [tab_cost.tex](tab_cost.tex) | tab:cost |  | Closed-loop resource ledger; protocols, counts and failures remain in the methods. |
| [tab_agent-beh.tex](tab_agent-beh.tex) | tab:agent-beh | tables/agent_oracle_behaviour | Detailed oracle behaviour; baseline regret and the cache-use summary remain. |
| [tab_k2replay.tex](tab_k2replay.tex) | tab:k2replay | tables/agent_k2replay | Per-family replay contrasts overlap the retained one-step/full-DP table; headline paired contrasts remain in the main text. |
| [item_17.tex](item_17.tex) |  |  | Expanded prompt illustration. The compact paper retains the intervention definitions; the common header and response schema are in app_bench-prompt.tex. |
| [item_18.tex](item_18.tex) |  |  | Exploratory pilot results, with original selection and truncation caveats. |
| [app_related-moved.tex](app_related-moved.tex) | app:related-moved-archived, app:impact-archived |  | Expanded related-work discussion; closest comparisons remain in the main paper and a focused appendix summary. |
| [app_bench-parsing.tex](app_bench-parsing.tex) |  |  | Parsing, truncation and timeout rules of the single-decision benchmark; fallback scoring remains summarised in the methods. |
| [app_bench-prompt.tex](app_bench-prompt.tex) |  |  | Prompt example of the single-decision benchmark; condition definitions remain in the appendix. |
| [tab_llm-main.tex](tab_llm-main.tex) | tab:llm-main | tables/llm_main | Full single-decision table (all runs, CIs, optimal/severe/MCWA); the compact table and every in-text estimate remain. |
| [tab_llm-contrasts.tex](tab_llm-contrasts.tex) | tab:llm-contrasts | tables/llm_contrasts | Paired single-decision contrasts; those used in the text are stated there with their intervals. |
| [tab_bench-ref.tex](tab_bench-ref.tex) | tab:bench-ref | tables/bench_reference | Benchmark reference policies; the MMI reference used in the text remains. |
| [tab_t_cells.tex](tab_t_cells.tex) | tab:t-cells | tables/agent_t_cells | Fault-wording results by cell; the planned paired contrasts remain in the appendix. |
| [tab_r_cells.tex](tab_r_cells.tex) | tab:r-cells | tables/agent_r_cells | Runbook results by cell; the planned paired contrasts remain in the appendix. |
| [tab_llm-contrasts-added.tex](tab_llm-contrasts-added.tex) | tab:llm-contrasts-added | tables/llm_contrasts_added | Paired contrasts of the three added single-decision runs; their levels are in tab_llm-main.tex. |
| [tab_agent-llm-terra.tex](tab_agent-llm-terra.tex) | tab:agent-llm-terra | tables/agent_llm | Per-cell levels of every closed-loop condition for terra (K3, K4, K4c and the VALUE family included); the paper reports their contrasts. |
| [tab_agent-llm-flash.tex](tab_agent-llm-flash.tex) | tab:agent-llm-flash | tables/agent_llm_flash | Per-cell levels of every closed-loop condition for flash; the paper reports their contrasts. |
| [tab_agent-llm-added.tex](tab_agent-llm-added.tex) | tab:agent-llm-added | tables/agent_llm_new | Per-cell levels of the added closed-loop runs; the paper reports their contrasts. |
| [tab_s1.tex](tab_s1.tex) | tab:s1-full | tables/tab_s1 | Full S1 grid behind the stress-test summary table. |
| [tab_s1_check.tex](tab_s1_check.tex) | tab:s1-check | tables/tab_s1_check | Exclusion-bound check for S1 (the bound needs (A4)). |
| [tab_s2.tex](tab_s2.tex) | tab:s2-full | tables/tab_s2 | Full S2 grid, including the zero violations of the corrected cap cited in the stress-test caption. |
| [tab_s3.tex](tab_s3.tex) | tab:s3-full | tables/tab_s3 | Full S3 grid. |
| [tab_s4.tex](tab_s4.tex) | tab:s4-full | tables/tab_s4 | Full S4 grid. |
| [tab_s5.tex](tab_s5.tex) | tab:s5-full | tables/tab_s5 | Full S5 grid. |
| [tab_agent-llm-added-contrasts.tex](tab_agent-llm-added-contrasts.tex) | tab:agent-llm-added-contrasts | tables/agent_llm_new_contrasts | Paired contrasts of the added closed-loop runs (Sonnet K2-K1 included); the main-text contrast table covers terra (medium) and flash. |
| [app_near-tie.tex](app_near-tie.tex) | app:near-tie, tab:near-tie | tables/agent_neartie | Near-tie sensitivity of regret attribution; headline summary retained in main.pdf. |
| [app_t.tex](app_t.tex) | app:t, tab:t-contrasts | tables/agent_t_contrasts | Fault wording vs. probability table; summary retained in Appendix D.7 of main.pdf. |
| [app_r.tex](app_r.tex) | app:r, tab:r-contrasts | tables/agent_r_contrasts | Runbook surface experiment; summary retained in Appendix D.7 of main.pdf. |
| [app_asif.tex](app_asif.tex) | app:asif, tab:asif, tab:asif-loss | tables/agent_z, tables/agent_zl | Post hoc as-if models (fault odds, losses, prior, ambiguity); summary retained in main.pdf. |
| [app_horizon.tex](app_horizon.tex) | app:horizon, tab:horizon | tables/agent_h | Post hoc effective planning horizon analysis; summary retained in main.pdf. |
| [app_sens.tex](app_sens.tex) | app:sens, tab:sens | tables/agent_sens | Post hoc scenario leave-one-out and Bonferroni sensitivity; summary retained in main.pdf. |
| [app_k2d.tex](app_k2d.tex) | app:k2d, tab:k2d | tables/agent_k2d | Post hoc independence-assuming decision-value planner (K2d/K2dp); summary retained in main.pdf. |
| [app_units.tex](app_units.tex) | app:units, tab:units | tables/agent_u | Post hoc realized outcomes in agent units; summary retained in main.pdf. |
