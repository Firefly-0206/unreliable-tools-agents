# Reference policies (exact, no LLM)

python bench/reference.py [--workers 4]   -> results/reference_policies.md / .json  one-step regret (optimal continuation)  Q(first action) - Q(best): the policy picks only the FIRST action, the                                         true-model optimum plays the rest. This is also what the LLM single-                                         decision score measures. closed-loop regret                      the policy picks EVERY action from its own belief / constraint until it                                         stops; expected total loss under the true model minus the optimum                                         (core.make_policy / core.closed_loop). The 140 selected items are category-balanced and over-sample cases where naive models fail; the full grid (432 tasks, every evidence pattern x parameter combination once) is the population reference. CIs: 95% percentile bootstrap over tasks, and over the 12 evidence patterns (cluster bootstrap) as a sensitivity check.

Policies (core.make_policy): fault_unaware / mmi / split plan with their own model at every step. dedup_history discards every later report from an already-used persistent source, including reports obtained during the episode; dedup_history_initial discards repeat-source reports in the initial history only and keeps every new report; forbid_repeat keeps all reports and never re-queries a used persistent source; dedup_both = dedup_history + forbid_repeat. Transient tool calls are separate sources (never discarded or forbidden). CL = closed loop.


### Selected 140 base tasks (category-balanced; NOT a population estimate) (n = 140)

| Policy | n | One-step regret (optimal continuation) | Closed-loop regret | Closed-loop regret, pattern-cluster CI | First action optimal % | Closed-loop optimal % | One-step severe % | CL P(false benign) % | CL P(escalate) % | CL call cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| optimal | 140 | 0 | 0 | - | 100 | 100 | - | 1.6 | 26.9 | 0.68 |
| fault_unaware | 140 | 1.64 [1.15, 2.22] | 2.13 [1.56, 2.78] | 2.13 [1.01, 3.13] | 46 | 41 | 26 | 9.2 | 0.7 | 0.38 |
| mmi | 140 | 0.45 [0.30, 0.64] | 1.11 [0.70, 1.58] | 1.11 [0.34, 2.11] | 64 | 58 | 9 | 4.5 | 18.4 | 0.61 |
| split | 24 | 1.47 [0.59, 2.62] | 1.85 [0.94, 2.97] | 1.85 [0.00, 2.35] | 25 | 25 | 25 | 7.3 | 7.9 | 0.80 |
| dedup_history | 140 | 0.21 [0.15, 0.27] | 0.73 [0.48, 1.00] | 0.73 [0.17, 1.60] | 71 | 69 | 0 | 1.3 | 37.9 | 1.17 |
| dedup_history_initial | 140 | 0.21 [0.15, 0.27] | 0.24 [0.17, 0.32] | 0.24 [0.06, 0.48] | 71 | 71 | 0 | 1.2 | 35.5 | 0.81 |
| forbid_repeat | 140 | 0.02 [0.01, 0.04] | 0.02 [0.01, 0.04] | 0.02 [0.00, 0.06] | 93 | 93 | 0 | 1.7 | 27.6 | 0.64 |
| dedup_both | 140 | 0.20 [0.13, 0.29] | 0.20 [0.13, 0.29] | 0.20 [0.06, 0.38] | 77 | 77 | 0 | 1.1 | 38.6 | 0.68 |

Paired closed-loop regret differences (task bootstrap; pattern-cluster bootstrap):

| A - B | d closed-loop regret | cluster CI |
| --- | --- | --- |
| dedup_history - forbid_repeat | 0.71 [0.46, 0.98] | [0.16, 1.58] |
| dedup_history_initial - forbid_repeat | 0.22 [0.15, 0.29] | [0.05, 0.44] |
| dedup_history - dedup_history_initial | 0.49 [0.28, 0.71] | [0.06, 1.12] |
| dedup_both - forbid_repeat | 0.18 [0.11, 0.27] | [0.05, 0.34] |
| mmi - dedup_history | 0.38 [-0.15, 0.95] | [-0.92, 1.61] |
| mmi - dedup_history_initial | 0.87 [0.44, 1.35] | [0.03, 1.90] |
| fault_unaware - mmi | 1.02 [0.54, 1.55] | [0.04, 2.17] |

### Full grid (all 432 math tasks) (n = 432)

| Policy | n | One-step regret (optimal continuation) | Closed-loop regret | Closed-loop regret, pattern-cluster CI | First action optimal % | Closed-loop optimal % | One-step severe % | CL P(false benign) % | CL P(escalate) % | CL call cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| optimal | 432 | 0 | 0 | - | 100 | 100 | - | 1.8 | 19.3 | 0.52 |
| fault_unaware | 432 | 1.02 [0.79, 1.27] | 1.38 [1.10, 1.66] | 1.38 [0.60, 2.16] | 62 | 57 | 19 | 7.0 | 0.5 | 0.29 |
| mmi | 432 | 0.47 [0.36, 0.59] | 0.87 [0.68, 1.05] | 0.87 [0.34, 1.46] | 65 | 62 | 11 | 4.4 | 12.3 | 0.43 |
| split | 108 | 0.87 [0.54, 1.23] | 1.06 [0.72, 1.44] | 1.06 [0.00, 2.19] | 55 | 49 | 19 | 5.9 | 2.6 | 0.51 |
| dedup_history | 432 | 0.18 [0.14, 0.21] | 0.59 [0.46, 0.72] | 0.59 [0.22, 1.14] | 75 | 73 | 0 | 1.4 | 27.7 | 0.94 |
| dedup_history_initial | 432 | 0.18 [0.14, 0.21] | 0.20 [0.16, 0.24] | 0.20 [0.06, 0.37] | 75 | 75 | 0 | 1.3 | 26.8 | 0.65 |
| forbid_repeat | 432 | 0.02 [0.01, 0.03] | 0.02 [0.01, 0.03] | 0.02 [0.00, 0.03] | 95 | 95 | 0 | 1.9 | 19.6 | 0.47 |
| dedup_both | 432 | 0.17 [0.13, 0.21] | 0.17 [0.13, 0.21] | 0.17 [0.05, 0.30] | 79 | 79 | 0 | 1.2 | 28.3 | 0.55 |

Paired closed-loop regret differences (task bootstrap; pattern-cluster bootstrap):

| A - B | d closed-loop regret | cluster CI |
| --- | --- | --- |
| dedup_history - forbid_repeat | 0.57 [0.45, 0.70] | [0.21, 1.12] |
| dedup_history_initial - forbid_repeat | 0.18 [0.14, 0.22] | [0.05, 0.35] |
| dedup_history - dedup_history_initial | 0.39 [0.29, 0.50] | [0.13, 0.78] |
| dedup_both - forbid_repeat | 0.15 [0.11, 0.19] | [0.05, 0.27] |
| mmi - dedup_history | 0.28 [0.04, 0.52] | [-0.55, 1.02] |
| mmi - dedup_history_initial | 0.67 [0.47, 0.86] | [0.10, 1.27] |
| fault_unaware - mmi | 0.51 [0.29, 0.75] | [-0.03, 1.26] |

### Full grid without near-ties (gap >= 0.05) (n = 410)

| Policy | n | One-step regret (optimal continuation) | Closed-loop regret | Closed-loop regret, pattern-cluster CI | First action optimal % | Closed-loop optimal % | One-step severe % | CL P(false benign) % | CL P(escalate) % | CL call cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| optimal | 410 | 0 | 0 | - | 100 | 100 | - | 1.8 | 19.5 | 0.47 |
| fault_unaware | 410 | 1.05 [0.80, 1.31] | 1.40 [1.11, 1.69] | 1.40 [0.60, 2.19] | 62 | 59 | 20 | 7.1 | 0.4 | 0.24 |
| mmi | 410 | 0.48 [0.37, 0.59] | 0.87 [0.67, 1.07] | 0.87 [0.33, 1.47] | 66 | 64 | 11 | 4.5 | 11.5 | 0.40 |
| split | 100 | 0.93 [0.59, 1.32] | 1.09 [0.74, 1.49] | 1.09 [0.00, 2.19] | 52 | 52 | 20 | 6.1 | 2.5 | 0.43 |
| dedup_history | 410 | 0.17 [0.14, 0.21] | 0.55 [0.43, 0.69] | 0.55 [0.17, 1.12] | 77 | 76 | 0 | 1.4 | 27.0 | 0.90 |
| dedup_history_initial | 410 | 0.17 [0.14, 0.21] | 0.20 [0.16, 0.24] | 0.20 [0.06, 0.37] | 77 | 77 | 0 | 1.3 | 26.0 | 0.62 |
| forbid_repeat | 410 | 0.01 [0.00, 0.01] | 0.01 [0.00, 0.01] | 0.01 [0.00, 0.02] | 98 | 98 | 0 | 1.9 | 19.7 | 0.46 |
| dedup_both | 410 | 0.16 [0.12, 0.20] | 0.16 [0.12, 0.21] | 0.16 [0.05, 0.29] | 82 | 82 | 0 | 1.3 | 27.7 | 0.54 |

Paired closed-loop regret differences (task bootstrap; pattern-cluster bootstrap):

| A - B | d closed-loop regret | cluster CI |
| --- | --- | --- |
| dedup_history - forbid_repeat | 0.54 [0.42, 0.68] | [0.17, 1.12] |
| dedup_history_initial - forbid_repeat | 0.19 [0.15, 0.23] | [0.05, 0.36] |
| dedup_history - dedup_history_initial | 0.35 [0.26, 0.46] | [0.10, 0.77] |
| dedup_both - forbid_repeat | 0.15 [0.11, 0.20] | [0.05, 0.27] |
| mmi - dedup_history | 0.32 [0.07, 0.56] | [-0.52, 1.05] |
| mmi - dedup_history_initial | 0.67 [0.46, 0.88] | [0.09, 1.29] |
| fault_unaware - mmi | 0.53 [0.30, 0.77] | [-0.05, 1.30] |

### Full grid, persistent structures only (n = 360)

| Policy | n | One-step regret (optimal continuation) | Closed-loop regret | Closed-loop regret, pattern-cluster CI | First action optimal % | Closed-loop optimal % | One-step severe % | CL P(false benign) % | CL P(escalate) % | CL call cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| optimal | 360 | 0 | 0 | - | 100 | 100 | - | 1.9 | 19.7 | 0.52 |
| fault_unaware | 360 | 0.96 [0.71, 1.21] | 1.23 [0.96, 1.52] | 1.23 [0.43, 2.04] | 62 | 59 | 20 | 6.7 | 0.4 | 0.29 |
| mmi | 360 | 0.56 [0.43, 0.70] | 1.04 [0.82, 1.28] | 1.04 [0.44, 1.69] | 58 | 54 | 13 | 4.9 | 11.3 | 0.41 |
| split | 108 | 0.87 [0.54, 1.23] | 1.06 [0.72, 1.44] | 1.06 [0.00, 2.19] | 55 | 49 | 19 | 5.9 | 2.6 | 0.51 |
| dedup_history | 360 | 0.21 [0.17, 0.25] | 0.71 [0.56, 0.86] | 0.71 [0.29, 1.32] | 70 | 68 | 0 | 1.3 | 29.7 | 1.02 |
| dedup_history_initial | 360 | 0.21 [0.17, 0.25] | 0.24 [0.20, 0.29] | 0.24 [0.08, 0.42] | 70 | 70 | 0 | 1.3 | 28.7 | 0.66 |
| forbid_repeat | 360 | 0.02 [0.01, 0.03] | 0.02 [0.01, 0.03] | 0.02 [0.00, 0.04] | 94 | 94 | 0 | 1.9 | 19.9 | 0.46 |
| dedup_both | 360 | 0.20 [0.15, 0.25] | 0.20 [0.15, 0.25] | 0.20 [0.08, 0.34] | 75 | 75 | 0 | 1.1 | 30.5 | 0.54 |

Paired closed-loop regret differences (task bootstrap; pattern-cluster bootstrap):

| A - B | d closed-loop regret | cluster CI |
| --- | --- | --- |
| dedup_history - forbid_repeat | 0.68 [0.54, 0.84] | [0.28, 1.30] |
| dedup_history_initial - forbid_repeat | 0.22 [0.17, 0.27] | [0.06, 0.40] |
| dedup_history - dedup_history_initial | 0.47 [0.35, 0.59] | [0.17, 0.93] |
| dedup_both - forbid_repeat | 0.18 [0.13, 0.23] | [0.06, 0.31] |
| mmi - dedup_history | 0.34 [0.03, 0.65] | [-0.70, 1.22] |
| mmi - dedup_history_initial | 0.80 [0.57, 1.05] | [0.12, 1.48] |
| fault_unaware - mmi | 0.19 [0.00, 0.40] | [-0.12, 0.59] |

### Selected tasks, persistent structures only (n = 111)

| Policy | n | One-step regret (optimal continuation) | Closed-loop regret | Closed-loop regret, pattern-cluster CI | First action optimal % | Closed-loop optimal % | One-step severe % | CL P(false benign) % | CL P(escalate) % | CL call cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| optimal | 111 | 0 | 0 | - | 100 | 100 | - | 1.6 | 27.5 | 0.61 |
| fault_unaware | 111 | 1.45 [0.90, 2.10] | 1.78 [1.21, 2.47] | 1.78 [0.72, 2.86] | 49 | 47 | 25 | 8.0 | 0.6 | 0.37 |
| mmi | 111 | 0.57 [0.38, 0.80] | 1.40 [0.89, 1.99] | 1.40 [0.51, 2.41] | 54 | 47 | 11 | 5.3 | 16.7 | 0.51 |
| split | 24 | 1.47 [0.59, 2.62] | 1.85 [0.94, 2.97] | 1.85 [0.00, 2.35] | 25 | 25 | 25 | 7.3 | 7.9 | 0.80 |
| dedup_history | 111 | 0.27 [0.19, 0.34] | 0.92 [0.62, 1.25] | 0.92 [0.27, 1.94] | 64 | 61 | 0 | 1.1 | 41.3 | 1.23 |
| dedup_history_initial | 111 | 0.27 [0.19, 0.34] | 0.31 [0.22, 0.40] | 0.31 [0.09, 0.57] | 64 | 64 | 0 | 1.0 | 38.4 | 0.77 |
| forbid_repeat | 111 | 0.03 [0.01, 0.05] | 0.03 [0.01, 0.05] | 0.03 [0.00, 0.08] | 91 | 91 | 0 | 1.7 | 28.3 | 0.56 |
| dedup_both | 111 | 0.26 [0.16, 0.36] | 0.26 [0.16, 0.36] | 0.26 [0.08, 0.44] | 71 | 71 | 0 | 0.9 | 42.2 | 0.61 |

Paired closed-loop regret differences (task bootstrap; pattern-cluster bootstrap):

| A - B | d closed-loop regret | cluster CI |
| --- | --- | --- |
| dedup_history - forbid_repeat | 0.89 [0.59, 1.22] | [0.25, 1.92] |
| dedup_history_initial - forbid_repeat | 0.28 [0.19, 0.37] | [0.07, 0.54] |
| dedup_history - dedup_history_initial | 0.61 [0.36, 0.89] | [0.08, 1.40] |
| dedup_both - forbid_repeat | 0.23 [0.14, 0.34] | [0.07, 0.41] |
| mmi - dedup_history | 0.48 [-0.19, 1.21] | [-1.13, 1.93] |
| mmi - dedup_history_initial | 1.09 [0.56, 1.71] | [0.08, 2.17] |
| fault_unaware - mmi | 0.38 [0.02, 0.82] | [-0.11, 0.97] |

### Transient tasks: source-aware vs tool-level dedup / forbid (closed-loop regret, full grid, n = 72)

Tool-level = the agent treats every call of the transient tool as the same source (the definition used before this revision).

| Policy | Source-aware | Tool-level |
| --- | --- | --- |
| dedup_history | 0.00 [0.00, 0.00] | 3.72 [3.12, 4.33] |
| dedup_history_initial | 0.00 [0.00, 0.00] | 0.85 [0.66, 1.04] |
| forbid_repeat | 0.00 [0.00, 0.00] | 0.27 [0.16, 0.38] |
| dedup_both | 0.00 [0.00, 0.00] | 1.27 [1.01, 1.51] |

Check: closed-loop optimal play reproduces the exact best value on every grid task (max |error| = 8.9e-16).
