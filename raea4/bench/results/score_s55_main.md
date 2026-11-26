# Bench v2 score (980 answers, 140 base tasks)

Regret = one-step regret (optimal continuation): Q(chosen first action) - Q(best), the rest played optimally (unparsed -> escalate). Means with 95% bootstrap CI over base tasks.

## Reference policies (exact, no LLM) on the same base tasks

Category-balanced selection, not a population estimate (full grid: results/reference_policies.md).

| Policy | One-step regret (opt. continuation) | Closed-loop regret | First action optimal | Closed-loop optimal | One-step severe |
| --- | --- | --- | --- | --- | --- |
| fault_unaware (n=140) | 1.64 [1.15, 2.22] | 2.13 [1.56, 2.78] | 46 [37, 54]% | 41 [33, 49]% | 26 [20, 34]% |
| mmi (n=140) | 0.45 [0.30, 0.64] | 1.11 [0.70, 1.58] | 64 [56, 71]% | 58 [50, 66]% | 9 [4, 14]% |
| split (n=24) | 1.47 [0.59, 2.62] | 1.85 [0.94, 2.97] | 25 [8, 42]% | 25 [8, 42]% | 25 [8, 42]% |
| dedup_history (n=140) | 0.21 [0.15, 0.27] | 0.73 [0.48, 1.00] | 71 [64, 79]% | 69 [61, 77]% | 0 [0, 0]% |
| dedup_history_initial (n=140) | 0.21 [0.15, 0.27] | 0.24 [0.17, 0.32] | 71 [64, 79]% | 71 [64, 79]% | 0 [0, 0]% |
| forbid_repeat (n=140) | 0.02 [0.01, 0.04] | 0.02 [0.01, 0.04] | 93 [88, 97]% | 93 [88, 97]% | 0 [0, 0]% |
| dedup_both (n=140) | 0.20 [0.13, 0.29] | 0.20 [0.13, 0.29] | 77 [70, 84]% | 77 [70, 84]% | 0 [0, 0]% |

## Run claude:claude-sonnet-5-5@high|cap0|s55

| Cond | n ans | One-step regret | Regret (parsed only) | Optimal % | Severe % | Escalate % | Query cost | Parsed % | |p-exact| | Over-count % (dir.) | Over-conf. % (2-sided) | Signed err | CBWA % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | 140 | 0.33 [0.09, 0.62] | 0.33 [0.09, 0.62] | 85 [79, 91] | 6 [3, 11] | 17 [11, 24] | 0.49 | 100 | 0.012 | 5 [1, 10] | 5 [1, 10] | +0.014 | 7 [3, 11] |
| L1 | 140 | 0.13 [0.04, 0.27] | 0.13 [0.04, 0.27] | 87 [81, 92] | 1 [0, 4] | 20 [14, 27] | 0.48 | 100 | 0.005 | 0 [0, 0] | 0 [0, 0] | +0.000 | 10 [5, 15] |
| L2 | 140 | 0.02 [0.00, 0.04] | 0.02 [0.00, 0.04] | 95 [91, 99] | 0 [0, 0] | 21 [14, 28] | 0.57 | 100 | 0.002 | 0 [0, 0] | 0 [0, 0] | +0.001 | 4 [1, 7] |
| L3 | 140 | 0.03 [0.00, 0.05] | 0.03 [0.00, 0.05] | 97 [94, 99] | 0 [0, 0] | 19 [13, 26] | 0.59 | 100 | 0.004 | 0 [0, 0] | 0 [0, 0] | +0.000 | 0 [0, 0] |
| C2 | 140 | 0.04 [0.01, 0.07] | 0.04 [0.01, 0.07] | 94 [90, 98] | 1 [0, 2] | 19 [13, 26] | 0.58 | 100 | 0.004 | 0 [0, 0] | 0 [0, 0] | +0.000 | 1 [0, 4] |
| X1 | 140 | 0.77 [0.45, 1.14] | 0.77 [0.45, 1.14] | 54 [46, 63] | 9 [4, 14] | 23 [16, 30] | 0.55 | 100 | 0.065 | 12 [5, 19] | 28 [19, 38] | -0.007 | 4 [1, 8] |
| H1 | 140 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 100 [100, 100] | 0 [0, 0] | 20 [14, 27] | 0.60 | 100 | 0.000 | 0 [0, 0] | 0 [0, 0] | -0.000 | 0 [0, 0] |

Paired contrasts (task-level; difference A - B; last column: pattern-cluster bootstrap CI of d regret):

| A - B | Tasks | n | d Regret | d Optimal (pts) | d Severe (pts) | d Regret, cluster CI |
| --- | --- | --- | --- | --- | --- | --- |
| L1 - P0 | all | 140 | -0.20 [-0.51, 0.07] | 2 [-5, 9] | -5 [-9, -1] | [-0.56, 0.08] |
| L2 - P0 | all | 140 | -0.31 [-0.60, -0.08] | 10 [4, 16] | -6 [-11, -3] | [-0.61, -0.07] |
| L3 - P0 | all | 140 | -0.30 [-0.59, -0.07] | 12 [6, 19] | -6 [-11, -3] | [-0.61, -0.06] |
| L2 - L1 | all | 140 | -0.11 [-0.25, -0.02] | 8 [2, 14] | -1 [-4, 0] | [-0.19, -0.02] |
| L3 - L2 | all | 140 | 0.01 [-0.01, 0.03] | 2 [-2, 6] | 0 [0, 0] | [-0.01, 0.03] |
| L2 - C2 | all | 140 | -0.02 [-0.06, 0.01] | 1 [-4, 6] | -1 [-2, 0] | [-0.08, 0.02] |
| L1 - X1 | all | 140 | -0.64 [-1.04, -0.29] | 33 [24, 41] | -7 [-12, -3] | [-1.28, -0.13] |
| X1 - P0 | all | 140 | 0.44 [0.17, 0.74] | -31 [-39, -22] | 2 [-3, 7] | [0.15, 0.80] |
| H1 - P0 | all | 140 | -0.33 [-0.62, -0.10] | 15 [10, 21] | -6 [-11, -3] | [-0.62, -0.10] |

One-step regret by exact-best category (task means; repeat right = best action re-queries a used persistent source; repeat right (transient) = best action re-calls the transient tool):

| Cond | declare_benign | declare_nonbenign | escalate | query_same_source | query_repeat_transient | query_new_source | repeat right (persistent) | repeat right (transient) | repeat wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | 0.00 | 0.00 | 1.21 | 0.22 | 0.08 | 0.29 | 0.22 | 0.08 | 1.01 |
| L1 | 0.00 | 0.00 | 0.38 | 0.15 | 0.05 | 0.16 | 0.15 | 0.05 | 0.11 |
| L2 | 0.00 | 0.02 | 0.00 | 0.03 | 0.07 | 0.00 | 0.03 | 0.07 | 0.01 |
| L3 | 0.00 | 0.00 | 0.02 | 0.00 | 0.16 | 0.00 | 0.00 | 0.16 | 0.00 |
| C2 | 0.00 | 0.00 | 0.10 | 0.03 | 0.00 | 0.08 | 0.03 | 0.00 | 0.13 |
| X1 | 0.79 | 0.01 | 2.30 | 0.55 | 0.15 | 0.44 | 0.55 | 0.15 | 1.61 |
| H1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Correct belief, wrong action: share of parsed answers, by chosen action category:

| Cond | declare_benign | declare_nonbenign | escalate | query_same_source | query_repeat_transient | query_new_source |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 4.3% | 0.0% | 0.7% | 0.0% | 0.0% | 2.1% |
| L1 | 4.3% | 0.0% | 2.1% | 2.9% | 0.0% | 0.7% |
| L2 | 1.4% | 0.0% | 0.7% | 1.4% | 0.0% | 0.0% |
| L3 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| C2 | 0.7% | 0.0% | 0.7% | 0.0% | 0.0% | 0.0% |
| X1 | 1.4% | 0.0% | 2.1% | 0.0% | 0.0% | 0.7% |
| H1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

Trade-off by severity variant (severe % / escalate % / mean query cost / one-step regret):

| Cond | sev3 | sev6 |
| --- | --- | --- |
| P0 | 5 / 13 / 0.29 / 0.09 | 8 / 21 / 0.65 / 0.51 |
| L1 | 2 / 16 / 0.32 / 0.15 | 1 / 23 / 0.60 / 0.10 |
| L2 | 0 / 15 / 0.42 / 0.03 | 0 / 26 / 0.69 / 0.01 |
| L3 | 0 / 13 / 0.44 / 0.02 | 0 / 24 / 0.71 / 0.03 |
| C2 | 0 / 15 / 0.42 / 0.00 | 1 / 22 / 0.71 / 0.07 |
| X1 | 8 / 29 / 0.34 / 0.56 | 9 / 18 / 0.72 / 0.93 |
| H1 | 0 / 13 / 0.45 / 0.00 | 0 / 26 / 0.72 / 0.00 |

## Fixed-budget view (per model and output cap)

| Model | Cap | Cond | n | Success % (finished + parsed) | One-step regret | Median out tok | Median reasoning tok | Median latency s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude:claude-sonnet-5-5@high | 0 | P0 | 140 | 100 [100, 100] | 0.33 [0.09, 0.62] | 1632 | 976 | 18.4 |
| claude:claude-sonnet-5-5@high | 0 | L1 | 140 | 100 [100, 100] | 0.13 [0.04, 0.27] | 1304 | 799 | 16.4 |
| claude:claude-sonnet-5-5@high | 0 | L2 | 140 | 100 [100, 100] | 0.02 [0.00, 0.04] | 1085 | 546 | 12.7 |
| claude:claude-sonnet-5-5@high | 0 | L3 | 140 | 100 [100, 100] | 0.03 [0.00, 0.05] | 317 | 90 | 9.2 |
| claude:claude-sonnet-5-5@high | 0 | C2 | 140 | 100 [100, 100] | 0.04 [0.01, 0.07] | 3634 | 2720 | 32.4 |
| claude:claude-sonnet-5-5@high | 0 | X1 | 140 | 100 [100, 100] | 0.77 [0.45, 1.14] | 1444 | 915 | 19.6 |
| claude:claude-sonnet-5-5@high | 0 | H1 | 140 | 100 [100, 100] | 0.00 [0.00, 0.00] | 291 | 0 | 11.3 |

## Cost (USD at off-peak list price, no cache discount; x2 at peak)

| Model | Answers | Input tok | Output tok (incl. reasoning) | Reasoning tok | USD | USD per answer |
| --- | --- | --- | --- | --- | --- | --- |
| claude:claude-sonnet-5-5@high | 980 | 2058235 | 1487840 | 988500 | 1.2014 | 0.00123 |

Unparsed / failed answers by condition and status (first attempts): {"claude:claude-sonnet-5-5@high|cap0|s55": {}}

Later attempts (never scored): {"claude:claude-sonnet-5-5@high|cap0|s55": {"keys_with_later_attempts": 0, "later_attempt_parsed_where_first_did_not": 0, "answers_after_failed_calls": 0}}
