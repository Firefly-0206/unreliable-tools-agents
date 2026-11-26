# Bench v2 score (536 answers, 134 base tasks)

Regret = one-step regret (optimal continuation): Q(chosen first action) - Q(best), the rest played optimally (unparsed -> escalate). Means with 95% bootstrap CI over base tasks.

## Reference policies (exact, no LLM) on the same base tasks

Category-balanced selection, not a population estimate (full grid: results/reference_policies.md).

| Policy | One-step regret (opt. continuation) | Closed-loop regret | First action optimal | Closed-loop optimal | One-step severe |
| --- | --- | --- | --- | --- | --- |
| fault_unaware (n=134) | 2.81 [1.94, 3.85] | 3.15 [2.27, 4.22] | 40 [31, 48]% | 38 [30, 46]% | 37 [29, 46]% |
| mmi (n=134) | 0.49 [0.30, 0.71] | 1.18 [0.72, 1.68] | 66 [58, 74]% | 65 [57, 73]% | 7 [3, 12]% |
| split (n=23) | 1.31 [0.61, 2.20] | 1.89 [0.96, 2.94] | 30 [13, 48]% | 30 [13, 48]% | 22 [9, 39]% |
| dedup_history (n=134) | 0.16 [0.11, 0.23] | 0.55 [0.31, 0.81] | 79 [72, 86]% | 78 [72, 85]% | 0 [0, 0]% |
| dedup_history_initial (n=134) | 0.16 [0.11, 0.23] | 0.18 [0.12, 0.26] | 79 [72, 86]% | 79 [72, 86]% | 0 [0, 0]% |
| forbid_repeat (n=134) | 0.01 [0.00, 0.02] | 0.01 [0.00, 0.02] | 98 [95, 100]% | 98 [95, 100]% | 0 [0, 0]% |
| dedup_both (n=134) | 0.22 [0.12, 0.33] | 0.22 [0.12, 0.33] | 81 [75, 87]% | 81 [75, 87]% | 0 [0, 0]% |

## Run deepseek-chat|cap8192|confirm

| Cond | n ans | One-step regret | Regret (parsed only) | Optimal % | Severe % | Escalate % | Query cost | Parsed % | |p-exact| | Over-count % (dir.) | Over-conf. % (2-sided) | Signed err | CBWA % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | 134 | 0.12 [0.07, 0.16] | 0.12 [0.07, 0.16] | 84 [78, 90] | 0 [0, 0] | 25 [17, 32] | 0.52 | 100 | 0.001 | 0 [0, 0] | 0 [0, 0] | +0.001 | 16 [10, 22] |
| L1 | 134 | 0.12 [0.07, 0.17] | 0.12 [0.07, 0.17] | 82 [75, 88] | 0 [0, 0] | 21 [14, 28] | 0.56 | 100 | 0.000 | 0 [0, 0] | 0 [0, 0] | -0.000 | 18 [12, 25] |
| L2 | 134 | 0.03 [0.01, 0.06] | 0.03 [0.01, 0.06] | 95 [91, 99] | 0 [0, 0] | 24 [17, 31] | 0.56 | 100 | 0.000 | 0 [0, 0] | 0 [0, 0] | -0.000 | 5 [1, 9] |
| H1 | 134 | 0.02 [0.00, 0.03] | 0.02 [0.00, 0.03] | 97 [94, 99] | 0 [0, 0] | 22 [16, 30] | 0.57 | 100 | 0.006 | 0 [0, 0] | 0 [0, 0] | -0.000 | 0 [0, 0] |

Paired contrasts (task-level; difference A - B; last column: pattern-cluster bootstrap CI of d regret):

| A - B | Tasks | n | d Regret | d Optimal (pts) | d Severe (pts) | d Regret, cluster CI |
| --- | --- | --- | --- | --- | --- | --- |
| L1 - P0 | all | 134 | 0.00 [-0.06, 0.07] | -1 [-9, 7] | 0 [0, 0] | [-0.04, 0.05] |
| L2 - P0 | all | 134 | -0.08 [-0.14, -0.03] | 11 [5, 18] | 0 [0, 0] | [-0.16, -0.01] |
| L2 - L1 | all | 134 | -0.09 [-0.14, -0.04] | 13 [6, 19] | 0 [0, 0] | [-0.17, -0.02] |
| H1 - P0 | all | 134 | -0.10 [-0.15, -0.05] | 13 [6, 20] | 0 [0, 0] | [-0.17, -0.03] |

One-step regret by exact-best category (task means; repeat right = best action re-queries a used persistent source; repeat right (transient) = best action re-calls the transient tool):

| Cond | declare_benign | declare_nonbenign | escalate | query_same_source | query_repeat_transient | query_new_source | repeat right (persistent) | repeat right (transient) | repeat wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | 0.14 | 0.00 | 0.03 | 0.27 | 0.16 | 0.24 | 0.27 | 0.16 | 0.13 |
| L1 | 0.08 | 0.00 | 0.17 | 0.36 | 0.14 | 0.19 | 0.36 | 0.14 | 0.17 |
| L2 | 0.00 | 0.07 | 0.00 | 0.00 | 0.10 | 0.02 | 0.00 | 0.10 | 0.01 |
| H1 | 0.00 | 0.04 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 | 0.00 | 0.03 |

Correct belief, wrong action: share of parsed answers, by chosen action category:

| Cond | declare_benign | declare_nonbenign | escalate | query_same_source | query_repeat_transient | query_new_source |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 0.7% | 0.0% | 4.5% | 9.0% | 0.7% | 1.5% |
| L1 | 0.7% | 0.0% | 3.7% | 12.7% | 0.7% | 0.0% |
| L2 | 0.7% | 0.0% | 3.0% | 1.5% | 0.0% | 0.0% |
| H1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

Trade-off by severity variant (severe % / escalate % / mean query cost / one-step regret):

| Cond | sev4 | sev8 |
| --- | --- | --- |
| P0 | 0 / 11 / 0.49 / 0.08 | 0 / 36 / 0.55 / 0.15 |
| L1 | 0 / 10 / 0.51 / 0.11 | 0 / 30 / 0.60 / 0.12 |
| L2 | 0 / 11 / 0.54 / 0.03 | 0 / 34 / 0.58 / 0.03 |
| H1 | 0 / 11 / 0.52 / 0.02 | 0 / 32 / 0.62 / 0.02 |

## Fixed-budget view (per model and output cap)

| Model | Cap | Cond | n | Success % (finished + parsed) | One-step regret | Median out tok | Median reasoning tok | Median latency s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-chat | 8192 | P0 | 134 | 100 [100, 100] | 0.12 [0.07, 0.16] | 2976 | 0 | 14.2 |
| deepseek-chat | 8192 | L1 | 134 | 100 [100, 100] | 0.12 [0.07, 0.17] | 2386 | 0 | 11.6 |
| deepseek-chat | 8192 | L2 | 134 | 100 [100, 100] | 0.03 [0.01, 0.06] | 1422 | 0 | 6.7 |
| deepseek-chat | 8192 | H1 | 134 | 100 [100, 100] | 0.02 [0.00, 0.03] | 105 | 0 | 1.6 |

## Cost (USD at off-peak list price, no cache discount; x2 at peak)

| Model | Answers | Input tok | Output tok (incl. reasoning) | Reasoning tok | USD | USD per answer |
| --- | --- | --- | --- | --- | --- | --- |
| deepseek-chat | 536 | 399453 | 922117 | 0 | 0.6132 | 0.00114 |

Unparsed / failed answers by condition and status (first attempts): {"deepseek-chat|cap8192|confirm": {}}

Later attempts (never scored): {"deepseek-chat|cap8192|confirm": {"keys_with_later_attempts": 0, "later_attempt_parsed_where_first_did_not": 0, "answers_after_failed_calls": 0}}
