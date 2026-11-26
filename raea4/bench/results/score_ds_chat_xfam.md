# Bench v2 score (560 answers, 140 base tasks)

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

## Run deepseek-chat|cap8192|xfam

| Cond | n ans | One-step regret | Regret (parsed only) | Optimal % | Severe % | Escalate % | Query cost | Parsed % | |p-exact| | Over-count % (dir.) | Over-conf. % (2-sided) | Signed err | CBWA % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | 140 | 0.12 [0.07, 0.18] | 0.12 [0.07, 0.18] | 82 [76, 89] | 0 [0, 0] | 23 [16, 30] | 0.50 | 100 | 0.001 | 0 [0, 0] | 0 [0, 0] | -0.001 | 18 [11, 24] |
| P1 | 140 | 0.14 [0.07, 0.23] | 0.14 [0.07, 0.23] | 81 [75, 88] | 1 [0, 2] | 21 [15, 29] | 0.49 | 100 | 0.004 | 1 [0, 4] | 3 [0, 6] | +0.002 | 17 [11, 24] |
| P2 | 140 | 0.10 [0.06, 0.15] | 0.10 [0.06, 0.15] | 85 [79, 91] | 0 [0, 0] | 21 [15, 29] | 0.56 | 100 | 0.002 | 0 [0, 0] | 1 [0, 4] | -0.003 | 14 [9, 20] |
| P3 | 140 | 0.28 [0.20, 0.36] | 0.26 [0.19, 0.34] | 65 [57, 73] | 0 [0, 0] | 29 [22, 37] | 0.54 | 99 | 0.050 | 3 [0, 6] | 29 [19, 40] | +0.005 | 9 [4, 14] |

Paired contrasts (task-level; difference A - B; last column: pattern-cluster bootstrap CI of d regret):

| A - B | Tasks | n | d Regret | d Optimal (pts) | d Severe (pts) | d Regret, cluster CI |
| --- | --- | --- | --- | --- | --- | --- |
| P1 - P0 | all | 140 | 0.02 [-0.07, 0.12] | -1 [-9, 7] | 1 [0, 2] | [-0.05, 0.10] |
| P2 - P0 | persistent | 111 | -0.02 [-0.10, 0.06] | 4 [-5, 13] | 0 [0, 0] | [-0.09, 0.05] |
| P3 - P0 | persistent | 111 | 0.15 [0.04, 0.25] | -17 [-27, -6] | 0 [0, 0] | [-0.00, 0.30] |
| P2 - P0 | transient | 29 | -0.03 [-0.18, 0.09] | 0 [-10, 10] | 0 [0, 0] | [-0.04, 0.00] |
| P3 - P0 | transient | 29 | 0.20 [-0.04, 0.47] | -17 [-34, 0] | 0 [0, 0] | [0.00, 0.25] |

One-step regret by exact-best category (task means; repeat right = best action re-queries a used persistent source; repeat right (transient) = best action re-calls the transient tool):

| Cond | declare_benign | declare_nonbenign | escalate | query_same_source | query_repeat_transient | query_new_source | repeat right (persistent) | repeat right (transient) | repeat wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | 0.12 | 0.11 | 0.07 | 0.25 | 0.10 | 0.14 | 0.25 | 0.10 | 0.09 |
| P1 | 0.06 | 0.00 | 0.26 | 0.32 | 0.05 | 0.22 | 0.32 | 0.05 | 0.14 |
| P2 | 0.09 | 0.06 | 0.05 | 0.21 | 0.05 | 0.19 | 0.21 | 0.05 | 0.12 |
| P3 | 0.30 | 0.25 | 0.14 | 0.55 | 0.42 | 0.23 | 0.55 | 0.42 | 0.23 |

Correct belief, wrong action: share of parsed answers, by chosen action category:

| Cond | declare_benign | declare_nonbenign | escalate | query_same_source | query_repeat_transient | query_new_source |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 5.7% | 0.0% | 4.3% | 7.1% | 0.0% | 0.7% |
| P1 | 5.7% | 0.0% | 2.9% | 6.4% | 0.0% | 2.1% |
| P2 | 2.1% | 0.0% | 2.9% | 6.4% | 0.0% | 2.9% |
| P3 | 2.2% | 0.0% | 4.3% | 2.2% | 0.0% | 0.0% |

Trade-off by severity variant (severe % / escalate % / mean query cost / one-step regret):

| Cond | sev3 | sev6 |
| --- | --- | --- |
| P0 | 0 / 18 / 0.37 / 0.09 | 0 / 27 / 0.60 / 0.14 |
| P1 | 2 / 13 / 0.35 / 0.17 | 0 / 28 / 0.59 / 0.11 |
| P2 | 0 / 15 / 0.45 / 0.07 | 0 / 27 / 0.65 / 0.13 |
| P3 | 0 / 18 / 0.52 / 0.30 | 0 / 38 / 0.56 / 0.26 |

## Fixed-budget view (per model and output cap)

| Model | Cap | Cond | n | Success % (finished + parsed) | One-step regret | Median out tok | Median reasoning tok | Median latency s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-chat | 8192 | P0 | 140 | 100 [100, 100] | 0.12 [0.07, 0.18] | 3094 | 0 | 15.3 |
| deepseek-chat | 8192 | P1 | 140 | 100 [100, 100] | 0.14 [0.07, 0.23] | 2886 | 0 | 14.5 |
| deepseek-chat | 8192 | P2 | 140 | 100 [100, 100] | 0.10 [0.06, 0.15] | 3104 | 0 | 15.0 |
| deepseek-chat | 8192 | P3 | 140 | 99 [98, 100] | 0.28 [0.20, 0.36] | 2996 | 0 | 14.8 |

## Cost (USD at off-peak list price, no cache discount; x2 at peak)

| Model | Answers | Input tok | Output tok (incl. reasoning) | Reasoning tok | USD | USD per answer |
| --- | --- | --- | --- | --- | --- | --- |
| deepseek-chat | 560 | 279290 | 1674124 | 0 | 1.0464 | 0.00187 |

Unparsed / failed answers by condition and status (first attempts): {"deepseek-chat|cap8192|xfam": {"P3:no_json": 1}}

Later attempts (never scored): {"deepseek-chat|cap8192|xfam": {"keys_with_later_attempts": 0, "later_attempt_parsed_where_first_did_not": 0, "answers_after_failed_calls": 0}}
