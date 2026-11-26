# Bench v2 score (420 answers, 140 base tasks)

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

## Run deepseek-reasoner|cap16384|think

| Cond | n ans | One-step regret | Regret (parsed only) | Optimal % | Severe % | Escalate % | Query cost | Parsed % | |p-exact| | Over-count % (dir.) | Over-conf. % (2-sided) | Signed err | CBWA % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | 140 | 0.56 [0.43, 0.70] | 0.10 [0.00, 0.28] | 31 [24, 39] | 1 [0, 2] | 1 [0, 2] | 0.08 | 33 | 0.003 | 4 [0, 12] | 4 [0, 12] | +0.005 | 2 [0, 7] |
| L2 | 140 | 0.54 [0.41, 0.67] | 0.00 [0.00, 0.00] | 31 [23, 39] | 0 [0, 0] | 1 [0, 2] | 0.04 | 31 | 0.000 | 0 [0, 0] | 0 [0, 0] | +0.000 | 0 [0, 0] |
| H1 | 140 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 100 [100, 100] | 0 [0, 0] | 20 [14, 27] | 0.60 | 100 | 0.000 | 0 [0, 0] | 0 [0, 0] | -0.000 | 0 [0, 0] |

Paired contrasts (task-level; difference A - B; last column: pattern-cluster bootstrap CI of d regret):

| A - B | Tasks | n | d Regret | d Optimal (pts) | d Severe (pts) | d Regret, cluster CI |
| --- | --- | --- | --- | --- | --- | --- |
| L2 - P0 | all | 140 | -0.02 [-0.14, 0.10] | -1 [-7, 5] | -1 [-2, 0] | [-0.15, 0.09] |
| H1 - P0 | all | 140 | -0.56 [-0.69, -0.43] | 69 [61, 76] | -1 [-2, 0] | [-0.70, -0.41] |

One-step regret by exact-best category (task means; repeat right = best action re-queries a used persistent source; repeat right (transient) = best action re-calls the transient tool):

| Cond | declare_benign | declare_nonbenign | escalate | query_same_source | query_repeat_transient | query_new_source | repeat right (persistent) | repeat right (transient) | repeat wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | 0.94 | 0.57 | 0.00 | 0.89 | 0.62 | 0.55 | 0.89 | 0.62 | 0.66 |
| L2 | 0.73 | 0.57 | 0.00 | 1.19 | 0.88 | 0.39 | 1.19 | 0.88 | 0.37 |
| H1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Correct belief, wrong action: share of parsed answers, by chosen action category:

| Cond | declare_benign | declare_nonbenign | escalate | query_same_source | query_repeat_transient | query_new_source |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 0.0% | 0.0% | 0.0% | 2.2% | 0.0% | 0.0% |
| L2 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| H1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

Trade-off by severity variant (severe % / escalate % / mean query cost / one-step regret):

| Cond | sev3 | sev6 |
| --- | --- | --- |
| P0 | 0 / 0 / 0.06 / 0.60 | 1 / 1 / 0.09 / 0.52 |
| L2 | 0 / 0 / 0.03 / 0.60 | 0 / 1 / 0.05 / 0.49 |
| H1 | 0 / 13 / 0.45 / 0.00 | 0 / 26 / 0.72 / 0.00 |

## Fixed-budget view (per model and output cap)

| Model | Cap | Cond | n | Success % (finished + parsed) | One-step regret | Median out tok | Median reasoning tok | Median latency s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-reasoner | 16384 | P0 | 140 | 33 [26, 41] | 0.56 [0.43, 0.70] | 16384 | 16384 | 93.1 |
| deepseek-reasoner | 16384 | L2 | 140 | 31 [23, 39] | 0.54 [0.41, 0.67] | 16384 | 16384 | 91.4 |
| deepseek-reasoner | 16384 | H1 | 140 | 100 [100, 100] | 0.00 [0.00, 0.00] | 361 | 272 | 3.6 |

## Cost (USD at off-peak list price, no cache discount; x2 at peak)

| Model | Answers | Input tok | Output tok (incl. reasoning) | Reasoning tok | USD | USD per answer |
| --- | --- | --- | --- | --- | --- | --- |
| deepseek-reasoner | 420 | 322938 | 3861680 | 3837131 | 2.3654 | 0.00563 |

Unparsed / failed answers by condition and status (first attempts): {"deepseek-reasoner|cap16384|think": {"P0:truncated": 94, "L2:truncated": 97}}

Later attempts (never scored): {"deepseek-reasoner|cap16384|think": {"keys_with_later_attempts": 0, "later_attempt_parsed_where_first_did_not": 0, "answers_after_failed_calls": 9}}
