# Bench v2 score (4 answers, 1 base tasks)

Regret = one-step regret (optimal continuation): Q(chosen first action) - Q(best), the rest played optimally (unparsed -> escalate). Means with 95% bootstrap CI over base tasks.

## Reference policies (exact, no LLM) on the same base tasks

Category-balanced selection, not a population estimate (full grid: results/reference_policies.md).

| Policy | One-step regret (opt. continuation) | Closed-loop regret | First action optimal | Closed-loop optimal | One-step severe |
| --- | --- | --- | --- | --- | --- |
| fault_unaware (n=1) | 11.99 [11.99, 11.99] | 11.99 [11.99, 11.99] | 0 [0, 0]% | 0 [0, 0]% | 100 [100, 100]% |
| mmi (n=1) | 0.99 [0.99, 0.99] | 13.41 [13.41, 13.41] | 0 [0, 0]% | 0 [0, 0]% | 0 [0, 0]% |
| dedup_history (n=1) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 100 [100, 100]% | 100 [100, 100]% | 0 [0, 0]% |
| dedup_history_initial (n=1) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 100 [100, 100]% | 100 [100, 100]% | 0 [0, 0]% |
| forbid_repeat (n=1) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 100 [100, 100]% | 100 [100, 100]% | 0 [0, 0]% |
| dedup_both (n=1) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 100 [100, 100]% | 100 [100, 100]% | 0 [0, 0]% |

## Run deepseek-chat|cap1500|smoke

| Cond | n ans | One-step regret | Regret (parsed only) | Optimal % | Severe % | Escalate % | Query cost | Parsed % | |p-exact| | Over-count % (dir.) | Over-conf. % (2-sided) | Signed err | CBWA % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2 | 1 | 0.00 [0.00, 0.00] | - | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0.00 | 0 | nan | - | - | +nan | - |
| L3 | 1 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 100 [100, 100] | 0 [0, 0] | 100 [100, 100] | 0.00 | 100 | 0.000 | 0 [0, 0] | 0 [0, 0] | +0.000 | 0 [0, 0] |
| T1 | 1 | 0.00 [0.00, 0.00] | - | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0.00 | 0 | nan | - | - | +nan | - |
| H1 | 1 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 100 [100, 100] | 0 [0, 0] | 100 [100, 100] | 0.00 | 100 | 0.000 | 0 [0, 0] | 0 [0, 0] | +0.000 | 0 [0, 0] |

Paired contrasts (task-level; difference A - B; last column: pattern-cluster bootstrap CI of d regret):

| A - B | Tasks | n | d Regret | d Optimal (pts) | d Severe (pts) | d Regret, cluster CI |
| --- | --- | --- | --- | --- | --- | --- |

One-step regret by exact-best category (task means; repeat right = best action re-queries a used persistent source; repeat right (transient) = best action re-calls the transient tool):

| Cond | declare_benign | declare_nonbenign | escalate | query_same_source | query_repeat_transient | query_new_source | repeat right (persistent) | repeat right (transient) | repeat wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2 | nan | nan | 0.00 | nan | nan | nan | nan | nan | 0.00 |
| L3 | nan | nan | 0.00 | nan | nan | nan | nan | nan | 0.00 |
| T1 | nan | nan | 0.00 | nan | nan | nan | nan | nan | 0.00 |
| H1 | nan | nan | 0.00 | nan | nan | nan | nan | nan | 0.00 |

Correct belief, wrong action: share of parsed answers, by chosen action category:

| Cond | declare_benign | declare_nonbenign | escalate | query_same_source | query_repeat_transient | query_new_source |
| --- | --- | --- | --- | --- | --- | --- |
| P2 | nan% | nan% | nan% | nan% | nan% | nan% |
| L3 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| T1 | nan% | nan% | nan% | nan% | nan% | nan% |
| H1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

Trade-off by severity variant (severe % / escalate % / mean query cost / one-step regret):

| Cond | sev3 | sev6 |
| --- | --- | --- |
| P2 | - | 0 / 0 / 0.00 / 0.00 |
| L3 | - | 0 / 100 / 0.00 / 0.00 |
| T1 | - | 0 / 0 / 0.00 / 0.00 |
| H1 | - | 0 / 100 / 0.00 / 0.00 |

## Fixed-budget view (per model and output cap)

| Model | Cap | Cond | n | Success % (finished + parsed) | One-step regret | Median out tok | Median reasoning tok | Median latency s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-chat | 1500 | P2 | 1 | 0 [0, 0] | 0.00 [0.00, 0.00] | 1500 | 0 | 7.4 |
| deepseek-chat | 1500 | L3 | 1 | 100 [100, 100] | 0.00 [0.00, 0.00] | 130 | 0 | 1.9 |
| deepseek-chat | 1500 | T1 | 1 | 0 [0, 0] | 0.00 [0.00, 0.00] | 458 | 0 | 4.9 |
| deepseek-chat | 1500 | H1 | 1 | 100 [100, 100] | 0.00 [0.00, 0.00] | 165 | 0 | 2.0 |

## Cost (USD at off-peak list price, no cache discount; x2 at peak)

| Model | Answers | Input tok | Output tok (incl. reasoning) | Reasoning tok | USD | USD per answer |
| --- | --- | --- | --- | --- | --- | --- |
| deepseek-chat | 4 | 3656 | 2253 | 0 | 0.0019 | 0.00048 |

Unparsed / failed answers by condition and status (first attempts): {"deepseek-chat|cap1500|smoke": {"P2:truncated": 1, "T1:tool_markup_in_content": 1}}

Later attempts (never scored): {"deepseek-chat|cap1500|smoke": {"keys_with_later_attempts": 0, "later_attempt_parsed_where_first_did_not": 0, "answers_after_failed_calls": 0}}
