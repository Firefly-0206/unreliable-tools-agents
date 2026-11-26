# Confirmation analysis (preregistered, see bench/PREREG_confirm.md)


## flash

| Cond | answers | parsed | one-step regret [task 95% CI] | cluster CI | optimal % |
| --- | --- | --- | --- | --- | --- |
| P0 | 134 | 1.00 | 0.115 [0.071, 0.164] | [0.055, 0.181] | 84 |
| L1 | 134 | 1.00 | 0.119 [0.073, 0.168] | [0.062, 0.182] | 82 |
| L2 | 134 | 1.00 | 0.032 [0.009, 0.059] | [0.000, 0.063] | 95 |
| H1 | 134 | 1.00 | 0.015 [0.003, 0.031] | [0.000, 0.037] | 97 |

| Hypothesis | Contrast | d regret | task CI | cluster CI | conservative | criterion |
| --- | --- | --- | --- | --- | --- | --- |
| Ha | L2-P0 | -0.083 | [-0.137, -0.034] | [-0.162, -0.015] | [-0.162, -0.015] | met |
| Hb | H1-P0 | -0.100 | [-0.153, -0.052] | [-0.169, -0.034] | [-0.169, -0.034] | met |
| S1 | L1-P0 | 0.003 | [-0.059, 0.065] | [-0.037, 0.050] | [-0.059, 0.065] | secondary |
| S2 | L2-L1 | -0.087 | [-0.139, -0.036] | [-0.167, -0.018] | [-0.167, -0.018] | secondary |
| S3 | H1-L2 | -0.017 | [-0.047, 0.012] | [-0.053, 0.022] | [-0.053, 0.022] | secondary |

Hc (P0, accurate benign marginal and non-optimal action): task-mean share 16.4% (task CI [10.4, 22.4], cluster CI [7.5, 25.5]); 22 of 134 parsed answers; criterion met.

H1 audit: {"n": 134, "ok": 134, "incomplete_model": 0, "other_failures": {}, "answers_with_any_default_or_clip": 0, "regret_if_planner_action_used_despite_defaults": {"mean": NaN, "task_ci": [NaN, NaN], "n": 0}}

## terra

| Cond | answers | parsed | one-step regret [task 95% CI] | cluster CI | optimal % |
| --- | --- | --- | --- | --- | --- |
| P0 | 134 | 1.00 | 0.026 [0.009, 0.046] | [0.007, 0.058] | 93 |
| L1 | 134 | 1.00 | 0.016 [0.003, 0.034] | [0.001, 0.029] | 96 |
| L2 | 134 | 1.00 | 0.011 [0.001, 0.026] | [0.000, 0.023] | 97 |
| H1 | 134 | 1.00 | 0.000 [0.000, 0.000] | [0.000, 0.000] | 100 |

| Hypothesis | Contrast | d regret | task CI | cluster CI | conservative | criterion |
| --- | --- | --- | --- | --- | --- | --- |
| Ha | L2-P0 | -0.015 | [-0.040, 0.007] | [-0.052, 0.011] | [-0.052, 0.011] | not met |
| Hb | H1-P0 | -0.026 | [-0.047, -0.010] | [-0.058, -0.007] | [-0.058, -0.007] | met |
| S1 | L1-P0 | -0.010 | [-0.034, 0.012] | [-0.047, 0.015] | [-0.047, 0.015] | secondary |
| S2 | L2-L1 | -0.005 | [-0.025, 0.016] | [-0.018, 0.006] | [-0.025, 0.016] | secondary |
| S3 | H1-L2 | -0.011 | [-0.026, -0.001] | [-0.023, 0.000] | [-0.026, 0.000] | secondary |

Hc (P0, accurate benign marginal and non-optimal action): task-mean share 7.5% (task CI [3.0, 11.9], cluster CI [3.4, 12.8]); 10 of 134 parsed answers; criterion met.

H1 audit: {"n": 134, "ok": 134, "incomplete_model": 0, "other_failures": {}, "answers_with_any_default_or_clip": 0, "regret_if_planner_action_used_despite_defaults": {"mean": NaN, "task_ci": [NaN, NaN], "n": 0}}
