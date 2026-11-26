# Confirmation analysis (preregistered, see bench/PREREG_confirm.md)


## sonnet

| Cond | answers | parsed | one-step regret [task 95% CI] | cluster CI | optimal % |
| --- | --- | --- | --- | --- | --- |
| P0 | 134 | 1.00 | 0.384 [0.098, 0.788] | [0.078, 0.904] | 85 |
| L2 | 134 | 1.00 | 0.028 [0.006, 0.054] | [0.000, 0.069] | 96 |
| H1 | 134 | 1.00 | 0.000 [0.000, 0.000] | [0.000, 0.000] | 100 |

| Hypothesis | Contrast | d regret | task CI | cluster CI | conservative | criterion |
| --- | --- | --- | --- | --- | --- | --- |
| Ha | L2-P0 | -0.356 | [-0.753, -0.072] | [-0.883, -0.041] | [-0.883, -0.041] | met |
| Hb | H1-P0 | -0.384 | [-0.785, -0.103] | [-0.904, -0.078] | [-0.904, -0.078] | met |
| S3 | H1-L2 | -0.028 | [-0.054, -0.007] | [-0.069, 0.000] | [-0.069, 0.000] | secondary |

Hc (P0, accurate benign marginal and non-optimal action): task-mean share 9.0% (task CI [4.5, 14.2], cluster CI [2.4, 15.7]); 12 of 134 parsed answers; criterion met.

H1 audit: {"n": 134, "ok": 134, "incomplete_model": 0, "other_failures": {}, "answers_with_any_default_or_clip": 0, "regret_if_planner_action_used_despite_defaults": {"mean": NaN, "task_ci": [NaN, NaN], "n": 0}}
