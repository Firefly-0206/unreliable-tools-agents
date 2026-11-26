# Code and data

This is the code and data release for the arXiv preprint named below. Local paths, account
names and times of day in the logs were removed as described under "What was removed".

Licences: code under MIT, everything else under CC BY 4.0; see `LICENSE`.

Code, task definitions, model outputs and result files for "Information Gathering with Unreliable Tools: Separating Modeling from Execution in LLM Agents". The archive keeps the
authors' working layout (`raea4/` for experiment code and results, `paper/` for the table and figure generators), and
every generator finds its inputs relative to its own file, so nothing needs editing before a run.

No API key is included; the runners read `DEEPSEEK_API_KEY` from the environment, and the CLI-served models need the
vendors' command-line clients. Nothing in "Reproducing without any model call" needs a key or a network connection.

## Requirements

Python 3.10 or later; the stored results were produced with Python 3.14.7 and the package versions in
`requirements.txt` (numpy 2.5.4, matplotlib 3.11.2; scipy 1.18.1 is needed only by `raea4/depth/experiments.py`,
`raea4/agentenv/analyze_repeat_direction.py` and `raea4/agentenv/be_analysis.py`). `pip install -r requirements.txt`.

## Rebuilding every table and figure of the paper

```
python reproduce.py            # all generators
python reproduce.py --quick    # leaves out make_agent_llm_tables.py, which re-solves the exact oracle (minutes)
```

`reproduce.py` runs each `paper/scripts/make_*.py` on the stored result files and logs, rewrites `paper/tables/` and
`paper/figs/`, and compares every rewritten table with the packaged one. It exits with status 0 when every generator
ran and no table changed. Two things are not compared byte for byte: figures (PDF files carry creation metadata) and
the run-record macros whose names end in `Sha`, `ShaShort`, `Date`, `Rows` or `ApiErrors`. Those macros record hashes,
modification dates and row counts of the original log files, which the anonymised copies here cannot reproduce (see
"What was removed"). A few of them are printed in the appendix (`\beRows`, `\agFlashApiErrors`,
`\agTerraUsageLimitRows`) and in the resource tables; they are records of the original runs, so `reproduce.py` puts every
such line back to its packaged value after the generators have run, and the tables stay exactly as packaged. The numbers, tables and figures of the main text come from
`make_tables.py`, `make_agent_llm_tables.py`, `make_defaults_table.py`, `make_anatomy_table.py`,
`make_anatomy_confirm_table.py`, `make_k2r_strict_outputs.py`, `make_example_outputs.py`, `make_x_outputs.py`,
`make_w_outputs.py`, `make_d2_outputs.py`, `make_d3_outputs.py`, `make_v_outputs.py`, `make_s_outputs.py`,
`make_m_outputs.py`, `make_review_outputs.py`, `make_u_outputs.py`, `make_hd_outputs.py`, `make_be_outputs.py`,
`make_overview_fig.py` and `make_s_fig.py`, which are run first.

## Layout

| Folder | Contents | Paper |
|---|---|---|
| `raea4/sim.py`, `exact.py`, `runs.py`, `contrasts.py`, `mechanism.py`, `results/` | simulator, exact evaluator, experiments X1-X6 | Sec. 4, App. B |
| `raea4/depth/`, `raea4/stress/` | planning-depth, history and stress experiments | App. B |
| `raea4/theory/` | numerical checks of the propositions and of the appendix theorem | Sec. 4, App. A |
| `raea4/bench/` | single-decision benchmark: 140 tasks, prompts, runner, scorer, held-out confirmation (analysis plan `PLAN_confirm.md`, see `frozen_confirm/README.md`) | App. C |
| `raea4/agentenv/` | closed-loop environment, exact oracle, runner, scorer, same-JSON replays, logs and scores | Sec. 5, App. D |
| `raea4/diag/` | exploratory pilot | appendix |
| `paper/scripts/` | scripts that turn result files into the paper's tables, macros and figures | all |
| `paper/tables/` | the generated tables and number macros used by the manuscript | all |
| `paper/figs/` | the generated PDF figures | all |
| `paper/additional_results/` | secondary tables and resource ledgers (token counts, latency, cost equivalents) left out of the compact appendix; see `INDEX.md` there | App. B-D |

## Reproducing the analyses without any model call

Everything below reads stored logs and result files only and writes the JSON files that the generators above read.

```
cd raea4/agentenv
python selftest.py                      # oracle self-test (exact DP against Monte Carlo)
python score.py logs/codex_terra_medium_k2decomp.jsonl logs/codex_terra_k2e_replay.jsonl --out results/tmp.json
python k2e_replay.py <K2e log> --out <new file>     # same-JSON replay (K2r); refuses to overwrite
python k2v_replay.py <K2e log> --out <new file>     # one-step planner on the same JSON (K2v)
python k2e_anatomy.py                   # step-level attribution of regret (Fig. 2c, App. D.4)
python k2r_strict_subset.py             # same-JSON replay on the strict subset of accurate JSONs (App. D.3), post hoc
python anatomy_confirm.py --selftest    # the confirmation analysis applied to the original logs
python anatomy_confirm.py               # held-out confirmation (episodes 25-49, App. D.5), plan in PLAN_anatomy_confirm.md
python near_tie_sensitivity.py          # sensitivity of the attribution to near ties (App. D.16), post hoc
python x_analysis.py                    # matched-marginal fault-rate sweep (Fig. 3, App. D.6), plan in PLAN_x_dependence.md
python y_analysis.py                    # randomised tool names/order and repeated generation (App. D.7.1), plan in PLAN_y_surface.md
python z_asif.py; python z_asif_soft.py     # exploratory as-if models of the under-use (App. D.16), post hoc
python z_asif_loss.py; python z_asif_loss.py --m   # as-if planners with altered losses, escalation cost or prior (App. D.16), post hoc
python r_robust.py                     # ambiguity-averse planner (phi + delta) and centred prior over phi (App. D.16), post hoc
python u_units.py main; python u_units.py held   # loss in agent units: verdicts, spend, realised loss (App. D.16), post hoc
RAEA_DOMAIN=2 python u_units.py d2; RAEA_DOMAIN=3 python u_units.py d3   # the same for the second and third domains (App. D.16), post hoc; about 30 minutes each
python h_horizon.py core; python h_horizon.py x; python h_horizon.py m   # effective planning horizon (App. D.16), post hoc
python w_analysis.py                    # sweep in which the best first query flips (App. D.8), plan in PLAN_w_flip.md
RAEA_DOMAIN=2 python d2_analysis.py       # second task domain (App. D.9), plan in PLAN_d2_domain.md; about 25 minutes per run
python d2_de_split.py                    # description and execution loss by JSON structure (App. D.9), post hoc
RAEA_DOMAIN=3 python d3_analysis.py       # third task domain, fact-checking (App. D.10), plan in PLAN_d3_domain.md
python v_analysis.py                    # cost, documentation style, open-weight model (App. D.11), plan in PLAN_v_cost_style.md
python t_analysis.py                    # fault wording (summary in App. D.7; full in paper/additional_results/app_t.tex), plan in PLAN_t_wording.md
python s_analysis.py                    # layered supply of exact quantities (Fig. 3, App. D.12), plan in PLAN_s_supply.md
python r_analysis.py                    # runbook surface (summary in App. D.7; full in paper/additional_results/app_r.tex), plan in PLAN_r_runbook.md
python hd_bayes.py; python hd_analysis.py   # rates in words instead of numbers (App. D.13), plan in PLAN_hd_hard_description.md; make_episodes_hd.py rebuilds the episodes
python be_points.py; python be_analysis.py   # beliefs and values stated at the logged decisions (App. D.14), plan in PLAN_be_belief_elicit.md; be_run.py made the forks (logs/be_*.jsonl)
python be_posthoc.py                    # one-step comparison of the stated values (App. D.14), post hoc
python scenario_sensitivity.py          # per-scenario and leave-one-scenario-out results, multiplicity (App. D.16), post hoc
python scenario_sensitivity.py --d2     # the same for the second domain; about 25 minutes per run
python scenario_sensitivity.py --d3     # the same for the third domain
python scenario_sensitivity.py --summarise   # recompute the summary from the cached values only
python k2v_summary.py; python k2r_struct_subset.py; python k2_audit_readings.py; python difficulty_matched.py   # inputs of the table generators (post hoc readings)
python m_analysis.py                    # fault and fault-free tasks matched on four root-level exact measures (App. D.15), plan in PLAN_m_matched.md
python k2d_summary.py                   # decision-value planner with independent checks on the same JSON (App. D.16), post hoc
```

Simulation: `python runs.py X1 X2 X3 X4 X5 X6` in `raea4/` (exact expected losses; minutes to hours).

## Re-running the models

`raea4/agentenv/run.py` and `raea4/bench/run.py` are resumable; the exact commands of each run are in
`raea4/RUNS_round5.md`, the `run_*.ps1` and `run_*.sh` files and the `PLAN_*.md` files. The `.ps1` launchers contain the
authors' absolute paths and need adjusting. Model identifiers, access routes, harness versions, dates and token counts
are in the paper's appendix. The models are proprietary and served through vendor clients, so re-runs need not
reproduce the stored outputs.

## Hidden material

`raea4/agentenv/episodes*.jsonl`, `ground_truth*.jsonl` and `export_hidden_outputs.json` hold the hidden states and
pre-drawn tool outputs. They are read by code only and are never placed in a model's context.

## What was removed from the packaged logs

For anonymity and size, the packaging script (not included) changes the copies in this archive, never the results:
user names in paths are replaced; wall-clock times of day and time-zone names are removed (dates are kept); operating-
system messages in a non-English locale are replaced by a placeholder; rows of closed-loop logs that ended in an API
error and were reissued are dropped (the analysis scripts ignore such rows, so every reported number is unchanged);
stderr files and logs of launches that failed before any model call are left out.

Each `frozen_*.sha256` lists the files as they were hashed before that experiment's model calls. In this archive
every plan, experiment-specific script and episode file matches its hash (for V, the dated amendment
`frozen_v_amend_kimi_thinking.sha256` re-hashes the plan, analysis and run scripts), except
`raea4/claude_backend.py`, which differs only by the path replacement; `agentenv/nestboot.py`, `agentenv/d2_analysis.py`
and `agentenv/d3_analysis.py`, corrected after the runs so that intervals over parameter draws of unequal size
count every episode once, as the point estimates do (the default of `nestboot.py` is unchanged); and the four logs
hashed in `frozen_be_belief_elicit.sha256`, whose times of day were removed by the packaging. The shared modules `agentenv/env.py`,
`agentenv/k2decomp.py` and `agentenv/run.py` were later extended for later experiments, so their current hashes
differ from the earlier freezes; the later plans describe these additive extensions and the checks that earlier
prompts and analyses were unchanged by them (for example `PLAN_x_dependence.md`, `PLAN_y_surface.md`,
`PLAN_d2_domain.md`).
