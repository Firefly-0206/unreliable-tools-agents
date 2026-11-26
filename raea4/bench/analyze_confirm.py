"""Preregistered analysis of the held-out confirmation run (see PREREG_confirm.md). No model is called.

  python bench/analyze_confirm.py --run flash=<flash.jsonl> --run terra=<terra.jsonl> \
         --items-file bench/items_confirm.jsonl --out bench/results/confirm_analysis

Uses the SAME scorer as the main runs (score.score_answer / load_records / parse_into / task bootstrap): first
record per run key, unparsed or failed answers fall back to `escalate` for regret, answers are averaged within
base task, 2000-replicate percentile bootstrap over base tasks, and a bootstrap over the 12 evidence patterns
(pattern-cluster) as the second interval. All intervals are two-sided 95%.
A primary criterion is met only if the CONSERVATIVE interval, the union (min lower, max upper) of the task and
the pattern-cluster interval, lies strictly on the predicted side of 0.
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as S                                       # noqa: E402
from run import load_items, parse_into                  # noqa: E402

PRIMARY = [("Ha", "L2", "P0", "regret", -1), ("Hb", "H1", "P0", "regret", -1)]
SECONDARY = [("S1", "L1", "P0", "regret"), ("S2", "L2", "L1", "regret"), ("S3", "H1", "L2", "regret")]

def rows_of(files, items):
    prim, later = S.load_records(files)
    rows = []
    for k, r in prim.items():
        it = items[r["item"]]
        if r.get("ok"): parse_into(r, it, r["cond"])
        else: r.update(parse_status="timeout" if S.is_timeout(r) else "api_error", action=None, p_benign=None)
        d = S.score_answer(it, r)
        d.update(item=r["item"], cond=r["cond"], status=r.get("parse_status"), pattern=it["pattern"],
                 fixes=len(r.get("h1_fixes") or []) if r["cond"] == "H1" else 0,
                 # sensitivity for H1: the planner's action even when the model JSON needed defaults
                 regret_planner=(it["Q"][r["h1_action_planner"]] - it["best_value"]) if r["cond"] == "H1" and r.get("h1_action_planner") else np.nan)
        rows.append(d)
    return rows

def cons(task, cluster):
    return (min(task[1], cluster[1]), max(task[2], cluster[2]))

def contrast(rows, items, A, B, m):
    ta = S.task_means([r for r in rows if r["cond"] == A], m); tb_ = S.task_means([r for r in rows if r["cond"] == B], m)
    diff = {i: ta[i] - tb_[i] for i in sorted(set(ta) & set(tb_)) if not (np.isnan(ta[i]) or np.isnan(tb_[i]))}
    t, c = S.tb(diff, items), S.tb(diff, items, cluster=True)
    return dict(mean=t[0], task_ci=[t[1], t[2]], cluster_ci=[c[1], c[2]], conservative=list(cons(t, c)), n_tasks=t[3])

def analyse(rows, items):
    out = dict(conditions={}, primary={}, secondary={}, H1_audit={}, Hc={})
    conds = [c for c in ["P0", "L1", "L2", "H1"] if any(r["cond"] == c for r in rows)]
    for c in conds:
        R = [r for r in rows if r["cond"] == c]
        reg = S.tb(S.task_means(R, "regret"), items); regc = S.tb(S.task_means(R, "regret"), items, cluster=True)
        opt = S.tb(S.task_means(R, "optimal"), items)
        out["conditions"][c] = dict(n_answers=len(R), parsed=float(np.mean([r["parsed"] for r in R])),
                                    regret=dict(mean=reg[0], task_ci=[reg[1], reg[2]], cluster_ci=[regc[1], regc[2]]),
                                    optimal=dict(mean=opt[0], task_ci=[opt[1], opt[2]]),
                                    statuses={s: sum(r["status"] == s for r in R) for s in sorted({r["status"] for r in R})})
    for nm, A, B, m, sign in PRIMARY:
        if A in conds and B in conds:
            x = contrast(rows, items, A, B, m); x["predicted_sign"] = "negative"
            x["criterion_met"] = bool(x["conservative"][1] < 0); out["primary"][nm] = dict(contrast=f"{A}-{B}", **x)
    for nm, A, B, m in SECONDARY:
        if A in conds and B in conds: out["secondary"][nm] = dict(contrast=f"{A}-{B}", **contrast(rows, items, A, B, m))
    if "P0" in conds:       # Hc: accurate benign-state marginal (|dp| <= 0.02) AND non-optimal action
        R = [r for r in rows if r["cond"] == "P0"]
        t = S.tb(S.task_means(R, "cbwa"), items); c = S.tb(S.task_means(R, "cbwa"), items, cluster=True)
        n_par = [r for r in R if r["parsed"]]
        out["Hc"] = dict(share_task_mean=t[0], task_ci=[t[1], t[2]], cluster_ci=[c[1], c[2]], conservative=list(cons(t, c)),
                         n_answers_with_marginal_accurate_and_action_nonoptimal=int(sum(r["cbwa"] == 1 for r in n_par)),
                         n_parsed=len(n_par), criterion_met=bool(cons(t, c)[0] > 0),
                         note="reports only the benign-state marginal; says nothing about the joint posterior")
    if "H1" in conds:
        R = [r for r in rows if r["cond"] == "H1"]
        pl = S.tb(S.task_means(R, "regret_planner"), items)
        out["H1_audit"] = dict(n=len(R), ok=sum(r["status"] == "ok" for r in R),
                               incomplete_model=sum(r["status"] == "incomplete_model" for r in R),
                               other_failures={s: sum(r["status"] == s for r in R) for s in sorted({r["status"] for r in R}) if s not in ("ok", "incomplete_model")},
                               answers_with_any_default_or_clip=sum(r["fixes"] > 0 for r in R),
                               regret_if_planner_action_used_despite_defaults=dict(mean=pl[0], task_ci=[pl[1], pl[2]], n=pl[3]))
    return out

def fmt(d): return f"{d['mean']:.3f} [{d['task_ci'][0]:.3f}, {d['task_ci'][1]:.3f}]"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=path/to/records.jsonl (repeatable)")
    ap.add_argument("--items-file", default="items_confirm.jsonl")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "confirm_analysis"))
    a = ap.parse_args()
    runs = [x.split("=", 1) for x in a.run]
    items = load_items(a.items_file)
    res, md = {}, ["# Confirmation analysis (preregistered, see bench/PREREG_confirm.md)\n"]
    for nm, fn in runs:
        o = analyse(rows_of([fn], items), items); res[nm] = o
        md += [f"\n## {nm}\n", "| Cond | answers | parsed | one-step regret [task 95% CI] | cluster CI | optimal % |", "| --- | --- | --- | --- | --- | --- |"]
        for c, v in o["conditions"].items():
            md.append(f"| {c} | {v['n_answers']} | {v['parsed']:.2f} | {fmt(v['regret'])} | [{v['regret']['cluster_ci'][0]:.3f}, {v['regret']['cluster_ci'][1]:.3f}] | {v['optimal']['mean'] * 100:.0f} |")
        md += ["\n| Hypothesis | Contrast | d regret | task CI | cluster CI | conservative | criterion |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for k, v in {**o["primary"], **o["secondary"]}.items():
            md.append(f"| {k} | {v['contrast']} | {v['mean']:.3f} | [{v['task_ci'][0]:.3f}, {v['task_ci'][1]:.3f}] | [{v['cluster_ci'][0]:.3f}, {v['cluster_ci'][1]:.3f}] | "
                      f"[{v['conservative'][0]:.3f}, {v['conservative'][1]:.3f}] | {'met' if v.get('criterion_met') else ('not met' if 'criterion_met' in v else 'secondary')} |")
        if o["Hc"]:
            h = o["Hc"]
            md.append(f"\nHc (P0, accurate benign marginal and non-optimal action): task-mean share {h['share_task_mean'] * 100:.1f}% "
                      f"(task CI [{h['task_ci'][0] * 100:.1f}, {h['task_ci'][1] * 100:.1f}], cluster CI [{h['cluster_ci'][0] * 100:.1f}, {h['cluster_ci'][1] * 100:.1f}]); "
                      f"{h['n_answers_with_marginal_accurate_and_action_nonoptimal']} of {h['n_parsed']} parsed answers; criterion {'met' if h['criterion_met'] else 'not met'}.")
        if o["H1_audit"]: md.append("\nH1 audit: " + json.dumps(o["H1_audit"], default=float))
    open(a.out + ".md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump(res, open(a.out + ".json", "w"), indent=1, default=float)
    print("\n".join(md))

if __name__ == "__main__":
    main()
