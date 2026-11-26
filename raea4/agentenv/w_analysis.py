"""
Accuracy sweep in which the optimal first query flips: analysis (PLAN_w_flip.md). No model call.

  python w_analysis.py                 # the three model runs
  python w_analysis.py --mock FILE     # any K2e log of the W episodes (code check only)

Per run: first action of the model (replica tool / log tool / other tool / submit at once), of the exact oracle and of
the exact planner on the model's JSON (K2r); regret of K2e and K2r under the true model; under-use of the replica tool
(regret at steps where the oracle would query it and the model does not) and of the log tool (same for the log tool).
Contrasts (nested bootstrap: scenarios, then parameter draws; 95%), fixed in the plan:
  W1  axis A, log accuracy 0.55 and 0.65 (oracle opens with the replica tool in every episode): model minus oracle share
      of episodes opening with the replica tool; replica under-use regret
  W2  axis A: model's replica first-call share at 0.55 minus at 0.95 (oracle: 100 points)
  W3  axis A, log accuracy 0.85 and 0.95 (oracle opens with the log tool in every episode): model minus oracle share of
      episodes opening with the log tool; share opening with the replica tool
  W4  axis B: model's replica first-call share at the drawn accuracy (X run, fault rate 0.10) minus at 0.65; model minus
      oracle at 0.75 and at 0.65
  W5  K2r - K2e, all 175 episodes
  W6  as-if prediction (w_predictions.json, written before the runs): share of episodes whose first action (five
      options) equals the prediction of the planner with replica fault odds x g, for the g fixed per run in the plan,
      against the exact oracle's (g = 1); paired difference
Output: results/w_analysis.json, results/w_analysis.txt
"""
import os
os.environ.setdefault("RAEA_EPISODES", "episodes_x.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_x.jsonl")
import argparse, json
import numpy as np
from env import HERE, load_episodes, load_truth
from nestboot import nested_ci
from x_analysis import parts, replay, REP
from y_analysis import load, RUNS

LOG = 2
G_PLAN = {"flash": 24, "terra": 4, "sonnet": 4, "mock": 4}      # fixed in PLAN_w_flip.md (from z_asif.py / z_asif_soft.py)
A_LEVELS = [0.55, 0.65, 0.75, 0.85, 0.95]; B_LEVELS = [0.75, 0.65]
P = lambda *a: os.path.join(HERE, *a)


def first_action(log):
    c = [x["j"] for x in log if x["type"] == "call"]; return c[0] if c else 4


def log_under(ep, log, truth):
    from oracle import score_trajectory
    s = score_trajectory(ep, log, truth); u = 0.0
    for st in s["steps"]:
        o = st["oracle"][0] == "probe" and st["oracle"][1] == LOG
        if st["action"][0] == "probe":
            if o and st["action"][1] != LOG: u += st["adv"]
        elif o: u += st["adv_stoptime"]
    return u


def ci(v, ids, scl):
    m, lo, hi = nested_ci(v, ids, scl); return dict(mean=float(np.mean(v)), lo=float(lo), hi=float(hi), n=len(set(ids)))


def analyse(key, wlog, xlog, EW, TW, EX, PRED):
    W = load(wlog, [0]); miss = [i for i in EW if (i, 0) not in W]
    assert not miss, f"{len(miss)} episodes missing, e.g. {miss[:3]}"
    X = load(xlog, [0]) if xlog else {}
    g = str(G_PLAN[key]); R = []
    for i in sorted(EW):
        ep, r = EW[i], W[(i, 0)]; m = parts(ep, r["log"], TW[i]); k = parts(ep, replay(ep, r["elicit"]["spec"]), TW[i]); fa = first_action(r["log"])
        d = dict(axis=ep["axis"], level=ep["level"], draw=ep["pair"], sc=ep["scenario"], m=m, r=k, fa=fa, log_under=log_under(ep, r["log"], TW[i]),
                 of=PRED[i]["1"], pf=PRED[i][g], kf=first_action(replay(ep, r["elicit"]["spec"])) if False else None,
                 usable=float(bool(r["elicit"]["parse_ok"])), no_submit=float(r["status"] != "submitted"))
        d["x_first"] = float(first_action(X[(ep["x_id"], 0)]["log"]) == REP) if (ep["x_id"], 0) in X else None
        R.append(d)
    C = lambda rows, f: ci([f(d) for d in rows], [d["draw"] for d in rows], [d["sc"] for d in rows])
    sel = lambda axis, levels: [d for d in R if d["axis"] == axis and any(abs(d["level"] - q) < 1e-9 for q in levels)]
    out = dict(n=len(R), usable=int(sum(d["usable"] for d in R)), no_submit=int(sum(d["no_submit"] for d in R)), g=G_PLAN[key], cells={}, contrasts={})
    for axis, levels in (("A", A_LEVELS), ("B", B_LEVELS)):
        for q in levels:
            rows = sel(axis, [q])
            out["cells"][f"{axis}|{q:.2f}"] = dict(
                n=len(rows), first_rep=C(rows, lambda d: float(d["fa"] == REP)), first_log=C(rows, lambda d: float(d["fa"] == LOG)),
                first_other=float(np.mean([d["fa"] in (1, 3) for d in rows])), first_stop=float(np.mean([d["fa"] == 4 for d in rows])),
                oracle_rep=float(np.mean([d["of"] == REP for d in rows])), oracle_log=float(np.mean([d["of"] == LOG for d in rows])),
                asif_rep=float(np.mean([d["pf"] == REP for d in rows])), asif_log=float(np.mean([d["pf"] == LOG for d in rows])),
                planner_rep=float(np.mean([d["r"]["first"] for d in rows])),
                regret=C(rows, lambda d: d["m"]["regret"]), K2r=C(rows, lambda d: d["r"]["regret"]), under_rep=C(rows, lambda d: d["m"]["under"]),
                under_log=C(rows, lambda d: d["log_under"]), over_rep=C(rows, lambda d: d["m"]["over"]), verdict=C(rows, lambda d: d["m"]["verdict"]),
                calls=C(rows, lambda d: d["m"]["calls"]))
    D = out["contrasts"]; lo = sel("A", [0.55, 0.65]); hi = sel("A", [0.85, 0.95])
    D["W1 first replica model-oracle|A 0.55,0.65"] = C(lo, lambda d: float(d["fa"] == REP) - float(d["of"] == REP))
    D["W1 under replica|A 0.55,0.65"] = C(lo, lambda d: d["m"]["under"])
    a = {d["draw"]: d for d in sel("A", [0.55])}; b = {d["draw"]: d for d in sel("A", [0.95])}; k = sorted(a)
    D["W2 first replica 0.55-0.95|A"] = ci([float(a[x]["fa"] == REP) - float(b[x]["fa"] == REP) for x in k], k, [a[x]["sc"] for x in k])
    D["W3 first log model-oracle|A 0.85,0.95"] = C(hi, lambda d: float(d["fa"] == LOG) - float(d["of"] == LOG))
    D["W3 first replica|A 0.85,0.95"] = C(hi, lambda d: float(d["fa"] == REP))
    D["W3 under log|A 0.85,0.95"] = C(hi, lambda d: d["log_under"])
    for q in B_LEVELS:
        D[f"W4 first replica model-oracle|B {q:.2f}"] = C(sel("B", [q]), lambda d: float(d["fa"] == REP) - float(d["of"] == REP))
    rb = [d for d in sel("B", [0.65]) if d["x_first"] is not None]
    if rb: D["W4 first replica drawn-0.65|B"] = C(rb, lambda d: d["x_first"] - float(d["fa"] == REP))
    D["W5 K2r-K2e|all"] = C(R, lambda d: d["r"]["regret"] - d["m"]["regret"]); D["regret|all"] = C(R, lambda d: d["m"]["regret"])
    D["W6 first-action match, as-if|all"] = C(R, lambda d: float(d["fa"] == d["pf"])); D["W6 first-action match, oracle|all"] = C(R, lambda d: float(d["fa"] == d["of"]))
    D["W6 first-action match, as-if minus oracle|all"] = C(R, lambda d: float(d["fa"] == d["pf"]) - float(d["fa"] == d["of"]))
    dis = [d for d in R if d["pf"] != d["of"]]
    out["w6_disagree"] = dict(n=len(dis), model_with_asif=float(np.mean([d["fa"] == d["pf"] for d in dis])) if dis else None,
                              model_with_oracle=float(np.mean([d["fa"] == d["of"] for d in dis])) if dis else None)
    return out


def fmt(d, pct=False):
    k = 100 if pct else 1; f = (lambda x: f"{k * x:+.0f}") if pct else (lambda x: f"{x:+.3f}")
    return f"{f(d['mean'])} [{f(d['lo'])}, {f(d['hi'])}]"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mock"); ap.add_argument("--runs", nargs="*", default=list(RUNS)); a = ap.parse_args()
    EW, TW = load_episodes(P("episodes_w.jsonl")), load_truth(P("ground_truth_w.jsonl")); EX = load_episodes(P("episodes_x.jsonl"))
    PRED = json.load(open(P("w_predictions.json")))["first_action"]
    jobs = [("mock", ("mock", None))] if a.mock else [(k, RUNS[k]) for k in a.runs]; out, L = {}, []
    for key, (label, stem) in jobs:
        o = analyse(key, a.mock, None, EW, TW, EX, PRED) if a.mock else analyse(key, P("logs", f"w_{stem}_k2e.jsonl"), P("logs", f"x_{stem}_k2e.jsonl"), EW, TW, EX, PRED)
        o["label"] = label; out[key] = o
        L += ["", f"== {label}   n={o['n']}  usable JSON {o['usable']}  no submit {o['no_submit']}  as-if g={o['g']}",
              f"{'cell':8s}{'first replica: model':>21s}{'oracle':>7s}{'as-if':>6s}{'planner':>8s}{'  first log: model':>19s}{'oracle':>7s}{'as-if':>6s}{'other':>6s}{'stop':>5s}{'K2e':>7s}{'K2r':>6s}{'under rep':>10s}{'under log':>10s}{'over rep':>9s}{'verdict':>8s}"]
        for cn, c in o["cells"].items():
            L.append(f"{cn:8s}{100 * c['first_rep']['mean']:20.0f}%{100 * c['oracle_rep']:6.0f}%{100 * c['asif_rep']:5.0f}%{100 * c['planner_rep']:7.0f}%"
                     f"{100 * c['first_log']['mean']:18.0f}%{100 * c['oracle_log']:6.0f}%{100 * c['asif_log']:5.0f}%{100 * c['first_other']:5.0f}%{100 * c['first_stop']:4.0f}%"
                     f"{c['regret']['mean']:7.2f}{c['K2r']['mean']:6.2f}{c['under_rep']['mean']:10.2f}{c['under_log']['mean']:10.2f}{c['over_rep']['mean']:9.2f}{c['verdict']['mean']:8.2f}")
        for name, d in o["contrasts"].items():
            pct = "first" in name; L.append(f"  {name:48s} {fmt(d, pct)}" + ("  (pp)" if pct else ""))
        w = o["w6_disagree"]
        if w["n"]: L.append(f"  W6 episodes where as-if and oracle predictions differ: n={w['n']}; model agrees with as-if {100 * w['model_with_asif']:.0f}%, with oracle {100 * w['model_with_oracle']:.0f}%")
    txt = "\n".join(L); print(txt)
    if not a.mock:
        json.dump(out, open(P("results", "w_analysis.json"), "w"), indent=1); open(P("results", "w_analysis.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
