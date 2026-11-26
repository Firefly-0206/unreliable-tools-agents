"""
Cost sweep, documentation-style variant and an open-weight model: analysis (PLAN_v_cost_style.md). No model call.

  python v_analysis.py                    # all runs that have logs
  python v_analysis.py --mock FILE        # any K2e log of the V episodes (code check only)

Part V (runs with a V log; reference cells XP10 / XP30 from the same run's X log):
  V1  cost of the replica tool, fault rate 0.10: model's replica first-call share at cost 1 (X run) minus at cost 3
      (oracle: 92 points); model minus oracle share at cost 2 and at cost 3
  V2  cost of the log tool, fault rate 0.30: model's replica first-call share at log cost 3 minus at log cost 1 (X run);
      model minus oracle share at log cost 3 (oracle: 100%); replica under-use regret there
  V3  field-list documentation, fault rates 0.10 and 0.30: model minus oracle replica first-call share; share under the
      field list minus under the sentences (X run, paired); regret difference; K2r - K2e; replica under-use regret
  V4  K2r - K2e over the 75 cost episodes; share of episodes whose first action equals the as-if prediction (replica
      fault odds x g, g fixed per run in the plan) minus the same for the exact oracle, cost episodes
Part K (open-weight run; X persistent-arm episodes; only cells with all 25 episodes are used):
  K1  K2r - K2e;  K2  model minus oracle replica first-call share at fault rates 0.10 and 0.30, replica under-use regret;
  K3  replica first-call share at 0.10 minus at 0.70;  K4 share of regret at steps where the replica tool is passed over
Nested bootstrap (scenarios, then parameter draws), 95%. Output: results/v_analysis.json, results/v_analysis.txt
"""
import os
os.environ.setdefault("RAEA_EPISODES", "episodes_x.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_x.jsonl")
import argparse, json
import numpy as np
import k2decomp as K
from env import HERE, load_episodes, load_truth
from nestboot import nested_ci
from x_analysis import parts, replay, REP
from y_analysis import load

LOG = 2
RUNS = {"flash": ("flash (non-thinking)", "ds_chat_fc"), "terra": ("gpt-6.1-terra (medium)", "codex_terra"), "sonnet": ("Sonnet 5.5 (high)", "claude_s55"),
        "kimi": ("Kimi K3.1 Pro (non-thinking)", "kimi_k31pro"),
        "kimit": ("Kimi K3.1 Pro (thinking)", "kimi_k31pro_think")}
G_PLAN = {"flash": 24, "terra": 4, "sonnet": 4, "mock": 4}       # as in PLAN_w_flip.md; none for the open-weight run
CELLS = ["VC2", "VC3", "VL3", "VS10", "VS30"]; COST = ["VC2", "VC3", "VL3"]; STYLE = ["VS10", "VS30"]
P = lambda *a: os.path.join(HERE, *a)


def first_action(log):
    c = [x["j"] for x in log if x["type"] == "call"]; return c[0] if c else 4


def ci(v, ids, scl):
    m, lo, hi = nested_ci(v, ids, scl); return dict(mean=float(np.mean(v)), lo=float(lo), hi=float(hi), n=len(set(ids)))


def row(ep, r, truth):
    m = parts(ep, r["log"], truth); k = parts(ep, replay(ep, r["elicit"]["spec"]), truth); o = replay(ep, K.truth_spec(ep))
    return dict(draw=ep["pair"], sc=ep["scenario"], m=m, r=k, fa=first_action(r["log"]), of=first_action(o),
                usable=float(bool(r["elicit"]["parse_ok"])), no_submit=float(r["status"] != "submitted"))


def cell(rows, C):
    return dict(n=len(rows), first_rep=C(rows, lambda d: float(d["fa"] == REP)), first_log=float(np.mean([d["fa"] == LOG for d in rows])),
                first_other=float(np.mean([d["fa"] in (1, 3) for d in rows])), first_stop=float(np.mean([d["fa"] == 4 for d in rows])),
                oracle_rep=float(np.mean([d["of"] == REP for d in rows])), planner_rep=float(np.mean([d["r"]["first"] for d in rows])),
                regret=C(rows, lambda d: d["m"]["regret"]), K2r=C(rows, lambda d: d["r"]["regret"]), under=C(rows, lambda d: d["m"]["under"]),
                over=C(rows, lambda d: d["m"]["over"]), verdict=C(rows, lambda d: d["m"]["verdict"]), calls=C(rows, lambda d: d["m"]["calls"]))


C = lambda rows, f: ci([f(d) for d in rows], [d["draw"] for d in rows], [d["sc"] for d in rows])


def paired(a, b, f):
    """mean over draws of f(a[draw]) - f(b[draw]); a, b: lists of rows of two cells with the same draws."""
    A = {d["draw"]: d for d in a}; B = {d["draw"]: d for d in b}; k = sorted(set(A) & set(B))
    return ci([f(A[x]) - f(B[x]) for x in k], k, [A[x]["sc"] for x in k])


def analyse_v(key, vlog, xlog, EV, TV, EX, TX, PRED, mock=False):
    V = load(vlog, [0]); ids = sorted(i for i in EV if (i, 0) in V) if mock else sorted(EV)
    miss = [i for i in ids if (i, 0) not in V]; assert not miss, f"{len(miss)} V episodes missing, e.g. {miss[:3]}"
    X = load(xlog, [0]) if xlog else {}
    R = {c: [] for c in CELLS + ["XP10", "XP30"]}
    for i in ids:
        d = row(EV[i], V[(i, 0)], TV[i]); d["pred"] = PRED[i]; R[EV[i]["cell"]].append(d)
    for i in sorted(EX):
        if EX[i]["family"] in ("XP10", "XP30") and (i, 0) in X: R[EX[i]["family"]].append(row(EX[i], X[(i, 0)], TX[i]))
    allv = [d for c in CELLS for d in R[c]]
    out = dict(n=len(allv), usable=int(sum(d["usable"] for d in allv)), no_submit=int(sum(d["no_submit"] for d in allv)), g=G_PLAN.get(key),
               cells={c: cell(rows, C) for c, rows in R.items() if rows}, contrasts={})
    D = out["contrasts"]; fr = lambda d: float(d["fa"] == REP); gap = lambda d: fr(d) - float(d["of"] == REP)
    if R["XP10"] and R["VC3"]: D["V1 first replica cost1-cost3|0.10"] = paired(R["XP10"], R["VC3"], fr)
    if R["XP10"] and R["VC2"]: D["V1 first replica cost1-cost2|0.10"] = paired(R["XP10"], R["VC2"], fr)
    for c in ("VC2", "VC3"):
        if R[c]: D[f"V1 first replica model-oracle|{c}"] = C(R[c], gap)
    if R["VL3"]:
        if R["XP30"]: D["V2 first replica logcost3-logcost1|0.30"] = paired(R["VL3"], R["XP30"], fr)
        D["V2 first replica model-oracle|VL3"] = C(R["VL3"], gap); D["V2 under|VL3"] = C(R["VL3"], lambda d: d["m"]["under"])
    st = R["VS10"] + R["VS30"]; px = R["XP10"] + R["XP30"]
    if st:
        D["V3 first replica model-oracle|spec"] = C(st, gap); D["V3 under|spec"] = C(st, lambda d: d["m"]["under"])
        D["V3 K2r-K2e|spec"] = C(st, lambda d: d["r"]["regret"] - d["m"]["regret"]); D["V3 regret|spec"] = C(st, lambda d: d["m"]["regret"])
        if px:
            # pair VS10 with XP10 and VS30 with XP30 (same draw, same fault rate): key the pairing on (cell fault rate, draw)
            def pr(f):
                v, k, s = [], [], []
                for a, b in ((R["VS10"], R["XP10"]), (R["VS30"], R["XP30"])):
                    B = {d["draw"]: d for d in b}
                    for d in a:
                        if d["draw"] in B: v.append(f(d) - f(B[d["draw"]])); k.append(d["draw"]); s.append(d["sc"])
                return ci(v, k, s)
            D["V3 first replica spec-prose|0.10,0.30"] = pr(fr); D["V3 regret spec-prose|0.10,0.30"] = pr(lambda d: d["m"]["regret"])
            D["first replica model-oracle|prose (X run)"] = C(px, gap)
    cs = [d for c in COST for d in R[c]]
    if cs:
        D["V4 K2r-K2e|cost"] = C(cs, lambda d: d["r"]["regret"] - d["m"]["regret"]); D["regret|cost"] = C(cs, lambda d: d["m"]["regret"])
        if out["g"]:
            g = str(out["g"])
            D["V4 first-action match, as-if|cost"] = C(cs, lambda d: float(d["fa"] == d["pred"][g]))
            D["V4 first-action match, oracle|cost"] = C(cs, lambda d: float(d["fa"] == d["pred"]["1"]))
            D["V4 first-action match, as-if minus oracle|cost"] = C(cs, lambda d: float(d["fa"] == d["pred"][g]) - float(d["fa"] == d["pred"]["1"]))
    return out


def analyse_k(xlog, EX, TX):
    X = load(xlog, [0]); R = {}
    for fam in ("XP10", "XP30", "XP50", "XP70"):
        ids = sorted(i for i in EX if EX[i]["family"] == fam)
        if all((i, 0) in X for i in ids): R[fam] = [row(EX[i], X[(i, 0)], TX[i]) for i in ids]
    allr = [d for rows in R.values() for d in rows]
    out = dict(n=len(allr), cells_complete=list(R), n_logged=len(X), usable=int(sum(d["usable"] for d in allr)), no_submit=int(sum(d["no_submit"] for d in allr)),
               cells={c: cell(rows, C) for c, rows in R.items()}, contrasts={})
    D = out["contrasts"]; fr = lambda d: float(d["fa"] == REP); gap = lambda d: fr(d) - float(d["of"] == REP)
    if allr:
        D["K1 K2r-K2e|complete cells"] = C(allr, lambda d: d["r"]["regret"] - d["m"]["regret"]); D["regret|complete cells"] = C(allr, lambda d: d["m"]["regret"])
        D["K2r regret|complete cells"] = C(allr, lambda d: d["r"]["regret"]); D["verdict|complete cells"] = C(allr, lambda d: d["m"]["verdict"])
        tot = sum(d["m"]["regret"] for d in allr); out["pass_share"] = float(sum(d["m"]["under"] for d in allr) / tot) if tot > 0 else None
    lo = R.get("XP10", []) + R.get("XP30", [])
    if "XP10" in R and "XP30" in R:
        D["K2 first replica model-oracle|0.10,0.30"] = C(lo, gap); D["K2 under|0.10,0.30"] = C(lo, lambda d: d["m"]["under"])
    if "XP10" in R and "XP70" in R: D["K3 first replica 0.10-0.70"] = paired(R["XP10"], R["XP70"], fr)
    return out


def fmt(d, pct=False):
    k = 100 if pct else 1; f = (lambda x: f"{k * x:+.0f}") if pct else (lambda x: f"{x:+.3f}")
    return f"{f(d['mean'])} [{f(d['lo'])}, {f(d['hi'])}]"


def table(o, L):
    L.append(f"{'cell':8s}{'first replica: model':>21s}{'oracle':>7s}{'planner':>8s}{'log':>5s}{'other':>6s}{'stop':>5s}{'K2e':>7s}{'K2r':>6s}{'under':>7s}{'over':>6s}{'verdict':>8s}{'calls':>6s}")
    for cn, c in o["cells"].items():
        L.append(f"{cn:8s}{100 * c['first_rep']['mean']:20.0f}%{100 * c['oracle_rep']:6.0f}%{100 * c['planner_rep']:7.0f}%{100 * c['first_log']:4.0f}%{100 * c['first_other']:5.0f}%"
                 f"{100 * c['first_stop']:4.0f}%{c['regret']['mean']:7.2f}{c['K2r']['mean']:6.2f}{c['under']['mean']:7.2f}{c['over']['mean']:6.2f}{c['verdict']['mean']:8.2f}{c['calls']['mean']:6.2f}")
    for name, d in o["contrasts"].items():
        pct = "first" in name; L.append(f"  {name:52s} {fmt(d, pct)}" + ("  (pp)" if pct else ""))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mock"); ap.add_argument("--runs", nargs="*", default=list(RUNS)); a = ap.parse_args()
    EV, TV = load_episodes(P("episodes_v.jsonl")), load_truth(P("ground_truth_v.jsonl")); EX, TX = load_episodes(P("episodes_x.jsonl")), load_truth(P("ground_truth_x.jsonl"))
    PRED = json.load(open(P("v_predictions.json")))["first_action"]; out, L = {}, []
    if a.mock:
        o = analyse_v("mock", a.mock, None, EV, TV, EX, TX, PRED, mock=True); L.append(f"== mock n={o['n']}"); table(o, L); print("\n".join(L)); return
    for key in a.runs:
        label, stem = RUNS[key]; vlog, xlog = P("logs", f"v_{stem}_k2e.jsonl"), P("logs", f"x_{stem}_k2e.jsonl"); o = dict(label=label)
        if os.path.exists(vlog):
            o["V"] = analyse_v(key, vlog, xlog if os.path.exists(xlog) else None, EV, TV, EX, TX, PRED)
            L += ["", f"== {label}: cost and style   n={o['V']['n']}  usable JSON {o['V']['usable']}  no submit {o['V']['no_submit']}  as-if g={o['V']['g']}"]; table(o["V"], L)
        if key == "kimit":           # thinking run: shards written by separate processes (run_v.sh kimit)
            import glob
            sh = sorted(glob.glob(P("logs", "x_kimi_k31pro_think_s*_k2e.jsonl")))
            if sh: open(xlog, "w", encoding="utf-8").write("".join(open(f, encoding="utf-8").read() for f in sh))
        if key in ("kimi", "kimit") and os.path.exists(xlog):
            o["K"] = analyse_k(xlog, EX, TX)
            L += ["", f"== {label}: X persistent arm   complete cells {o['K']['cells_complete']}  n={o['K']['n']} (logged {o['K']['n_logged']})  usable JSON {o['K']['usable']}  "
                      f"no submit {o['K']['no_submit']}  passed-over share {o['K'].get('pass_share')}"]; table(o["K"], L)
        if len(o) > 1: out[key] = o
    txt = "\n".join(L); print(txt)
    json.dump(out, open(P("results", "v_analysis.json"), "w"), indent=1); open(P("results", "v_analysis.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
