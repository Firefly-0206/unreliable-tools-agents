"""
Difficulty-matched fault / fault-free experiment: analysis (PLAN_m_matched.md). No model call.

  python m_analysis.py                 # the three model runs
  python m_analysis.py --mock FILE     # any K2e log of the M episodes (code check only)

Episodes (make_episodes_m.py): 50 pairs. MF: persistent fault on tool 0 (the replica tool), rate 0.30. MZ: no fault; the
accuracies of tool 0 and the log tool are set so that the first decision matches MF's (best first action tool 0, same
second-best action, same one-step solvability, margin within 0.02, oracle value within 0.30). Condition K2e.
Trajectories are scored under the true model of each episode.

Per arm: share of episodes opening with tool 0 (model, oracle, planner on the model's JSON); share of suboptimal first
actions and first-step loss Q(first action) - Q(best) at the root; regret K2e and K2r; under-use of tool 0; share of
JSONs used as written with every parameter within 0.02 (k2r_strict_subset.flags).
Contrasts, paired by parameter draw, nested bootstrap (scenarios, then draws), 95%:
  M1  MF - MZ: share of suboptimal first actions (primary); first-step loss
  M2  MF - MZ: regret; under-use regret of tool 0
  M3  K2r - K2e within each arm
  M4  M1 on the pairs whose two JSONs are both within tolerance
Output: results/m_analysis.json, results/m_analysis.txt
"""
import os
os.environ.setdefault("RAEA_EPISODES", "episodes_m.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_m.jsonl")
import argparse, json
import numpy as np
from env import HERE, load_episodes, load_truth, build_model
from oracle import qvalues
from x_analysis import parts, replay, ci, fmt, REP
from k2r_strict_subset import flags
from r_analysis import load

RUNS = {"flash": ("flash (non-thinking)", "m_ds_chat_fc_k2e.jsonl"), "terra": ("gpt-6.1-terra (medium)", "m_codex_terra_k2e.jsonl"),
        "sonnet": ("Sonnet 5.5 (high)", "m_claude_s55_k2e.jsonl")}
ARMS = ["MF", "MZ"]


def row(ep, r, truth):
    spec = r["elicit"]["spec"]; M = build_model(ep["params"]); q = qvalues(M, M.prior.copy(), float(M.budget))
    acts = {k: v for k, v in q.items() if k[0] == "probe"}; acts[("stop",)] = min(v for k, v in q.items() if k[0] == "stop")
    best = min(acts, key=acts.get); calls = [x["j"] for x in r["log"] if x["type"] == "call"]; a = ("probe", calls[0]) if calls else ("stop",)
    f = flags(ep, spec)
    return dict(arm=ep["arm"], draw=ep["pair"], sc=ep["scenario"], m=parts(ep, r["log"], truth), r=parts(ep, replay(ep, spec), truth),
                oracle_first=float(best == ("probe", REP)), subopt=float(acts[a] - acts[best] > 1e-9), loss1=float(acts[a] - acts[best]),
                first_other=float(bool(calls) and calls[0] != REP), first_stop=float(not calls),
                usable=float(bool(r["elicit"]["parse_ok"])), no_submit=float(r["status"] != "submitted"), tol=float(f["as_written_correct"]))


def analyse(path, eps, truth, mock=False):
    recs = load(path); ids = sorted(eps)
    if mock:                                         # code check on a partial log: keep complete pairs only
        ok = {eps[i]["pair"] for i in ids if i in recs}; ok = {p for p in ok if all(i in recs for i in ids if eps[i]["pair"] == p)}
        ids = [i for i in ids if eps[i]["pair"] in ok]
    missing = [i for i in ids if i not in recs]
    assert not missing, f"{len(missing)} episodes missing, e.g. {missing[:3]}"
    R = {i: row(eps[i], recs[i], truth[i]) for i in ids}
    by = {a: {d["draw"]: d for d in R.values() if d["arm"] == a} for a in ARMS}; draws = sorted(by["MF"]); assert draws == sorted(by["MZ"])
    sc = {x: by["MF"][x]["sc"] for x in draws}
    C = lambda rows, f: ci([f(d) for d in rows], [d["draw"] for d in rows], [d["sc"] for d in rows])
    Pd = lambda f, keep=draws: ci([f(by["MF"][x]) - f(by["MZ"][x]) for x in keep], list(keep), [sc[x] for x in keep])
    out = dict(n=len(ids), pairs=len(draws), usable=int(sum(d["usable"] for d in R.values())), no_submit=int(sum(d["no_submit"] for d in R.values())),
               cells={}, contrasts={})
    for a in ARMS:
        rows = list(by[a].values())
        out["cells"][a] = dict(n=len(rows), first_model=C(rows, lambda d: d["m"]["first"]), first_oracle=float(np.mean([d["oracle_first"] for d in rows])),
                               first_planner=float(np.mean([d["r"]["first"] for d in rows])), subopt=C(rows, lambda d: d["subopt"]), loss1=C(rows, lambda d: d["loss1"]),
                               first_other=float(np.mean([d["first_other"] for d in rows])), first_stop=float(np.mean([d["first_stop"] for d in rows])),
                               regret=C(rows, lambda d: d["m"]["regret"]), K2r=C(rows, lambda d: d["r"]["regret"]), under=C(rows, lambda d: d["m"]["under"]),
                               verdict=C(rows, lambda d: d["m"]["verdict"]), calls=C(rows, lambda d: d["m"]["calls"]), tol=float(np.mean([d["tol"] for d in rows])))
    D = out["contrasts"]
    D["M1 subopt first MF-MZ"] = Pd(lambda d: d["subopt"]); D["M1 first-step loss MF-MZ"] = Pd(lambda d: d["loss1"])
    D["M2 regret MF-MZ"] = Pd(lambda d: d["m"]["regret"]); D["M2 under MF-MZ"] = Pd(lambda d: d["m"]["under"])
    for a in ARMS: D[f"M3 K2r-K2e|{a}"] = C(list(by[a].values()), lambda d: d["r"]["regret"] - d["m"]["regret"])
    both = [x for x in draws if by["MF"][x]["tol"] == 1.0 and by["MZ"][x]["tol"] == 1.0]; out["pairs_tol"] = len(both)
    if len(both) >= 2: D["M4 subopt first MF-MZ|both within tolerance"] = Pd(lambda d: d["subopt"], both)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mock"); ap.add_argument("--runs", nargs="*", default=list(RUNS)); a = ap.parse_args()
    eps, truth = load_episodes(), load_truth(); out = {}; L = []
    jobs = [("mock", ("mock", a.mock))] if a.mock else [(k, RUNS[k]) for k in a.runs]
    for key, (label, fn) in jobs:
        o = analyse(fn if a.mock else os.path.join(HERE, "logs", fn), eps, truth, bool(a.mock)); o["label"] = label; out[key] = o
        L += ["", f"== {label}   n={o['n']} ({o['pairs']} pairs; {o['pairs_tol']} with both JSONs within tolerance)  usable JSON {o['usable']}  no submit {o['no_submit']}",
              f"{'arm':5s}{'first tool 0: model':>20s}{'oracle':>8s}{'planner':>9s}{'subopt first':>14s}{'loss1':>7s}{'other/stop':>12s}{'K2e':>7s}{'K2r':>7s}{'under':>7s}{'verdict':>9s}{'calls':>7s}{'JSON tol':>10s}"]
        for an, c in o["cells"].items():
            L.append(f"{an:5s}{100 * c['first_model']['mean']:19.0f}%{100 * c['first_oracle']:7.0f}%{100 * c['first_planner']:8.0f}%{100 * c['subopt']['mean']:13.0f}%"
                     f"{c['loss1']['mean']:7.2f}{100 * c['first_other']:8.0f}/{100 * c['first_stop']:.0f}%{c['regret']['mean']:7.2f}{c['K2r']['mean']:7.2f}{c['under']['mean']:7.2f}"
                     f"{c['verdict']['mean']:9.2f}{c['calls']['mean']:7.2f}{100 * c['tol']:9.0f}%")
        for name, d in o["contrasts"].items():
            pct = "subopt" in name; L.append(f"  {name:46s} {fmt(d, pct=pct)}" + ("  (percentage points)" if pct else ""))
    txt = "\n".join(L); print(txt)
    if not a.mock:
        json.dump(out, open(os.path.join(HERE, "results", "m_analysis.json"), "w"), indent=1)
        open(os.path.join(HERE, "results", "m_analysis.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
