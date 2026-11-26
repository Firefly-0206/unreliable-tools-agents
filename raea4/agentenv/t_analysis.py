"""
Fault-wording experiment: analysis (PLAN_t_wording.md). No model call.

  python t_analysis.py                 # the three model runs
  python t_analysis.py --mock FILE     # any K2 log of the T episodes (code check only)

Condition K2 (documentation only, no written model). Every trajectory is scored with oracle.score_trajectory under the
true model (independent per-call fault), which is the same in the three cells of a draw.

Cells: TS sentences of the X sweep; TD probabilities, replica law as fresh / stale snapshot; TT probabilities, replica
law as one table by true status (no mention of staleness). Fault rates 0.10 and 0.30, pooled in the contrasts.
  T1  model minus oracle share of episodes whose first call is the replica tool, per cell; under-use regret per cell
  T2  TT minus TD (paired by draw and fault rate): first-call share, under-use regret, regret
  T3  TD minus TS (paired): the same three quantities
Nested bootstrap (scenarios, then parameter draws), 95%. Output: results/t_analysis.json, results/t_analysis.txt
"""
import os
os.environ.setdefault("RAEA_EPISODES", "episodes_t.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_t.jsonl")
import argparse, json
import numpy as np
from env import HERE, load_episodes, load_truth
from x_analysis import parts, ci, fmt, REP

RUNS = {"flash": ("flash (non-thinking)", "t_ds_chat_fc_k2.jsonl"), "terra": ("gpt-6.1-terra (medium)", "t_codex_terra_k2.jsonl"),
        "sonnet": ("Sonnet 5.5 (high)", "t_claude_s55_k2.jsonl")}
PHIS = [0.10, 0.30]; ARMS = ["TS", "TD", "TT"]


def load(path):
    recs = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["condition"] != "K2" or r["status"].startswith("api_error") or r.get("sample", 0) != 0: continue
        recs[r["episode"]] = r
    return recs


def analyse(path, eps, truth, first):
    recs = load(path); ids = sorted(eps); missing = [i for i in ids if i not in recs]
    assert not missing, f"{len(missing)} episodes missing, e.g. {missing[:3]}"
    R = {}
    for i in ids:
        ep, r = eps[i], recs[i]
        R[i] = dict(arm=ep["arm"], phi=ep["phi"], key=(ep["pair"], ep["phi"]), draw=ep["pair"], sc=ep["scenario"],
                    m=parts(ep, r["log"], truth[i]), oracle_first=float(first[i] == ["probe", REP]),
                    no_submit=float(r["status"] != "submitted"))
    sel = lambda arm, phis=PHIS: [d for d in R.values() if d["arm"] == arm and any(abs(d["phi"] - p) < 1e-9 for p in phis)]
    C = lambda rows, f: ci([f(d) for d in rows], [d["draw"] for d in rows], [d["sc"] for d in rows])
    out = dict(n=len(ids), no_submit=int(sum(d["no_submit"] for d in R.values())), cells={}, contrasts={})
    for arm in ARMS:
        for p in PHIS + [None]:
            rows = sel(arm, [p] if p else PHIS); c = dict(n=len(rows))
            c["first_model"] = C(rows, lambda d: d["m"]["first"]); c["first_oracle"] = float(np.mean([d["oracle_first"] for d in rows]))
            c["used_model"] = float(np.mean([d["m"]["used"] for d in rows]))
            for m in ["regret", "under", "over", "verdict", "calls", "rep_calls"]: c[m] = C(rows, lambda d, m=m: d["m"][m])
            out["cells"][f"{arm}|{p:.2f}" if p else f"{arm}|all"] = c
        rows = sel(arm)
        out["contrasts"][f"T1 first model-oracle|{arm}"] = C(rows, lambda d: d["m"]["first"] - d["oracle_first"])
        out["contrasts"][f"T1 under|{arm}"] = C(rows, lambda d: d["m"]["under"])
    for tag, a, b in [("T2", "TT", "TD"), ("T3", "TD", "TS"), ("T2+T3", "TT", "TS")]:
        A = {d["key"]: d for d in sel(a)}; B = {d["key"]: d for d in sel(b)}; ks = sorted(A)
        for name in ["first", "under", "regret"]:
            out["contrasts"][f"{tag} {name} {a}-{b}"] = ci([A[k]["m"][name] - B[k]["m"][name] for k in ks], [A[k]["draw"] for k in ks], [A[k]["sc"] for k in ks])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mock"); ap.add_argument("--runs", nargs="*", default=list(RUNS)); a = ap.parse_args()
    eps, truth = load_episodes(), load_truth(); first = json.load(open(os.path.join(HERE, "t_oracle_first.json"))); out = {}; L = []
    jobs = [("mock", ("mock", a.mock))] if a.mock else [(k, RUNS[k]) for k in a.runs]
    for key, (label, fn) in jobs:
        o = analyse(fn if a.mock else os.path.join(HERE, "logs", fn), eps, truth, first); o["label"] = label; out[key] = o
        L += ["", f"== {label}   n={o['n']}  no submit {o['no_submit']}",
              f"{'cell':9s}{'first: model':>13s}{'oracle':>8s}{'used':>7s}{'regret':>8s}{'under':>8s}{'over':>8s}{'verdict':>9s}{'calls':>7s}{'rep calls':>10s}"]
        for cname, c in o["cells"].items():
            L.append(f"{cname:9s}{100 * c['first_model']['mean']:12.0f}%{100 * c['first_oracle']:7.0f}%{100 * c['used_model']:6.0f}%"
                     f"{c['regret']['mean']:8.2f}{c['under']['mean']:8.2f}{c['over']['mean']:8.2f}{c['verdict']['mean']:9.2f}"
                     f"{c['calls']['mean']:7.2f}{c['rep_calls']['mean']:10.2f}")
        for name, d in o["contrasts"].items():
            L.append(f"  {name:28s} {fmt(d, pct='first' in name)}" + ("  (percentage points)" if "first" in name else ""))
    txt = "\n".join(L); print(txt)
    if not a.mock:
        json.dump(out, open(os.path.join(HERE, "results", "t_analysis.json"), "w"), indent=1)
        open(os.path.join(HERE, "results", "t_analysis.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
