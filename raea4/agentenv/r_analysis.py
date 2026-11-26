"""
Runbook case study: analysis (PLAN_r_runbook.md). No model call.

  python r_analysis.py                 # the three model runs
  python r_analysis.py --mock FILE     # any K2e log of the R episodes (code check only; surface contrast skipped)

Episodes RB10/RB30/RB70: the persistent-arm X episodes with the tools documented as runbook entries (rates as counts)
and tool results as service responses. Condition K2e. Trajectories are scored under the true model.

Per fault rate: replica first-call share (model, oracle, planner on the model's JSON), regret K2e and K2r, under-use,
over-use, verdict regret. Recovery of the written model: share of episodes whose JSON was used as written and has
every parameter within 0.02 (k2r_strict_subset.flags), maximum fault-rate and accuracy errors.
Contrasts, nested bootstrap (scenarios, then draws), 95%:
  R1  K2r - K2e, all episodes; and on the episodes whose JSON is within tolerance
  R2  fault rates 0.10 and 0.30: model minus oracle first-call share; under-use regret
  R3  first-call share at 0.10 minus at 0.70
  R4  runbook minus original surface (the same run's X log), paired by draw and fault rate: first-call share at
      0.10 and 0.30, regret over all three rates, share of JSONs within tolerance
Output: results/r_analysis.json, results/r_analysis.txt
"""
import os
os.environ.setdefault("RAEA_EPISODES", "episodes_r.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_r.jsonl")
import argparse, json
import numpy as np
from env import HERE, load_episodes, load_truth, build_model
from oracle import qvalues
from x_analysis import parts, replay, ci, fmt, REP
from k2r_strict_subset import flags

RUNS = {"flash": ("flash (non-thinking)", "r_ds_chat_fc_k2e.jsonl", "x_ds_chat_fc_k2e.jsonl"),
        "terra": ("gpt-6.1-terra (medium)", "r_codex_terra_k2e.jsonl", "x_codex_terra_k2e.jsonl"),
        "sonnet": ("Sonnet 5.5 (high)", "r_claude_s55_k2e.jsonl", "x_claude_s55_k2e.jsonl")}
PHIS = [0.10, 0.30, 0.70]; LOW = [0.10, 0.30]


def load(path):
    recs = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["condition"] != "K2e" or r["status"].startswith("api_error") or r.get("sample", 0) != 0: continue
        recs[r["episode"]] = r
    return recs


def rows_of(recs, eps, truth, ids):
    R = {}
    for i in ids:
        ep, r = eps[i], recs[i]; spec = r["elicit"]["spec"]; M = build_model(ep["params"]); q = qvalues(M, M.prior.copy(), float(M.budget))
        f = flags(ep, spec)
        R[i] = dict(phi=ep["phi"], key=(ep["pair"], ep["phi"]), draw=ep["pair"], sc=ep["scenario"], m=parts(ep, r["log"], truth[i]),
                    r=parts(ep, replay(ep, spec), truth[i]), oracle_first=float(min(q, key=q.get) == ("probe", REP)),
                    usable=float(bool(r["elicit"]["parse_ok"])), no_submit=float(r["status"] != "submitted"),
                    tol=float(f["as_written_correct"]), phi_max=f["phi_max"], acc_max=f["acc_max"],
                    invalid=sum(x["type"] == "invalid" for x in r["log"]))
    return R


def analyse(path, xpath, eps, truth):
    recs = load(path); ids = sorted(eps); missing = [i for i in ids if i not in recs]
    assert not missing, f"{len(missing)} episodes missing, e.g. {missing[:3]}"
    R = rows_of(recs, eps, truth, ids)
    sel = lambda phis, pred=lambda d: True: [d for d in R.values() if any(abs(d["phi"] - p) < 1e-9 for p in phis) and pred(d)]
    C = lambda rows, f: ci([f(d) for d in rows], [d["draw"] for d in rows], [d["sc"] for d in rows])
    out = dict(n=len(ids), usable=int(sum(d["usable"] for d in R.values())), no_submit=int(sum(d["no_submit"] for d in R.values())),
               invalid=int(sum(d["invalid"] for d in R.values())), n_tol=int(sum(d["tol"] for d in R.values())), cells={}, contrasts={})
    for p in PHIS:
        rows = sel([p]); c = dict(n=len(rows))
        c["first_model"] = C(rows, lambda d: d["m"]["first"]); c["first_oracle"] = float(np.mean([d["oracle_first"] for d in rows]))
        c["first_planner"] = float(np.mean([d["r"]["first"] for d in rows]))
        for m in ["regret", "under", "over", "verdict", "calls"]: c[m] = C(rows, lambda d, m=m: d["m"][m])
        c["K2r"] = C(rows, lambda d: d["r"]["regret"]); c["tol"] = float(np.mean([d["tol"] for d in rows]))
        c["phi_max"] = float(np.mean([d["phi_max"] for d in rows])); c["acc_max"] = float(np.mean([d["acc_max"] for d in rows]))
        out["cells"][f"{p:.2f}"] = c
    D = out["contrasts"]; allr = sel(PHIS); tolr = sel(PHIS, lambda d: d["tol"] == 1.0); low = sel(LOW)
    D["R1 K2r-K2e|all"] = C(allr, lambda d: d["r"]["regret"] - d["m"]["regret"])
    if tolr: D["R1 K2r-K2e|within tolerance"] = C(tolr, lambda d: d["r"]["regret"] - d["m"]["regret"]); D["K2r|within tolerance"] = C(tolr, lambda d: d["r"]["regret"])
    D["regret K2e|all"] = C(allr, lambda d: d["m"]["regret"]); D["regret K2r|all"] = C(allr, lambda d: d["r"]["regret"])
    D["R2 first model-oracle|low"] = C(low, lambda d: d["m"]["first"] - d["oracle_first"]); D["R2 under|low"] = C(low, lambda d: d["m"]["under"])
    a, b = {d["draw"]: d for d in sel([0.10])}, {d["draw"]: d for d in sel([0.70])}; k = sorted(a)
    D["R3 first 0.10-0.70"] = ci([a[x]["m"]["first"] - b[x]["m"]["first"] for x in k], k, [a[x]["sc"] for x in k])
    D["first at 0.70"] = C(sel([0.70]), lambda d: d["m"]["first"]); D["over at 0.70"] = C(sel([0.70]), lambda d: d["m"]["over"])
    if xpath:                                    # the same run on the original surface (X sweep, persistent arm)
        import env
        xe = env.load_episodes(os.path.join(HERE, "episodes_x.jsonl")); xt = env.load_truth(os.path.join(HERE, "ground_truth_x.jsonl"))
        xids = [eps[i]["x_id"] for i in ids]; XR = rows_of(load(xpath), xe, xt, xids); pair = {i: XR[eps[i]["x_id"]] for i in ids}
        def P(sub, f):
            s = [i for i in ids if any(abs(eps[i]["phi"] - p) < 1e-9 for p in sub)]
            return ci([f(R[i]) - f(pair[i]) for i in s], [R[i]["draw"] for i in s], [R[i]["sc"] for i in s])
        D["R4 first runbook-original|low"] = P(LOW, lambda d: d["m"]["first"]); D["R4 regret runbook-original|all"] = P(PHIS, lambda d: d["m"]["regret"])
        D["R4 under runbook-original|low"] = P(LOW, lambda d: d["m"]["under"]); D["R4 tol runbook-original|all"] = P(PHIS, lambda d: d["tol"])
        out["original"] = dict(n_tol=int(sum(d["tol"] for d in XR.values())), regret=float(np.mean([d["m"]["regret"] for d in XR.values()])),
                               first_low=float(np.mean([pair[i]["m"]["first"] for i in ids if eps[i]["phi"] < 0.5])))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mock"); ap.add_argument("--runs", nargs="*", default=list(RUNS)); a = ap.parse_args()
    eps, truth = load_episodes(), load_truth(); out = {}; L = []
    jobs = [("mock", ("mock", a.mock, None))] if a.mock else [(k, RUNS[k]) for k in a.runs]
    for key, (label, fn, xfn) in jobs:
        o = analyse(fn if a.mock else os.path.join(HERE, "logs", fn), None if a.mock else os.path.join(HERE, "logs", xfn), eps, truth)
        o["label"] = label; out[key] = o
        L += ["", f"== {label}   n={o['n']}  usable JSON {o['usable']}  within tolerance {o['n_tol']}  no submit {o['no_submit']}  invalid {o['invalid']}"
              + (f"   [original surface: within tolerance {o['original']['n_tol']}, regret {o['original']['regret']:.2f}]" if "original" in o else ""),
              f"{'phi':6s}{'first: model':>13s}{'oracle':>8s}{'planner':>9s}{'K2e':>8s}{'K2r':>8s}{'under':>8s}{'over':>8s}{'verdict':>9s}{'calls':>7s}{'tol':>6s}{'phi err':>9s}{'acc err':>9s}"]
        for cname, c in o["cells"].items():
            L.append(f"{cname:6s}{100 * c['first_model']['mean']:12.0f}%{100 * c['first_oracle']:7.0f}%{100 * c['first_planner']:8.0f}%"
                     f"{c['regret']['mean']:8.2f}{c['K2r']['mean']:8.2f}{c['under']['mean']:8.2f}{c['over']['mean']:8.2f}{c['verdict']['mean']:9.2f}"
                     f"{c['calls']['mean']:7.2f}{100 * c['tol']:5.0f}%{c['phi_max']:9.3f}{c['acc_max']:9.3f}")
        for name, d in o["contrasts"].items():
            pct = "first" in name or " tol " in name
            L.append(f"  {name:34s} {fmt(d, pct=pct)}" + ("  (percentage points)" if pct else ""))
    txt = "\n".join(L); print(txt)
    if not a.mock:
        json.dump(out, open(os.path.join(HERE, "results", "r_analysis.json"), "w"), indent=1)
        open(os.path.join(HERE, "results", "r_analysis.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
