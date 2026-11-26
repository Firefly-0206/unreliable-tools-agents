"""
Matched-marginal dependence experiment: analysis (PLAN_x_dependence.md). No model call.

  set RAEA_EPISODES=episodes_x.jsonl, RAEA_TRUTH=ground_truth_x.jsonl (done here before importing env)
  python x_analysis.py                 # the three model runs
  python x_analysis.py --mock FILE     # any K2e log of the X episodes (code check only)

Per run it reads the K2e log, replays the exact planner on each stored JSON (K2r, as k2e_replay.py does; and K2r*, the
same with a group counted as faulty whenever its written phi is positive, whatever its shared_fault flag), and scores
every trajectory with oracle.score_trajectory under the TRUE model of the episode's arm.

Quantities per arm (XP persistent, XI independent per call) and fault rate:
  first      share of episodes whose first call is the replica tool: model, true-model oracle, planner on the JSON
  under      regret at steps where the oracle would query the replica tool and the model probes another tool or stops
  over       regret at steps where the model queries the replica tool and the oracle would not
  regret     K2e, K2r, K2r*
Contrasts (nested bootstrap: scenarios, then parameter draws; 95%), all fixed in the plan:
  D1  low fault rates (0.10, 0.30): model minus oracle first-call share, per arm; XP minus XI of the model's share and of
      under (paired by draw)
  D2  model first-call share at 0.10 minus at 0.70, per arm (the oracle's is 1); over at 0.70
  D3  K2r - K2e per arm, pooled over fault rates
Output: results/x_analysis.json, results/x_analysis.txt
"""
import os
os.environ.setdefault("RAEA_EPISODES", "episodes_x.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_x.jsonl")
import argparse, copy, json
from collections import defaultdict
import numpy as np
from env import HERE, Env, SCENARIOS, load_episodes, load_truth, build_model, system_prompt
from oracle import score_trajectory, qvalues
from nestboot import nested_ci
import k2decomp as K

RUNS = {"flash": ("flash (non-thinking)", "x_ds_chat_fc_k2e.jsonl"), "terra": ("gpt-6.1-terra (medium)", "x_codex_terra_k2e.jsonl"),
        "sonnet": ("Sonnet 5.5 (high)", "x_claude_s55_k2e.jsonl")}
PHIS = [0.10, 0.30, 0.50, 0.70]; ARMS = ["XP", "XI"]; REP = 0      # tool 0 is the replica tool in every X episode


def load(path):
    recs = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["condition"] != "K2e" or r["status"].startswith("api_error") or r.get("sample", 0) != 0: continue
        recs[r["episode"]] = r
    return recs


def parts(ep, log, truth):
    s = score_trajectory(ep, log, truth); under = over = 0.0
    for st in s["steps"]:
        o_rep = st["oracle"][0] == "probe" and st["oracle"][1] == REP
        if st["action"][0] == "probe":
            m_rep = st["action"][1] == REP
            if o_rep and not m_rep: under += st["adv"]
            if m_rep and not o_rep: over += st["adv"]
        elif o_rep:
            under += st["adv_stoptime"]
    calls = [x["j"] for x in log if x["type"] == "call"]
    return dict(regret=s["regret"], under=under, over=over, first=float(bool(calls) and calls[0] == REP),
                used=float(REP in calls), calls=len(calls), rep_calls=calls.count(REP), verdict=sum(st.get("adv_verdict", 0.0) for st in s["steps"]))


def replay(ep, spec):
    env = Env(ep, "K2e", "text"); K.plan_and_play(ep, spec, env); return env.log


def lenient(ep, raw):
    """K2r*: the written JSON with shared_fault forced to true in every group whose written phi is a positive number."""
    obj = K.extract_json(raw) if isinstance(raw, str) else raw
    if not isinstance(obj, dict): return None
    obj = copy.deepcopy(obj); om = obj.get("observation_model")
    for g in (om.get("tool_groups") if isinstance(om, dict) and isinstance(om.get("tool_groups"), list) else []):
        if isinstance(g, dict):
            p = K._num(g.get("phi"))
            if p is not None and p > 0: g["shared_fault"] = True
    return K.normalize_model(ep, obj, K.parse_docs(ep, system_prompt(ep, "K2e", "text")))


def ci(v, ids, scl):
    m, lo, hi = nested_ci(v, ids, scl); return dict(mean=float(np.mean(v)), lo=float(lo), hi=float(hi), n=len(set(ids)))


def analyse(path, eps, truth):
    recs = load(path); ids = sorted(eps); missing = [i for i in ids if i not in recs]
    assert not missing, f"{len(missing)} episodes missing, e.g. {missing[:3]}"
    R = {}
    for i in ids:
        ep, r = eps[i], recs[i]; spec = r["elicit"]["spec"]; M = build_model(ep["params"]); q = qvalues(M, M.prior.copy(), float(M.budget))
        d = dict(arm=ep["arm"], phi=ep["phi"], draw=ep["pair"], sc=ep["scenario"], m=parts(ep, r["log"], truth[i]),
                 r=parts(ep, replay(ep, spec), truth[i]), oracle_first=float(min(q, key=q.get) == ("probe", REP)),
                 usable=float(bool(r["elicit"]["parse_ok"])), no_submit=float(r["status"] != "submitted"),
                 json_persistent=None, json_faulty=None)
        ls = lenient(ep, r["elicit"].get("raw")); d["rs"] = parts(ep, replay(ep, ls if ls is not None else spec), truth[i])
        g = [x for x in spec["groups"] if REP in x["tools"]][0]
        d["json_faulty"] = float(g["shared_fault"] and g["phi"] > 0); d["json_persistent"] = float(bool(g["persistent"]))
        d["json_phi_err"] = abs((g["phi"] if g["shared_fault"] else 0.0) - ep["phi"])
        R[i] = d
    sel = lambda arm, phis: [d for d in R.values() if d["arm"] == arm and any(abs(d["phi"] - p) < 1e-9 for p in phis)]
    C = lambda rows, f: ci([f(d) for d in rows], [d["draw"] for d in rows], [d["sc"] for d in rows])
    out = dict(n=len(ids), usable=int(sum(d["usable"] for d in R.values())), no_submit=int(sum(d["no_submit"] for d in R.values())), cells={})
    for arm in ARMS:
        for p in PHIS:
            rows = sel(arm, [p]); c = dict(n=len(rows))
            c["first_model"] = C(rows, lambda d: d["m"]["first"]); c["first_oracle"] = float(np.mean([d["oracle_first"] for d in rows]))
            c["first_planner"] = float(np.mean([d["r"]["first"] for d in rows])); c["used_model"] = float(np.mean([d["m"]["used"] for d in rows]))
            for m in ["regret", "under", "over", "verdict", "calls", "rep_calls"]: c[m] = C(rows, lambda d, m=m: d["m"][m])
            c["K2r"] = C(rows, lambda d: d["r"]["regret"]); c["K2rs"] = C(rows, lambda d: d["rs"]["regret"])
            c["json_faulty"] = float(np.mean([d["json_faulty"] for d in rows])); c["json_persistent"] = float(np.mean([d["json_persistent"] for d in rows]))
            c["json_phi_err"] = float(np.mean([d["json_phi_err"] for d in rows]))
            out["cells"][f"{arm}|{p:.2f}"] = c
    by = lambda arm, p: {d["draw"]: d for d in sel(arm, [p])}
    D = {}
    for arm in ARMS:
        lowr = sel(arm, [0.10, 0.30])
        D[f"D1 first model-oracle|{arm}"] = C(lowr, lambda d: d["m"]["first"] - d["oracle_first"])
        D[f"D1 under|{arm}"] = C(lowr, lambda d: d["m"]["under"])
        a, b = by(arm, 0.10), by(arm, 0.70); k = sorted(a)
        D[f"D2 first 0.10-0.70|{arm}"] = ci([a[x]["m"]["first"] - b[x]["m"]["first"] for x in k], k, [a[x]["sc"] for x in k])
        D[f"D2 over at 0.70|{arm}"] = C(sel(arm, [0.70]), lambda d: d["m"]["over"])
        D[f"D2 first at 0.70|{arm}"] = C(sel(arm, [0.70]), lambda d: d["m"]["first"])
        allr = sel(arm, PHIS)
        D[f"D3 K2r-K2e|{arm}"] = C(allr, lambda d: d["r"]["regret"] - d["m"]["regret"])
        D[f"D3 K2r*-K2e|{arm}"] = C(allr, lambda d: d["rs"]["regret"] - d["m"]["regret"])
        D[f"regret|{arm}"] = C(allr, lambda d: d["m"]["regret"])
    for name, f in [("first", lambda d: d["m"]["first"]), ("under", lambda d: d["m"]["under"]), ("regret", lambda d: d["m"]["regret"])]:
        v, idv, scv = [], [], []
        for p in [0.10, 0.30]:
            a, b = by("XP", p), by("XI", p)
            for x in sorted(a): v.append(f(a[x]) - f(b[x])); idv.append(x); scv.append(a[x]["sc"])
        D[f"D1 {name} XP-XI"] = ci(v, idv, scv)
    out["contrasts"] = D
    return out


def fmt(d, pct=False):
    k = 100 if pct else 1; f = (lambda x: f"{k * x:+.0f}") if pct else (lambda x: f"{x:+.3f}")
    return f"{f(d['mean'])} [{f(d['lo'])}, {f(d['hi'])}]"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mock"); ap.add_argument("--runs", nargs="*", default=list(RUNS)); a = ap.parse_args()
    eps, truth = load_episodes(), load_truth(); out = {}; L = []
    jobs = [("mock", ("mock", a.mock))] if a.mock else [(k, RUNS[k]) for k in a.runs]
    for key, (label, fn) in jobs:
        o = analyse(fn if a.mock else os.path.join(HERE, "logs", fn), eps, truth); o["label"] = label; out[key] = o
        L += ["", f"== {label}   n={o['n']}  usable JSON {o['usable']}  no submit {o['no_submit']}",
              f"{'cell':9s}{'first: model':>13s}{'oracle':>8s}{'planner':>9s}{'K2e':>8s}{'K2r':>8s}{'K2r*':>8s}{'under':>8s}{'over':>8s}{'verdict':>9s}{'calls':>7s}{'JSON faulty/persist':>21s}"]
        for cname, c in o["cells"].items():
            L.append(f"{cname:9s}{100 * c['first_model']['mean']:12.0f}%{100 * c['first_oracle']:7.0f}%{100 * c['first_planner']:8.0f}%"
                     f"{c['regret']['mean']:8.2f}{c['K2r']['mean']:8.2f}{c['K2rs']['mean']:8.2f}{c['under']['mean']:8.2f}{c['over']['mean']:8.2f}"
                     f"{c['verdict']['mean']:9.2f}{c['calls']['mean']:7.2f}{100 * c['json_faulty']:13.0f}% /{100 * c['json_persistent']:4.0f}%")
        for name, d in o["contrasts"].items():
            L.append(f"  {name:28s} {fmt(d, pct='first' in name)}" + ("  (percentage points)" if "first" in name else ""))
    txt = "\n".join(L); print(txt)
    if not a.mock:
        json.dump(out, open(os.path.join(HERE, "results", "x_analysis.json"), "w"), indent=1)
        open(os.path.join(HERE, "results", "x_analysis.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
