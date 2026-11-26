"""
Second task domain: analysis (PLAN_d2_domain.md). No model call.

  python d2_analysis.py [--workers 4]                # the three model runs
  python d2_analysis.py --mock FILE [--limit N]      # any K2e log of the D2 episodes (code check only)

Per run and episode: the K2e trajectory is scored against the exact oracle of the true model; the exact planner is
replayed on the stored JSON in the same world (K2r); both give regret and its parts. Tool 0 is the replica tool (A).
Contrasts (nested bootstrap: scenarios, then parameter draws; 95%), fixed in the plan:
  R1  executor gap: K2r - K2e on the 100 DN episodes (and on all 200)
  R2  DX, replica fault rates 0.10 and 0.30: model minus oracle share of episodes opening with the replica tool;
      regret at steps where the oracle would query the replica tool and the model does not (under-use)
  R3  DX: the model's replica first-call share at 0.10 minus at 0.70 (oracle: 100 points); over-use regret at 0.70
  R4  DN (descriptive): share of regret at steps where the replica tool is passed over; verdict regret
Output: results/d2_analysis.json, results/d2_analysis.txt
"""
import os
os.environ["RAEA_DOMAIN"] = "2"
os.environ.setdefault("RAEA_EPISODES", "episodes_d2.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_d2.jsonl")
import argparse, json
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = {"flash": ("flash (non-thinking)", "ds_chat_fc"), "terra": ("gpt-6.1-terra (medium)", "codex_terra"), "sonnet": ("Sonnet 5.5 (high)", "claude_s55")}
PHIS = [0.10, 0.30, 0.50, 0.70]; REP, FEED, NT = 0, 1, 5
P = lambda *a: os.path.join(HERE, *a)


def first_action(log):
    c = [x["j"] for x in log if x["type"] == "call"]; return c[0] if c else NT


def work(job):
    """One episode of one run: score the model's trajectory and the planner's replay; frees the exact models afterwards."""
    ep, truth, rec = job
    import oracle
    from x_analysis import parts, replay
    spec = rec["elicit"]["spec"]; m = parts(ep, rec["log"], truth); oracle._MODELS.clear()
    rl = replay(ep, spec); r = parts(ep, rl, truth); oracle._MODELS.clear()
    gA = [g for g in spec["groups"] if REP in g["tools"]][0]; gB = [g for g in spec["groups"] if FEED in g["tools"]][0]
    return ep["id"], dict(m=m, r=r, fa=first_action(rec["log"]), kf=first_action(rl), usable=float(bool(rec["elicit"]["parse_ok"])),
                          no_submit=float(rec["status"] != "submitted"),
                          json_ok=float(len(gA["tools"]) == 1 and len(gB["tools"]) == 1 and bool(gA["shared_fault"]) and bool(gB["shared_fault"])
                                        and bool(gA["persistent"]) and bool(gB["persistent"])
                                        and all(not g["shared_fault"] for g in spec["groups"] if REP not in g["tools"] and FEED not in g["tools"])))


def ci(v, ids, scl):
    from nestboot import nested_ci
    m, lo, hi = nested_ci(v, ids, scl, weight="episode"); return dict(mean=float(m), lo=float(lo), hi=float(hi), n=len(set(ids)))


def analyse(path, EP, TR, ORC, workers, limit=None, mock=False):
    recs = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["condition"] == "K2e" and not r["status"].startswith("api_error") and r.get("sample", 0) == 0: recs[r["episode"]] = r
    ids = sorted(EP)[:limit] if limit else sorted(EP)
    if mock: ids = [i for i in ids if i in recs]      # code check on a partial mock log
    miss = [i for i in ids if i not in recs]
    assert not miss, f"{len(miss)} episodes missing, e.g. {miss[:3]}"
    with Pool(workers) as pool: res = dict(pool.imap_unordered(work, [(EP[i], TR[i], recs[i]) for i in ids], chunksize=1))
    R = [dict(res[i], id=i, part=EP[i]["part"], phi=EP[i]["phi"], draw=EP[i]["pair"], sc=EP[i]["scenario"], of=ORC[i]["first"]) for i in ids]
    C = lambda rows, f: ci([f(d) for d in rows], [d["draw"] for d in rows], [d["sc"] for d in rows])
    cell = lambda rows: dict(
        n=len(rows), first_rep=C(rows, lambda d: float(d["fa"] == REP)), oracle_rep=float(np.mean([d["of"] == REP for d in rows])),
        planner_rep=float(np.mean([d["kf"] == REP for d in rows])),
        first_model=[float(np.mean([d["fa"] == j for d in rows])) for j in range(NT + 1)], first_oracle=[float(np.mean([d["of"] == j for d in rows])) for j in range(NT + 1)],
        regret=C(rows, lambda d: d["m"]["regret"]), K2r=C(rows, lambda d: d["r"]["regret"]), under=C(rows, lambda d: d["m"]["under"]),
        over=C(rows, lambda d: d["m"]["over"]), verdict=C(rows, lambda d: d["m"]["verdict"]), calls=C(rows, lambda d: d["m"]["calls"]),
        rep_calls=C(rows, lambda d: d["m"]["rep_calls"]), json_ok=float(np.mean([d["json_ok"] for d in rows])))
    DN = [d for d in R if d["part"] == "DN"]; DX = lambda ps: [d for d in R if d["part"] == "DX" and any(abs(d["phi"] - p) < 1e-9 for p in ps)]
    out = dict(n=len(R), usable=int(sum(d["usable"] for d in R)), no_submit=int(sum(d["no_submit"] for d in R)), cells={}, contrasts={})
    if DN: out["cells"]["DN"] = cell(DN)
    for p in PHIS:
        if DX([p]): out["cells"][f"DX|{p:.2f}"] = cell(DX([p]))
    D = out["contrasts"]
    if DN:
        D["R1 K2r-K2e|DN"] = C(DN, lambda d: d["r"]["regret"] - d["m"]["regret"]); D["regret|DN"] = C(DN, lambda d: d["m"]["regret"])
        D["R4 under|DN"] = C(DN, lambda d: d["m"]["under"]); D["R4 verdict|DN"] = C(DN, lambda d: d["m"]["verdict"])
        out["pass_share_DN"] = float(np.sum([d["m"]["under"] for d in DN]) / max(np.sum([d["m"]["regret"] for d in DN]), 1e-12))
        D["first replica model-oracle|DN"] = C(DN, lambda d: float(d["fa"] == REP) - float(d["of"] == REP))
    D["R1 K2r-K2e|all"] = C(R, lambda d: d["r"]["regret"] - d["m"]["regret"])
    lo = DX([0.10, 0.30])
    if lo:
        D["R2 first replica model-oracle|DX 0.10,0.30"] = C(lo, lambda d: float(d["fa"] == REP) - float(d["of"] == REP)); D["R2 under|DX 0.10,0.30"] = C(lo, lambda d: d["m"]["under"])
        a = {d["draw"]: d for d in DX([0.10])}; b = {d["draw"]: d for d in DX([0.70])}; k = sorted(set(a) & set(b))
        if k: D["R3 first replica 0.10-0.70|DX"] = ci([float(a[x]["fa"] == REP) - float(b[x]["fa"] == REP) for x in k], k, [a[x]["sc"] for x in k])
        if DX([0.70]): D["R3 over at 0.70|DX"] = C(DX([0.70]), lambda d: d["m"]["over"])
    return out


def fmt(d, pct=False):
    k = 100 if pct else 1; f = (lambda x: f"{k * x:+.0f}") if pct else (lambda x: f"{x:+.3f}")
    return f"{f(d['mean'])} [{f(d['lo'])}, {f(d['hi'])}]"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mock"); ap.add_argument("--limit", type=int); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--runs", nargs="*", default=list(RUNS)); a = ap.parse_args()
    from env import load_episodes, load_truth
    EP, TR = load_episodes(P("episodes_d2.jsonl")), load_truth(P("ground_truth_d2.jsonl")); ORC = json.load(open(P("d2_oracle.json")))
    jobs = [("mock", ("mock", None))] if a.mock else [(k, RUNS[k]) for k in a.runs]; out, L = {}, []
    names = ["A", "B", "log", "prim", "probe", "none"]
    for key, (label, stem) in jobs:
        o = analyse(a.mock if a.mock else P("logs", f"d2_{stem}_k2e.jsonl"), EP, TR, ORC, a.workers, a.limit, bool(a.mock)); o["label"] = label; out[key] = o
        L += ["", f"== {label}   n={o['n']}  usable JSON {o['usable']}  no submit {o['no_submit']}",
              f"{'cell':9s}{'first A: model':>15s}{'oracle':>7s}{'planner':>8s}{'K2e':>7s}{'K2r':>6s}{'under':>7s}{'over':>6s}{'verdict':>8s}{'calls':>6s}{'JSON ok':>8s}   first action model | oracle (A/B/log/prim/probe/none %)"]
        for cn, c in o["cells"].items():
            L.append(f"{cn:9s}{100 * c['first_rep']['mean']:14.0f}%{100 * c['oracle_rep']:6.0f}%{100 * c['planner_rep']:7.0f}%{c['regret']['mean']:7.2f}{c['K2r']['mean']:6.2f}"
                     f"{c['under']['mean']:7.2f}{c['over']['mean']:6.2f}{c['verdict']['mean']:8.2f}{c['calls']['mean']:6.2f}{100 * c['json_ok']:7.0f}%   "
                     + "/".join(f"{100 * x:.0f}" for x in c["first_model"]) + " | " + "/".join(f"{100 * x:.0f}" for x in c["first_oracle"]))
        for name, d in o["contrasts"].items():
            pct = "first" in name; L.append(f"  {name:46s} {fmt(d, pct)}" + ("  (pp)" if pct else ""))
        if "pass_share_DN" in o: L.append(f"  R4 share of DN regret at passed-over replica steps: {100 * o['pass_share_DN']:.0f}%")
    txt = "\n".join(L); print(txt)
    if not a.mock:
        json.dump(out, open(P("results", "d2_analysis.json"), "w"), indent=1); open(P("results", "d2_analysis.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
