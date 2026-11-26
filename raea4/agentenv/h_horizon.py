"""
Post hoc (added 2026-11-18 after a mock review; no plan fixed in advance; no model call): effective planning horizon.

At every logged decision of a K2e run (calls and the stop decision; the verdict itself is not compared) the exact belief
under the true model is recomputed and the action is compared with the k-step lookahead policy pi_k (setup: depth-k
truncation of V, terminal risk at depth 0; k = 1 is VOI-1, k = inf the DP), optionally on a model whose replica fault
odds are multiplied by g (the as-if planner of z_asif.py). An action agrees with pi_k if its depth-k value is within
1e-6 of the best (ties count as agreement). Agreement is reported over all decisions, over first decisions and over
later decisions, by set and arm. The horizon is "fitted" per run as the k with the highest agreement on later decisions.

Sets (domain 1): core  CORE episodes of the decomposition runs (n=100 per run; fault families + fault-free twin)
                 x     fault-rate sweep, persistent and independent arms (200)
                 m     matched fault (MF) / fault-free (MZ) pairs (100)
  python h_horizon.py core|x|m      -> results/h_horizon_<set>.json, .txt
"""
import os, sys
SET = sys.argv[1] if len(sys.argv) > 1 else "core"
if SET == "x": os.environ.setdefault("RAEA_EPISODES", "episodes_x.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_x.jsonl")
if SET == "m": os.environ.setdefault("RAEA_EPISODES", "episodes_m.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_m.jsonl")
import copy, json
from multiprocessing import Pool
import numpy as np
from env import HERE, CORE, load_episodes
import k2decomp as KD

P = lambda *a: os.path.join(HERE, *a)
REP = 0
KS = [1, 2, 3, 4, "inf"]
GS = [1, 4]
LOGS = {"core": {"flash": "ds_chat_fc_k2decomp.jsonl", "terra": "codex_terra_medium_k2decomp.jsonl", "sonnet": "claude_s55.jsonl"},
        "x": {"flash": "x_ds_chat_fc_k2e.jsonl", "terra": "x_codex_terra_k2e.jsonl", "sonnet": "x_claude_s55_k2e.jsonl"},
        "m": {"flash": "m_ds_chat_fc_k2e.jsonl", "terra": "m_codex_terra_k2e.jsonl", "sonnet": "m_claude_s55_k2e.jsonl"}}[SET]
EPS = {"core": "episodes.jsonl", "x": "episodes_x.jsonl", "m": "episodes_m.jsonl"}[SET]


def load(fn):
    recs = {}
    for line in open(P("logs", fn), encoding="utf-8"):
        r = json.loads(line)
        if r["condition"] != "K2e" or r["status"].startswith("api_error") or r.get("sample", 0) != 0: continue
        recs[r["episode"]] = r
    return recs


def model_g(ep, g):
    spec = copy.deepcopy(KD.truth_spec(ep))
    for grp in spec["groups"]:
        if REP in grp["tools"] and grp["phi"] > 0:
            phi = grp["phi"]; grp["phi"] = g * phi / (1 - phi + g * phi)
    return KD.build_recovered_model(ep, spec)


def qk(M, b, B, k):
    """depth-k values of every legal action; stop = best terminal action."""
    q = {"stop": M.term(b)[0]}
    for j, t in enumerate(M.tools):
        if t["cost"] > B + 1e-9: continue
        v = t["cost"]
        for o, p in enumerate(M.pred(b, j)):
            if p > 1e-12:
                nb = M.update(b, j, o); v += p * (M.V(nb, B - t["cost"])[0] if k == "inf" else M.Vd(nb, B - t["cost"], k - 1)[0])
        q[j] = v
    return q


def episode(args):
    ep, log = args
    res = []
    for g in GS:
        M = model_g(ep, g); b = M.prior.copy(); B = float(M.budget); n = 0
        for e in log:
            if e["type"] not in ("call", "submit"): continue
            a = e["j"] if e["type"] == "call" else "stop"
            agree = {}
            for k in KS:
                q = qk(M, b, B, k); agree[str(k)] = float(a in q and q[a] <= min(q.values()) + 1e-6)
            res.append(dict(g=g, step=n, agree=agree))
            if e["type"] == "submit": break
            nb = M.update(b, e["j"], e["o"]); b = nb if nb is not None else M.prior.copy(); B -= M.tools[e["j"]]["cost"]; n += 1
    return ep["id"], res


def main():
    eps = load_episodes(P(EPS)); out, L = dict(set=SET, runs={}), [f"Effective horizon, set {SET} (post hoc): % of decisions agreeing with pi_k (ties agree)"]
    for key, fn in LOGS.items():
        recs = load(fn)
        ids = sorted(i for i, r in recs.items() if (SET != "core" or r["family"] in CORE))
        with Pool(12) as pool: R = dict(pool.map(episode, [(eps[i], recs[i]["log"]) for i in ids], chunksize=2))
        arms = {"all": ids}
        if SET == "core": arms.update(fault=[i for i in ids if eps[i]["family"] != "TRANSIENT"], faultfree=[i for i in ids if eps[i]["family"] == "TRANSIENT"])
        if SET == "x": arms.update(persistent=[i for i in ids if eps[i]["arm"] == "XP"], independent=[i for i in ids if eps[i]["arm"] == "XI"])
        if SET == "m": arms.update(MF=[i for i in ids if eps[i]["arm"] == "MF"], MZ=[i for i in ids if eps[i]["arm"] == "MZ"])
        o = {}
        L += ["", f"== {key} ({fn}, n={len(ids)})", f"  {'arm':12s}{'g':>3s} {'steps':>9s} " + "".join(f"{'k=' + str(k):>8s}" for k in KS) + "   (first / later decisions)"]
        for an, sel in arms.items():
            for g in GS:
                for part, cond in (("first", lambda s: s == 0), ("later", lambda s: s > 0)):
                    rows = [r for i in sel for r in R[i] if r["g"] == g and cond(r["step"])]
                    if not rows: continue
                    ag = {str(k): float(np.mean([r["agree"][str(k)] for r in rows])) for k in KS}
                    o[f"{an}|g{g}|{part}"] = dict(n=len(rows), agree=ag)
                    L.append(f"  {an:12s}{g:3d} {part + ' ' + str(len(rows)):>9s} " + "".join(f"{100 * ag[str(k)]:8.0f}" for k in KS))
        best = {c: max(KS, key=lambda k: o[c]["agree"][str(k)]) for c in o if c.endswith("later")}
        o["best_k_later"] = {c: str(v) for c, v in best.items()}
        out["runs"][key] = o
    txt = "\n".join(L); print(txt)
    json.dump(out, open(P("results", f"h_horizon_{SET}.json"), "w"), indent=1); open(P("results", f"h_horizon_{SET}.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
