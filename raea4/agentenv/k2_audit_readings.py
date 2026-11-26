"""K2 identifiability audit, quantitative part (no LLM calls).

The K2 documents do not state some parts of the observation model (see K2_AUDIT.md). For each part we build the alternative
reading that is consistent with the K2 text, let the EXACT planner of that reading act, and evaluate it exactly under the TRUE
model. The resulting regret is what an agent loses even with perfect planning if it resolves that ambiguity differently.

  R0 true            tools on one replica share a persistent session fault (name-sharing reading, = the oracle)
  R1 per-tool flags  each tool of a replica has its OWN persistent stale flag (names carry no sharing information)
  R2 no persistence  the documented incidence is a per-call probability (stale emission each call, no session flag)
  R3 emission alt    stale calls report the stuck value w.p. .97 and ONE other status w.p. .03 (leftover mass not spread evenly)
  R4 no staleness    the incidence is ignored (fault-unaware planner)
  R5 forbid repeats  true model but never repeat a cache call (the 'repeat = redundant' heuristic)
Usage: python k2_audit_readings.py [--n 50] [--workers 4]  -> results/k2_audit_readings.json / .txt"""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from env import HERE, load_episodes, srcmap, F_STUCK
from oracle import exact_eval, BayesPol, true_model
import k2decomp as K

FAMS = ["PERSIST", "SHARED", "MIXED", "TRANSIENT", "TRANSIENT_P"]

def reading_spec(ep, name):
    sp = K.truth_spec(ep)
    for g in sp["groups"]:
        if g["tools"][0] in [j for j, t in enumerate(ep["params"]["tools"]) if t["src"] == "cache"] and g["shared_fault"]:
            if name == "R1":
                pass                                          # handled below (split group)
            if name == "R2": g["persistent"] = False
            if name == "R3": g["emission"] = [F_STUCK, 1 - F_STUCK, 0.0]
            if name == "R4": g["shared_fault"] = False
    if name == "R1":
        new = []
        for g in sp["groups"]:
            if g["shared_fault"] and len(g["tools"]) > 1: new += [dict(g, tools=[j]) for j in g["tools"]]
            else: new.append(g)
        sp["groups"] = new
    return sp

def one(eid):
    ep = load_episodes()[eid]; M = true_model(ep); sm = srcmap(ep["params"]); out = {}
    v0 = M.V(M.prior.copy(), float(M.budget))[0]
    for r in ["R1", "R2", "R3", "R4"]:
        pol = BayesPol(K.build_recovered_model(ep, reading_spec(ep, r)), sm)
        out[r] = exact_eval(M, pol, sm)["total"] - v0
    pol = BayesPol(K.build_recovered_model(ep, K.truth_spec(ep)), sm, forbid={"c"})
    out["R5"] = exact_eval(M, pol, sm)["total"] - v0
    out["R0"] = exact_eval(M, BayesPol(K.build_recovered_model(ep, K.truth_spec(ep)), sm), sm)["total"] - v0
    return eid, out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=50); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(); eps = load_episodes()
    ids = [i for f in FAMS for i in sorted(e for e in eps if eps[e]["family"] == f)[:a.n]]
    with ProcessPoolExecutor(a.workers) as ex: res = dict(ex.map(one, ids, chunksize=4))
    cols = ["R0", "R1", "R2", "R3", "R4", "R5"]; table = {}
    lines = [f"{'family':12s}" + "".join(f"{c:>8s}" for c in cols) + "   (exact regret of the exact planner of each reading, true model; first {} episodes)".format(a.n)]
    for f in FAMS:
        rows = [res[i] for i in ids if eps[i]["family"] == f]; table[f] = {c: float(np.mean([r[c] for r in rows])) for c in cols}
        lines.append(f"{f:12s}" + "".join(f"{table[f][c]:8.2f}" for c in cols))
    core = [f for f in FAMS if f != "TRANSIENT_P"]
    table["CORE"] = {c: float(np.mean([table[f][c] for f in core])) for c in cols}
    lines.append(f"{'CORE':12s}" + "".join(f"{table['CORE'][c]:8.2f}" for c in cols) + "   (PERSIST SHARED MIXED TRANSIENT, equal weights)")
    json.dump(dict(table=table, per_episode=res), open(os.path.join(HERE, "results", "k2_audit_readings.json"), "w"), indent=1)
    open(os.path.join(HERE, "results", "k2_audit_readings.txt"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))

if __name__ == "__main__":
    main()

