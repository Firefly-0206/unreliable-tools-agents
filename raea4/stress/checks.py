"""Per-instance checks of the benign-commitment exclusion condition against exact posteriors after runs of benign
reports from the fault group, and depth profiles (VOI-1/2/3/DP regret). Writes results/checks.json.
Exclusion condition: pi0/(1-pi0) * cap < tau*/(1-tau*), with cap = 1/phi (paper) and, for S2 (fault correlated
with s), also the corrected cap 1/min_{s!=0} P(z=1|s). A violation = condition holds but the posterior after a
benign-only run reaches tau*; counted for runs of any length n<=20 (theory statement) and within budget
(n <= B / group-tool cost = 6)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_stress as R

def excl(exp, env, seed, fams, pars, n=200):
    out = {}
    for fam in fams:
        for par in pars:
            c = dict(n=n, condition_holds=0, violated=0, violated_inB=0, crossing_when_not_holding=0,
                     crossing_when_not_holding_inB=0)
            if exp == "S2": c.update(minS_condition_holds=0, minS_violated=0, minS_violated_inB=0)
            for i in range(n):
                sp, true, ag, src, d = env(fam, R.inst_rng(seed, n, i), par)
                phi = list(sp["groups"].values())[0]; pi0 = sp["prior"][0]; tau = R.TAU(sp)
                odds = tau / (1 - tau)
                if pi0 / (1 - pi0) / phi < odds:
                    c["condition_holds"] += 1; c["violated"] += d["crosses_tau"]; c["violated_inB"] += d["crosses_tau_inB"]
                else:
                    c["crossing_when_not_holding"] += d["crosses_tau"]
                    c["crossing_when_not_holding_inB"] += d["crosses_tau_inB"]
                if exp == "S2" and pi0 / (1 - pi0) / d["phi_s1"] < odds:     # P(z=1|s) equal for all s != 0
                    c["minS_condition_holds"] += 1; c["minS_violated"] += d["crosses_tau"]
                    c["minS_violated_inB"] += d["crosses_tau_inB"]
            out[f"{exp} {fam} {par}"] = c
            print(exp, fam, par, c, flush=True)
    return out

def prop1(n=200):
    """Prop. 1 diagnostics for S1, aggregated: max phi*ratio and number of instances where the 1/phi cap fails,
    over the full grid and within budget."""
    out = {}
    for fam in ("RETRY", "SHARED"):
        for f in (0.5, 0.7, 0.85, "q", 0.95, 0.97, 0.99):
            rows = []
            for i in range(n):
                sp = R.spec(fam, R.inst_rng(201, n, i))
                for t in sp["tools"]:
                    if t.get("group"): t["femit"] = R.emit_benign(sp["K"], t["q"] if f == "q" else f)
                rows.append(R.prop1_check(R.build(sp), sp))
            out[f"S1 {fam} {f}"] = dict(
                max_exact_over_middle=float(max(r["exact_over_middle"] for r in rows)),
                max_ratio_times_phi=float(max(r["max_ratio_times_phi"] for r in rows)),
                max_ratio_times_phi_inB=float(max(r["max_ratio_times_phi_inB"] for r in rows)),
                cap_fails=int(sum(r["max_ratio_times_phi"] > 1 + 1e-9 for r in rows)),
                cap_fails_inB=int(sum(r["max_ratio_times_phi_inB"] > 1 + 1e-9 for r in rows)))
            print(f"S1 {fam} {f}", out[f"S1 {fam} {f}"], flush=True)
    return out

def depth():
    out = {}
    for e in ("S1", "S2", "S3", "S4", "S5"):
        res = json.load(open(os.path.join(R.HERE, "results", f"{e}.json")))
        for fam, cells in res.items():
            for par, c in cells.items():
                P = c["policies"]
                for a in {p[3:-1] for p in P if p.startswith("DP[")} | {"true"}:
                    ks = [f"VOI1[{a}]", f"VOI2[{a}]", f"VOI3[{a}]", f"DP[{a}]" if a != "true" else "Oracle-DP"]
                    if all(k in P for k in ks): out[f"{e} {fam} {par} {a}"] = [P[k]["regret"] for k in ks]
    return out

if __name__ == "__main__":
    fams = ("RETRY", "SHARED")
    res = dict(exclusion={**excl("S1", R.env_S1, 201, fams, (0.5, 0.7, 0.85, "q", 0.95, 0.97, 0.99)),
                          **excl("S2", R.env_S2, 202, fams, (-0.5, 0.0, 0.5, 1.0)),
                          **excl("S5", R.env_S5, 205, fams, (0.0, 0.5))},
               prop1=prop1(), depth_VOI1_VOI2_VOI3_DP=depth())
    json.dump(res, open(os.path.join(R.HERE, "results", "checks.json"), "w"), indent=1)
