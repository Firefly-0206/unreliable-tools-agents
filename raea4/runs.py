"""Round-4 experiments with exact evaluation. Usage: python runs.py X1 ... X6"""
import json, sys, time
import numpy as np
from sim import build, mmi, fault_unaware, split_true, llm_sim
from exact import run, summarize, with_dedup, fitted

M = {
    "true": lambda c: build(c["sp_test"]),
    "fault_unaware": lambda c: fault_unaware(c["sp"]),
    "mmi": lambda c: mmi(c["sp"]),
    "split_true": lambda c: split_true(c["sp"]),
    "oracle_dedup": lambda c: with_dedup(build(c["sp"]), c["sp"]),
    "unaware_dedup": lambda c: with_dedup(fault_unaware(c["sp"]), c["sp"]),
    "llm_composite": lambda c: llm_sim(c["sp"], inflate=0.5, temper=1.5),
    "composite_dedup": lambda c: with_dedup(llm_sim(c["sp"], inflate=0.5, temper=1.5), c["sp"]),
    "fit": lambda c: fitted(c["sp"], c["S"][:100], c["O"][:100]),
    "fit_dedup": lambda c: with_dedup(fitted(c["sp"], c["S"][:100], c["O"][:100]), c["sp"]),
    "fit_wrongmech": lambda c: fitted(c["sp"], c["S"][:100], c["O"][:100], "wrong_mechanism"),
    "fit_split": lambda c: fitted(c["sp"], c["S"][:100], c["O"][:100], "split_shared"),
    "fit_map": lambda c: fitted(c["sp"], c["S"][:100], c["Omap"][:100], eps=0.1),
    "fit_omit": lambda c: fitted(c["sp"], c["S"][:100], c["O"][:100], omit=c["omit"]),
}

def pols_for(models, planners=(("DP", ("dp",)), ("VOI1", ("look", 1)))):
    out = {"Oracle-DP": (("dp",), "true")}
    for m in models:
        for pn, pol in planners:
            if m == "true" and pn == "DP": continue
            out[f"{pn}[{m}]"] = (pol, m)
    return out

def X1():
    models = ["true", "fault_unaware", "mmi", "split_true", "oracle_dedup", "unaware_dedup", "fit", "fit_dedup",
              "llm_composite", "composite_dedup"]
    return {fam: summarize(run(fam, M, pols_for(models), 300, 101, n_cal_max=100)) for fam in ["RETRY", "SHARED", "IND"]}

def X2():
    models = ["true", "llm_composite", "fit", "fit_wrongmech", "fit_split", "fit_map", "fit_omit"]
    pols = pols_for(models)
    pols.update({"Fixed-3[true]": (("fixed", 3), "true"), "Thr0.9[true]": (("thr", 0.9), "true"),
                 "Thr0.9[llm_composite]": (("thr", 0.9), "llm_composite")})
    return {fam: summarize(run(fam, M, pols, 200, 102, n_cal_max=100)) for fam in ["RETRY", "SHARED", "IND"]}

def X3():
    out = {}
    for fam in ["RETRY", "SHARED"]:
        out[fam] = {}
        for n in [10, 30, 100, 300, 1000]:
            mods = {"true": M["true"],
                    "fit": lambda c, n=n: fitted(c["sp"], c["S"][:n], c["O"][:n]),
                    "fit_wrongmech": lambda c, n=n: fitted(c["sp"], c["S"][:n], c["O"][:n], "wrong_mechanism"),
                    "fit_split": lambda c, n=n: fitted(c["sp"], c["S"][:n], c["O"][:n], "split_shared"),
                    "fit_map": lambda c, n=n: fitted(c["sp"], c["S"][:n], c["Omap"][:n], eps=0.1)}
            pols = {"Oracle-DP": (("dp",), "true")}
            pols.update({f"DP[{m}]": (("dp",), m) for m in mods if m != "true"})
            # same seed for every n: same instances and the same raw calibration log, only the prefix changes
            out[fam][str(n)] = summarize(run(fam, mods, pols, 100, 103, n_cal_max=1000))
            print(fam, n, flush=True)
    return out

def shift_fault(sp, rng):
    sp2 = dict(sp); sp2["groups"] = {g: min(2 * p, 0.7) for g, p in sp["groups"].items()}; return sp2

def shift_primary(sp, rng):
    sp2 = dict(sp); tools = [dict(t) for t in sp["tools"]]; tools[0]["q"] -= 0.15; sp2["tools"] = tools; return sp2

def X4():
    mods = {"true": M["true"], "llm_composite": M["llm_composite"], "fit_stale": M["fit"],
            "fit_refit": lambda c: fitted(c["sp_test"], c["S_post"][:100], c["O_post"][:100])}
    pols = {"Oracle-DP": (("dp",), "true")}
    pols.update({f"DP[{m}]": (("dp",), m) for m in mods if m != "true"})
    out = {}
    for fam in ["RETRY", "SHARED"]:
        out[fam] = {}
        for name, sh in [("none", None), ("fault rate x2", shift_fault), ("primary tool degraded", shift_primary)]:
            out[fam][name] = summarize(run(fam, mods, pols, 150, 104, n_cal_max=100, shift=sh or (lambda sp, r: sp)))
            print(fam, name, flush=True)
    return out

def X5():
    models = ["true", "fault_unaware", "mmi", "split_true", "oracle_dedup", "unaware_dedup"]
    pols = pols_for(models, planners=(("DP", ("dp",)),))
    out = {}
    for fam, sevs in [("RETRY", [3]), ("SHARED", [1, 3, 10])]:
        out[fam] = {}
        for sev in sevs:
            out[fam][str(sev)] = {}
            for phi in [0.02, 0.05, 0.1, 0.2, 0.3, 0.4]:
                out[fam][str(sev)][str(phi)] = summarize(run(fam, M, pols, 100, 105, spec_kw=dict(sev_mult=sev, phi=phi)))
                print(fam, sev, phi, flush=True)
    return out

def X6():
    pols = {"Oracle-DP": (("dp",), "true"), "VOI-1": (("look", 1), "true"), "VOI-2": (("look", 2), "true"),
            "VOI-3": (("look", 3), "true"), "Fixed-1": (("fixed", 1), "true"), "Fixed-3": (("fixed", 3), "true"),
            "Thr-0.8": (("thr", 0.8), "true"), "Thr-0.95": (("thr", 0.95), "true"), "No-probe": (("noprobe",), "true")}
    out = {}
    for fam in ["IND", "RETRY", "SHARED", "COMP", "COMP3"]:
        t = time.time(); out[fam] = summarize(run(fam, M, pols, 150, 106)); print(fam, round(time.time() - t), flush=True)
    return out

if __name__ == "__main__":
    for e in sys.argv[1:]:
        t = time.time(); res = globals()[e]()
        json.dump(res, open(f"results/{e}.json", "w"), indent=1)
        print("done", e, round(time.time() - t), flush=True)
