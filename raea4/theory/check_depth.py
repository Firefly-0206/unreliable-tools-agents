"""Theorem 2 (depth amplification under MMI) and Proposition 5 (simulation-lemma converse), checked against
sim.Model / exact.evaluate_exact.  python check_depth.py [n_random] [n_family]"""
import sys, json
import numpy as np
from multiprocessing import Pool
from common import sim, exact, minimal_spec, theorem2_constants

TOL = 1e-9

def act(M, pol, b, B):
    return M.Vd(b, B, pol[1])[1] if pol[0] == "look" else M.V(b, B)[1]

def zero_path(M, pol, B, c):
    """Follow the all-benign path of A reports; return (#reports when the agent stops, terminal action)."""
    b, n = M.prior.copy(), 0
    while True:
        a = act(M, pol, b, B)
        if a[0] == "stop": return n, a[1]
        assert a[1] == 0
        b = M.update(b, 0, 0); B -= c; n += 1

def check_instance(par):
    k = theorem2_constants(**par)
    sp = minimal_spec(**par); true = sim.build(sp); ag = sim.mmi(sp)
    Vstar = true.V(true.prior, sp["budget"])[0]; d_esc = par["ce"] - Vstar
    pols = [("look", d) for d in range(1, k["N"] + 1)] + [("dp",)]
    reg = {}; ok = {}
    for pol in pols:
        r = exact.evaluate_exact(true, ag, pol, sp); reg[pol] = r["total"] - Vstar; sev = r["severe"]
        depth = pol[1] if pol[0] == "look" else k["N"]
        if depth < k["mstar"]:
            root = act(ag, pol, ag.prior, sp["budget"])
            ok[pol] = (root == ("stop", 2)) and abs(reg[pol] - d_esc) < 1e-7
        elif depth >= k["mD"]:
            n, a = zero_path(ag, pol, sp["budget"], par["c"])
            p = par; P1N = (1 - p["pi0"]) * ((1 - p["phi"]) * (1 - p["q"]) ** k["N"] + p["phi"] * p["f"] ** k["N"])
            gam_cf = p["phi"] * p["f"] ** k["N"] * (p["lam"] * p["phi"] / (p["phi"] + p["pi0"] / (1 - p["pi0"])) - p["ce"])
            ok[pol] = (a == 0 and k["mstar"] <= n <= k["N"] and reg[pol] - d_esc >= k["gap_lb"] - TOL
                       and sev >= P1N - TOL and k["Gamma"] >= gam_cf - TOL > 0 and d_esc <= p["ce"] * k["PS"] + TOL)
    ok["no_declare_bad"] = declare_bad_never_chosen(par)
    w0 = par["pi0"] / (1 - par["pi0"])
    if w0 < k["Omega"]:
        ok["mstar_formula"] = int(np.ceil(np.log(k["Omega"] / w0) / np.log(k["ell"]) - 1e-12)) == k["mstar"]
    return dict(par=par, const={a: (float(v) if v is not None else None) for a, v in k.items()},
                d_esc=float(d_esc), regret={str(p): float(v) for p, v in reg.items()}, ok={str(p): bool(v) for p, v in ok.items()})

def random_par(rng):
    # wide ranges on purpose: high q (up to .99) and N up to 8 exercise near-zero posteriors
    q = rng.uniform(.55, .99); c = rng.uniform(.05, .6); N = int(rng.integers(2, 9))
    return dict(pi0=rng.uniform(.15, .85), phi=rng.uniform(.05, .6), q=q, f=rng.uniform(q, 1.0), c=c,
                lam=rng.uniform(10, 80), ce=rng.uniform(1, 8), B=N * c + 1e-6)

def declare_bad_never_chosen(par):
    """Sanity check that the disabled 'declare bad' action is never optimal under the true model (Theorem 2 setting)."""
    sp = minimal_spec(**par); N = int(np.floor(par["B"] / par["c"] + 1e-9))
    import itertools
    for M in (sim.build(sp), sim.mmi(sp)):
        for n in range(N + 1):
            for seq in itertools.product([0, 1], repeat=n):
                b = M.prior.copy()
                for o in seq: b = M.update(b, 0, o)
                if b is not None and M.term(b)[1] == 1: return False
    return True

# --------------------------------------------------------------- converse (small phi) ----------
def tv_exact(par, N):
    """Exact total variation between the true and MMI laws of (s, o_1..o_N) for the single source."""
    import itertools
    pi = [par["pi0"], 1 - par["pi0"]]; phi, q, f = par["phi"], par["q"], par["f"]
    h = lambda s, o: q if o == s else 1 - q
    e = lambda o: f if o == 0 else 1 - f
    tv = 0.0
    for s in (0, 1):
        for seq in itertools.product([0, 1], repeat=N):
            pt = (1 - phi) * np.prod([h(s, o) for o in seq]) + phi * np.prod([e(o) for o in seq])
            pm = np.prod([(1 - phi) * h(s, o) + phi * e(o) for o in seq])
            tv += pi[s] * abs(pt - pm)
    return tv / 2

def check_converse(par):
    sp = minimal_spec(**par); true = sim.build(sp); ag = sim.mmi(sp)
    N = int(np.floor(par["B"] / par["c"] + 1e-9))
    Vstar = true.V(true.prior, sp["budget"])[0]
    reg = exact.evaluate_exact(true, ag, ("dp",), sp)["total"] - Vstar
    tv = tv_exact(par, N); tvb = par["phi"] + 1 - (1 - par["phi"]) ** N
    Lmax = max(par["lam"], par["ce"]) + par["c"] * N
    return dict(par=par, regret=float(reg), tv=float(tv), tv_bound=float(tvb), bound=float(2 * Lmax * tv),
                ok=bool(tv <= tvb + TOL and reg <= 2 * Lmax * tv + TOL))

# --------------------------------------------------------------- paper families: regret vs depth --
def family_instance(args):
    fam, seed, depths = args
    rng = np.random.default_rng(seed); sp = sim.spec(fam, rng); true = sim.build(sp); ag = sim.mmi(sp)
    Vstar = true.V(true.prior, sp["budget"])[0]; out = {}
    for name, M in (("MMI", ag), ("exact", true)):
        for pol in [("look", d) for d in depths] + [("dp",)]:
            r = exact.evaluate_exact(true, M, pol, sp)
            out[f"{name} {pol}"] = (r["total"] - Vstar, r["severe"], r["repeat"])
    return out

if __name__ == "__main__":
    n_rand = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    n_fam = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    res = {}
    print("== Theorem 2: worked instances ==")
    worked = [dict(pi0=.5, phi=.3, q=.9, f=.97, c=.25, lam=30, ce=4, B=1.5),
              dict(pi0=.6, phi=.25, q=.9, f=.97, c=.2, lam=30, ce=4, B=1.2)]
    res["worked"] = [check_instance(p) for p in worked]
    for r in res["worked"]:
        print(r["par"]); print("  ", {a: (round(v, 4) if isinstance(v, float) else v) for a, v in r["const"].items()})
        print("   regret by depth:", {p: round(v, 3) for p, v in r["regret"].items()}, "| checks:", r["ok"])

    print("== Condition D(m) vs call cost (phi=.3, q=.9, f=.97, lam=30, c_e=4, N=6 calls) ==")
    res["cost_sweep"] = []
    for pi0 in (.5, .6):
        for c in (.2, .3, .4, .5, .6, .8, 1.0):
            p = dict(pi0=pi0, phi=.3, q=.9, f=.97, c=c, lam=30, ce=4, B=6 * c + 1e-6); k = theorem2_constants(**p)
            sp = minimal_spec(**p); true = sim.build(sp); ag = sim.mmi(sp); Vs = true.V(true.prior, sp["budget"])[0]
            rg = [float(exact.evaluate_exact(true, ag, pol, sp)["total"] - Vs) for pol in [("look", d) for d in range(1, 7)] + [("dp",)]]
            print(f"  pi0={pi0} c={c}: m*={k['mstar']} m_D={k['mD']}  MMI regret k=1..6,DP: {[round(x, 3) for x in rg]}")
            res["cost_sweep"].append(dict(pi0=pi0, c=c, mstar=k["mstar"], mD=k["mD"], regret=[float(x) for x in rg]))

    print("== Theorem 2: random instances satisfying the hypotheses ==")
    rng = np.random.default_rng(1); pars = []
    while len(pars) < n_rand:
        p = random_par(rng); k = theorem2_constants(**p)
        if k["excl"] and k["mstar"] and k["mD"] and k["mstar"] <= k["N"]: pars.append(p)
    with Pool(4) as pool: rr = pool.map(check_instance, pars)
    allok = all(all(r["ok"].values()) for r in rr)
    nchk = sum(len(r["ok"]) for r in rr)
    pos = np.mean([r["const"]["gap_lb"] > 0 for r in rr])
    PS0 = [r for r in rr if r["const"]["PS"] == 0]
    mono = np.mean([all(np.diff([r["regret"][str(("look", d))] for d in range(1, int(r["const"]["N"]) + 1)]) >= -1e-9) for r in rr])
    step = np.mean([min([r["regret"][str(("look", d))] for d in range(int(r["const"]["mD"]), int(r["const"]["N"]) + 1)]   # every k >= m_D
                        + [r["regret"][str(("dp",))]]) > r["d_esc"] + 1e-6 for r in rr])
    nhq = sum(r["par"]["q"] > .95 for r in rr); nN8 = sum(r["const"]["N"] >= 7 for r in rr)
    print(f"  instances {len(rr)} (q > .95 in {nhq}, N >= 7 in {nN8}), depth-claims checked {nchk}, all hold: {allok}")
    print(f"  explicit lower bound c+Gamma-ce*P(S) > 0 in {pos:.0%}; P(S) = 0 (clean case) in {len(PS0)} instances")
    print(f"  regret(k) - regret(k<m*) > 0 for every k >= m_D in {step:.0%}; regret non-decreasing in k over 1..N in {mono:.0%}")
    res["random"] = dict(n=len(rr), checks=nchk, all_ok=allok, frac_bound_positive=float(pos), n_PS0=len(PS0),
                         frac_strict_increase=float(step), frac_monotone=float(mono),
                         failures=[r for r in rr if not all(r["ok"].values())][:5])

    print("== Proposition 5 (converse): rare faults ==")
    rng = np.random.default_rng(2); cp = []
    for _ in range(60):
        p = random_par(rng); p["phi"] = float(rng.uniform(.001, .05)); cp.append(p)
    with Pool(4) as pool: cr = pool.map(check_converse, cp)
    print(f"  {len(cr)} instances; TV <= phi + 1-(1-phi)^N and regret(MMI-DP) <= 2 Lmax TV in all: {all(r['ok'] for r in cr)}")
    print(f"  max regret {max(r['regret'] for r in cr):.4f}; mean regret / bound {np.mean([r['regret'] / r['bound'] for r in cr]):.4f}")
    res["converse"] = dict(n=len(cr), all_ok=all(r["ok"] for r in cr), max_regret=max(r["regret"] for r in cr),
                           rows=cr[:10])

    print("== Paper families: MMI and exact regret by depth (true parameters, exact evaluation) ==")
    res["families"] = {}
    for fam, depths in (("RETRY", [1, 2, 3, 4]), ("SHARED", [1, 2, 3])):
        with Pool(4) as pool: fr = pool.map(family_instance, [(fam, 1000 + i, depths) for i in range(n_fam)])
        keys = fr[0].keys(); summ = {}
        for kk in keys:
            a = np.array([x[kk] for x in fr]); d = a[:, 0]
            # instance bootstrap: one evaluation per independently drawn instance, resample instances
            boots = d[np.random.default_rng(0).integers(0, len(d), (2000, len(d)))].mean(1)
            summ[kk] = dict(regret=float(d.mean()), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)),
                            severe=float(a[:, 1].mean()), repeat=float(a[:, 2].mean()))
            print(f"  {fam:6s} {kk:18s} regret {d.mean():.3f} [{summ[kk]['lo']:.3f}, {summ[kk]['hi']:.3f}]  severe {a[:, 1].mean():.3f}  repeat {a[:, 2].mean():.2f}")
        res["families"][fam] = dict(n=n_fam, summary=summ)
    json.dump(res, open("check_depth.json", "w"), indent=1, default=str)
