"""
Theory-applicability stress tests S1-S5. Usage: python run_stress.py S1 S2 ... [--n 100] [--workers 4]
Exact expected-loss evaluation in the TRUE (assumption-violating) environment, paired regret against the
Oracle-DP of that environment, cluster bootstrap over instances (exact.summarize).
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (GModel, from_model, mmi_of, mmi_time, emit_benign, evaluate, dedup_of, summarize2, spec,  # noqa
                  build, source_of, healthy, STUCK)
from sim import fault_unaware, mmi, split_true                                                          # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
PLANNERS = (("DP", ("dp",)), ("VOI1", ("look", 1)))
TAU = lambda sp: 1 - sp["loss"][sp["K"], 1] / sp["loss"][0, 1]          # tau* = 1 - c_esc / lambda_sev

def inst_rng(seed, n, i):
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(n)[i].spawn(3)[0])

def srcmap(sp):
    return {j: source_of(t, j) for j, t in enumerate(sp["tools"])}

# ------------------------------------------------------------------ benign-run diagnostics --
def benign_runs(true, sp, nmax=20):
    """Exact Bayes factors / posteriors after runs of benign reports from the fault group only.
    Group tools are called alternately (n_j split as evenly as possible). Valid for static environments."""
    G = [j for j, t in enumerate(sp["tools"]) if t.get("group")]
    ps = true.marg_s(true.prior); out = []
    for n in range(1, nmax + 1):
        nj = {j: n // len(G) + (1 if k < n % len(G) else 0) for k, j in enumerate(G)}
        like = np.ones(true.nH)
        for j in G: like *= true.lik[j][:, 0] ** nj[j]
        PO = true.marg_s(true.prior * like) / np.maximum(ps, 1e-300)      # P(O | s)
        post = true.marg_s(true.prior * like); post /= post.sum()
        out.append(dict(n=n, nj=nj, ratio=[PO[0] / PO[s] for s in range(1, true.K)], p0=post[0]))
    return out

def n_budget(sp):
    """Most group reports an agent can collect within the budget (B / cheapest group-tool cost)."""
    return int(sp["budget"] // min(t["cost"] for t in sp["tools"] if t.get("group")))

def prop1_check(true, sp):
    """Proposition 1 on a grid of call counts: exact ratio <= middle expression (always), middle <= 1/phi (A4).
    The grid (n<=20 RETRY; n_A,n_B<=10 SHARED) exceeds the budget; *_inB restricts to n <= n_budget(sp)."""
    G = [j for j, t in enumerate(sp["tools"]) if t.get("group")]
    phi = list(sp["groups"].values())[0]; nB = n_budget(sp)
    q = {j: sp["tools"][j]["q"] for j in G}; f = {j: sp["tools"][j]["femit"][0] for j in G}
    ps = true.marg_s(true.prior)
    worst_mid, worst_bound, worst_inB, first_fail = 0.0, 0.0, 0.0, None
    grid = [(a,) for a in range(1, 21)] if len(G) == 1 else [(a, b) for a in range(11) for b in range(11) if a + b]
    for ns in grid:
        like = np.ones(true.nH)
        for j, n in zip(G, ns): like *= true.lik[j][:, 0] ** n
        PO = true.marg_s(true.prior * like) / np.maximum(ps, 1e-300)
        mid = 1 + (1 - phi) / phi * np.prod([(q[j] / f[j]) ** n for j, n in zip(G, ns)])
        r = max(PO[0] / PO[s] for s in range(1, true.K))
        worst_mid = max(worst_mid, r / mid)
        worst_bound = max(worst_bound, r * phi)
        if sum(ns) <= nB: worst_inB = max(worst_inB, r * phi)
        if r * phi > 1 + 1e-9 and (first_fail is None or sum(ns) < first_fail): first_fail = sum(ns)
    pi0 = ps[0]; tau = TAU(sp)
    return dict(exact_over_middle=worst_mid, max_ratio_times_phi=worst_bound, max_ratio_times_phi_inB=worst_inB,
                first_fail_n=first_fail, a4=all(f[j] >= q[j] - 1e-12 for j in G),
                excl_cond=bool(pi0 / (1 - pi0) / phi < tau / (1 - tau)))

def run_diag(runs, sp):
    """Crossing of tau* by the exact posterior after benign-only group runs: unrestricted (n <= 20, the
    theory statement) and within budget (n <= n_budget, what the agent can actually reach)."""
    tau = TAU(sp); p = np.array([r["p0"] for r in runs]); pB = p[:n_budget(sp)]
    return dict(max_p0=float(p.max()), p0_n20=float(p[-1]), crosses_tau=bool(p.max() >= tau),
                n_cross=int(np.argmax(p >= tau) + 1) if (p >= tau).any() else None,
                max_p0_inB=float(pB.max()), crosses_tau_inB=bool(pB.max() >= tau))

# ------------------------------------------------------------------ environments -----------
def env_S1(fam, rng, f):
    sp = spec(fam, rng)
    for t in sp["tools"]:
        if t.get("group"): t["femit"] = emit_benign(sp["K"], t["q"] if f == "q" else f)
    true = build(sp); src = srcmap(sp)
    ag = {"true": true, "fault_unaware": fault_unaware(sp), "mmi": mmi(sp),
          "oracle_dedup": dedup_of(build(sp), src)}
    if fam == "SHARED": ag["split_shared"] = split_true(sp)
    d = prop1_check(true, sp); d.update(run_diag(benign_runs(true, sp), sp))
    return sp, true, ag, src, d

def kappa_eff(kappa, phi, pi0):
    if kappa >= 0: return min(kappa, pi0 / (1 - pi0), 1 / phi - 1)
    return -min(-kappa, 1.0, (1 / phi - 1) * pi0 / (1 - pi0))

def env_S2(fam, rng, kappa):
    """P(z=1|s!=0) = phi(1+k), P(z=1|s=0) = phi(1 - k(1-pi0)/pi0): marginal fault rate stays phi.
    k = kappa clipped per instance so that both probabilities lie in [0, 1]."""
    sp = spec(fam, rng); src = srcmap(sp)
    a1 = build(sp); phi = list(sp["groups"].values())[0]; pi = sp["prior"]
    k = kappa_eff(kappa, phi, pi[0])
    pz = lambda s: phi * (1 + k) if s else phi * (1 - k * (1 - pi[0]) / pi[0])
    prior = np.array([pi[s] * (pz(s) if z[0] else 1 - pz(s)) for s, z in a1.H])
    true = from_model(a1, prior=prior)
    ag = {"true": true, "assume_A1": build(sp), "fault_unaware": fault_unaware(sp), "mmi": mmi_of(true),
          "oracle_dedup": dedup_of(from_model(a1, prior=prior), src)}
    d = dict(kappa_eff=k, phi_s0=pz(0), phi_s1=pz(1)); d.update(run_diag(benign_runs(true, sp), sp))
    return sp, true, ag, src, d

def env_S3(fam, rng, rho):
    """SHARED backend: flags (zS, zA, zB); tool A faulty iff zS or zA, B iff zS or zB.
    phi_S = rho*phi, phi_own = 1-(1-phi)/(1-phi_S), so each tool's marginal fault rate is phi."""
    sp = spec("SHARED", rng); src = srcmap(sp); K = sp["K"]
    phi = sp["groups"]["g"]; pS = rho * phi; pO = 1 - (1 - phi) / (1 - pS)
    H = [(s, z) for s in range(K) for z in __import__("itertools").product([0, 1], repeat=3)]
    pr = lambda p, z: p if z else 1 - p
    prior = [sp["prior"][s] * pr(pS, z[0]) * pr(pO, z[1]) * pr(pO, z[2]) for s, z in H]
    lik = []
    for j, t in enumerate(sp["tools"]):
        L = np.zeros((len(H), n_obs_(t, K)))
        for i, (s, z) in enumerate(H):
            bad = (j == 0 and (z[0] or z[1])) or (j == 1 and (z[0] or z[2]))
            L[i] = t["femit"] if bad else healthy(t["kind"], t["q"], K, s)
        lik.append(L)
    mk = lambda: GModel(K, H, prior, lik, sp["tools"], sp["loss"], sp["budget"])
    true = mk()
    ag = {"true": true, "assume_shared": build(sp), "assume_split": split_true(sp), "mmi": mmi_of(true),
          "oracle_dedup": dedup_of(mk(), src)}
    return sp, true, ag, src, dict(phi_shared=pS, phi_own=pO)

def n_obs_(t, K):
    return K if t["kind"] == "sym" else 2

def n_calls_max(sp):
    return int(sp["budget"] // min(t["cost"] for t in sp["tools"]))

def env_S4(fam, rng, par):
    """Hidden-Markov fault: after every call each faulty flag recovers w.p. r (and a healthy one fails
    w.p. onset). s stays static. mmi uses the time-0 per-call marginal (sim.mmi); mmi_t uses the true
    per-call marginal at the call's index t (marginal-matched at every call under recovery/onset)."""
    r, onset = par
    sp = spec(fam, rng); src = srcmap(sp); m = build(sp)
    T = np.zeros((m.nH, m.nH))
    step = lambda a, b: (r if b == 0 else 1 - r) if a else (onset if b else 1 - onset)
    for i, (s, z) in enumerate(m.H):
        for i2, (s2, z2) in enumerate(m.H):
            if s == s2: T[i, i2] = np.prod([step(a, b) for a, b in zip(z, z2)])
    true = from_model(m, trans=T)
    ag = {"true": true, "static_fault": build(sp), "fault_unaware": fault_unaware(sp), "mmi": mmi(sp),
          "mmi_t": mmi_time(true, n_calls_max(sp)), "oracle_dedup": dedup_of(from_model(m, trans=T), src)}
    return sp, true, ag, src, {}

def env_S5(fam, rng, psi, eta=0.03):
    """Value-sharing cache: a stale source repeats a stale copy v of the state value, v ~ psi*delta_s +
    (1-psi)*prior (drawn once per episode), reported with symbol noise eta. Hidden state (s, z, v)."""
    sp = spec(fam, rng); src = srcmap(sp); K = sp["K"]; pi = sp["prior"]
    phi = list(sp["groups"].values())[0]
    H = [(s, z, v) for s in range(K) for z in (0, 1) for v in range(K)]
    prior = [pi[s] * (phi if z else 1 - phi) * (psi * (v == s) + (1 - psi) * pi[v]) for s, z, v in H]
    lik = []
    for t in sp["tools"]:
        L = np.zeros((len(H), K))
        for i, (s, z, v) in enumerate(H):
            if t.get("group") and z:
                L[i] = eta / (K - 1); L[i, v] = 1 - eta
            else:
                L[i] = healthy(t["kind"], t["q"], K, s)
        lik.append(L)
    mk = lambda: GModel(K, H, prior, lik, sp["tools"], sp["loss"], sp["budget"])
    true = mk()
    ag = {"true": true, "benign_stuck": build(sp), "fault_unaware": fault_unaware(sp), "mmi": mmi_of(true),
          "oracle_dedup": dedup_of(mk(), src), "stuck_dedup": dedup_of(build(sp), src)}
    d = run_diag(benign_runs(true, sp), sp)
    d.update({"stuck_" + k: v for k, v in run_diag(benign_runs(build(sp), sp), sp).items()})
    return sp, true, ag, src, d

ENVS = {"S1": env_S1, "S2": env_S2, "S3": env_S3, "S4": env_S4, "S5": env_S5}

# ------------------------------------------------------------------ worker / driver --------
def work(args):
    exp, fam, par, seed, n, i, extra = args
    sp, true, ag, src, diag = ENVS[exp](fam, inst_rng(seed, n, i), par)
    out = {"Oracle-DP": evaluate(true, true, ("dp",), src)}
    for name, m in ag.items():
        for pn, pol in PLANNERS:
            if name == "true" and pn == "DP": continue
            out[f"{pn}[{name}]"] = evaluate(true, m, pol, src)
        for d in extra.get(name, ()):                       # extra lookahead depths
            out[f"VOI{d}[{name}]"] = evaluate(true, m, ("look", d), src)
    return (fam, str(par), i, out, diag)

CELLS = {
    "S1": [(fam, f) for fam in ("RETRY", "SHARED") for f in (0.5, 0.7, 0.85, "q", 0.95, 0.97, 0.99)],
    "S2": [(fam, k) for fam in ("RETRY", "SHARED") for k in (-0.5, 0.0, 0.5, 1.0)],
    "S3": [("SHARED", r) for r in (0.0, 0.25, 0.5, 0.75, 1.0)],
    "S4": [(fam, p) for fam in ("RETRY", "SHARED") for p in ((0.0, 0.0), (0.1, 0.0), (0.3, 0.0), (0.5, 0.0), (0.1, 0.05))],
    "S5": [(fam, p) for fam in ("RETRY", "SHARED") for p in (0.0, 0.5)],
}
SEEDS = {"S1": 201, "S2": 202, "S3": 203, "S4": 204, "S5": 205}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("exps", nargs="+"); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--depth", type=int, nargs="*", default=[])
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    for exp in a.exps:
        t0 = time.time()
        extra = {m: a.depth for m in ("true", "mmi", "mmi_t", "fault_unaware", "benign_stuck", "static_fault", "assume_A1",
                                      "assume_shared", "assume_split", "split_shared")} if a.depth else {}
        tasks = [(exp, fam, par, SEEDS[exp], a.n, i, extra) for fam, par in CELLS[exp] for i in range(a.n)]
        raw = {}
        with Pool(a.workers) as pool:
            for fam, par, i, out, diag in pool.imap_unordered(work, tasks, chunksize=2):
                raw.setdefault(fam, {}).setdefault(par, []).append((i, out, diag))
        res = {}
        for fam, cells in raw.items():
            res[fam] = {}
            for par, rows in cells.items():
                rows.sort(key=lambda x: x[0])
                per = {p: {k: np.array([r[1][p][k] for r in rows]) for k in rows[0][1][p]} for p in rows[0][1]}
                s = summarize2(per)
                dg = {}
                for k in rows[0][2]:
                    v = [r[2][k] for r in rows]
                    num = [x for x in v if x is not None and not isinstance(x, dict)]
                    dg[k] = dict(mean=float(np.mean(num)), max=float(np.max(num)), n_valid=len(num)) if num else None
                res[fam][par] = dict(policies=s, diag=dg)
        json.dump(res, open(os.path.join(HERE, "results", f"{exp}.json"), "w"), indent=1)
        print("done", exp, round(time.time() - t0), "s", flush=True)

if __name__ == "__main__":
    main()
