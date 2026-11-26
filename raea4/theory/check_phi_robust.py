"""Proposition 4: robustness of the benign-commitment exclusion condition to an estimated fault rate.
The condition w0 / phi < Omega is a threshold on log(phi) - log(w0); with estimates (phi_hat, w0_hat) it is
classified correctly whenever |log phi_hat/phi| + |log w0_hat/w0| < |log(phi Omega / w0)| (triangle inequality,
so this is not an empirical check; we only report how often the sufficient condition applies).
Empirical parts: agreement of the plug-in classification vs the majority-class baseline, and the false-certification
rate of a percentile-bootstrap lower confidence bound, conditional on the condition being false. EM fits via sim.fit."""
import json, sys
import numpy as np
from multiprocessing import Pool
from common import sim, exact

NS = [25, 50, 100, 400, 1000]
CERT_NS = [100, 400]                # calibration sizes at which the bootstrap certificate is computed
ALPHAS = [.05, .025]

def excl(w0, phi, Om): return w0 / phi < Om

def one(args):
    fam, seed, n_boot = args
    ss = np.random.SeedSequence(seed); r_inst, r_cal, r_b = (np.random.default_rng(x) for x in ss.spawn(3))
    sp = sim.spec(fam, r_inst); true = sim.build(sp)
    L = sp["loss"]; K = sp["K"]; lam, ce = L[0, 1], L[K, 0]; Om = (lam - ce) / ce
    g = list(sp["groups"])[0]; phi = sp["groups"][g]; w0 = sp["prior"][0] / (1 - sp["prior"][0])
    S, O = exact.raw_calibration(true, sp, max(NS), r_cal)
    out = dict(true=bool(excl(w0, phi, Om)), margin=float(abs(np.log(phi * Om / w0))), fits={})
    for n in NS:
        tools, groups, prior = sim.fit(sp, S[:n], O[:n], sim.true_groups(sp))
        ph, wh = groups[g], prior[0] / (1 - prior[0])
        err = abs(np.log(ph / phi)) + abs(np.log(wh / w0))
        rec = dict(phi_hat=float(ph), w0_hat=float(wh), est=bool(excl(wh, ph, Om)), err=float(err))
        if n in CERT_NS:                                          # percentile-bootstrap one-sided lower bound
            stats = []
            for _ in range(n_boot):
                idx = r_b.integers(0, n, n); _, gb, pb = sim.fit(sp, S[:n][idx], O[:n][idx], sim.true_groups(sp))
                stats.append(np.log(gb[g]) - np.log(pb[0] / (1 - pb[0])))
            for a in ALPHAS:                                      # certify log(phi/w0) > log(1/Omega)
                rec[f"cert_{a}"] = bool(np.percentile(stats, 100 * a) > np.log(1 / Om))
        out["fits"][n] = rec
    return out

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, float(c - h)), min(1.0, float(c + h)))

if __name__ == "__main__":
    n_inst = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    res = dict(n_inst=n_inst, n_boot=n_boot)
    for fam in ("RETRY", "SHARED"):
        with Pool(4) as pool: rr = pool.map(one, [(fam, 5000 + i, n_boot) for i in range(n_inst)])
        tr = np.array([r["true"] for r in rr]); base = max(tr.mean(), 1 - tr.mean())
        print(f"[{fam}] exclusion condition holds (true parameters) in {tr.mean():.1%} of {n_inst} instances; "
              f"majority-class baseline ('always holds') agreement {base:.1%}")
        res[fam] = dict(true_rate=float(tr.mean()), baseline=float(base), by_n={}, certificate={})
        for n in NS:
            est = np.array([r["fits"][n]["est"] for r in rr]); agree = (est == tr).mean()
            bal = .5 * (est[tr].mean() + (~est[~tr]).mean())
            suff = np.array([r["fits"][n]["err"] < r["margin"] for r in rr])
            fa = int(np.sum(est & ~tr)); fm = int(np.sum(~est & tr))
            print(f"   n={n:5d}: agreement {agree:.1%} (baseline {base:.1%}, gain {agree - base:+.1%}), balanced accuracy {bal:.1%}; "
                  f"false exclusions {fa}/{int((~tr).sum())}, missed {fm}/{int(tr.sum())}; error < margin (guarantee applies) in {suff.mean():.0%}")
            res[fam]["by_n"][n] = dict(agree=float(agree), gain_over_baseline=float(agree - base), balanced_acc=float(bal),
                                       false_excl=fa, missed=fm, suff=float(suff.mean()))
        for n in CERT_NS:
            for a in ALPHAS:
                c = np.array([r["fits"][n][f"cert_{a}"] for r in rr])
                nf = int((~tr).sum()); wrong = int(np.sum(c & ~tr)); right = int(np.sum(c & tr))
                lo, hi = wilson(wrong, nf)
                print(f"   certificate n={n}, alpha={a}, {n_boot} resamples: certified {int(c.sum())}/{n_inst}; "
                      f"false certification among false-condition instances {wrong}/{nf} = {wrong / nf:.1%} "
                      f"[Wilson 95% {lo:.1%}, {hi:.1%}]; power {right}/{int(tr.sum())} = {right / tr.sum():.1%}")
                res[fam]["certificate"][f"n{n}_a{a}"] = dict(certified=int(c.sum()), wrong=wrong, n_false=nf,
                                                           rate=wrong / nf, ci=[lo, hi], power=right / float(tr.sum()))
    json.dump(res, open("check_phi_robust.json", "w"), indent=1)
