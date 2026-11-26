"""Bayes reference of the hard-description experiment (PLAN_hd_hard_description.md). Offline, no model call.

A planner that reads every word as its range: each rate is uniform over the range of its word (env.HD_PHI_BINS,
env.HD_ACC_BINS; the generator's ranges), discretised on a grid of G_PHI points for the fault rate and G_ACC points for
each tool's accuracy (centres of equal sub-intervals). The hidden state becomes (s, z, theta), theta = grid point of
(phi, q_0..q_3); prior over s as documented (numbers), theta uniform, z ~ Bernoulli(phi_theta). Exact Bayes-optimal
dynamic programming on this model (the same sim.Model value recursion as the oracle). Played in each draw's world with
the pre-drawn outputs, scored under the TRUE model (x_analysis.parts), like every other trajectory.

  python hd_bayes.py --selftest     degenerate grid (one point = the true rates): must reproduce the oracle exactly
  python hd_bayes.py                every draw (HQ episodes; the reference is the same for HQ and HI, whose words agree)
Output: results/hd_bayes.json  {episode id of the draw's HQ episode: dict(regret, under, over, first, calls, log)}
"""
import os
os.environ.setdefault("RAEA_EPISODES", "episodes_hd.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_hd.jsonl")
import argparse, itertools, json, sys, time
from multiprocessing import Pool
import numpy as np
from env import HERE, K, LOSS, BUDGET, N_TOOLS, HD_PHI_BINS, HD_ACC_BINS, Env, build_model, stuck_emission, load_episodes, load_truth
from sim import Model, healthy

G_PHI, G_ACC = 5, 3


def grid(lo, hi, n):
    return [lo + (hi - lo) * (i + 0.5) / n for i in range(n)]


def bayes_model(ep, phis=None, accs=None):
    """phis: list of fault-rate grid points; accs: per tool, list of accuracy grid points (defaults: the word ranges)."""
    par, w = ep["params"], ep["words"]
    phis = phis if phis is not None else grid(*HD_PHI_BINS[w["phi"]], G_PHI)
    accs = accs if accs is not None else [grid(*HD_ACC_BINS[a], G_ACC) for a in w["acc"]]
    thetas = list(itertools.product(phis, *accs))
    H = [(s, z, t) for t in range(len(thetas)) for s in range(K) for z in (0, 1)]
    M = object.__new__(Model)
    M.K, M.loss, M.budget, M.temper = K, np.asarray(LOSS), float(BUDGET), 1.0
    M.tools = [dict(kind="sym", q=None, cost=t["cost"]) for t in par["tools"]]
    M.H, M.nH = H, len(H); M.s_of = np.array([h[0] for h in H])
    pr = np.array([par["prior"][s] * (thetas[t][0] if z else 1 - thetas[t][0]) for s, z, t in H]); M.prior = pr / pr.sum()
    em = np.asarray(stuck_emission(0))
    M.lik = []
    for j, t in enumerate(par["tools"]):
        L = np.zeros((M.nH, K))
        for i, (s, z, th) in enumerate(H):
            L[i] = em if (t["src"] == "cache" and z) else healthy("sym", thetas[th][1 + j], K, s)
        M.lik.append(L)
    M.cache = {}
    return M


def play(ep, M):
    env = Env(ep, "K2e", "text"); b = M.prior.copy(); verdicts = env.sc["states"] + ["escalate"]
    for _ in range(4 * N_TOOLS + 4):
        if env.done: break
        a = M.V(b, env.budget)[1]
        if a[0] == "stop": env.submit(verdicts[a[1]]); break
        env.call(env.names[a[1]]); b = M.update(b, a[1], env.hist[-1][1])
    return env.log


def one(args):
    from x_analysis import parts
    ep, tr = args; t0 = time.time(); M = bayes_model(ep); log = play(ep, M)
    out = dict(parts(ep, log, tr), log=log, V0_bayes=float(M.V(M.prior.copy(), float(BUDGET))[0]), sec=time.time() - t0)
    return ep["id"], out


def selftest(eps, truth):
    from x_analysis import parts
    n = 0
    for e in eps.values():
        if e["arm"] != "HQ": continue
        par = e["params"]; Mb = bayes_model(e, phis=[par["phi"]], accs=[[t["q"]] for t in par["tools"]]); Mt = build_model(par)
        vb, vt = Mb.V(Mb.prior.copy(), float(BUDGET)), Mt.V(Mt.prior.copy(), float(BUDGET))
        assert abs(vb[0] - vt[0]) < 1e-9 and vb[1] == vt[1], (e["id"], vb, vt)
        lb = play(e, Mb); pb = parts(e, lb, truth[e["id"]])
        assert pb["regret"] < 1e-9, (e["id"], pb["regret"])
        n += 1
    print(f"selftest ok: degenerate grid reproduces the oracle's value and first action, regret 0, in {n} draws")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args(); eps, truth = load_episodes(), load_truth()
    if a.selftest: return selftest(eps, truth)
    jobs = [(e, truth[i]) for i, e in sorted(eps.items()) if e["arm"] == "HQ"]
    with Pool(a.workers) as pool: res = dict(pool.map(one, jobs, chunksize=1))
    json.dump(res, open(os.path.join(HERE, "results", "hd_bayes.json"), "w"), indent=1)
    r = list(res.values())
    print(f"Bayes reference over {len(r)} draws: regret {np.mean([x['regret'] for x in r]):.3f}, first call replica "
          f"{100 * np.mean([x['first'] for x in r]):.0f}%, max {max(x['sec'] for x in r):.0f} s per draw")


if __name__ == "__main__":
    main()
