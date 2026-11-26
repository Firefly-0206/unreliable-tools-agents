"""Checks that the environment and the oracle agree exactly.
 1. Oracle-DP exact_eval == V(b0, B) == exact.evaluate_exact (the existing evaluator).
 2. Monte Carlo through Env (fresh worlds from the episode's params): mean realised loss of a non-oracle
    policy == its exact expected loss, and mean advantage sum == its exact expected regret.
 3. Oracle policy driven through Env has zero advantage at every step.
 4. Coherent discard (planner knows it drops repeat cache reports) == forbid repeat cache calls, exactly.
 5. Regret split: info + verdict == regret on every MC trajectory."""
import sys, os
import numpy as np
from env import Env, srcmap, sample_world, load_episodes, sim_spec
from oracle import exact_eval, baseline_policies, score_trajectory, true_model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exact as EX  # noqa: E402

def drive(ep, pol, rng):
    w, table = sample_world(ep["params"], rng, true_model(ep))
    e2 = dict(ep, table=table); env = Env(e2); st = pol.init(); n = 0
    while True:
        a = pol.act(st, env.budget, n)
        if a[0] == "stop":
            env.submit((env.sc["states"] + ["escalate"])[a[1]]); break
        j = a[1]; env.call(env.names[j]); o = env.hist[-1][1]; st = pol.update(st, j, o); n += 1
    return score_trajectory(e2, env.log, w)

def main(n_mc=4000):
    eps = load_episodes(); ok = True; sds = []
    for eid in ["PERSIST-000", "TRANSIENT-001", "SHARED-002", "MIXED-003", "VALUE-004", "TRANSIENT_P-005"]:
        ep = eps[eid]; par = ep["params"]; M = true_model(ep); sm = srcmap(par); pols = baseline_policies(par)
        v0 = M.V(M.prior.copy(), 6.0)[0]; ex = exact_eval(M, pols["Oracle-DP"], sm)["total"]
        old = EX.evaluate_exact(M, M, ("dp",), sim_spec(par))["total"]
        print(f"{eid}: V0={v0:.5f} exact_eval={ex:.5f} exact.py={old:.5f}"); ok &= abs(v0 - ex) < 1e-8 and abs(v0 - old) < 1e-8
        da = exact_eval(M, pols["Oracle+DiscardCache-aware-DP"], sm)["total"]; fc = exact_eval(M, pols["Oracle+ForbidCache-DP"], sm)["total"]
        print(f"   coherent discard {da:.6f} == forbid-cache {fc:.6f}"); ok &= abs(da - fc) < 1e-9
        for name in ["MMI-DP", "Fixed2-A", "FaultUnaware-DP"]:
            pol = baseline_policies(par)[name]; e = exact_eval(M, pol, sm)
            rng = np.random.default_rng(1); rs = [drive(ep, pol, rng) for _ in range(n_mc)]
            L = np.array([r["realised"] for r in rs]); R = np.array([r["regret"] for r in rs])
            split = max(abs(sum(s["adv"] for s in r["steps"]) - sum(s.get("adv_stoptime", 0) + s.get("adv_verdict", 0)
                        + (s["adv"] if s["action"][0] == "probe" else 0) for s in r["steps"])) for r in rs)
            ok &= split < 1e-9
            zL = (L.mean() - e["total"]) / (L.std() / np.sqrt(n_mc) + 1e-12); zR = (R.mean() - (e["total"] - v0)) / (R.std() / np.sqrt(n_mc) + 1e-12)
            print(f"   {name:16s} exact {e['total']:.3f}  MC loss {L.mean():.3f} (z={zL:+.2f})  exact regret {e['total']-v0:.3f}  "
                  f"MC adv-sum {R.mean():.3f} (z={zR:+.2f}, sd {R.std():.2f} vs loss sd {L.std():.2f})")
            if e["total"] - v0 > 1e-9: sds.append((R.std(), L.std()))
            ok &= abs(zL) < 4 and abs(zR) < 4
        rng = np.random.default_rng(2); rs = [drive(ep, pols["Oracle-DP"], rng) for _ in range(200)]
        mx = max(abs(s["adv"]) for r in rs for s in r["steps"]); print(f"   Oracle max |advantage| = {mx:.2e}"); ok &= mx < 1e-6
    sds = np.array(sds)
    print(f"Policies with nonzero regret: adv-sum sd {sds[:, 0].min():.2f}-{sds[:, 0].max():.2f} (median {np.median(sds[:, 0]):.2f}) "
          f"vs realised-loss sd {sds[:, 1].min():.2f}-{sds[:, 1].max():.2f} (median {np.median(sds[:, 1]):.2f}); "
          f"ratio loss/adv sd {np.min(sds[:, 1] / sds[:, 0]):.1f}-{np.max(sds[:, 1] / sds[:, 0]):.1f}")
    print("SELFTEST", "PASSED" if ok else "FAILED")

if __name__ == "__main__":
    main()
