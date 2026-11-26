"""Exact expected metrics of the non-LLM baseline policies on every episode's parameters.
Regret CIs: nested bootstrap over scenarios, then episodes within scenario (nestboot.py).
Twin contrasts: a policy's regret in a fault family minus its regret in the phi = 0 twin of the same episode.
Usage: python baselines.py [--workers 4]   ->  results/baselines.json, results/baselines.txt"""
import argparse, json, os
from multiprocessing import Pool
import numpy as np
from env import HERE, FAMILIES, TWINS, build_model, srcmap, load_episodes, load_truth
from oracle import exact_eval, baseline_policies, BASELINES, qvalues
from nestboot import nested_ci, paired_ci

def one(ep):
    par = ep["params"]; true = build_model(par); sm = srcmap(par)
    r = {n: exact_eval(true, p, sm) for n, p in baseline_policies(par).items()}
    q = qvalues(true, true.prior.copy(), float(true.budget)); V = min(q.values())
    r["_first"] = dict(oracle=list(min(q, key=q.get)), q_probe_minus_V=q.get(("probe", 3), np.nan) - V)
    return ep["id"], r

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    eps = load_episodes(); truth = load_truth(); E = list(eps.values())
    with Pool(a.workers) as pool:
        per = dict(pool.map(one, E, chunksize=4))
    sc = {i: eps[i]["scenario"] for i in eps}
    reg = {n: {i: per[i][n]["total"] - per[i]["Oracle-DP"]["total"] for i in eps} for n in BASELINES}
    summ, lines = {}, []
    for fam in FAMILIES:
        ids = [i for i in eps if eps[i]["family"] == fam]; summ[fam] = {}
        for n in BASELINES:
            r = {k: float(np.mean([per[i][n][k] for i in ids])) for k in per[ids[0]][n]}
            r["regret"], r["regret_lo"], r["regret_hi"] = nested_ci([reg[n][i] for i in ids], ids, [sc[i] for i in ids])
            summ[fam][n] = r
        summ[fam]["_oracle_first"] = dict(
            probe_first=float(np.mean([per[i]["_first"]["oracle"] == ["probe", 3] for i in ids])),
            q_probe_minus_V=float(np.mean([per[i]["_first"]["q_probe_minus_V"] for i in ids])),
            probe_first_output_equals_s=float(np.mean([eps[i]["table"][3][0] == truth[i]["s"] for i in ids])),
            n=len(ids), scenarios={s: sum(sc[i] == s for i in ids) for s in sorted(set(sc[i] for i in ids))})
    # twin contrasts: regret(fault family) - regret(phi = 0 twin), paired by episode index
    twin = {}
    for t, base in TWINS.items():
        for n in BASELINES:
            a_ = {eps[i]["pair"]: reg[n][i] for i in eps if eps[i]["family"] == base}
            b_ = {eps[i]["pair"]: reg[n][i] for i in eps if eps[i]["family"] == t}
            m, lo, hi, k = paired_ci(a_, b_, {p: sc[p] for p in a_})
            twin[f"{base}-minus-{t}|{n}"] = dict(diff=m, lo=lo, hi=hi, n=k)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(dict(per_episode=per, by_family=summ, twin_contrasts=twin),
              open(os.path.join(HERE, "results", "baselines.json"), "w"), indent=1)
    P = lambda s: (print(s), lines.append(s))
    P("Exact expected total loss, regret vs Oracle-DP [95% nested-bootstrap CI over scenarios > episodes]")
    P(f"{'policy':32s}" + "".join(f"{f:>28s}" for f in FAMILIES))
    for n in BASELINES:
        P(f"{n:32s}" + "".join(f"{summ[f][n]['total']:7.2f} ({summ[f][n]['regret']:+5.2f} [{summ[f][n]['regret_lo']:+5.2f},{summ[f][n]['regret_hi']:+5.2f}])"
                               for f in FAMILIES))
    P("\nPer policy and family: calls / repeat-cache / repeat-log / P(used cache) / P(probe) / severe / esc")
    for n in ["Oracle-DP", "MMI-DP", "Oracle+ForbidAny-DP", "Oracle+ForbidCache-DP", "Oracle+DiscardCache-aware-DP",
              "Oracle+DiscardCache-unaware-DP", "MMI+DiscardCache-unaware-DP", "Probe-once"]:
        P(n)
        for f in FAMILIES:
            r = summ[f][n]
            P(f"  {f:12s} {r['calls']:.2f} / {r['rep_cache']:.2f} / {r['rep_log']:.2f} / {r['used_cache']:.2f} / "
              f"{r['probe_used']:.2f} / {r['severe']:.3f} / {r['esc']:.3f}")
    P("\nOracle first action at the prior: P(probe first) / mean Q(probe first)-V / P(probe 1st output == s) / scenarios")
    for f in FAMILIES:
        r = summ[f]["_oracle_first"]
        P(f"  {f:12s} {r['probe_first']:.2f} / {r['q_probe_minus_V']:.2f} / {r['probe_first_output_equals_s']:.3f} / {r['scenarios']}")
    P("\nTwin contrasts (regret in fault family minus regret in its phi=0 twin, same parameters) [95% CI]")
    for k, v in twin.items():
        if k.split("|")[1] in ("MMI-DP", "FaultUnaware-DP", "Oracle+ForbidAny-DP", "Oracle+ForbidCache-DP", "Fixed2-A",
                               "Probe-once", "Oracle-VOI1", "Oracle+DiscardCache-unaware-DP"):
            P(f"  {k:60s} {v['diff']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}] n={v['n']}")
    open(os.path.join(HERE, "results", "baselines.txt"), "w").write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
