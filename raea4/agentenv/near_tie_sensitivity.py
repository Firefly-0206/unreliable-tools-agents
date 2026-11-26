"""
Near-tie sensitivity of the regret attribution (post hoc, no model call).  python near_tie_sensitivity.py

The attribution labels a lossy step "replica tool passed over" when the oracle's argmin action at that step is a probe
of a replica (cache-backed) tool and the model does something else (k2e_anatomy.split). If the oracle's best
replica action and its best non-replica action are nearly tied, that label is fragile. For every such step we recompute
the exact Q-values (oracle.qvalues, the scorer's own) and its margin

    margin = min_{a not a replica probe} Q*(b,a) - min_{a replica probe} Q*(b,a)  (>= 0 at labelled steps)

and report, for eps in EPS, the share of K2e regret on labelled steps whose margin is at least eps (steps below eps
are moved to "rest"). Episodes: the held-out confirmation logs (CORE index 25..49) and the original logs (0..24),
for terra (medium) and Sonnet, the two runs with the attribution criterion C2. CIs: nested bootstrap (nestboot).
Output: results/near_tie_sensitivity.json, results/near_tie_sensitivity.txt
"""
import json, os
import numpy as np
from env import HERE, load_episodes, load_truth, srcmap
from oracle import score_trajectory, true_model, qvalues
import oracle
from nestboot import nested_ci
from k2e_anatomy import split
from anatomy_confirm import load as load_confirm, RUNS as CRUNS

EPS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
KEYS = ["terra", "sonnet"]


def margins(ep, log, s, src):
    """Margin at each scored step whose oracle action is a replica probe (None elsewhere), replaying the beliefs."""
    M = true_model(ep); b = M.prior.copy(); B = float(M.budget); out = []
    calls = [e for e in log if e["type"] in ("call", "submit") or e["type"] not in ("invalid",)]
    k = 0
    for e in log:
        if e["type"] == "invalid":
            continue
        q = qvalues(M, b, B)
        rep = [v for a, v in q.items() if a[0] == "probe" and src[a[1]] == "c"]
        non = [v for a, v in q.items() if not (a[0] == "probe" and src[a[1]] == "c")]
        st = s["steps"][k]; k += 1
        o_cache = st["oracle"][0] == "probe" and src[st["oracle"][1]] == "c"
        out.append(min(non) - min(rep) if (o_cache and rep) else None)
        if e["type"] == "call":
            b = M.update(b, e["j"], e["o"]); B -= M.tools[e["j"]]["cost"]
        else:
            break
    if len(out) < len(s["steps"]):  # forced final stop row
        q = qvalues(M, b, B); st = s["steps"][len(out)]
        rep = [v for a, v in q.items() if a[0] == "probe" and src[a[1]] == "c"]
        non = [v for a, v in q.items() if not (a[0] == "probe" and src[a[1]] == "c")]
        o_cache = st["oracle"][0] == "probe" and src[st["oracle"][1]] == "c"
        out.append(min(non) - min(rep) if (o_cache and rep) else None)
    return out


def pass_regret(s, src, mg, eps):
    """Regret on labelled pass-over steps whose margin >= eps (same step accounting as k2e_anatomy.split)."""
    tot = 0.0
    for st, m in zip(s["steps"], mg):
        if m is None or m < eps:
            continue
        if st["action"][0] == "probe":
            if src[st["action"][1]] != "c": tot += st["adv"]
        else:
            tot += st["adv_stoptime"]
    return tot


def main():
    eps_all, truth = load_episodes(), load_truth()
    out, L = {}, ["Near-tie sensitivity of the passed-over-replica share (post hoc; K2e; exact Q* margins)", ""]
    for key in KEYS:
        label, ce, cr, oe, orr, crit = CRUNS[key]
        for part, fn, lo, hi in [("held-out", ce, 25, 50), ("original", oe, 0, 25)]:
            e = load_confirm(fn, "K2e", lo, hi); ids = sorted(e); scl = [eps_all[i]["scenario"] for i in ids]
            rows = []
            for i in ids:
                ep = eps_all[i]; src = srcmap(ep["params"])
                s = score_trajectory(ep, e[i]["log"], truth[i]); d = split(s, src); mg = margins(ep, e[i]["log"], s, src)
                assert abs(pass_regret(s, src, mg, 0.0) - d["pass_cache"]) < 1e-9, (key, i)
                lab = [m for m in mg if m is not None]
                rows.append(dict(regret=d["regret"], pr={eps: pass_regret(s, src, mg, eps) for eps in EPS},
                                 lab_steps=len(lab), lab_small=sum(m < 0.05 for m in lab), margins=lab))
                oracle._MODELS.clear()
            R = float(np.sum([r["regret"] for r in rows])); res = dict(n=len(ids), regret=R / len(ids))
            for eps in EPS:
                share = float(np.sum([r["pr"][eps] for r in rows]) / R)
                c = nested_ci([r["pr"][eps] - (r["regret"] - r["pr"][eps]) for r in rows], ids, scl)
                res[f"eps={eps:g}"] = dict(share=share, pass_minus_rest=dict(mean=float(c[0]), lo=float(c[1]), hi=float(c[2])))
            allm = [m for r in rows for m in r["margins"]]
            res["labelled_steps"] = len(allm); res["median_margin"] = float(np.median(allm)) if allm else None
            res["share_steps_margin_lt_0.05"] = float(np.mean([m < 0.05 for m in allm])) if allm else None
            out[f"{key}|{part}"] = res
            L.append(f"{label} [{part}] n={len(ids)} mean regret {res['regret']:.3f}; labelled steps {len(allm)}, "
                     f"median margin {res['median_margin']:.3f}, margin<0.05: {100 * res['share_steps_margin_lt_0.05']:.0f}%")
            for eps in EPS:
                r = res[f"eps={eps:g}"]; c = r["pass_minus_rest"]
                L.append(f"   eps={eps:<5g} share {100 * r['share']:5.1f}%   pass-rest {c['mean']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}]")
            L.append("")
    json.dump(out, open(os.path.join(HERE, "results", "near_tie_sensitivity.json"), "w"), indent=1)
    open(os.path.join(HERE, "results", "near_tie_sensitivity.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
