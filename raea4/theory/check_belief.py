"""Section T3: the current s-marginal is not a sufficient statistic for the next information-gathering action.
Lemma 3 (MMI agrees with the exact model at the root, to one step), Lemma 4 (the two-step covariance gap),
Proposition 3 (Pi-sufficiency only in the degenerate case) and the constructive example (Example 1)."""
import json
import numpy as np
from common import sim

TOL = 1e-10
rng = np.random.default_rng(11)
res = {}

def qvalues(M, b, B, names):
    """Exact action values: terminal actions from R, probes from c_j + E V(b', B - c_j) under model M."""
    ps = M.marg_s(b); Q = {}
    for a in range(M.K + 1): Q[f"term{a}"] = float(M.loss[a] @ ps)
    for j, t in enumerate(M.tools):
        if t["cost"] > B + 1e-9: continue
        Q[f"query_{names[j]}"] = float(t["cost"] + sum(p * M.V(M.update(b, j, o), B - t["cost"])[0]
                                                       for o, p in enumerate(M.pred(b, j)) if p > 1e-12))
    return Q

def q1(M, b, B):
    """Depth-1 lookahead action values (VOI-1)."""
    out = {"stop": M.term(b)[0]}
    for j, t in enumerate(M.tools):
        if t["cost"] <= B + 1e-9:
            out[j] = t["cost"] + sum(p * M.term(M.update(b, j, o))[0] for o, p in enumerate(M.pred(b, j)) if p > 1e-12)
    return out

# ------------------------------------------------------- Lemma 3: root equivalence of MMI ----------------
n, bad = 0, {"marg": 0, "pred": 0, "post1": 0, "voi1": 0}; differs_after = []
for fam in ("RETRY", "SHARED"):
    for _ in range(100):
        sp = sim.spec(fam, rng); E = sim.build(sp); Mm = sim.mmi(sp); n += 1
        bE, bM = E.prior, Mm.prior
        bad["marg"] += np.abs(E.marg_s(bE) - Mm.marg_s(bM)).max() > TOL
        for j in range(len(sp["tools"])):
            bad["pred"] += np.abs(E.pred(bE, j) - Mm.pred(bM, j)).max() > TOL
            for o in range(E.lik[j].shape[1]):
                e1, m1 = E.update(bE, j, o), Mm.update(bM, j, o)
                bad["post1"] += np.abs(E.marg_s(e1) - Mm.marg_s(m1)).max() > TOL
                if sp["tools"][j].get("group"):                      # next call to a same-group tool
                    for j2, t2 in enumerate(sp["tools"]):
                        if t2.get("group") == sp["tools"][j]["group"]:
                            differs_after.append(np.abs(E.pred(e1, j2) - Mm.pred(m1, j2)).max())
        a, b_ = q1(E, bE, sp["budget"]), q1(Mm, bM, sp["budget"])
        bad["voi1"] += max(abs(a[k] - b_[k]) for k in a) > 1e-9
print(f"[Lemma 3] {n} RETRY/SHARED instances: mismatches at the root -- s-marginal {bad['marg']}, one-step predictive "
      f"{bad['pred']}, s-posterior after one report {bad['post1']}, VOI-1 action values {bad['voi1']}")
print(f"   after one group report, the next same-group predictive differs (max abs diff): median "
      f"{np.median(differs_after):.3f}, min {np.min(differs_after):.3f}")
res["lemma3"] = dict(n=n, mismatches=bad, after_one_diff_median=float(np.median(differs_after)),
                     after_one_diff_min=float(np.min(differs_after)))

# ------------------------------------------------------- Lemma 4: covariance identity -----------------
viol = 0; m = 0
for _ in range(400):
    sp = sim.spec("SHARED", rng); K = sp["K"]; phi = sp["groups"]["g"]
    for s in range(K):
        oh = np.eye(K)[s]
        E = sim.build(sp, prior=oh)
        Mm = sim.mmi(dict(sp, prior=oh))
        for j, j2 in ((0, 0), (0, 1), (1, 1)):
            for o in range(K):
                for o2 in range(K):
                    pe = E.pred(E.prior, j)[o] * E.pred(E.update(E.prior, j, o), j2)[o2]
                    pm = Mm.pred(Mm.prior, j)[o] * Mm.pred(Mm.update(Mm.prior, j, o), j2)[o2]
                    h1 = sim.healthy("sym", sp["tools"][j]["q"], K, s); h2 = sim.healthy("sym", sp["tools"][j2]["q"], K, s)
                    e1, e2 = sp["tools"][j]["femit"], sp["tools"][j2]["femit"]
                    pred_gap = phi * (1 - phi) * (h1[o] - e1[o]) * (h2[o2] - e2[o2])
                    m += 1; viol += abs((pe - pm) - pred_gap) > 1e-12
print(f"[Lemma 4] P_exact(o,o'|s) - P_MMI(o,o'|s) = phi(1-phi)(h(o)-e(o))(h'(o')-e'(o')): {m - viol}/{m} hold")
res["lemma4"] = dict(n=m, violations=int(viol))

# ------------------------------------------------------- Proposition 3(b): Pi-sufficiency -------------
STATES = ["healthy", "degraded", "down"]
LOSS = sim.loss_matrix(3, 10, 30, 4); fe = sim.STUCK(3)
def diag_model(phi, q, costA=1, B=4):
    tl = [dict(kind="sym", q=q, cost=costA, group="g", femit=fe), dict(kind="sym", q=0.8, cost=2)]
    return sim.Model(3, np.array([.5, .25, .25]), tl, {"g": phi}, LOSS, B)
viol = 0; m = 0
for _ in range(300):
    M = diag_model(rng.uniform(.05, .5), rng.uniform(.7, .95), costA=5)       # A unaffordable with budget 4
    ps = rng.dirichlet(np.ones(3)); x1, x2 = rng.random(3) * ps, rng.random(3) * ps
    b1 = np.ravel(np.c_[ps - x1, x1]); b2 = np.ravel(np.c_[ps - x2, x2])
    m += 1; viol += abs(M.V(b1, 4)[0] - M.V(b2, 4)[0]) > 1e-9 or M.V(b1, 4)[1] != M.V(b2, 4)[1]
print(f"[Prop 3b] only fault-free tools affordable: V and the optimal action depend on b only through Pi(b): {m - viol}/{m}")
res["prop3b"] = dict(n=m, violations=int(viol))

# ------------------------------------------------------- Examples 1-2 (constructive) ------------------
names = ["A", "V"]
fmt = lambda Q: {k.replace("term0", "declare_healthy").replace("term1", "declare_degraded").replace("term2", "declare_down")
                 .replace("term3", "escalate"): round(v, 3) for k, v in Q.items()}
h_of = lambda q: np.array([sim.healthy("sym", q, 3, s) for s in range(3)])
STN = ["healthy", "degraded", "down"]

def example(phi, q, hist, B=4, label=""):
    M = diag_model(phi, q, B=B); b = M.prior.copy()
    for o in hist: b = M.update(b, 0, o)
    ps = M.marg_s(b); x = b[1::2]                                        # x_s = b(s, stale)
    QE = qvalues(M, b, B, names)
    # 1-parameter family with the same s-marginal AND the same one-step predictive for every tool:
    # pred_A(o) = sum_s p_s h_s(o) + sum_s x_s (e(o) - h_s(o)), so x moves along the null space of D
    D = (fe[None, :] - h_of(q)).T
    nvec = np.linalg.svd(D)[2][-1]
    feas = [t for t in np.linspace(-3, 3, 6001) if np.all(x + t * nvec >= -1e-12) and np.all(x + t * nvec <= ps + 1e-12)]
    fam = []
    for t in feas[::10] + [feas[-1]]:
        xt = np.clip(x + t * nvec, 0, ps); bt = np.ravel(np.c_[ps - xt, xt])
        Q = qvalues(M, bt, B, names)
        fam.append(dict(t=float(t), p_stale=float(xt.sum()), p_stale_given_s=(xt / ps).round(3).tolist(),
                        best=fmt({min(Q, key=Q.get): 0}).popitem()[0], Q=fmt(Q),
                        pred_gap=float(max(np.abs(M.pred(bt, j) - M.pred(b, j)).max() for j in range(2)))))
    acts = sorted(set(f["best"] for f in fam))
    print(f"[{label}] phi={phi}, q={q}, A reported {[STN[o] for o in hist]}, budget {B}")
    print(f"   exact joint: P(s)={np.round(ps, 4)}, P(stale)={x.sum():.4f}, P(stale|s)={np.round(x / ps, 3)}")
    print(f"   exact Q = {fmt(QE)}")
    print(f"   same Pi AND same one-step predictive for A and V (max pred diff {max(f['pred_gap'] for f in fam):.1e}): "
          f"P(stale) {min(f['p_stale'] for f in fam):.3f}..{max(f['p_stale'] for f in fam):.3f}; optimal actions {acts}")
    sw = [f for i, f in enumerate(fam[1:]) if f["best"] != fam[i]["best"]]
    for f in [fam[0]] + sw + [fam[-1]]:
        print(f"     P(stale)={f['p_stale']:.3f} P(stale|s)={f['p_stale_given_s']} best={f['best']:12s} "
              f"Q(A)={f['Q']['query_A']:.3f} Q(V)={f['Q']['query_V']:.3f} Q(esc)={f['Q']['escalate']:.3f}")
    # Pi-matched only: stale flag independent of s, and stale only when healthy
    alt = {}
    for nm, xt in (("stale independent of s", ps * x.sum()), ("stale only if healthy", np.r_[min(x.sum(), ps[0]), 0, 0])):
        bt = np.ravel(np.c_[ps - xt, xt]); Q = qvalues(M, bt, B, names); alt[nm] = fmt(Q)
        print(f"   same Pi only, {nm}: best={min(alt[nm], key=alt[nm].get)}  Q={alt[nm]}")
    Mm = sim.mmi(dict(family="D", K=3, prior=np.array([.5, .25, .25]), tools=M.tools, groups={"g": phi}, loss=LOSS, budget=B))
    bm = Mm.prior.copy()
    for o in hist: bm = Mm.update(bm, 0, o)
    QM = qvalues(Mm, bm, B, names)
    print(f"   MMI after the same history: P(s)={np.round(Mm.marg_s(bm), 4)}, best={min(fmt(QM), key=fmt(QM).get)}")
    return dict(phi=phi, q=q, hist=hist, budget=B, ps=ps.tolist(), p_stale=float(x.sum()), p_stale_given_s=(x / ps).tolist(),
                Q_exact=fmt(QE), family_actions=acts, family=fam, pi_only=alt, mmi_ps=Mm.marg_s(bm).tolist(), mmi_Q=fmt(QM))

res["example_ok_ok_degraded"] = example(0.3, 0.9, [0, 0, 1], label="Example 1")
res["example_retry3"] = example(0.3, 0.9, [0, 0, 0], label="Example 2")
res["example_moderate_retry2"] = example(0.15, 0.85, [0, 0], label="Example 3")

# ------------------------------------------------------- pilot re-query items: where the error enters -------------
# Planner with the CORRECT current s-marginal but an independent (MMI) model of FUTURE same-source reports:
# Lemma 4 says its post-repeat posterior is too high, so re-querying A looks too valuable.
REG = {"frequent": (.3, .9), "moderate": (.15, .85), "rare": (.05, .75)}
def post_repeat(regime, hist, B=4):
    phi, q = REG[regime]; M = diag_model(phi, q, B=B); b = M.prior.copy()
    for o in hist: b = M.update(b, 0, o)
    ps = M.marg_s(b); QE = fmt(qvalues(M, b, B, names))
    Mh = sim.mmi(dict(family="D", K=3, prior=ps, tools=M.tools, groups={"g": phi}, loss=LOSS, budget=B))
    QH = fmt(qvalues(Mh, Mh.prior, B, names))
    # 'mixed': exact next-report predictive (as 5 of 7 pilot chat answers had) but hybrid posteriors and continuation
    cA = M.tools[0]["cost"]
    QX = cA + sum(p * Mh.V(Mh.update(Mh.prior, 0, o), B - cA)[0] for o, p in enumerate(M.pred(b, 0)) if p > 1e-12)
    r = dict(regime=regime, hist=[STN[o] for o in hist], p_healthy=float(ps[0]), p_stale=float(b[1::2].sum()),
             pred_healthy_exact=float(M.pred(b, 0)[0]), pred_healthy_hybrid=float(Mh.pred(Mh.prior, 0)[0]),
             post_repeat_exact=float(M.marg_s(M.update(b, 0, 0))[0]), post_repeat_hybrid=float(Mh.marg_s(Mh.update(Mh.prior, 0, 0))[0]),
             QA_exact=QE["query_A"], QA_hybrid=QH["query_A"], QA_mixed=float(QX), best_exact=min(QE, key=QE.get), best_hybrid=min(QH, key=QH.get))
    print(f"   {regime:8s} A:{'/'.join(r['hist']):24s} P(h)={r['p_healthy']:.3f} next P(healthy) exact {r['pred_healthy_exact']:.3f} "
          f"hybrid {r['pred_healthy_hybrid']:.3f} | P(h | +healthy) exact {r['post_repeat_exact']:.3f} hybrid {r['post_repeat_hybrid']:.3f} | "
          f"Q(query_A) exact {r['QA_exact']:.2f} hybrid {r['QA_hybrid']:.2f} mixed {r['QA_mixed']:.2f} | best exact {r['best_exact']}, hybrid {r['best_hybrid']}")
    return r
print("[Pilot items] correct current Pi(b) + independent model of future same-source reports ('hybrid'):")
res["pilot_requery"] = [post_repeat("frequent", [0, 0, 0]), post_repeat("frequent", [0, 0, 1]),
                        post_repeat("moderate", [0, 0]), post_repeat("rare", [0])]
json.dump(res, open("check_belief.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
