"""
Validation pipeline (round 4).

Fixes relative to the earlier runs:
  * Exact expected-loss evaluation of every policy (no episode sampling).
  * Separate random streams per instance: one for the task instance, one for calibration data.
    Changing the calibration size never changes which instances are drawn.
  * One raw calibration log per instance, shared by every assumed structure; mapping errors are
    derived from that same raw log; calibration-size results use prefixes of one long log.
  * Dedup is a wrapper that can be put on ANY model (oracle model, fault-unaware model, fitted model).
  * Per-instance paired regret with cluster-bootstrap 95% intervals over instances.
"""
import numpy as np
from sim import (Model, spec, build, fit, mmi, fault_unaware, split_true, llm_sim, true_groups, source_of,
                 target, n_obs)

# ------------------------------------------------------------------ dedup wrapper ----------
def with_dedup(m, sp):
    """Only the first observation from each TRUE source is used; repeat-source probes are never planned."""
    m.dedup = True
    m.srcmap = {j: source_of(t, j) for j, t in enumerate(sp["tools"])}
    m.ddcache = {}
    return m

def dd_V(M, b, B, used):
    k = (tuple(np.round(b, 9)), round(B, 6), used)
    if k in M.ddcache: return M.ddcache[k]
    best, a = M.term(b); out = (best, ("stop", a))
    for j, t in enumerate(M.tools):
        if t["cost"] > B + 1e-9 or M.srcmap[j] in used: continue
        q = t["cost"]
        for o, p in enumerate(M.pred(b, j)):
            if p > 1e-12: q += p * dd_V(M, M.update(b, j, o), B - t["cost"], used | {M.srcmap[j]})[0]
        if q < out[0] - 1e-9: out = (q, ("probe", j))
    M.ddcache[k] = out; return out

def dd_Vd(M, b, B, d, used):
    best, a = M.term(b); out = (best, ("stop", a))
    if d == 0: return out
    for j, t in enumerate(M.tools):
        if t["cost"] > B + 1e-9 or M.srcmap[j] in used: continue
        q = t["cost"]
        for o, p in enumerate(M.pred(b, j)):
            if p > 1e-12: q += p * dd_Vd(M, M.update(b, j, o), B - t["cost"], d - 1, used | {M.srcmap[j]})[0]
        if q < out[0] - 1e-9: out = (q, ("probe", j))
    return out

def act(pol, M, b, B, hist):
    k = pol[0]
    if getattr(M, "dedup", False):
        used = frozenset(M.srcmap[j] for j in hist)
        if k == "dp": return dd_V(M, b, B, used)[1]
        if k == "look": return dd_Vd(M, b, B, pol[1], used)[1]
    if k == "dp": return M.V(b, B)[1]
    if k == "look": return M.Vd(b, B, pol[1])[1]
    if k == "fixed":
        if len(hist) < pol[1] and M.tools[0]["cost"] <= B + 1e-9: return ("probe", 0)
        return ("stop", M.term(b)[1])
    if k == "thr":
        ps = M.marg_s(b)
        if ps.max() >= pol[1]: return ("stop", int(np.argmax(ps)))
        c = [(M.info_gain_s(b, j) / t["cost"], j) for j, t in enumerate(M.tools) if t["cost"] <= B + 1e-9]
        return ("probe", max(c)[1]) if c else ("stop", M.term(b)[1])
    if k == "noprobe": return ("stop", M.term(b)[1])
    raise ValueError(pol)

# ------------------------------------------------------------------ exact evaluation --------
def evaluate_exact(true, agent, pol, sp):
    """Expected total loss, severe-error probability, expected calls, escalation probability and expected
    repeat-source calls of policy `pol` driven by `agent`, in environment `true`. Exact: sums over all
    observation sequences with their true probabilities (and over mapping noise, if the agent has it)."""
    eps = getattr(agent, "map_eps", 0.0)
    src = {j: source_of(t, j) for j, t in enumerate(sp["tools"])}
    memo = {}

    def rec(bt, ba, B, hist):
        used = frozenset(src[j] for j in hist)
        key = (tuple(np.round(bt, 9)), tuple(np.round(ba, 9)), round(B, 6), used,
               len(hist) if pol[0] == "fixed" else 0)
        if key in memo: return memo[key]
        a = act(pol, agent, ba, B, hist)
        if a[0] == "stop":
            ps = true.marg_s(bt); L = float(true.loss[a[1]] @ ps)
            out = np.array([L, (1 - ps[0]) if a[1] == 0 else 0.0, 0.0, 1.0 if a[1] == true.K else 0.0, 0.0])
        else:
            j = a[1]; c = true.tools[j]["cost"]; m = true.lik[j].shape[1]
            repeat = 1.0 if src[j] in used else 0.0
            out = np.array([c, 0.0, 1.0, 0.0, repeat])
            skip = getattr(agent, "dedup", False) and src[j] in used
            for o, p in enumerate(true.pred(bt, j)):
                if p < 1e-12: continue
                bt2 = true.update(bt, j, o)
                chans = [(o, 1.0)] if eps == 0 else [(o, 1 - eps)] + [(x, eps / (m - 1)) for x in range(m) if x != o]
                for oa, w in chans:
                    if skip:
                        ba2 = ba
                    else:
                        ba2 = agent.update(ba, j, oa)
                        if ba2 is None: ba2 = agent.prior.copy()
                    out = out + p * w * rec(bt2, ba2, B - c, hist + (j,))
        memo[key] = out
        return out

    r = rec(true.prior.copy(), agent.prior.copy(), true.budget, ())
    return dict(total=r[0], severe=r[1], calls=r[2], esc=r[3], repeat=r[4])

# ------------------------------------------------------------------ calibration -------------
def raw_calibration(true, sp, n_max, rng, calls=3):
    """One raw labelled log per instance: n_max episodes, `calls` calls to every tool."""
    S, O = [], []
    for _ in range(n_max):
        i = rng.choice(true.nH, p=true.prior); S.append(true.H[i][0]); row = []
        for j in range(len(sp["tools"])):
            m = true.lik[j].shape[1]
            row += [int(rng.choice(m, p=true.lik[j][i])) for _ in range(calls)]
        O.append(row)
    return np.array(S), np.array(O)

def corrupt(sp, O, eps, rng, calls=3):
    """Mapping errors derived from the SAME raw log: each output replaced by a wrong symbol w.p. eps."""
    O2 = O.copy()
    for j, t in enumerate(sp["tools"]):
        m = n_obs(t["kind"], sp["K"])
        for c in range(j * calls, (j + 1) * calls):
            flip = rng.random(len(O)) < eps
            shift = rng.integers(1, m, size=len(O))
            O2[flip, c] = (O2[flip, c] + shift[flip]) % m
    return O2

def fitted(sp, S, O, structure="correct", omit=None, eps=0.0):
    ag = true_groups(sp)
    if structure == "split_shared": ag = {j: (f"{g}{j}" if g else None) for j, g in ag.items()}
    if structure == "no_faults": ag = {j: None for j in ag}
    tools, groups, prior = fit(sp, S, O, ag, emission="uniform" if structure == "wrong_mechanism" else "stuck")
    m = Model(sp["K"], prior, tools, groups, sp["loss"], sp["budget"], omit=omit)
    m.map_eps = eps
    return m

# ------------------------------------------------------------------ runner ------------------
def run(family, models, policies, n_inst, seed, spec_kw=None, n_cal_max=0, shift=None):
    """models: name -> builder(ctx) where ctx has sp, sp_test, true, true_cal, S, O, Omap, omit, rng_cal.
    policies: name -> (pol, model_name). Returns per-instance arrays of metrics."""
    ss = np.random.SeedSequence(seed)
    out = {p: [] for p in policies}
    for child in ss.spawn(n_inst):
        s_inst, s_cal, s_misc = child.spawn(3)
        r_inst, r_cal, r_misc = (np.random.default_rng(x) for x in (s_inst, s_cal, s_misc))
        sp = spec(family, r_inst, **(spec_kw or {}))
        sp_test = shift(sp, r_inst) if shift else sp
        true_cal = build(sp); true = build(sp_test) if shift else true_cal
        ctx = dict(sp=sp, sp_test=sp_test, true=true, true_cal=true_cal, rng=r_misc)
        if n_cal_max:
            S, O = raw_calibration(true_cal, sp, n_cal_max, r_cal)
            ctx.update(S=S, O=O, Omap=corrupt(sp, O, 0.1, r_misc),
                       omit=int(r_misc.integers(1, sp["K"])) if r_misc.random() < 0.1 else None)
            if shift:
                S2, O2 = raw_calibration(true, sp_test, n_cal_max, r_cal)
                ctx.update(S_post=S2, O_post=O2)
        built = {}
        for p, (pol, mn) in policies.items():
            if mn not in built: built[mn] = models[mn](ctx)
            out[p].append(evaluate_exact(true, built[mn], pol, sp_test))
    return {p: {k: np.array([r[k] for r in rs]) for k in rs[0]} for p, rs in out.items()}

def summarize(res, ref="Oracle-DP", n_boot=2000, seed=0):
    """Mean of each metric, and per-instance paired regret vs `ref` with a cluster-bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    base = res[ref]["total"]; n = len(base)
    idx = rng.integers(0, n, size=(n_boot, n))
    out = {}
    for p, r in res.items():
        d = r["total"] - base
        boots = d[idx].mean(1)
        out[p] = dict(total=float(r["total"].mean()), regret=float(d.mean()),
                      lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)),
                      severe=float(r["severe"].mean()), calls=float(r["calls"].mean()), esc=float(r["esc"].mean()),
                      repeat=float(r["repeat"].mean()), n=n)
    return out
