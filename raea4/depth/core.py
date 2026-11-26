"""
Shared machinery for the depth / baseline / cost experiments (D1-D5).

Extends ../sim.py and ../exact.py without editing them:
  * one memoised planner Q() for k-step lookahead (VOI-k) and budgeted DP, with an optional
    "forbid repeat-source probes" constraint carried through the planner's own recursion;
  * one exact evaluator that separates the two operations the draft calls "dedup":
      discard = the agent's belief ignores any observation from a source already observed
                (naive: its planner still plans with the normal model; consistent: plan_discard=True,
                the planner plans with the discard dynamics, so a repeat is worth nothing to it),
      forbid  = the planner never plans, and the agent never executes, a probe of a source already
                observed in this task;
    it also supports an inherited prefix history (forced calls, free of charge) and reports cost,
    decision loss, severe / ordinary-error / escalation probabilities and repeat-source calls;
  * a data-driven independent model (per-call emission table fitted from the labelled log);
  * an instance builder that reproduces exact.run()'s random streams, and a 4-process runner.
"""
import os, sys
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from sim import Model, spec, build, mmi, fault_unaware, split_true, source_of, n_obs, target  # noqa: E402
from exact import raw_calibration, fitted  # noqa: E402

WORKERS = 4

# ------------------------------------------------------------------ planner -------------------
def Q(M, src, b, B, d, used, forbid, discard=False):
    """(value, action) of depth-d lookahead (d=None: budgeted DP) under model M.
    forbid: repeat-source probes are excluded from the plan (used = sources already observed).
    discard: the planner knows its belief will ignore repeat-source reports (consistent discard), so a
    repeat probe leaves the belief unchanged and only costs."""
    cache = M.__dict__.setdefault("qcache", {})
    track = forbid or discard
    key = (tuple(np.round(b, 9)), round(B, 6), d, used if track else None, forbid, discard)
    if key in cache: return cache[key]
    best, a = M.term(b); out = (best, ("stop", a))
    if d != 0:
        nd = None if d is None else d - 1
        for j, t in enumerate(M.tools):
            if t["cost"] > B + 1e-9 or (forbid and src[j] in used): continue
            v = t["cost"]; nu = used | {src[j]} if track else used
            if discard and src[j] in used:
                v += Q(M, src, b, B - t["cost"], nd, nu, forbid, discard)[0]
            else:
                for o, p in enumerate(M.pred(b, j)):
                    if p > 1e-12: v += p * Q(M, src, M.update(b, j, o), B - t["cost"], nd, nu, forbid, discard)[0]
            if v < out[0] - 1e-9: out = (v, ("probe", j))
    cache[key] = out
    return out

def act(pol, M, src, b, B, hist, forbid, discard=False):
    k = pol[0]; used = frozenset(src[j] for j in hist)
    if k == "dp": return Q(M, src, b, B, None, used, forbid, discard)[1]
    if k == "look": return Q(M, src, b, B, pol[1], used, forbid, discard)[1]
    if k == "fixed":
        if len(hist) < pol[1] and M.tools[0]["cost"] <= B + 1e-9: return ("probe", 0)
        return ("stop", M.term(b)[1])
    if k == "thr":
        ps = M.marg_s(b)
        if ps.max() >= pol[1]: return ("stop", int(np.argmax(ps)))
        c = [(M.info_gain_s(b, j) / t["cost"], j) for j, t in enumerate(M.tools)
             if t["cost"] <= B + 1e-9 and not (forbid and src[j] in used)]
        return ("probe", max(c)[1]) if c else ("stop", M.term(b)[1])
    raise ValueError(pol)

# ------------------------------------------------------------------ evaluator -----------------
KEYS = ["total", "cost", "dloss", "severe", "wrong", "esc", "calls", "repeat"]

def tool_sources(sp):
    """Source map keyed on tool identity only (no knowledge of shared backends)."""
    return {j: f"tool{j}" for j in range(len(sp["tools"]))}

def pooled(agent, obs, asrc):
    """'Keep all reports, group by source': each source contributes the geometric mean of the
    likelihoods of its reports (one report's worth of weight), sources multiplied as independent."""
    lb = np.log(np.maximum(agent.prior, 1e-300)); by = {}
    for j, o in obs: by.setdefault(asrc[j], []).append(np.log(np.maximum(agent.lik[j][:, o], 1e-300)))
    for v in by.values(): lb = lb + np.mean(v, axis=0)
    b = np.exp(lb - lb.max()); return b / b.sum()

def evaluate(true, agent, pol, sp, discard=False, forbid=False, prefix=(), plan_discard=False, pool=False, asrc=None):
    """Exact expectations of KEYS for policy `pol` driven by `agent` in environment `true`.
    prefix: tool indices called before the agent takes over (inherited history; no cost, no budget).
    discard: belief ignores repeat-source reports; plan_discard: the planner also plans with those
    dynamics (consistent discard; without it the planner assumes repeats update the belief = naive).
    pool: belief keeps all reports but pools them per source (see pooled()).
    asrc: the agent's source map for discard/forbid/pool (default: the true grouping, incl. hidden
    shared backends); `repeat` is always counted with the true grouping."""
    src = {j: source_of(t, j) for j, t in enumerate(sp["tools"])}
    asrc = asrc or src
    memo = {}

    def rec(bt, ba, B, hist, npre, obs):
        used = frozenset(src[j] for j in hist); aused = frozenset(asrc[j] for j in hist)
        key = (tuple(np.round(bt, 9)), tuple(np.round(ba, 9)), round(B, 6), used, aused, npre,
               len(hist) if pol[0] == "fixed" else 0, obs if pool else None)
        if key in memo: return memo[key]
        if npre < len(prefix):
            j, c, forced = prefix[npre], 0.0, True
        else:
            a = act(pol, agent, asrc, ba, B, hist, forbid, discard and plan_discard); forced = False
            if a[0] == "stop":
                ps = true.marg_s(bt); L = float(true.loss[a[1]] @ ps); K = true.K
                r = np.zeros(len(KEYS))
                r[0] = r[2] = L
                if a[1] == 0: r[3] = 1 - ps[0]
                elif a[1] < K: r[4] = 1 - ps[a[1]]
                else: r[5] = 1.0
                memo[key] = r; return r
            j = a[1]; c = true.tools[j]["cost"]
        out = np.zeros(len(KEYS))
        if not forced:
            out[0] = out[1] = c; out[6] = 1.0; out[7] = 1.0 if src[j] in used else 0.0
        skip = discard and asrc[j] in aused
        for o, p in enumerate(true.pred(bt, j)):
            if p < 1e-12: continue
            if pool: ba2 = pooled(agent, obs + ((j, o),), asrc)
            else: ba2 = ba if skip else agent.update(ba, j, o)
            if ba2 is None: ba2 = agent.prior.copy()
            out = out + p * rec(true.update(bt, j, o), ba2, B - c, hist + (j,), npre + forced, obs + ((j, o),))
        memo[key] = out
        return out

    r = rec(true.prior.copy(), agent.prior.copy(), true.budget, (), 0, ())
    return dict(zip(KEYS, r))

# ------------------------------------------------------------------ repeat over-valuation -----
def gross(X, b, j):
    """One-step gross value of probing j at belief b under model X (stop loss now - expected stop loss after)."""
    return X.term(b)[0] - sum(p * X.term(X.update(b, j, o))[0] for o, p in enumerate(X.pred(b, j)) if p > 1e-12)

def overvalue_max(M, true, sp):
    """Max over every repeat-source pair (j then k, same true source, incl. cross-tool j != k) of the
    model-minus-truth gross value of probing k after one benign report from j, from the prior."""
    src = {j: source_of(t, j) for j, t in enumerate(sp["tools"])}; best = -np.inf
    for j in src:
        for k in src:
            if src[j] != src[k]: continue
            bm, bt = M.update(M.prior, j, 0), true.update(true.prior, j, 0)
            best = max(best, gross(M, bm, k) - gross(true, bt, k))
    return float(best)

def overvalue_plan(true, M, pol, sp):
    """Expected (under the truth) sum, over the repeat-source probes M's policy actually executes, of the
    model-minus-truth gross value of that probe at the beliefs reached (M's belief vs the true posterior)."""
    src = {j: source_of(t, j) for j, t in enumerate(sp["tools"])}; memo = {}
    def rec(bt, ba, B, hist):
        used = frozenset(src[j] for j in hist)
        key = (tuple(np.round(bt, 9)), tuple(np.round(ba, 9)), round(B, 6), used)
        if key in memo: return memo[key]
        a = act(pol, M, src, ba, B, hist, False)
        if a[0] == "stop": memo[key] = 0.0; return 0.0
        j = a[1]; out = (gross(M, ba, j) - gross(true, bt, j)) if src[j] in used else 0.0
        for o, p in enumerate(true.pred(bt, j)):
            if p < 1e-12: continue
            ba2 = M.update(ba, j, o)
            out += p * rec(true.update(bt, j, o), M.prior.copy() if ba2 is None else ba2, B - true.tools[j]["cost"], hist + (j,))
        memo[key] = out; return out
    return float(rec(true.prior.copy(), M.prior.copy(), true.budget, ()))

# ------------------------------------------------------------------ models --------------------
def indep_fit(sp, S, O, calls=3, alpha=0.5):
    """Data-driven independent model: per-call P(o | s) table per tool from the labelled log, calls
    multiplied as independent, no fault flags (a fitted analogue of MMI)."""
    K = sp["K"]
    tools = [dict(t, group=None, femit=None) for t in sp["tools"]]
    m = Model(K, (np.bincount(S, minlength=K) + 1.0) / (len(S) + K), tools, {}, sp["loss"], sp["budget"])
    for j, t in enumerate(sp["tools"]):
        E = np.full((K, n_obs(t["kind"], K)), alpha)
        o = O[:, j * calls:(j + 1) * calls]
        for c in range(calls): np.add.at(E, (S, o[:, c]), 1.0)
        m.lik[j] = E / E.sum(1, keepdims=True)
    return m

def instance(family, seed, idx, n_inst, n_cal=0, spec_kw=None, budget=None):
    """Same random streams as exact.run(): instance idx of SeedSequence(seed).spawn(n_inst)."""
    child = np.random.SeedSequence(seed).spawn(n_inst)[idx]
    s_inst, s_cal, s_misc = child.spawn(3)
    r_inst, r_cal = np.random.default_rng(s_inst), np.random.default_rng(s_cal)
    sp = spec(family, r_inst, **(spec_kw or {}))
    if budget is not None: sp["budget"] = budget
    true = build(sp)
    S = O = None
    if n_cal: S, O = raw_calibration(true, sp, n_cal, r_cal)
    return sp, true, S, O

def model(name, sp, S=None, O=None):
    if name == "true": return build(sp)
    if name == "mmi": return mmi(sp)
    if name == "fault_unaware": return fault_unaware(sp)
    if name == "split_true": return split_true(sp)
    if name.startswith("fit_wrongmech"): return fitted(sp, S[:_n(name)], O[:_n(name)], "wrong_mechanism")
    if name.startswith("fit_indep_a"):                        # e.g. fit_indep_a0.001@30: smoothing alpha
        return indep_fit(sp, S[:_n(name)], O[:_n(name)], alpha=float(name.split("@")[0][len("fit_indep_a"):]))
    if name.startswith("fit_indep"): return indep_fit(sp, S[:_n(name)], O[:_n(name)])
    if name.startswith("fit_nofault"): return fitted(sp, S[:_n(name)], O[:_n(name)], "no_faults")
    if name.startswith("fit"): return fitted(sp, S[:_n(name)], O[:_n(name)])
    raise ValueError(name)

def _n(name):
    return int(name.rsplit("@", 1)[1]) if "@" in name else 100

# ------------------------------------------------------------------ runner / stats ------------
def pmap(fn, jobs):
    with Pool(WORKERS) as p:
        return p.map(fn, jobs, chunksize=1)

def stack(rows):
    """rows: list (instances) of dict policy -> dict metric -> float. -> policy -> metric -> array."""
    return {p: {k: np.array([r[p][k] for r in rows]) for k in rows[0][p]} for p in rows[0]}

def boot(d, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bs = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def summarize(res, ref="Oracle-DP"):
    """Mean of every metric plus paired regret vs ref with a cluster-bootstrap 95% CI over instances."""
    base = res[ref]["total"]; out = {}
    for p, r in res.items():
        m, lo, hi = boot(r["total"] - base)
        out[p] = {k: float(v.mean()) for k, v in r.items()}
        out[p].update(regret=m, lo=lo, hi=hi, n=len(base))
    return out
