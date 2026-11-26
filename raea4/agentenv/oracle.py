"""
Exact oracle quantities for the closed-loop environment.

  qvalues(M, b, B)          exact Q of every legal action at belief b with remaining budget B
  score_trajectory(ep, log) per-step exact posterior, Q(chosen) - V (advantage), realised loss
  exact_eval(true, pol)     exact expected metrics of a baseline policy (sum over all outcome sequences)

For any (possibly stochastic) history-dependent policy, E[sum_t A_t] = E[loss] - V(b0, B) where
A_t = Q*(b_t, B_t, a_t) - V*(b_t, B_t) (performance-difference identity with the Bayes-optimal Q*). The
realised advantage sum is therefore an unbiased estimate of the episode's expected regret.

The advantage of a terminal step is split into
  stop-timing  min_a Q*(stop, a) - V*      (stopping while the oracle would still gather evidence)
  verdict      Q*(stop, a_chosen) - min_a Q*(stop, a)   (wrong verdict given the belief)
Information-gathering regret = probe-step advantages + stop-timing; commitment regret = verdict part.
"""
import numpy as np
from env import build_model, sim_spec, srcmap, LOSS, K
from sim import mmi, fault_unaware

_MODELS = {}

def true_model(ep):
    if ep["id"] not in _MODELS: _MODELS[ep["id"]] = build_model(ep["params"])
    return _MODELS[ep["id"]]

def qvalues(M, b, B):
    ps = M.marg_s(b); q = {("stop", a): float(M.loss[a] @ ps) for a in range(M.loss.shape[0])}
    for j, t in enumerate(M.tools):
        if t["cost"] > B + 1e-9: continue
        v = t["cost"]
        for o, p in enumerate(M.pred(b, j)):
            if p > 1e-12: v += p * M.V(M.update(b, j, o), B - t["cost"])[0]
        q[("probe", j)] = float(v)
    return q

def p_stale(M, b):
    return float(sum(w for w, h in zip(b, M.H) if (h[1] if isinstance(h[1], int) else any(h[1]))))

def score_trajectory(ep, log, truth):
    """log: Env.log. An episode that ends without a valid submit is flagged forced=True and scored, for the loss
    columns, as the WORST verdict at the final belief (a non-compliance penalty, not a cheap escalation)."""
    M = true_model(ep); b = M.prior.copy(); B = float(M.budget); steps = []; cost = 0.0
    src = srcmap(ep["params"]); used = {}
    final = None
    def stop_rec(q, V, a):
        qs = min(v for k, v in q.items() if k[0] == "stop")
        return dict(action=["stop", a], Q=q[("stop", a)], adv=q[("stop", a)] - V, adv_stoptime=qs - V,
                    adv_verdict=q[("stop", a)] - qs)
    for e in log:
        if e["type"] == "invalid":
            continue
        q = qvalues(M, b, B); V = min(q.values()); best = min(q, key=q.get)
        rec = dict(post=[round(float(x), 6) for x in M.marg_s(b)], p_stale=round(p_stale(M, b), 6), budget=B, V=V,
                   oracle=list(best))
        if e["type"] == "call":
            j, o = e["j"], e["o"]; a = ("probe", j)
            rec.update(action=["probe", j], Q=q[a], adv=q[a] - V,
                       repeat_src=used.get(src[j], 0) > 0, src=src[j])
            used[src[j]] = used.get(src[j], 0) + 1
            b = M.update(b, j, o); B -= M.tools[j]["cost"]; cost += M.tools[j]["cost"]
            steps.append(rec)
        else:
            rec.update(stop_rec(q, V, e["a"])); steps.append(rec)
            final = e["a"]; break
    forced = final is None
    if forced:
        q = qvalues(M, b, B); V = min(q.values())
        final = max(range(M.loss.shape[0]), key=lambda a: q[("stop", a)])
        rec = dict(post=[round(float(x), 6) for x in M.marg_s(b)], p_stale=round(p_stale(M, b), 6), budget=B, V=V,
                   oracle=list(min(q, key=q.get)), forced=True)
        rec.update(stop_rec(q, V, final)); steps.append(rec)
    s = truth["s"]
    return dict(steps=steps, verdict=final, forced=forced, cost=cost, dloss=float(LOSS[final, s]),
                realised=float(LOSS[final, s] + cost), regret=float(sum(st["adv"] for st in steps)),
                V0=float(M.V(M.prior.copy(), float(M.budget))[0]),
                severe=bool(final == 0 and s != 0), esc=bool(final == K), correct=bool(final == s),
                used_cache="c" in used)

# ------------------------------------------------------------------ baseline policies ---------
def V_rules(M, b, B, used, forbid, discard, memo):
    """Optimal value under source rules. forbid: set of source names that may not be queried twice (None = any
    source). discard: sources whose repeat observations are dropped from the belief; the planner KNOWS this, so
    a repeat call leaves its belief unchanged (coherent discard)."""
    k = (tuple(np.round(b, 9)), round(B, 6), used)
    if k in memo: return memo[k]
    best, a = M.term(b); out = (best, ("stop", a))
    for j, t in enumerate(M.tools):
        s = M.srcmap[j]; rep = s in used
        if t["cost"] > B + 1e-9 or (rep and _in(s, forbid)): continue
        q = t["cost"]
        if rep and _in(s, discard):
            q += V_rules(M, b, B - t["cost"], used, forbid, discard, memo)[0]
        else:
            for o, p in enumerate(M.pred(b, j)):
                if p > 1e-12: q += p * V_rules(M, M.update(b, j, o), B - t["cost"], used | {s}, forbid, discard, memo)[0]
        if q < out[0] - 1e-9: out = (q, ("probe", j))
    memo[k] = out; return out

def _in(s, scope):
    return scope is not None and (scope == "any" or s in scope)

class BayesPol:
    """Belief-based planner on model M.
    forbid:  None | "any" | {sources}: never call a used source of that scope again.
    discard: None | "any" | {sources}: drop repeat observations of that scope from the belief.
    aware:   the planner knows its discard rule (coherent). aware=False plans with M.V, which assumes every
             observation updates the belief (the planner does not know it will discard)."""
    def __init__(self, M, sm, plan="dp", forbid=None, discard=None, aware=True):
        self.M, self.plan, self.forbid, self.discard, self.aware = M, plan, forbid, discard, aware
        M.srcmap = sm; self.memo = {}
    def init(self): return (self.M.prior.copy(), frozenset())
    def key(self, st): return (tuple(np.round(st[0], 9)), st[1])
    def act(self, st, B, n):
        b, used = st; M = self.M
        if self.plan == "voi1":
            return voi1(M, b, B, used, self.forbid)
        if self.forbid is not None or (self.discard is not None and self.aware):
            return V_rules(M, b, B, used, self.forbid, self.discard if self.aware else None, self.memo)[1]
        return M.V(b, B)[1]
    def update(self, st, j, o):
        b, used = st; s = self.M.srcmap[j]
        if not (s in used and _in(s, self.discard)):
            b2 = self.M.update(b, j, o); b = b2 if b2 is not None else self.M.prior.copy()
        return (b, used | {s})

def voi1(M, b, B, used, forbid=None):
    best, a = M.term(b); out = (best, ("stop", a))
    for j, t in enumerate(M.tools):
        if t["cost"] > B + 1e-9 or (M.srcmap[j] in used and _in(M.srcmap[j], forbid)): continue
        q = t["cost"] + sum(p * M.term(M.update(b, j, o))[0] for o, p in enumerate(M.pred(b, j)) if p > 1e-12)
        if q < out[0] - 1e-9: out = (q, ("probe", j))
    return out[1]

class FixedPol:
    """Call `tool` k times (while affordable), then take the Bayes action of model M."""
    def __init__(self, M, tool, k): self.M, self.tool, self.k = M, tool, k
    def init(self): return self.M.prior.copy()
    def key(self, st): return tuple(np.round(st, 9))
    def act(self, b, B, n):
        if n < self.k and self.M.tools[self.tool]["cost"] <= B + 1e-9: return ("probe", self.tool)
        return ("stop", self.M.term(b)[1])
    def update(self, b, j, o): return self.M.update(b, j, o)

FIELDS = ["total", "severe", "esc", "calls", "cost", "rep_cache", "rep_log", "used_cache", "probe_used", "dloss"]

def exact_eval(true, pol, sm):
    """Exact expectation over hidden state and all outcome sequences.
    rep_cache: repeat calls to the cache source; rep_log: repeat calls to the log source (tool 2);
    used_cache: P(at least one cache-tool call); probe_used: P(live probe called); dloss: verdict loss."""
    memo = {}
    def rec(bt, st, B, used, n):
        key = (tuple(np.round(bt, 9)), pol.key(st), round(B, 6), used, n)
        if key in memo: return memo[key]
        a = pol.act(st, B, n)
        if a[0] == "stop":
            ps = true.marg_s(bt); dl = true.loss[a[1]] @ ps
            out = np.array([dl, (1 - ps[0]) if a[1] == 0 else 0.0, float(a[1] == K), 0, 0, 0, 0,
                            float("c" in used), float("tool3" in used), dl])
        else:
            j = a[1]; c = true.tools[j]["cost"]; s = sm[j]
            out = np.array([c, 0, 0, 1, c, float(s == "c" and s in used), float(s == "tool2" and s in used), 0, 0, 0])
            for o, p in enumerate(true.pred(bt, j)):
                if p < 1e-12: continue
                out = out + p * rec(true.update(bt, j, o), pol.update(st, j, o), B - c, used | {s}, n + 1)
        memo[key] = out; return out
    r = rec(true.prior.copy(), pol.init(), float(true.budget), frozenset(), 0)
    return dict(zip(FIELDS, map(float, r)))

BASELINES = ["Oracle-DP", "Oracle-VOI1", "FaultUnaware-DP", "MMI-DP", "MMI-VOI1",
             "Oracle+ForbidAny-DP", "Oracle+ForbidCache-DP", "MMI+ForbidAny-DP", "MMI+ForbidCache-DP",
             "Oracle+DiscardCache-aware-DP", "Oracle+DiscardCache-unaware-DP", "MMI+DiscardCache-unaware-DP",
             "Oracle+Dedup-DP", "Fixed2-A", "Fixed3-A", "Probe-once", "Prior-only"]

def baseline_policies(par):
    sp = sim_spec(par); sm = srcmap(par); C = {"c"}
    T = lambda: build_model(par)
    return {
        "Oracle-DP": BayesPol(T(), sm), "Oracle-VOI1": BayesPol(T(), sm, plan="voi1"),
        "FaultUnaware-DP": BayesPol(fault_unaware(sp), sm), "MMI-DP": BayesPol(mmi(sp), sm),
        "MMI-VOI1": BayesPol(mmi(sp), sm, plan="voi1"),
        "Oracle+ForbidAny-DP": BayesPol(T(), sm, forbid="any"),
        "Oracle+ForbidCache-DP": BayesPol(T(), sm, forbid=C),
        "MMI+ForbidAny-DP": BayesPol(mmi(sp), sm, forbid="any"),
        "MMI+ForbidCache-DP": BayesPol(mmi(sp), sm, forbid=C),
        "Oracle+DiscardCache-aware-DP": BayesPol(T(), sm, discard=C, aware=True),
        "Oracle+DiscardCache-unaware-DP": BayesPol(T(), sm, discard=C, aware=False),
        "MMI+DiscardCache-unaware-DP": BayesPol(mmi(sp), sm, discard=C, aware=False),
        "Oracle+Dedup-DP": BayesPol(T(), sm, forbid="any", discard="any"),
        "Fixed2-A": FixedPol(T(), 0, 2), "Fixed3-A": FixedPol(T(), 0, 3),
        "Probe-once": FixedPol(T(), 3, 1), "Prior-only": FixedPol(T(), 0, 0)}
