"""
Stress-test environments that violate the assumptions of Section 4 (A1-A4, static faults, benign-stuck
emissions). Everything is built on sim.Model: a GModel is a Model whose hidden-state set, prior and
likelihood tables are given explicitly, optionally with a Markov transition on the hidden state that is
applied after every call (used for mid-episode fault recovery). sim.py / exact.py are imported, not edited.
"""
import itertools, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import Model, spec, build, healthy, n_obs, source_of, STUCK            # noqa: E402
from exact import act as exact_act, dd_V, dd_Vd, with_dedup, summarize        # noqa: E402

# ------------------------------------------------------------------ general model ---------
class GModel(Model):
    """Model with explicit hidden states H (tuples whose first entry is s), prior, likelihoods and an
    optional transition matrix T (b' = b_post @ T after every call)."""
    def __init__(self, K, H, prior, lik, tools, loss, budget, trans=None):
        self.K, self.tools, self.loss, self.budget, self.temper = K, tools, np.asarray(loss), budget, 1.0
        self.H, self.nH = list(H), len(H)
        self.s_of = np.array([h[0] for h in self.H])
        self.prior = np.asarray(prior, float) / np.sum(prior)
        self.lik, self.trans, self.cache = lik, trans, {}

    def update(self, b, j, o):
        bp = Model.update(self, b, j, o)
        if bp is None or self.trans is None: return bp
        return bp @ self.trans

def from_model(m, trans=None, prior=None):
    return GModel(m.K, m.H, m.prior if prior is None else prior, m.lik, m.tools, m.loss, m.budget, trans)

def mmi_of(true):
    """Generic marginal-matched independence: per-call P_j(o|s) of the TRUE model at time 0, multiplied as
    independent across calls. Equals sim.mmi for the paper's families."""
    K = true.K; ps = true.marg_s(true.prior)
    lik = []
    for L in true.lik:
        M = np.zeros((K, L.shape[1]))
        for i, h in enumerate(true.H): M[h[0]] += true.prior[i] * L[i]
        lik.append(M / np.maximum(ps, 1e-300)[:, None])
    return GModel(K, [(s,) for s in range(K)], ps, lik, true.tools, true.loss, true.budget)

def mmi_time(true, ncalls):
    """Time-varying marginal-matched independence for environments with a fault transition: the t-th call
    uses the TRUE per-call marginal P_t(o|s), computed from the unconditional belief prior @ T^t; calls are
    multiplied as independent. Hidden state (s, t), t advanced deterministically after every call."""
    K = true.K; H = [(s, t) for t in range(ncalls + 1) for s in range(K)]
    lik = [np.zeros((len(H), L.shape[1])) for L in true.lik]
    b = true.prior.copy()
    for t in range(ncalls + 1):
        ps = np.maximum(true.marg_s(b), 1e-300)
        for j, L in enumerate(true.lik):
            M = np.zeros((K, L.shape[1]))
            for i, h in enumerate(true.H): M[h[0]] += b[i] * L[i]
            lik[j][t * K:(t + 1) * K] = M / ps[:, None]
        if true.trans is not None: b = b @ true.trans
    prior = np.r_[true.marg_s(true.prior), np.zeros(ncalls * K)]
    T = np.zeros((len(H), len(H)))
    for i, (s, t) in enumerate(H): T[i, min(t + 1, ncalls) * K + s] = 1.0
    return GModel(K, H, prior, lik, true.tools, true.loss, true.budget, trans=T)

def emit_benign(K, f):
    return np.r_[f, np.full(K - 1, (1 - f) / (K - 1))]

# ------------------------------------------------------------------ exact evaluation ------
def evaluate(true, agent, pol, src):
    """As exact.evaluate_exact, but `src` (tool -> source id) is passed explicitly and the environment may be
    any GModel (state-correlated faults, partial sharing, fault recovery, value-sharing caches).
    Returns expected total loss, severe-error prob, expected calls, escalation prob, expected repeat-source
    calls, expected probe cost."""
    memo = {}

    def rec(bt, ba, B, hist):
        used = frozenset(src[j] for j in hist)
        key = (tuple(np.round(bt, 9)), tuple(np.round(ba, 9)), round(B, 6), used)
        if key in memo: return memo[key]
        a = exact_act(pol, agent, ba, B, hist)
        if a[0] == "stop":
            ps = true.marg_s(bt); L = float(true.loss[a[1]] @ ps)
            out = np.array([L, (1 - ps[0]) if a[1] == 0 else 0.0, 0.0, 1.0 if a[1] == true.K else 0.0, 0.0, 0.0])
        else:
            j = a[1]; c = true.tools[j]["cost"]
            out = np.array([c, 0.0, 1.0, 0.0, 1.0 if src[j] in used else 0.0, c])
            skip = getattr(agent, "dedup", False) and src[j] in used
            for o, p in enumerate(true.pred(bt, j)):
                if p < 1e-12: continue
                ba2 = ba if skip else agent.update(ba, j, o)
                if ba2 is None: ba2 = agent.prior.copy()
                out = out + p * rec(true.update(bt, j, o), ba2, B - c, hist + (j,))
        memo[key] = out
        return out

    r = rec(true.prior.copy(), agent.prior.copy(), true.budget, ())
    return dict(total=r[0], severe=r[1], calls=r[2], esc=r[3], repeat=r[4], cost=r[5])

def dedup_of(m, src):
    m.dedup = True; m.srcmap = dict(src); m.ddcache = {}; return m

def summarize2(res, ref="Oracle-DP"):
    out = summarize(res, ref)
    for p in out: out[p]["cost"] = float(res[p]["cost"].mean())
    return out
