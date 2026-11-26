"""
Budgeted evidence acquisition with persistent, possibly shared, tool faults.

Hidden state h = (s, z): s in {0..K-1} is the task-relevant world state (s = 0 is "benign"),
z[g] in {0,1} is a persistent fault flag for fault group g. A fault group is a set of tools that
share a failure source: one tool retried (same-tool retries) or several tools on one backend.
Assumptions (used by Proposition 2): z independent of s; the fault emission distribution of a tool
does not depend on s; outputs are i.i.d. given the full hidden state h.
"""
import itertools
import numpy as np

STUCK = lambda K: np.r_[0.97, np.full(K - 1, 0.03 / (K - 1))]   # faulty tool reports "benign"

def target(kind, s):
    return s if kind == "sym" else (s >> int(kind[3:])) & 1

def n_obs(kind, K):
    return K if kind == "sym" else 2

def healthy(kind, q, K, s):
    m = n_obs(kind, K); d = np.full(m, (1 - q) / (m - 1)); d[target(kind, s)] = q; return d

# ------------------------------------------------------------------ model ----
class Model:
    def __init__(self, K, prior_s, tools, groups, loss, budget, temper=1.0, omit=None):
        """tools: list of dict(kind,q,cost,group,femit); groups: dict g -> phi."""
        self.K, self.tools, self.loss, self.budget, self.temper = K, tools, np.asarray(loss), budget, temper
        self.G = sorted(g for g, phi in groups.items() if phi > 0)
        zs = list(itertools.product([0, 1], repeat=len(self.G)))
        self.H = [(s, z) for s in range(K) for z in zs]
        self.nH = len(self.H)
        self.s_of = np.array([h[0] for h in self.H])
        pr = np.array([prior_s[s] * np.prod([groups[g] if z[k] else 1 - groups[g] for k, g in enumerate(self.G)])
                       for s, z in self.H])
        if omit is not None:
            pr[self.s_of == omit] = 0
        self.prior = pr / pr.sum()
        self.lik = []
        for t in tools:
            L = np.zeros((self.nH, n_obs(t["kind"], K)))
            for i, (s, z) in enumerate(self.H):
                g = t.get("group")
                if t.get("marg_phi") is not None:     # marginal-matched: per-call mixture, no persistent flag
                    mp = t["marg_phi"]; L[i] = (1 - mp) * healthy(t["kind"], t["q"], K, s) + mp * t["femit"]
                elif g in self.G and z[self.G.index(g)]:
                    L[i] = t["femit"]
                else:
                    L[i] = healthy(t["kind"], t["q"], K, s)
            self.lik.append(L)
        self.cache = {}

    def marg_s(self, b):
        return np.bincount(self.s_of, weights=b, minlength=self.K)

    def term(self, b):
        el = self.loss @ self.marg_s(b); a = int(np.argmin(el)); return el[a], a

    def update(self, b, j, o):
        bp = b * self.lik[j][:, o]; z = bp.sum()
        if z <= 0: return None
        bp = bp / z
        if self.temper != 1.0:
            ps = self.marg_s(bp); pt = ps ** self.temper; pt /= pt.sum()
            sc = np.where(ps[self.s_of] > 0, pt[self.s_of] / np.maximum(ps[self.s_of], 1e-300), 0)
            bp = bp * sc; bp /= bp.sum()
        return bp

    def pred(self, b, j):
        return b @ self.lik[j]

    def V(self, b, B):
        k = (tuple(np.round(b, 9)), round(B, 6))
        if k in self.cache: return self.cache[k]
        best, a = self.term(b); out = (best, ("stop", a))
        for j, t in enumerate(self.tools):
            if t["cost"] > B + 1e-9: continue
            q = t["cost"]
            for o, p in enumerate(self.pred(b, j)):
                if p > 1e-12: q += p * self.V(self.update(b, j, o), B - t["cost"])[0]
            if q < out[0] - 1e-9: out = (q, ("probe", j))
        self.cache[k] = out; return out

    def Vd(self, b, B, d):
        best, a = self.term(b); out = (best, ("stop", a))
        if d == 0: return out
        for j, t in enumerate(self.tools):
            if t["cost"] > B + 1e-9: continue
            q = t["cost"]
            for o, p in enumerate(self.pred(b, j)):
                if p > 1e-12: q += p * self.Vd(self.update(b, j, o), B - t["cost"], d - 1)[0]
            if q < out[0] - 1e-9: out = (q, ("probe", j))
        return out

    def info_gain_s(self, b, j):
        ps = self.marg_s(b); H0 = -np.sum(ps[ps > 0] * np.log2(ps[ps > 0])); ig = 0.0
        for o, p in enumerate(self.pred(b, j)):
            if p < 1e-12: continue
            bp = b * self.lik[j][:, o]; q = self.marg_s(bp / bp.sum())
            ig += p * (H0 + np.sum(q[q > 0] * np.log2(q[q > 0])))
        return ig

# --------------------------------------------------------------- families ----
def loss_matrix(K, err, sev, esc):
    L = np.full((K + 1, K), float(err)); L[np.arange(K), np.arange(K)] = 0; L[0, 1:] = sev; L[K] = esc
    return L

def spec(family, rng, sev_mult=3.0, phi=None, esc=None, cost_mult=1.0):
    K = 3; err = 10.0; esc_ = 4.0; B = 6; groups = {}
    if family == "IND":
        prior = rng.dirichlet([3, 2, 2])
        tools = [dict(kind="sym", q=rng.uniform(.6, .8), cost=1), dict(kind="sym", q=rng.uniform(.75, .92), cost=2)]
    elif family == "RETRY":                      # one cheap tool whose retries share a persistent fault
        prior = rng.dirichlet([4, 2, 2])
        groups = {"a": rng.uniform(.2, .4) if phi is None else phi}
        tools = [dict(kind="sym", q=rng.uniform(.85, .95), cost=1, group="a"),
                 dict(kind="sym", q=rng.uniform(.7, .85), cost=2)]
    elif family == "SHARED":                     # two different-looking tools on one backend + one independent
        prior = rng.dirichlet([4, 2, 2])
        groups = {"g": rng.uniform(.15, .35) if phi is None else phi}
        tools = [dict(kind="sym", q=rng.uniform(.85, .95), cost=1, group="g"),
                 dict(kind="sym", q=rng.uniform(.8, .9), cost=1, group="g"),
                 dict(kind="sym", q=rng.uniform(.75, .85), cost=3)]
    elif family == "COMP":
        K = 4; prior = rng.dirichlet([20] * 4); err = rng.uniform(15, 25); esc_ = rng.uniform(6, 9)
        tools = [dict(kind="bit0", q=rng.uniform(.85, .95), cost=1), dict(kind="bit1", q=rng.uniform(.85, .95), cost=1),
                 dict(kind="sym", q=rng.uniform(.85, .92), cost=4)]
    elif family == "COMP3":
        K = 8; prior = rng.dirichlet([20] * 8); err = rng.uniform(15, 25); esc_ = rng.uniform(7, 10)
        tools = [dict(kind=f"bit{i}", q=rng.uniform(.88, .96), cost=1) for i in range(3)]
    for t in tools:
        t["cost"] *= cost_mult
        if t.get("group"): t["femit"] = STUCK(K)
    return dict(family=family, K=K, prior=prior, tools=tools, groups=groups,
                loss=loss_matrix(K, err, sev_mult * err, esc if esc is not None else esc_), budget=B)

def build(sp, **kw):
    return Model(sp["K"], kw.pop("prior", sp["prior"]), kw.pop("tools", sp["tools"]),
                 kw.pop("groups", sp["groups"]), sp["loss"], sp["budget"], **kw)

# ------------------------------------------------------- belief sources -------
def llm_sim(sp, inflate=0.0, temper=1.0):
    """Simulated direct-LLM model: treats every observation as independent noise (no fault groups)."""
    tools = [dict(t, q=1 - (1 - t["q"]) * (1 - inflate), group=None) for t in sp["tools"]]
    return build(sp, tools=tools, groups={}, temper=temper)

def calib_logs(true, sp, n, rng, eps, calls=3):
    S, O = [], []
    for _ in range(n):
        i = rng.choice(true.nH, p=true.prior); S.append(true.H[i][0]); row = []
        for j, t in enumerate(sp["tools"]):
            m = true.lik[j].shape[1]
            for _ in range(calls):
                o = rng.choice(m, p=true.lik[j][i]); row.append(mapnoise(o, m, eps, rng))
        O.append(row)
    return np.array(S), np.array(O)

def mapnoise(o, m, eps, rng):
    if eps > 0 and rng.random() < eps:
        return int(rng.choice([x for x in range(m) if x != o]))
    return o

def fit(sp, S, O, assumed_groups, emission="stuck", calls=3, iters=60):
    """EM fit of reliabilities, fault rates and fault emissions under an ASSUMED dependence structure."""
    K, tools = sp["K"], sp["tools"]
    cols = {j: slice(j * calls, (j + 1) * calls) for j in range(len(tools))}
    corr = {j: (O[:, cols[j]] == np.array([target(t["kind"], s) for s in S])[:, None]) for j, t in enumerate(tools)}
    q = {j: corr[j].mean() for j in range(len(tools))}
    groups, femit = {}, {}
    gnames = sorted(set(g for g in assumed_groups.values() if g))
    for g in gnames:
        mem = [j for j, gg in assumed_groups.items() if gg == g]
        phi = 0.2; fe = {j: np.full(n_obs(tools[j]["kind"], K), 1 / n_obs(tools[j]["kind"], K)) for j in mem}
        if emission == "stuck":
            for j in mem: fe[j] = np.bincount(O[:, cols[j]].ravel(), minlength=len(fe[j])) + 0.5; fe[j] /= fe[j].sum()
        for _ in range(iters):
            lh = np.zeros(len(S)); lf = np.zeros(len(S))
            for j in mem:
                m = len(fe[j]); c = corr[j].sum(1); w = calls - c
                lh += c * np.log(max(q[j], 1e-6)) + w * np.log(max((1 - q[j]) / (m - 1), 1e-9))
                lf += np.log(fe[j][O[:, cols[j]]]).sum(1)
            a = np.log(phi + 1e-12) + lf; b_ = np.log(1 - phi + 1e-12) + lh
            r = 1 / (1 + np.exp(np.clip(b_ - a, -50, 50)))
            phi = float(np.clip(r.mean(), 1e-4, 0.95))
            for j in mem:
                wh = (1 - r)[:, None]
                q[j] = float(np.clip((wh * corr[j]).sum() / max(wh.sum() * calls, 1e-9), 0.34, 0.999))
                if emission == "stuck":
                    m = len(fe[j]); cnt = np.array([(r[:, None] * (O[:, cols[j]] == x)).sum() for x in range(m)]) + 0.5
                    fe[j] = cnt / cnt.sum()
        groups[g] = phi; femit.update(fe)
    new_tools = [dict(t, q=q[j], group=assumed_groups.get(j), femit=femit.get(j)) for j, t in enumerate(tools)]
    prior = (np.bincount(S, minlength=K) + 1.0) / (len(S) + K)
    return new_tools, groups, prior

def true_groups(sp):
    return {j: t.get("group") for j, t in enumerate(sp["tools"])}

def stat_fit(n=100, structure="correct", eps=0.0, omit_p=0.0):
    """structure: correct | wrong_mechanism | split_shared | no_faults"""
    def f(sp_cal, true_cal, rng):
        S, O = calib_logs(true_cal, sp_cal, n, rng, eps)
        ag = true_groups(sp_cal)
        if structure == "split_shared":
            ag = {j: (f"{g}{j}" if g else None) for j, g in ag.items()}
        if structure == "no_faults":
            ag = {j: None for j in ag}
        tools, groups, prior = fit(sp_cal, S, O, ag, emission="uniform" if structure == "wrong_mechanism" else "stuck")
        omit = int(rng.integers(1, sp_cal["K"])) if rng.random() < omit_p else None
        m = Model(sp_cal["K"], prior, tools, groups, sp_cal["loss"], sp_cal["budget"], omit=omit)
        m.map_eps = eps
        return m
    return f

# ------------------------------------------------- dependence controls ---------
def fault_unaware(sp):
    """phi = 0: removes both single-call unreliability and persistence."""
    return llm_sim(sp)

def mmi(sp):
    """Marginal-matched independence: each call has the TRUE marginal P(o|s), calls multiplied as independent."""
    tools = [dict(t, marg_phi=sp["groups"].get(t.get("group"), 0.0) if t.get("group") else None, group=None)
             for t in sp["tools"]]
    return build(sp, tools=tools, groups={})

def split_true(sp):
    """True parameters, but every tool gets its own persistent flag (cross-tool sharing ignored)."""
    tools, groups = [], {}
    for j, t in enumerate(sp["tools"]):
        g = t.get("group")
        if g:
            groups[f"{g}{j}"] = sp["groups"][g]; tools.append(dict(t, group=f"{g}{j}"))
        else:
            tools.append(dict(t))
    return build(sp, tools=tools, groups=groups)

class Dedup(Model):
    """True model, but only the first observation from each source is used; repeat-source probes are excluded."""
    dedup = True
    def src(self, j):
        return source_of(self.tools[j], j)
    def V(self, b, B, used=frozenset()):
        k = (tuple(np.round(b, 9)), round(B, 6), used)
        if k in self.cache: return self.cache[k]
        best, a = self.term(b); out = (best, ("stop", a))
        for j, t in enumerate(self.tools):
            if t["cost"] > B + 1e-9 or self.src(j) in used: continue
            q = t["cost"]
            for o, p in enumerate(self.pred(b, j)):
                if p > 1e-12: q += p * self.V(self.update(b, j, o), B - t["cost"], used | {self.src(j)})[0]
            if q < out[0] - 1e-9: out = (q, ("probe", j))
        self.cache[k] = out; return out
    def Vd(self, b, B, d, used=frozenset()):
        best, a = self.term(b); out = (best, ("stop", a))
        if d == 0: return out
        for j, t in enumerate(self.tools):
            if t["cost"] > B + 1e-9 or self.src(j) in used: continue
            q = t["cost"]
            for o, p in enumerate(self.pred(b, j)):
                if p > 1e-12: q += p * self.Vd(self.update(b, j, o), B - t["cost"], d - 1, used | {self.src(j)})[0]
            if q < out[0] - 1e-9: out = (q, ("probe", j))
        return out

def dedup(sp):
    return Dedup(sp["K"], sp["prior"], sp["tools"], sp["groups"], sp["loss"], sp["budget"])

# --------------------------------------------------------------- policies -----
def act(pol, M, b, B, hist):
    k = pol[0]
    if getattr(M, "dedup", False):
        used = frozenset(M.src(j) for j in hist)
        if k == "dp": return M.V(b, B, used)[1]
        if k == "look": return M.Vd(b, B, pol[1], used)[1]
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

def source_of(t, j):
    return t.get("group") or f"tool{j}"

def episode(true, agent, pol, rng):
    i = rng.choice(true.nH, p=true.prior); s = true.H[i][0]
    bt, ba, B = true.prior.copy(), agent.prior.copy(), true.budget
    cost, hist, lowinfo, steps, eps = 0.0, [], 0, [], getattr(agent, "map_eps", 0.0)
    div = np.nan                                   # observations seen when agent's P(benign) first departs from truth
    while True:
        steps.append((agent.marg_s(ba), s))
        a = act(pol, agent, ba, B, hist)
        if a[0] == "stop":
            dl = true.loss[a[1], s]
            r = dict(total=dl + cost, cost=cost, calls=len(hist), severe=(a[1] == 0 and s != 0),
                     premature=true.V(bt, B)[1][0] == "probe", esc=a[1] == true.K, lowinfo=lowinfo, steps=steps,
                     benign_commit=a[1] == 0, div=div, overconf=(agent.marg_s(ba)[0] - true.marg_s(bt)[0]) if a[1] == 0 else np.nan)
            return r
        j = a[1]
        src = source_of(true.tools[j], j)
        if any(source_of(true.tools[k], k) == src for k in hist) and true.info_gain_s(bt, j) < 0.05:
            lowinfo += 1
        o = rng.choice(true.lik[j].shape[1], p=true.lik[j][i])
        oa = mapnoise(o, true.lik[j].shape[1], eps, rng)
        cost += true.tools[j]["cost"]; B -= true.tools[j]["cost"]; hist.append(j)
        bt = true.update(bt, j, o)
        if getattr(agent, "dedup", False) and any(agent.src(k) == agent.src(j) for k in hist[:-1]):
            if np.isnan(div) and abs(agent.marg_s(ba)[0] - true.marg_s(bt)[0]) > 0.01: div = len(hist)
            continue                                   # dedup agent discards repeat-source observations
        nb = agent.update(ba, j, oa); ba = nb if nb is not None else agent.prior.copy()
        if np.isnan(div) and abs(agent.marg_s(ba)[0] - true.marg_s(bt)[0]) > 0.01: div = len(hist)

def calib(steps):
    conf = np.array([p.max() for p, s in steps]); ok = np.array([p.argmax() == s for p, s in steps])
    e = 0.0
    for lo in np.linspace(0, .9, 10):
        m = (conf > lo) & (conf <= lo + .1)
        if m.any(): e += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return e

def evaluate(family, policies, sources, n_inst=40, n_ep=300, seed=0, spec_kw=None, shift=None):
    """policies: name -> (pol, source). sources: name -> f(sp_cal, true_cal, rng) -> Model.
    shift: function(sp, rng) -> test spec (calibration uses the unshifted spec)."""
    rng = np.random.default_rng(seed); out = {n: [] for n in policies}; st = {n: [] for n in policies}
    for _ in range(n_inst):
        sp = spec(family, rng, **(spec_kw or {}))
        true_cal = build(sp)
        sp_test = shift(sp, rng) if shift else sp
        true = build(sp_test) if shift else true_cal
        ags = {}
        for sn, f in sources.items():
            if sn == "true": ags[sn] = true
            elif sn.endswith("_refit"): ags[sn] = f(sp_test, true, rng)
            else: ags[sn] = f(sp, true_cal, rng)
        seeds = rng.integers(1 << 31, size=n_ep)
        for name, (pol, sn) in policies.items():
            for es in seeds:
                r = episode(true, ags[sn], pol, np.random.default_rng(es))
                st[name].extend(r.pop("steps")); out[name].append(r)
    res = {}
    for n, rs in out.items():
        g = lambda k: float(np.mean([r[k] for r in rs]))
        calls = sum(r["calls"] for r in rs)
        oc = [r["overconf"] for r in rs if r["benign_commit"]]
        res[n] = dict(total=g("total"), se=float(np.std([r["total"] for r in rs]) / np.sqrt(len(rs))),
                      cost=g("cost"), calls=g("calls"), severe=g("severe"), premature=g("premature"), esc=g("esc"),
                      lowinfo=sum(r["lowinfo"] for r in rs) / max(calls, 1), ece=calib(st[n]),
                      overconf=float(np.mean(oc)) if oc else 0.0,
                      div_share=float(np.mean([not np.isnan(r["div"]) for r in rs])),
                      div_first1=float(np.mean([r["div"] == 1 for r in rs if not np.isnan(r["div"])])) if any(not np.isnan(r["div"]) for r in rs) else 0.0)
    if "Oracle-DP" in res:
        for n in res: res[n]["regret"] = res[n]["total"] - res["Oracle-DP"]["total"]
    return res

SOURCES = {
    "fault_unaware": lambda sp, t, rng: fault_unaware(sp),
    "mmi": lambda sp, t, rng: mmi(sp),
    "split_true": lambda sp, t, rng: split_true(sp),
    "dedup": lambda sp, t, rng: dedup(sp),
    "true": None,
    "llm_indep": lambda sp, t, rng: llm_sim(sp),
    "llm_composite": lambda sp, t, rng: llm_sim(sp, inflate=0.5, temper=1.5),
    "fit": stat_fit(100),
    "fit_wrongmech": stat_fit(100, "wrong_mechanism"),
    "fit_split": stat_fit(100, "split_shared"),
    "fit_map": stat_fit(100, eps=0.1),
    "fit_omit": stat_fit(100, omit_p=0.1),
    "fit_nofault": stat_fit(100, "no_faults"),
}
