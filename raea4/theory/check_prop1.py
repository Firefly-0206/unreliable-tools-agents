"""Proposition 1 and its generalizations (Lemma 1, Propositions 1a-1d), checked on random instances.
Likelihood ratios are read off sim.Model posteriors wherever sim can express the model; the stale-value and
recovery models (which sim cannot express) use the enumerators in common.py, cross-checked against sim first."""
import json
import numpy as np
from common import sim, seq_lik, recovery_lik

TOL = 1e-9
rng = np.random.default_rng(7)
res = {}

def lr_from_model(M, b0, obs, s, s2):
    """P(obs | s) / P(obs | s2), from the posterior/prior odds of sim.Model (b0 over (s, z), any joint)."""
    b = b0.copy()
    for j, o in obs: b = M.update(b, j, o)
    p0, p = M.marg_s(b0), M.marg_s(b)
    return (p[s] / p[s2]) / (p0[s] / p0[s2])

def group_model(K, qs, femits, phi, extra_tool=None):
    tools = [dict(kind="sym", q=q, cost=1, group="g", femit=fe) for q, fe in zip(qs, femits)]
    return sim.Model(K, np.full(K, 1 / K), tools, {"g": phi}, sim.loss_matrix(K, 10, 30, 4), 6)

# ---------------------------------------------------------- P1-gen: general two-sided bound ----
n_chk, viol = 0, 0
for _ in range(3000):
    K = int(rng.integers(2, 5)); J = int(rng.integers(1, 4)); phi = rng.uniform(.01, .9)
    qs = rng.uniform(1 / K + .01, .99, J); fes = [rng.dirichlet(np.ones(K) * rng.uniform(.2, 3)) for _ in range(J)]
    M = group_model(K, qs, fes, phi)
    obs = [(int(rng.integers(J)), int(rng.integers(K))) for _ in range(int(rng.integers(1, 9)))]
    s, s2 = rng.choice(K, 2, replace=False)
    lr = lr_from_model(M, M.prior, obs, s, s2)
    H = lambda st: np.prod([sim.healthy("sym", qs[j], K, st)[o] for j, o in obs])
    F = np.prod([fes[j][o] for j, o in obs])
    lo = min(1, H(s) / H(s2)); hi = min(max(1, H(s) / H(s2)), 1 + (1 - phi) / phi * H(s) / F)
    n_chk += 1; viol += not (lo * (1 - 1e-9) - TOL <= lr <= hi * (1 + 1e-9) + TOL)
print(f"[P1-gen] min(1,Hs/Hs') <= LR <= min(max(1,Hs/Hs'), 1+(1-phi)/phi*Hs/F): {n_chk - viol}/{n_chk} hold")
res["P1_general"] = dict(n=n_chk, violations=viol)

# ---------------------------------------------------------- P1a: the f vs q boundary (all-benign runs) --
def run_lr(K, q, f, phi, n):
    fe = np.r_[f, np.full(K - 1, (1 - f) / (K - 1))]
    M = group_model(K, [q], [fe], phi)
    return np.array([min(lr_from_model(M, M.prior, [(0, 0)] * m, 0, s) for s in range(1, K)) for m in range(1, n + 1)]), \
           np.array([max(lr_from_model(M, M.prior, [(0, 0)] * m, 0, s) for s in range(1, K)) for m in range(1, n + 1)])

out = {"f>q": [0, 0], "f=q": [0, 0], "f<q": [0, 0]}
for _ in range(300):
    K = int(rng.integers(2, 4)); q = rng.uniform(.6, .93); phi = rng.uniform(.05, .6); n = 60
    for reg, f in (("f>q", rng.uniform(q + .01, 1.0)), ("f=q", q), ("f<q", rng.uniform(max(.5, (1 - q) / (K - 1) + .05), q - .05))):
        lo_s, lr = run_lr(K, q, f, phi, n)
        m = np.arange(1, n + 1)
        if reg == "f>q":
            ok = np.all(lr <= 1 + (1 - phi) / phi * (q / f) ** m + TOL) and np.all(lr <= 1 / phi + TOL)
        elif reg == "f=q":
            ok = np.all(lr <= 1 / phi + TOL) and np.all(np.diff(lr) >= -TOL) and lr[-1] > 0.95 / phi
        else:
            r = (1 - q) / (K - 1)
            lb = 0.5 * np.minimum((q / r) ** m, (1 - phi) / phi * (q / max(f, r)) ** m)
            ok = np.all(lo_s >= lb - TOL) and lo_s[-1] > 1 / phi
        out[reg][0] += 1; out[reg][1] += bool(ok)
for reg, (a, b) in out.items(): print(f"[P1a] {reg}: claims hold in {b}/{a}")
# worked numbers (paper regime phi=.3, q=.9, K=3) for the three cases
K, q, phi = 3, .9, .3
wk = {}
for f in (.97, .9, .8):
    lmin, lmax = run_lr(K, q, f, phi, 12)
    w0 = .5 / .5; post = w0 * lmin / (1 + w0 * lmin)
    wk[f] = dict(LR_worst=[round(x, 3) for x in lmin], post_benign_vs_worst=[round(x, 3) for x in post])
    print(f"   f={f}: LR (worst non-benign s) n=1..12:", [round(float(x), 2) for x in lmin])
res["P1a"] = dict(counts=out, worked=wk)

# ---------------------------------------------------------- Lemma 1 / P1b: state-dependent fault rates ---
n_chk = viol = viol_marg = 0
for _ in range(1500):
    K = int(rng.integers(2, 4)); q = rng.uniform(.6, .95); f = rng.uniform(q, 1.0); n = int(rng.integers(1, 30))
    phis = rng.uniform(.02, .7, K); pis = rng.dirichlet(np.ones(K) * 2)
    fe = np.r_[f, np.full(K - 1, (1 - f) / (K - 1))]
    M = group_model(K, [q], [fe], .5)                                  # flag prior overridden below
    b0 = np.array([pis[s] * (phis[s] if z[0] else 1 - phis[s]) for s, z in M.H])
    for s in range(1, K):
        lr = lr_from_model(M, b0, [(0, 0)] * n, 0, s)
        n_chk += 1; viol += lr > 1 / phis[s] + TOL
        phibar = float(pis @ phis); viol_marg += lr > 1 / phibar + TOL
# long-run limit phi_0 / phi_s
K, q, f = 3, .9, .97; phis = np.array([.1, .4, .4]); M = group_model(K, [q], [np.r_[f, .015, .015]], .5)
b0 = np.array([[.5, .25, .25][s] * (phis[s] if z[0] else 1 - phis[s]) for s, z in M.H])
lim = lr_from_model(M, b0, [(0, 0)] * 300, 0, 1)
print(f"[P1b] LR(0 vs s) <= 1/phi_s: {n_chk - viol}/{n_chk} hold; the marginal-phi bound 1/phibar fails in {viol_marg}/{n_chk}")
print(f"   long run (phi_0=.1, phi_s=.4): LR after 300 reports {lim:.4f}, predicted phi_0/phi_s = {phis[0] / phis[1]:.4f}")
M = group_model(2, [.9], [np.r_[.97, .03]], .5); pis, phis = np.array([.5, .5]), np.array([.6, .05])
b0 = np.array([pis[s] * (phis[s] if z[0] else 1 - phis[s]) for s, z in M.H]); cex = lr_from_model(M, b0, [(0, 0)] * 300, 0, 1)
print(f"   counterexample pi=(.5,.5), phi_s=(.6,.05): LR after 300 reports {cex:.3f} > 1/phibar = {1 / (pis @ phis):.3f}")
res["P1b"] = dict(counterexample=cex, n=n_chk, violations=int(viol), marginal_bound_violations=int(viol_marg), long_run=lim, predicted=phis[0] / phis[1])

# ---------------------------------------------------------- partial sharing (Lemma 1 with P(E) = phi pi^nB) --
n_chk = viol = 0; grow = 0
for _ in range(600):
    K = int(rng.integers(2, 4)); qa, qb = rng.uniform(.6, .95, 2); f = rng.uniform(max(qa, qb), 1.0)
    phi = rng.uniform(.05, .6); pib = rng.uniform(.2, .95); nA = int(rng.integers(0, 4)); nB = int(rng.integers(1, 8))
    fe = np.r_[f, np.full(K - 1, (1 - f) / (K - 1))]
    def emit(s, c, j):
        h = sim.healthy("sym", qa if j == 0 else qb, K, s)
        if c == 0: return h
        return fe if j == 0 else pib * fe + (1 - pib) * h           # B hits the shared cache w.p. pib per call
    cf = np.tile([1 - phi, phi], (K, 1))
    obs = [(0, 0)] * nA + [(1, 0)] * nB
    L = seq_lik(cf, emit, obs)
    for s in range(1, K):
        n_chk += 1; viol += L[0] / L[s] > 1 / (phi * pib ** nB) + TOL
    L2 = seq_lik(cf, emit, [(0, 0)] * nA + [(1, 0)] * 40)
    grow += L2[0] / L2[1] > 1 / phi
print(f"[P1c partial sharing] LR <= 1/(phi pi^nB): {n_chk - viol}/{n_chk} hold; long B-runs exceed 1/phi in {grow}/600")
res["partial"] = dict(n=n_chk, violations=int(viol), exceed_1_over_phi=int(grow))

# ---------------------------------------------------------- stale value --------------------------------
# cross-check the enumerator against sim on the standard stuck-at-benign model first
K, q, f, phi = 3, .9, .97, .3; fe = np.r_[f, .015, .015]
M = group_model(K, [q], [fe], phi)
L = seq_lik(np.tile([1 - phi, phi], (K, 1)), lambda s, c, j: fe if c else sim.healthy("sym", q, K, s), [(0, 0)] * 5)
xchk = abs(L[0] / L[1] - lr_from_model(M, M.prior, [(0, 0)] * 5, 0, 1))
n_chk = viol = viol_t = 0; lim_err = []
for _ in range(1500):
    K = int(rng.integers(2, 4)); q = rng.uniform(.6, .95); qs = q if rng.random() < .5 else rng.uniform(q, .99)   # qs: accuracy of reading the stale value
    phis = rng.uniform(.03, .6, K); rho = rng.dirichlet(np.ones(K), K)                    # rho[s, v] = P(v | z=1, s)
    indep = rng.random() < .5
    if indep: rho = np.tile(rho[0], (K, 1)); phis = np.full(K, phis[0])
    cfg = np.array([[1 - phis[s]] + [phis[s] * rho[s, v] for v in range(K)] for s in range(K)])   # c = 0 healthy, 1+v stale v
    emit = lambda s, c, j: sim.healthy("sym", q, K, s) if c == 0 else sim.healthy("sym", qs, K, c - 1)
    n = int(rng.integers(1, 25)); L = seq_lik(cfg, emit, [(0, 0)] * n)
    for s in range(1, K):
        lr = L[0] / L[s]
        if qs >= q:                                                    # (A4*) holds
            n_chk += 1; viol += lr > 1 / (phis[s] * rho[s, 0]) + TOL
            if indep:
                bt = 1 + (1 - phis[0]) / (phis[0] * rho[0, 0]) * (q / qs) ** n; viol_t += lr > bt + TOL
                if s == 1 and qs == q:
                    Linf = seq_lik(cfg, emit, [(0, 0)] * 400); b1 = 1 + (1 - phis[0]) / (phis[0] * rho[0, 0]); lim_err.append(abs(Linf[0] / Linf[s] - b1) / b1)
print(f"[P1d stale value] enumerator vs sim cross-check |diff| = {xchk:.2e}")
print(f"   LR <= 1/(phi_s rho(0|s)): {n_chk - viol}/{n_chk}; value independent of s: LR <= 1+(1-phi)/(phi rho0)(q/qs)^n violations {viol_t}; "
      f"with qs = q the long-run limit equals 1+(1-phi)/(phi rho0) (max rel. err {max(lim_err):.1e}, {len(lim_err)} cases)")
res["stale"] = dict(xcheck=xchk, n=n_chk, violations=int(viol), tight_violations=int(viol_t), limit_max_rel_err=max(lim_err))

# ---------------------------------------------------------- recovery -----------------------------------
K, q, f, phi = 3, .9, .97, .3
Lr = recovery_lik(K, phi, 0.0, [q, .05, .05], f, 6)
M = group_model(K, [q], [np.r_[f, .015, .015]], phi)
xchk = abs(Lr[0] / Lr[1] - lr_from_model(M, M.prior, [(0, 0)] * 6, 0, 1))
n_chk = viol = div_ok = div_n = 0
for _ in range(800):
    K = int(rng.integers(2, 4)); q = rng.uniform(.6, .95); f = rng.uniform(q, 1.0); phi = rng.uniform(.05, .6)
    rho = rng.uniform(0, .5); r = (1 - q) / (K - 1); h = [q] + [r] * (K - 1); x = q / ((1 - rho) * f)
    lrs = []
    for n in (1, 2, 5, 10, 20, 60):
        L = recovery_lik(K, phi, rho, h, f, n); lr = L[0] / L[1]; lrs.append(lr)
        if x < 1:
            n_chk += 1; viol += lr > 1 + (1 - phi) * (1 - rho) / phi * x ** n + rho * x / (1 - x) + TOL
    if x > 1.02:
        div_n += 1; L = recovery_lik(K, phi, rho, h, f, 200); div_ok += L[0] / L[1] > 1 / phi * 10
print(f"[P1e recovery] rho=0 cross-check vs sim |diff| = {xchk:.2e}; x<1 bound holds {n_chk - viol}/{n_chk}; "
      f"x>1: LR exceeds 10/phi after 200 reports in {div_ok}/{div_n}")
# boundary x = 1 with rho > 0, i.e. (1-rho) f = q: closed form, linear growth (unlike rho = 0, f = q, which saturates at 1/phi)
def lr_x1(phi, rho, q, r, n):
    num = (1 - phi) * (1 - rho) + phi * (1 + (n - 1) * rho)
    return num / ((1 - phi) * (1 - rho) * (r / q) ** n + phi * (1 + rho * sum((r / q) ** i for i in range(1, n))))
def lb_x1(phi, rho, q, r, n):
    return ((1 - phi) * (1 - rho) + phi * (1 + (n - 1) * rho)) / ((1 - phi) * (1 - rho) * (r / q) ** n + phi * (1 + rho * r / (q - r)))
x1_err = []; x1_lb = x1_inc = x1_exceed = x1_n = 0
NS = (1, 2, 5, 10, 20, 50, 100, 200, 400)
for _ in range(400):
    K = int(rng.integers(2, 4)); q = rng.uniform(.6, .95); rho = rng.uniform(.01, 1 - q - 1e-3); f = q / (1 - rho)
    phi = rng.uniform(.05, .6); r = (1 - q) / (K - 1); h = [q] + [r] * (K - 1)
    lrs = []
    for n in NS:
        L = recovery_lik(K, phi, rho, h, f, n); lr = L[0] / L[1]; lrs.append(lr)
        x1_err.append(abs(lr - lr_x1(phi, rho, q, r, n)) / lr); x1_lb += lr >= lb_x1(phi, rho, q, r, n) - TOL
    x1_n += 1; x1_inc += bool(np.all(np.diff(lrs) > 0)); x1_exceed += lrs[-1] > 1 / phi
ex = [round(float(recovery_lik(3, .3, .1, [.9, .05, .05], 1.0, n)[0] / recovery_lik(3, .3, .1, [.9, .05, .05], 1.0, n)[1]), 2) for n in (5, 20, 50, 100, 200, 400)]
print(f"   x=1, rho>0 ((1-rho)f = q), {x1_n} instances: closed form matches the enumerator (max rel. err {max(x1_err):.1e}); "
      f"lower bound holds {x1_lb}/{x1_n * len(NS)}; LR strictly increasing over n=1..400 in {x1_inc}/{x1_n}; LR(400) > 1/phi in {x1_exceed}/{x1_n}")
print(f"   example phi=.3, q=.9, r=.05, rho=.1, f=1 (x=1): LR at n=5,20,50,100,200,400 = {ex} (1/phi = 3.33)")
res["recovery"] = dict(xcheck=xchk, n=n_chk, violations=int(viol), diverge=div_ok, diverge_n=div_n,
                       x1=dict(n=x1_n, closed_form_max_rel_err=max(x1_err), lower_bound_holds=int(x1_lb), n_checks=x1_n * len(NS),
                               increasing=int(x1_inc), exceed_1_over_phi_at_400=int(x1_exceed), example=ex))
json.dump(res, open("check_prop1.json", "w"), indent=1, default=float)
