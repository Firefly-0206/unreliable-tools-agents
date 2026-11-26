"""Shared helpers for the theory checks. sim.py / exact.py are imported read-only from the parent folder."""
import itertools, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sim, exact                                   # noqa: E402

BIG = 1e250    # loss of a disabled terminal action. 1e12 was too small: with q ~ .99 and N >= 7, P(s=0|h) falls
               # below 1e-13 and "declare bad" became the exact optimum, so the model was no longer certify-or-escalate

# ------------------------------------------------------------ generic latent enumerator ----
def seq_lik(cfgs, emit, obs):
    """P(obs | s) for every s, for a group with a STATIC latent configuration c.
    cfgs: (K, C) array P(c | s); emit(s, c, j) -> distribution over outputs of tool j; obs: [(j, o), ...]."""
    K, C = cfgs.shape
    out = np.zeros(K)
    for s in range(K):
        for c in range(C):
            if cfgs[s, c] == 0: continue
            p = cfgs[s, c]
            for j, o in obs: p *= emit(s, c, j)[o]
            out[s] += p
    return out

def recovery_lik(K, phi, rho, h, f, n):
    """P(n benign reports | s) when a fault that is present at call 1 recovers before each later call w.p. rho
    (no new faults). h[s] = healthy P(o0 | s), f = faulty P(o0). Exact sum over the fault's lifetime."""
    out = np.zeros(K)
    for s in range(K):
        p = (1 - phi) * h[s] ** n
        for k in range(1, n + 1):                   # fault lasts exactly k calls (k = n: never recovers)
            pk = (1 - rho) ** (k - 1) * (rho if k < n else 1.0)
            p += phi * pk * f ** k * h[s] ** (n - k)
        out[s] = p
    return out

# ------------------------------------------------------------ minimal model (Theorem 2) ----
def minimal_spec(pi0, phi, q, f, c, lam, ce, B, verifier=None, err=BIG):
    """K = 2. Source A (group 'a', stuck-at-benign w.p. f when faulty); optional verifier dict(q=, cost=).
    Terminal actions: certify benign (loss lam if s = 1), escalate (ce); 'declare bad' disabled unless err given."""
    tools = [dict(kind="sym", q=q, cost=c, group="a", femit=np.array([f, 1 - f]))]
    if verifier: tools.append(dict(kind="sym", q=verifier["q"], cost=verifier["cost"]))
    return dict(family="MIN", K=2, prior=np.array([pi0, 1 - pi0]), tools=tools, groups={"a": phi},
                loss=sim.loss_matrix(2, err, lam, ce), budget=B)

def mmi_quantities(pi0, phi, q, f):
    a0 = (1 - phi) * q + phi * f; a1 = (1 - phi) * (1 - q) + phi * f
    return a0, a1

def exact_lr(phi, q, f, n0, n1):
    """Exact benign likelihood ratio of a sequence with n0 benign and n1 bad reports from A (K = 2)."""
    num = (1 - phi) * q ** n0 * (1 - q) ** n1 + phi * f ** n0 * (1 - f) ** n1
    den = (1 - phi) * (1 - q) ** n0 * q ** n1 + phi * f ** n0 * (1 - f) ** n1
    return num / den

def theorem2_constants(pi0, phi, q, f, c, lam, ce, B):
    """Every quantity named in Theorem 2: m*, m_D, Gamma, P(S), and the predicted gap c + Gamma - ce P(S)."""
    N = int(np.floor(B / c + 1e-9)); w0 = pi0 / (1 - pi0); Om = (lam - ce) / ce
    a0, a1 = mmi_quantities(pi0, phi, q, f); ell = a0 / a1
    mstar = next((m for m in range(1, 200) if w0 * ell ** m >= Om), None)
    PM = lambda i: pi0 * a0 ** i + (1 - pi0) * a1 ** i                     # MMI P(0^i)
    def D(m):
        return c * sum(PM(i) for i in range(m)) < ce * PM(m) - lam * (1 - pi0) * a1 ** m - 1e-12
    mD = next((m for m in range(mstar, N + 1) if D(m)), None) if mstar is not None else None
    excl = w0 / phi < Om                                                    # Proposition 1 / Corollary 1
    PT = lambda n: pi0 * ((1 - phi) * q ** n + phi * f ** n) + (1 - pi0) * ((1 - phi) * (1 - q) ** n + phi * f ** n)
    P1 = lambda n: (1 - pi0) * ((1 - phi) * (1 - q) ** n + phi * f ** n)
    Gamma = min(lam * P1(n) - ce * PT(n) for n in range(mstar, N + 1)) if (mstar and mstar <= N) else np.nan
    # P(S): probability that some prefix with >= 1 bad report has exact benign odds >= Omega
    PS = 0.0
    for seq in itertools.product([0, 1], repeat=N):
        hit, n0, n1 = False, 0, 0
        for o in seq:
            n0 += o == 0; n1 += o == 1
            if n1 >= 1 and w0 * exact_lr(phi, q, f, n0, n1) >= Om: hit = True; break
        if hit:
            n0 = seq.count(0); n1 = N - n0
            PS += pi0 * ((1 - phi) * q ** n0 * (1 - q) ** n1 + phi * f ** n0 * (1 - f) ** n1) + \
                  (1 - pi0) * ((1 - phi) * (1 - q) ** n0 * q ** n1 + phi * f ** n0 * (1 - f) ** n1)
    return dict(N=N, Omega=Om, ell=ell, a0=a0, a1=a1, mstar=mstar, mD=mD, excl=excl, Gamma=Gamma, PS=PS,
                gap_lb=(c + Gamma - ce * PS) if mD else np.nan)

def eval_policy(sp, agent, pol):
    true = sim.build(sp)
    return exact.evaluate_exact(true, agent, pol, sp)
