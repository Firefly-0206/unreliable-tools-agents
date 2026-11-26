"""
Exploratory, post hoc (no plan fixed in advance; no model call): first-action models compared by held-out likelihood.

The first action of an episode is one of five options: one of the four tools, or submitting at once.
  M0  noise only:        P(a) ~ exp(-Q*(a) / tau), Q* the exact action values of the TRUE model               (1 parameter)
  M1  pessimism + noise: P(a) ~ exp(-Q_g(a) / tau), Q_g the exact action values of the model whose replica
                         fault odds are multiplied by g (z_asif.py)                                           (2 parameters)
  U   uniform over the five options                                                                           (0 parameters)
Parameters are fitted per run by maximum likelihood on the PERSISTENT arm of the sweep (100 episodes) and evaluated,
unchanged, on the independent arm (100), the randomised-surface episodes (100) and the four further samples of the
repeated instances (80). Reported: mean log-likelihood per first action (nats; higher is better) and the predicted share
of episodes opening with the replica tool.
Output: results/z_asif_soft.json, results/z_asif_soft.txt
"""
import os
os.environ.setdefault("RAEA_EPISODES", "episodes_x.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_x.jsonl")
import copy, json
import numpy as np
from env import HERE, load_episodes
from oracle import qvalues
import k2decomp as K
from x_analysis import REP
from y_analysis import load, RUNS, B_SAMPLES

GRID_G = [1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
GRID_T = [0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10]
PHIS = [0.10, 0.30, 0.50, 0.70]
P = lambda *a: os.path.join(HERE, *a)


def qvec(ep, g):
    spec = copy.deepcopy(K.truth_spec(ep))
    for grp in spec["groups"]:
        if REP in grp["tools"]: phi = grp["phi"]; grp["phi"] = g * phi / (1 - phi + g * phi)
    M = K.build_recovered_model(ep, spec); q = qvalues(M, M.prior.copy(), float(M.budget))
    return np.array([q.get(("probe", j), np.inf) for j in range(4)] + [min(v for k, v in q.items() if k[0] == "stop")])


def probs(q, tau):
    z = -(q - q.min()) / tau; e = np.exp(z); return e / e.sum()


def first_action(log):
    c = [x["j"] for x in log if x["type"] == "call"]; return c[0] if c else 4


def main():
    EX = load_episodes(P("episodes_x.jsonl")); EY = load_episodes(P("episodes_y.jsonl")); ids = sorted(EX)
    Q = {g: {i: qvec(EX[i], g) for i in ids} for g in GRID_G}
    arm = lambda a: [i for i in ids if EX[i]["arm"] == a]
    out, L = {}, []
    for key, (label, stem) in RUNS.items():
        X = load(P("logs", f"x_{stem}_k2e.jsonl"), [0]); Y = load(P("logs", f"y_{stem}_k2e.jsonl"), [0]); R = load(P("logs", f"xrep_{stem}_k2e.jsonl"), B_SAMPLES[1:])
        sets = dict(fit_persistent=[(i, first_action(X[(i, 0)]["log"])) for i in arm("XP")],
                    independent=[(i, first_action(X[(i, 0)]["log"])) for i in arm("XI")],
                    randomised=[(EY[j]["x_id"], first_action(Y[(j, 0)]["log"])) for j in sorted(EY)],
                    repeats=[(i, first_action(r["log"])) for (i, s), r in sorted(R.items())])
        ll = lambda g, t, data: float(np.mean([np.log(max(probs(Q[g][i], t)[a], 1e-12)) for i, a in data]))
        t0 = max(GRID_T, key=lambda t: ll(1, t, sets["fit_persistent"]))
        g1, t1 = max(((g, t) for g in GRID_G for t in GRID_T), key=lambda gt: ll(gt[0], gt[1], sets["fit_persistent"]))
        o = dict(label=label, M0=dict(tau=t0), M1=dict(g=g1, tau=t1), loglik={}, replica_share={})
        for name, data in sets.items():
            o["loglik"][name] = dict(M0=ll(1, t0, data), M1=ll(g1, t1, data), U=float(np.log(1 / 5)), n=len(data))
        for a in ("XP", "XI"):
            for p in PHIS:
                c = [i for i in arm(a) if abs(EX[i]["phi"] - p) < 1e-9]
                o["replica_share"][f"{a}|{p:.2f}"] = dict(obs=float(np.mean([first_action(X[(i, 0)]["log"]) == REP for i in c])),
                                                         M0=float(np.mean([probs(Q[1][i], t0)[REP] for i in c])), M1=float(np.mean([probs(Q[g1][i], t1)[REP] for i in c])))
        out[key] = o
        L += ["", f"== {label}: M0 tau={t0};  M1 g={g1}, tau={t1}",
              "  mean log-likelihood per first action (nats)   M0 noise   M1 pessimism+noise   uniform"]
        L += [f"  {n:16s} (n={d['n']:3d})              {d['M0']:9.3f}{d['M1']:14.3f}{d['U']:14.3f}" for n, d in o["loglik"].items()]
        L.append("  replica first-call share, obs / M0 / M1 (%): " + "; ".join(f"{c}: {100 * d['obs']:.0f}/{100 * d['M0']:.0f}/{100 * d['M1']:.0f}" for c, d in o["replica_share"].items()))
    txt = "\n".join(L); print(txt)
    json.dump(out, open(P("results", "z_asif_soft.json"), "w"), indent=1); open(P("results", "z_asif_soft.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
