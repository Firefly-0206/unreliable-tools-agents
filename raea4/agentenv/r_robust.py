"""
Post hoc (added 2026-11-18 after a mock review; no plan fixed in advance; no model call): is the under-use what a
planner that is uncertain about the documented fault rate would do?

(1) Bayes over the fault rate. In the persistent arm the replica's stale flag is drawn once per session with
    probability phi, and reports depend on phi only through that flag. A planner with any prior over phi therefore
    acts exactly as the oracle with phi replaced by the prior mean: a prior centred on the documented rate changes
    nothing. This is an identity of the model class (P(stale) = E[phi]), so it is stated, not simulated.
(2) Ambiguity aversion. A planner that plans for the upper end of an ambiguity set, phi + delta (the stale replica
    reports the benign status, the costly direction), is fitted like z_asif.py: delta chosen per run on the persistent
    arm of the sweep X by first-call agreement, then applied unchanged to the independent arm, the randomised surface
    and the repeated instances. Reported: the delta needed, what phi = 0.1 / 0.3 then becomes, the fit, calls,
    escalations and regret under the true model.

  python r_robust.py      -> results/r_robust.json, .txt
"""
import os
os.environ.setdefault("RAEA_EPISODES", "episodes_x.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_x.jsonl")
import json
import numpy as np
import z_asif_loss as Z
from env import load_episodes, load_truth

P = Z.P
DELTAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
PHIS = Z.PHIS


def var(dphi, g=1.0):
    d = Z.variant("g", g); d["dphi"] = float(dphi); return d


def main():
    EX, TX = load_episodes(P("episodes_x.jsonl")), load_truth(P("ground_truth_x.jsonl")); EY = load_episodes(P("episodes_y.jsonl"))
    ids = sorted(EX); vs = {dl: var(dl) for dl in DELTAS}
    pred = Z.solve(EX, TX, {("dphi", dl): d for dl, d in vs.items()})
    pv = lambda dl, i: pred[(i, Z.key(vs[dl]))]
    arm = lambda a: [i for i in ids if EX[i]["arm"] == a]
    cell = lambda a, p: [i for i in ids if EX[i]["arm"] == a and abs(EX[i]["phi"] - p) < 1e-9]
    out, L = dict(deltas=DELTAS, runs={}), ["Ambiguity-averse planner (plans for phi + delta), fitted on the persistent arm of X (post hoc)"]
    from x_analysis import parts
    L.append("(2) behaviour by delta, persistent arm: replica first-call % at phi 0.1/0.3/0.5/0.7 | calls | escalation % | regret (all 200)")
    for dl in DELTAS:
        fs = [100 * np.mean([pv(dl, i)["first"] for i in cell("XP", p)]) for p in PHIS]
        L.append(f"  delta={dl:<5}" + "".join(f"{x:5.0f}" for x in fs) + f"   | {np.mean([pv(dl, i)['calls'] for i in arm('XP')]):.2f} | "
                 f"{100 * np.mean([pv(dl, i)['escalated'] for i in arm('XP')]):3.0f} | {np.mean([pv(dl, i)['regret'] for i in ids]):.2f}")
    for rk, (label, stem) in Z.RUNS.items():
        X = Z.load(P("logs", f"x_{stem}_k2e.jsonl"), [0]); Y = Z.load(P("logs", f"y_{stem}_k2e.jsonl"), [0])
        R = dict(X); R.update(Z.load(P("logs", f"xrep_{stem}_k2e.jsonl"), Z.B_SAMPLES[1:]))
        obs = {i: parts(EX[i], X[(i, 0)]["log"], TX[i]) for i in ids}
        yobs = {EY[j]["x_id"]: float([x["j"] for x in Y[(j, 0)]["log"] if x["type"] == "call"][:1] == [Z.REP]) for j in EY}
        bids = sorted({i for (i, s) in R if s > 0})
        share = {i: float(np.mean([[x["j"] for x in R[(i, s)]["log"] if x["type"] == "call"][:1] == [Z.REP] for s in Z.B_SAMPLES])) for i in bids}
        m = lambda dl, sel: float(np.mean([pv(dl, i)["first"] == obs[i]["first"] for i in sel]))
        dh = max(DELTAS, key=lambda dl: (round(m(dl, arm("XP")), 9), -dl))
        rec = dict(label=label, delta=dh, fit_persistent=m(dh, arm("XP")), independent=m(dh, arm("XI")),
                   randomised=float(np.mean([pv(dh, i)["first"] == yobs[i] for i in yobs])),
                   repeats_majority=float(np.mean([(share[i] > 0.5) == bool(pv(dh, i)["first"]) for i in bids])),
                   oracle_fit=m(0.0, arm("XP")), oracle_indep=m(0.0, arm("XI")),
                   delta_indep=max(DELTAS, key=lambda dl: (round(m(dl, arm("XI")), 9), -dl)),
                   calls=float(np.mean([pv(dh, i)["calls"] for i in arm("XP")])), esc=float(np.mean([pv(dh, i)["escalated"] for i in arm("XP")])),
                   regret=float(np.mean([pv(dh, i)["regret"] for i in ids])), regret_obs=float(np.mean([obs[i]["regret"] for i in ids])),
                   calls_obs=float(np.mean([obs[i]["calls"] for i in arm("XP")])),
                   profile={str(dl): m(dl, arm("XP")) for dl in DELTAS})
        out["runs"][rk] = rec
        L += ["", f"== {label}: delta fitted on persistent arm = {dh} (phi 0.1 -> {min(.95, .1 + dh):.2f}, 0.3 -> {min(.95, .3 + dh):.2f}); "
              f"independent arm alone would pick {rec['delta_indep']}",
              f"  first-call agreement %: fit {100 * rec['fit_persistent']:.0f} (oracle {100 * rec['oracle_fit']:.0f}), independent {100 * rec['independent']:.0f} "
              f"(oracle {100 * rec['oracle_indep']:.0f}), randomised {100 * rec['randomised']:.0f}, repeats {100 * rec['repeats_majority']:.0f}",
              f"  calls {rec['calls']:.2f} (observed {rec['calls_obs']:.2f}), escalation {100 * rec['esc']:.0f}%, regret {rec['regret']:.2f} (observed {rec['regret_obs']:.2f})",
              "  agreement by delta: " + ", ".join(f"{dl}: {100 * v:.0f}" for dl, v in zip(DELTAS, rec["profile"].values()))]
    txt = "\n".join(L); print(txt)
    json.dump(out, open(P("results", "r_robust.json"), "w"), indent=1); open(P("results", "r_robust.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
