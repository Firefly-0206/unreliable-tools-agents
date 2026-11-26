"""
Exploratory, post hoc (no plan fixed in advance; no model call): can ONE parameter describe the under-use of the replica tool?

As-if model: the agent plans exactly, but on a model whose replica fault ODDS are multiplied by g >= 1
    phi' = g * phi / (1 - phi + g * phi)        (g = 1: the exact oracle; larger g: more pessimistic about the replica)
and everything else (prior, accuracies, costs, losses, budget, persistence or not) is the true model.
For each g on a grid the as-if planner is played in the true world of every X episode (same pre-drawn outputs).

Fit: for each run, g is chosen on the PERSISTENT arm of the fault-rate sweep (100 episodes, 4 fault rates) to minimise the
number of episodes whose first-call choice (replica tool or not) differs from the run's.
Checks on data not used for the fit: the independent arm (100 episodes), the randomised-surface episodes (Y, 100) and the
five-sample instances (20). Not fitted at all: the regret of the as-if planner against the run's regret.
Baselines: g = 1 (exact oracle) and "never opens with the replica tool".
Output: results/z_asif.json, results/z_asif.txt
"""
import os
os.environ.setdefault("RAEA_EPISODES", "episodes_x.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_x.jsonl")
import copy, json
import numpy as np
from env import HERE, Env, load_episodes, load_truth
import k2decomp as K
from x_analysis import parts, REP
from y_analysis import load, RUNS, B_SAMPLES

GRID = [1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
PHIS = [0.10, 0.30, 0.50, 0.70]
P = lambda *a: os.path.join(HERE, *a)


def asif(ep, truth, g):
    spec = copy.deepcopy(K.truth_spec(ep))
    for grp in spec["groups"]:
        if REP in grp["tools"]: phi = grp["phi"]; grp["phi"] = g * phi / (1 - phi + g * phi)
    env = Env(ep, "K2e", "text"); K.plan_and_play(ep, spec, env); return parts(ep, env.log, truth)


def main():
    EX, TX = load_episodes(P("episodes_x.jsonl")), load_truth(P("ground_truth_x.jsonl"))
    EY = load_episodes(P("episodes_y.jsonl"))
    ids = sorted(EX); pred = {g: {i: asif(EX[i], TX[i], g) for i in ids} for g in GRID}
    arm = lambda a: [i for i in ids if EX[i]["arm"] == a]
    cell = lambda a, p: [i for i in ids if EX[i]["arm"] == a and abs(EX[i]["phi"] - p) < 1e-9]
    out, L = dict(grid=GRID), []
    L.append("as-if planner, share of episodes opening with the replica tool (persistent | independent; fault rate 0.1 0.3 0.5 0.7)")
    for g in GRID:
        L.append(f"  g={g:<5} " + " ".join(f"{100 * np.mean([pred[g][i]['first'] for i in cell('XP', p)]):4.0f}" for p in PHIS) + "  | "
                 + " ".join(f"{100 * np.mean([pred[g][i]['first'] for i in cell('XI', p)]):4.0f}" for p in PHIS))
    out["asif_first"] = {str(g): {f"{a}|{p:.2f}": float(np.mean([pred[g][i]["first"] for i in cell(a, p)])) for a in ("XP", "XI") for p in PHIS} for g in GRID}
    for key, (label, stem) in RUNS.items():
        X = load(P("logs", f"x_{stem}_k2e.jsonl"), [0]); Y = load(P("logs", f"y_{stem}_k2e.jsonl"), [0]); R = dict(X); R.update(load(P("logs", f"xrep_{stem}_k2e.jsonl"), B_SAMPLES[1:]))
        obs = {i: parts(EX[i], X[(i, 0)]["log"], TX[i]) for i in ids}
        match = lambda g, sel: float(np.mean([pred[g][i]["first"] == obs[i]["first"] for i in sel]))
        prof = {g: match(g, arm("XP")) for g in GRID}; g_hat = max(GRID, key=lambda g: (prof[g], -g))
        never = lambda sel, o=obs: float(np.mean([o[i]["first"] == 0 for i in sel]))
        # randomised surface: same parameters as the X episode x_id, model's choice from the Y run
        yobs = {EY[j]["x_id"]: float(bool([x["j"] for x in Y[(j, 0)]["log"] if x["type"] == "call"][:1] == [REP])) for j in EY}
        ym = lambda g: float(np.mean([pred[g][i]["first"] == yobs[i] for i in yobs]))
        # repeated generation: share of the five samples opening with the replica tool, against the as-if prediction
        bids = sorted({i for (i, s) in R if s > 0}); share = {i: float(np.mean([[x["j"] for x in R[(i, s)]["log"] if x["type"] == "call"][:1] == [REP] for s in B_SAMPLES])) for i in bids}
        o = dict(label=label, g_hat=g_hat, profile={str(g): prof[g] for g in GRID},
                 match=dict(fit_persistent=prof[g_hat], independent=match(g_hat, arm("XI")), randomised=ym(g_hat),
                            repeats_majority=float(np.mean([(share[i] > 0.5) == bool(pred[g_hat][i]["first"]) for i in bids]))),
                 oracle=dict(fit_persistent=prof[1], independent=match(1, arm("XI")), randomised=ym(1),
                             repeats_majority=float(np.mean([(share[i] > 0.5) == bool(pred[1][i]["first"]) for i in bids]))),
                 never=dict(fit_persistent=never(arm("XP")), independent=never(arm("XI")), randomised=float(np.mean([v == 0 for v in yobs.values()])),
                            repeats_majority=float(np.mean([share[i] <= 0.5 for i in bids]))),
                 cells={})
        for a in ("XP", "XI"):
            for p in PHIS:
                c = cell(a, p)
                o["cells"][f"{a}|{p:.2f}"] = dict(first_obs=float(np.mean([obs[i]["first"] for i in c])), first_asif=float(np.mean([pred[g_hat][i]["first"] for i in c])),
                                                  regret_obs=float(np.mean([obs[i]["regret"] for i in c])), regret_asif=float(np.mean([pred[g_hat][i]["regret"] for i in c])),
                                                  calls_obs=float(np.mean([obs[i]["calls"] for i in c])), calls_asif=float(np.mean([pred[g_hat][i]["calls"] for i in c])),
                                                  rep_calls_obs=float(np.mean([obs[i]["rep_calls"] for i in c])), rep_calls_asif=float(np.mean([pred[g_hat][i]["rep_calls"] for i in c])))
        o["g_hat_independent"] = max(GRID, key=lambda g: (match(g, arm("XI")), -g)); o["g_hat_both"] = max(GRID, key=lambda g: (match(g, ids), -g))
        o["regret_obs"] = float(np.mean([obs[i]["regret"] for i in ids])); o["regret_asif"] = float(np.mean([pred[g_hat][i]["regret"] for i in ids]))
        out[key] = o
        L += ["", f"== {label}: g fitted on the persistent arm = {g_hat}  (independent arm alone: {o['g_hat_independent']}; both arms: {o['g_hat_both']})",
              "  match rate by g (persistent arm): " + ", ".join(f"{g}: {100 * prof[g]:.0f}" for g in GRID),
              f"  first-call match %      fit(persistent)  independent  randomised  repeats(majority)",
              *(f"  {n:22s}{100 * d['fit_persistent']:12.0f}{100 * d['independent']:13.0f}{100 * d['randomised']:12.0f}{100 * d['repeats_majority']:14.0f}"
                for n, d in (("as-if, fitted g", o["match"]), ("exact oracle (g=1)", o["oracle"]), ("never the replica", o["never"]))),
              "  cell       first obs / as-if (%)   regret obs / as-if   calls obs / as-if   replica calls obs / as-if"]
        for cn, c in o["cells"].items():
            L.append(f"  {cn:9s}{100 * c['first_obs']:8.0f} /{100 * c['first_asif']:4.0f}{c['regret_obs']:16.2f} /{c['regret_asif']:5.2f}{c['calls_obs']:12.2f} /{c['calls_asif']:5.2f}{c['rep_calls_obs']:14.2f} /{c['rep_calls_asif']:5.2f}")
        L.append(f"  regret, all 200 episodes: observed {o['regret_obs']:.2f}, as-if {o['regret_asif']:.2f}")
    txt = "\n".join(L); print(txt)
    json.dump(out, open(P("results", "z_asif.json"), "w"), indent=1); open(P("results", "z_asif.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
