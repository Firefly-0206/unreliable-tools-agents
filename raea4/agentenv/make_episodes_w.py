"""Episodes of the accuracy sweep in which the optimal first query flips (PLAN_w_flip.md).

Same 25 parameter draws, scenarios, hidden-state and output uniforms as the X sweep (seed 2027); persistent fault;
original tool names and order. One accuracy is changed per episode:
  axis A  fault rate 0.30, accuracy of the LOG tool set to 0.55 / 0.65 / 0.75 / 0.85 / 0.95   (ids WA55-000 ...)
  axis B  fault rate 0.10, fresh accuracy of the REPLICA tool set to 0.75 / 0.65              (ids WB75-000 ...)
Tool outputs are re-drawn from the changed likelihoods with the X episode's uniforms (common random numbers).
Also writes the first-call predictions of the exact oracle and of the as-if planner (replica fault odds x g, z_asif.py)
for g in G_PRED: w_predictions.json. Usage: python make_episodes_w.py
"""
import copy, json, os
import numpy as np
from env import HERE, SCENARIO_LIST, BUDGET, sample_params, sample_world, build_model, ticket_id

SEED, N = 2027, 25
AXES = [("A", 0.30, 2, [0.55, 0.65, 0.75, 0.85, 0.95]), ("B", 0.10, 0, [0.75, 0.65])]   # axis, phi, tool index changed, levels
G_PRED = [1, 4, 12, 24]


def main():
    X = {e["id"]: e for e in map(json.loads, open(os.path.join(HERE, "episodes_x.jsonl"), encoding="utf-8"))}
    eps, truth, seen = [], [], {e["ticket"] for e in X.values()}
    for k, child in enumerate(np.random.SeedSequence([SEED, 0]).spawn(N)):
        s_par, s_world = child.spawn(2); base = sample_params("PERSIST", np.random.default_rng(s_par))
        chk = dict(base, family="XP30", phi=0.30, stuck="benign", tools=[dict(t) for t in base["tools"]])
        assert sample_world(chk, np.random.default_rng(s_world), build_model(chk))[1] == X[f"XP30-{k:03d}"]["table"], "X episode not reproduced"
        for axis, phi, j, levels in AXES:
            for q in levels:
                fam = f"W{axis}{int(round(100 * q)):02d}"; eid = f"{fam}-{k:03d}"
                par = dict(base, family=fam, phi=phi, stuck="benign", tools=[dict(t) for t in base["tools"]]); par["tools"][j]["q"] = q
                M = build_model(par); w, table = sample_world(par, np.random.default_rng(s_world), M)
                salt = 0; t = ticket_id(eid)
                while t in seen: salt += 1; t = ticket_id(eid, salt)
                seen.add(t)
                eps.append(dict(id=eid, ticket=t, family=fam, scenario=SCENARIO_LIST[k % len(SCENARIO_LIST)], pair=f"X-{k:03d}", arm="XP", phi=phi,
                                axis=axis, level=q, x_id=f"XP{int(round(100 * phi)):02d}-{k:03d}", params=par, table=table))
                truth.append(dict(id=eid, **w, V0=float(M.V(M.prior.copy(), float(BUDGET))[0])))
    for name, rows in (("episodes_w.jsonl", eps), ("ground_truth_w.jsonl", truth)):
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
    # predictions fixed before the runs
    from env import Env
    import k2decomp as K
    T = {r["id"]: r for r in truth}; pred = {}
    for e in eps:
        row = {}
        for g in G_PRED:
            spec = copy.deepcopy(K.truth_spec(e))
            for grp in spec["groups"]:
                if 0 in grp["tools"]: p = grp["phi"]; grp["phi"] = g * p / (1 - p + g * p)
            env = Env(e, "K2e", "text"); K.plan_and_play(e, spec, env); c = [x["j"] for x in env.log if x["type"] == "call"]
            row[str(g)] = c[0] if c else 4
        pred[e["id"]] = row
    json.dump(dict(g=G_PRED, first_action=pred), open(os.path.join(HERE, "w_predictions.json"), "w"), indent=0)
    print(f"wrote {len(eps)} episodes")
    for axis, phi, j, levels in AXES:
        for q in levels:
            ids = [e["id"] for e in eps if e["axis"] == axis and abs(e["level"] - q) < 1e-9]
            print(f"  axis {axis} level {q:.2f}: first action replica/primary/log/probe/stop (%) "
                  + "  ".join(f"g={g}: " + "/".join(f"{100 * np.mean([pred[i][str(g)] == a for i in ids]):.0f}" for a in range(5)) for g in G_PRED))


if __name__ == "__main__":
    main()
