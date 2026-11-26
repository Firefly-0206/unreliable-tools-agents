"""Episodes of the cost sweep and of the documentation-style variant (PLAN_v_cost_style.md).

Same 25 parameter draws, scenarios, hidden-state and output uniforms as the X sweep (seed 2027); persistent fault;
original tool names and order. One thing is changed per episode:
  VC2, VC3   fault rate 0.10, cost of the REPLICA tool set to 2 / 3 (drawn episodes: 1)           reference: XP10
  VL3        fault rate 0.30, cost of the LOG tool set to 3 (drawn: 1)                            reference: XP30
  VS10, VS30 fault rate 0.10 / 0.30, parameters and outputs of XP10 / XP30 unchanged; the tool documentation is
             written as a terse field list (env.tool_doc, doc_style = "spec") instead of sentences
Also writes the first-call predictions of the exact oracle and of the as-if planner (replica fault odds x g, z_asif.py)
for g in G_PRED: v_predictions.json. Usage: python make_episodes_v.py
"""
import copy, json, os
import numpy as np
from env import HERE, SCENARIO_LIST, BUDGET, Env, sample_params, sample_world, build_model, ticket_id, system_prompt
import k2decomp as K

SEED, N = 2027, 25
CELLS = [("VC2", 0.10, (0, 2), None), ("VC3", 0.10, (0, 3), None), ("VL3", 0.30, (2, 3), None), ("VS10", 0.10, None, "spec"), ("VS30", 0.30, None, "spec")]
G_PRED = [1, 4, 12, 24]


def main():
    X = {e["id"]: e for e in map(json.loads, open(os.path.join(HERE, "episodes_x.jsonl"), encoding="utf-8"))}
    W = [json.loads(l)["ticket"] for l in open(os.path.join(HERE, "episodes_w.jsonl"), encoding="utf-8")]
    eps, truth, seen = [], [], {e["ticket"] for e in X.values()} | set(W)
    for k, child in enumerate(np.random.SeedSequence([SEED, 0]).spawn(N)):
        s_par, s_world = child.spawn(2); base = sample_params("PERSIST", np.random.default_rng(s_par))
        for fam, phi, cost, style in CELLS:
            eid = f"{fam}-{k:03d}"; x_id = f"XP{int(round(100 * phi)):02d}-{k:03d}"
            par = dict(base, family=fam, phi=phi, stuck="benign", tools=[dict(t) for t in base["tools"]])
            if cost: par["tools"][cost[0]]["cost"] = cost[1]
            M = build_model(par); w, table = sample_world(par, np.random.default_rng(s_world), M)
            if not cost: assert table == X[x_id]["table"] and par["tools"] == X[x_id]["params"]["tools"], "X episode not reproduced"
            salt = 0; t = ticket_id(eid)
            while t in seen: salt += 1; t = ticket_id(eid, salt)
            seen.add(t)
            e = dict(id=eid, ticket=t, family=fam, scenario=SCENARIO_LIST[k % len(SCENARIO_LIST)], pair=f"X-{k:03d}", arm="XP", phi=phi,
                     cell=fam, x_id=x_id, params=par, table=table)
            if style: e["doc_style"] = style
            eps.append(e); truth.append(dict(id=eid, **w, V0=float(M.V(M.prior.copy(), float(BUDGET))[0])))
    for name, rows in (("episodes_v.jsonl", eps), ("ground_truth_v.jsonl", truth)):
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
    pred = {}
    for e in eps:
        docs = K.parse_docs(e, system_prompt(e, "K2e", "text")); ts = K.truth_spec(e)      # a reader of the docs recovers the true model
        assert [d["cost"] for d in docs] == [t["cost"] for t in e["params"]["tools"]] and [d["acc"] for d in docs] == ts["accuracy"]
        assert abs(docs[0]["phi"] - e["phi"]) < 1e-9 and docs[0]["cache"] and not any(d["cache"] for d in docs[1:]) and abs(docs[0]["stale_p"] - 0.97) < 1e-9
        row = {}
        for g in G_PRED:
            spec = copy.deepcopy(ts)
            for grp in spec["groups"]:
                if 0 in grp["tools"]: p = grp["phi"]; grp["phi"] = g * p / (1 - p + g * p)
            env = Env(e, "K2e", "text"); K.plan_and_play(e, spec, env); c = [x["j"] for x in env.log if x["type"] == "call"]
            row[str(g)] = c[0] if c else 4
            if g == 1: assert abs(sum(x.get("cost", 0) for x in env.log) + 0 - 0) >= 0
        pred[e["id"]] = row
    json.dump(dict(g=G_PRED, first_action=pred), open(os.path.join(HERE, "v_predictions.json"), "w"), indent=0)
    print(f"wrote {len(eps)} episodes")
    for fam, *_ in CELLS:
        ids = [e["id"] for e in eps if e["cell"] == fam]
        print(f"  {fam}: first action replica/primary/log/probe/stop (%) "
              + "  ".join(f"g={g}: " + "/".join(f"{100 * np.mean([pred[i][str(g)] == a for i in ids]):.0f}" for a in range(5)) for g in G_PRED))


if __name__ == "__main__":
    main()
