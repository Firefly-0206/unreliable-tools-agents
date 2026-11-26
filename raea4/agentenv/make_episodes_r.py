"""Episodes of the runbook case study (PLAN_r_runbook.md).

Same 25 parameter draws, scenarios, hidden statuses and output uniforms as the PERSISTENT arm of the X sweep (seed
2027) at fault rates 0.10, 0.30 and 0.70. Parameters, tool outputs and the exact oracle are those of the X episode;
only the surface differs:
  doc_style = "runbook"  every tool documented as a runbook entry; rates given as counts (env.runbook_doc)
  out_style = "raw"      tool results as service responses with fields that do not bear on the decision (env.raw_output)
Checks done here: the X episode is reproduced; the counts in the runbook text, read back by regular expressions, give
exactly the true fault rate, stale emission and accuracies; the planner on the true model has regret 0 by construction.
Usage: python make_episodes_r.py -> episodes_r.jsonl, ground_truth_r.jsonl
"""
import json, os, re
import numpy as np
from env import HERE, SCENARIO_LIST, BUDGET, sample_params, sample_world, build_model, ticket_id, system_prompt, tool_doc, Env

SEED, N = 2027, 25
PHIS = [0.10, 0.30, 0.70]


def main():
    seen = set()
    for f in ("episodes_x.jsonl", "episodes_y.jsonl", "episodes_w.jsonl", "episodes_v.jsonl", "episodes_t.jsonl"):
        seen |= {json.loads(l)["ticket"] for l in open(os.path.join(HERE, f), encoding="utf-8")}
    X = {e["id"]: e for e in map(json.loads, open(os.path.join(HERE, "episodes_x.jsonl"), encoding="utf-8"))}
    eps, truth, words = [], [], []
    for k, child in enumerate(np.random.SeedSequence([SEED, 0]).spawn(N)):
        s_par, s_world = child.spawn(2); base = sample_params("PERSIST", np.random.default_rng(s_par))
        for phi in PHIS:
            fam = f"RB{int(round(100 * phi)):02d}"; eid = f"{fam}-{k:03d}"; x_id = f"XP{int(round(100 * phi)):02d}-{k:03d}"
            par = dict(base, family=fam, phi=phi, stuck="benign", tools=[dict(t) for t in base["tools"]])
            M = build_model(par); w, table = sample_world(par, np.random.default_rng(s_world), M)
            assert table == X[x_id]["table"] and par["tools"] == X[x_id]["params"]["tools"] and par["prior"] == X[x_id]["params"]["prior"], "X episode not reproduced"
            salt = 0; t = ticket_id(eid)
            while t in seen: salt += 1; t = ticket_id(eid, salt)
            seen.add(t)
            e = dict(id=eid, ticket=t, family=fam, scenario=SCENARIO_LIST[k % len(SCENARIO_LIST)], pair=f"X-{k:03d}", arm="RB", phi=phi,
                     cell=fam, x_id=x_id, params=par, table=table, doc_style="runbook", out_style="raw")
            eps.append(e); truth.append(dict(id=eid, **w, V0=float(M.V(M.prior.copy(), float(BUDGET))[0])))
            # read the counts back from the text
            d0 = tool_doc(e, 0)
            a, b = map(int, re.search(r"in (\d+) of the last (\d+) on-call sessions the replica was not refreshing", d0).groups()); assert abs(a / b - phi) < 1e-12
            a, b = map(int, re.search(r"of (\d+) such answers sampled, (\d+) were", d0).groups()); assert abs(b / a - 0.97) < 1e-12
            for j in range(4):
                dj = tool_doc(e, j); a, b = map(int, re.search(r"(\d+) of (\d+) answers matched the real status", dj).groups())
                assert abs(a / b - par["tools"][j]["q"]) < 1e-12 and f"Cost: {par['tools'][j]['cost']}." in dj
                assert ("replica" in dj.split("Data path:")[1].split("Spot checks")[0] and j == 0) or j > 0
            sp = system_prompt(e, "K2e", "text"); words.append(len(sp.split()))
            env = Env(e, "K2e", "text"); out = json.loads(env.call(env.names[0]).split("\n")[0])
            assert out["result"][env.sc["tools"][0]["field"]] == env.sc["tools"][0]["values"][table[0][0]] and out["budget_remaining"] == BUDGET - par["tools"][0]["cost"]
    for name, rows in (("episodes_r.jsonl", eps), ("ground_truth_r.jsonl", truth)):
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
    xw = [len(system_prompt(X[e["x_id"]], "K2e", "text").split()) for e in eps]
    print(f"wrote {len(eps)} episodes; system prompt words: runbook {np.mean(words):.0f}, original {np.mean(xw):.0f}")


if __name__ == "__main__":
    main()
