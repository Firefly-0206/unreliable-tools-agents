"""Episodes of the fault-wording experiment (PLAN_t_wording.md).

Same 25 parameter draws, scenarios, hidden-state and output uniforms as the INDEPENDENT arm of the X sweep (seed 2027;
every replica call is stale independently with probability phi), at the fault rates 0.10 and 0.30, where the exact
oracle opens with the replica tool. Parameters, outputs and the exact oracle are identical in the three cells of a
draw; only the tool documentation differs:
  TS<phi>  sentences of the X sweep (XI arm), unchanged
  TD<phi>  every tool's report law as probabilities; the replica tool's law stated as fresh / stale snapshot
  TT<phi>  as TD, but the replica tool's law stated as ONE table of report probabilities by true status
           (the exact mixture of the fresh and stale laws); the documentation never mentions staleness or a fault
Checks done here: the X episode is reproduced; the table of TT is the per-call law of the true model to 1e-9.
Usage: python make_episodes_t.py -> episodes_t.jsonl, ground_truth_t.jsonl
"""
import json, os, re
import numpy as np
from env import HERE, SCENARIO_LIST, SCENARIOS, BUDGET, K, sample_params, sample_world, build_model, ticket_id, system_prompt, tool_doc
from oracle import qvalues

SEED, N = 2027, 25
PHIS = [0.10, 0.30]
CELLS = [("TS", None), ("TD", "tdecomp"), ("TT", "tmixed")]


def main():
    seen = set()
    for f in ("episodes_x.jsonl", "episodes_y.jsonl", "episodes_w.jsonl", "episodes_v.jsonl"):
        seen |= {json.loads(l)["ticket"] for l in open(os.path.join(HERE, f), encoding="utf-8")}
    X = {e["id"]: e for e in map(json.loads, open(os.path.join(HERE, "episodes_x.jsonl"), encoding="utf-8"))}
    eps, truth, first = [], [], {}
    for k, child in enumerate(np.random.SeedSequence([SEED, 0]).spawn(N)):
        s_par, s_world = child.spawn(2); base = sample_params("PERSIST", np.random.default_rng(s_par))
        for phi in PHIS:
            x_id = f"XI{int(round(100 * phi)):02d}-{k:03d}"
            par0 = dict(base, phi=phi, stuck="percall", tools=[dict(t) for t in base["tools"]])
            M = build_model(par0); w, table = sample_world(par0, np.random.default_rng(s_world), M)
            assert table == X[x_id]["table"] and par0["tools"] == X[x_id]["params"]["tools"] and par0["prior"] == X[x_id]["params"]["prior"], "X episode not reproduced"
            q = qvalues(M, M.prior.copy(), float(M.budget)); best = min(q, key=q.get)
            for tag, style in CELLS:
                fam = f"{tag}{int(round(100 * phi)):02d}"; eid = f"{fam}-{k:03d}"
                par = dict(par0, family=fam, tools=[dict(t) for t in par0["tools"]])
                salt = 0; t = ticket_id(eid)
                while t in seen: salt += 1; t = ticket_id(eid, salt)
                seen.add(t)
                e = dict(id=eid, ticket=t, family=fam, scenario=SCENARIO_LIST[k % len(SCENARIO_LIST)], pair=f"X-{k:03d}", arm=tag, phi=phi,
                         cell=fam, x_id=x_id, params=par, table=table)
                if style: e["doc_style"] = style
                eps.append(e); truth.append(dict(id=eid, **w, V0=float(M.V(M.prior.copy(), float(BUDGET))[0])))
                first[eid] = list(best)
                sp = system_prompt(e, "K2", "text")
                if tag == "TS":      # the X prompt, up to the ticket-free system prompt
                    assert sp == system_prompt(X[x_id], "K2", "text")
                if tag == "TT":
                    assert not re.search(r"stale|fault|fresh|snapshot|incident", sp, re.I), "TT prompt mentions a fault"
                    doc = tool_doc(e, 0); vals = SCENARIOS[e["scenario"]]["tools"][0]["values"]
                    for s in range(K):
                        row = doc.split(f"if the true status is {SCENARIOS[e['scenario']]['states'][s]}: ")[1].split(";")[0].rstrip(".")
                        p = [float(x.split()[-1]) for x in row.split(", ")]
                        i = M.H.index((s, ())) if (s, ()) in M.H else [h for h in range(M.nH) if M.s_of[h] == s][0]
                        assert np.allclose(p, M.lik[0][i], atol=1e-9) and abs(sum(p) - 1) < 1e-9, (eid, p, M.lik[0][i])
    for name, rows in (("episodes_t.jsonl", eps), ("ground_truth_t.jsonl", truth)):
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
    json.dump(first, open(os.path.join(HERE, "t_oracle_first.json"), "w"), indent=0)
    print(f"wrote {len(eps)} episodes")
    for phi in PHIS:
        ids = [e["id"] for e in eps if e["arm"] == "TT" and abs(e["phi"] - phi) < 1e-9]
        print(f"  phi={phi:.2f}: oracle opens with the replica tool in {100 * np.mean([first[i] == ['probe', 0] for i in ids]):.0f}% of draws")


if __name__ == "__main__":
    main()
