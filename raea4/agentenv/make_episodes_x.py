"""Episodes of the matched-marginal dependence experiment (PLAN_x_dependence.md).

For each of N parameter draws (PERSIST structure: one replica tool, a primary-database tool, a log tool, the live probe)
and each fault rate phi in PHIS, two arms with IDENTICAL single-call laws:
  XP<phi>  persistent fault: the replica is stale for the whole session with probability phi (as in PERSIST)
  XI<phi>  independent fault: every replica call is stale independently with probability phi
Same prior, accuracies, costs, scenario, hidden status and uniforms within a draw (common random numbers), so the arms
differ only in whether repeated replica reports are dependent. Writes episodes_x.jsonl and ground_truth_x.jsonl
(hidden; read by code only). Usage: python make_episodes_x.py [--n 25] [--seed 2027]
"""
import argparse, json, os
import numpy as np
from env import HERE, SCENARIO_LIST, BUDGET, sample_params, sample_world, build_model, ticket_id

PHIS = [0.10, 0.30, 0.50, 0.70]
ARMS = [("XP", "benign"), ("XI", "percall")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=25); ap.add_argument("--seed", type=int, default=2027)
    a = ap.parse_args(); eps, truth, seen = [], [], set()
    for k, child in enumerate(np.random.SeedSequence([a.seed, 0]).spawn(a.n)):
        s_par, s_world = child.spawn(2)
        base = sample_params("PERSIST", np.random.default_rng(s_par))
        for phi in PHIS:
            for tag, stuck in ARMS:
                fam = f"{tag}{int(round(100 * phi)):02d}"; eid = f"{fam}-{k:03d}"
                par = dict(base, family=fam, phi=phi, stuck=stuck, tools=[dict(t) for t in base["tools"]])
                M = build_model(par); w, table = sample_world(par, np.random.default_rng(s_world), M)
                salt = 0; t = ticket_id(eid)
                while t in seen: salt += 1; t = ticket_id(eid, salt)
                seen.add(t)
                eps.append(dict(id=eid, ticket=t, family=fam, scenario=SCENARIO_LIST[k % len(SCENARIO_LIST)], pair=f"X-{k:03d}",
                                arm=tag, phi=phi, params=par, table=table))
                truth.append(dict(id=eid, **w, V0=float(M.V(M.prior.copy(), float(BUDGET))[0])))
    for name, rows in (("episodes_x.jsonl", eps), ("ground_truth_x.jsonl", truth)):
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
    print(f"wrote {len(eps)} episodes")


if __name__ == "__main__":
    main()
