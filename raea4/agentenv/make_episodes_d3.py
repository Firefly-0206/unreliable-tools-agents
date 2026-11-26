"""Episodes of the third task domain (PLAN_d3_domain.md; env3_data.py): fact-checking a claim with two checkable
parts (four statuses), six tools of which two answer one yes/no sub-question each, two outlets sharing one wire item
(persistent shared fault, reports 'accurate'), a fact-check aggregator with its own outage (arbitrary reports), budget 7.
Generated from make_episodes_d2.py by substitution; ids keep the DN / DX family labels.

  DN   100 parameter draws with the replica fault rate drawn from its range (0.15-0.40)        ids DN-000 .. DN-099
  DX   25 further draws x replica fault rate 0.10 / 0.30 / 0.50 / 0.70 (same draw, same uniforms) ids DX10-100 ..
Writes episodes_d3.jsonl, ground_truth_d3.jsonl (hidden) and d3_oracle.json (the exact oracle's first action and value
per episode, computed before any model call). Usage: python make_episodes_d2.py [--workers 4]
"""
import os
os.environ["RAEA_DOMAIN"] = "3"
import argparse, json
from multiprocessing import Pool
import numpy as np
from env import HERE, SCENARIO_LIST, BUDGET, sample_params, sample_world, build_model, ticket_id

SEED, N_NAT, N_CURVE = 2031, 100, 25
PHIS = [0.10, 0.30, 0.50, 0.70]


def solve(ep):
    from oracle import qvalues
    M = build_model(ep["params"]); q = qvalues(M, M.prior.copy(), float(M.budget)); a = min(q, key=q.get)
    return ep["id"], dict(first=a[1] if a[0] == "probe" else len(ep["params"]["tools"]), V0=float(q[a]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    eps, truth, seen = [], [], set()
    for k, child in enumerate(np.random.SeedSequence([SEED, 0]).spawn(N_NAT + N_CURVE)):
        s_par, s_world = child.spawn(2); base = sample_params("D3", np.random.default_rng(s_par))
        variants = [("DN", base["phi"])] if k < N_NAT else [(f"DX{int(round(100 * p)):02d}", p) for p in PHIS]
        for fam, phi in variants:
            eid = f"{fam}-{k:03d}"; par = dict(base, family=fam, phi=phi, tools=[dict(t) for t in base["tools"]])
            M = build_model(par); w, table = sample_world(par, np.random.default_rng(s_world), M); M.cache.clear()
            salt = 0; t = ticket_id(eid)
            while t in seen: salt += 1; t = ticket_id(eid, salt)
            seen.add(t)
            eps.append(dict(id=eid, ticket=t, family=fam, scenario=SCENARIO_LIST[k % len(SCENARIO_LIST)], pair=f"E-{k:03d}",
                            part="DN" if k < N_NAT else "DX", phi=phi, params=par, table=table))
            truth.append(dict(id=eid, **w))
    with Pool(a.workers) as pool: orc = dict(pool.imap_unordered(solve, eps, chunksize=2))
    for t in truth: t["V0"] = orc[t["id"]]["V0"]
    for name, rows in (("episodes_d3.jsonl", eps), ("ground_truth_d3.jsonl", truth)):
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
    json.dump(orc, open(os.path.join(HERE, "d3_oracle.json"), "w"), indent=0)
    print(f"wrote {len(eps)} episodes")
    names = ["outlet A", "outlet B", "figure chk", "source chk", "aggregator", "office", "submit"]
    for label, ids in [("DN", [e["id"] for e in eps if e["part"] == "DN"])] + [(f"DX phi={p:.2f}", [e["id"] for e in eps if e["part"] == "DX" and abs(e["phi"] - p) < 1e-9]) for p in PHIS]:
        print(f"  {label:13s} oracle first action (%): " + ", ".join(f"{n} {100 * np.mean([orc[i]['first'] == j for i in ids]):.0f}" for j, n in enumerate(names))
              + f";  oracle loss {np.mean([orc[i]['V0'] for i in ids]):.2f}")


if __name__ == "__main__":
    main()
