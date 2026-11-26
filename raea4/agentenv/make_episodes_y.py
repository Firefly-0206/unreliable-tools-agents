"""Episodes of the surface-randomisation check (PLAN_y_surface.md).

For every X episode at fault rates 0.10 and 0.30 (25 draws x 2 rates x 2 arms = 100), one copy with the SAME parameters,
hidden status and pre-drawn tool outputs, and a randomised surface:
  - the four tools are listed in a random order (prompt, cost line, function schemas);
  - they are renamed tool_a .. tool_d by a second, independent random permutation.
Tool descriptions, output fields and values, costs and documentation sentences are unchanged. One permutation pair per
parameter draw (shared by its four episodes, so the two arms stay paired); the list position of the replica tool is
balanced over draws (positions 1-4: 7, 6, 6, 6 draws). Tool indices in logs stay canonical (0 = replica tool).
Writes episodes_y.jsonl and ground_truth_y.jsonl. Usage: python make_episodes_y.py [--seed 2028]
"""
import argparse, json, os
import numpy as np
from env import HERE, ticket_id

PHIS = (0.10, 0.30)
LETTERS = ["tool_a", "tool_b", "tool_c", "tool_d"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=2028); a = ap.parse_args()
    X = [json.loads(l) for l in open(os.path.join(HERE, "episodes_x.jsonl"), encoding="utf-8")]
    T = {r["id"]: r for r in map(json.loads, open(os.path.join(HERE, "ground_truth_x.jsonl"), encoding="utf-8"))}
    draws = sorted({e["pair"] for e in X}); rng = np.random.default_rng(a.seed)
    slots = [k % 4 for k in range(len(draws))]; rng.shuffle(slots); surf = {}
    for d, slot in zip(draws, slots):
        rest = [int(x) for x in rng.permutation([1, 2, 3])]; order = rest[:slot] + [0] + rest[slot:]
        names = [LETTERS[int(i)] for i in rng.permutation(4)]          # names[j] = name of tool index j
        surf[d] = (order, names)
    eps, truth, seen = [], [], {e["ticket"] for e in X}
    for e in X:
        if not any(abs(e["phi"] - p) < 1e-9 for p in PHIS): continue
        order, names = surf[e["pair"]]; eid = "Y" + e["id"][1:]; salt = 0; t = ticket_id(eid)
        while t in seen: salt += 1; t = ticket_id(eid, salt)
        seen.add(t)
        eps.append(dict(e, id=eid, ticket=t, family="Y" + e["family"][1:], x_id=e["id"], base_scenario=e["scenario"],
                        scenario=f"{e['scenario']}~{e['pair']}", order=order, tool_names=names, replica_position=order.index(0) + 1))
        truth.append(dict(T[e["id"]], id=eid))
    for name, rows in (("episodes_y.jsonl", eps), ("ground_truth_y.jsonl", truth)):
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
    print(f"wrote {len(eps)} episodes; replica positions", [sum(s[0].index(0) == p for s in surf.values()) for p in range(4)])


if __name__ == "__main__":
    main()
