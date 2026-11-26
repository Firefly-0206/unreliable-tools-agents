"""Mechanism curves on fixed worked examples (no sampling). Writes results/mechanism.json."""
import json
import numpy as np
from sim import build, mmi, fault_unaware, split_true, loss_matrix, STUCK

def example(phi=0.3, q=0.9, shared=False):
    K = 3; femit = STUCK(K)
    if shared:   # A and B on one backend, C independent
        tools = [dict(kind="sym", q=q, cost=1, group="g", femit=femit), dict(kind="sym", q=q, cost=1, group="g", femit=femit),
                 dict(kind="sym", q=0.8, cost=3)]
    else:
        tools = [dict(kind="sym", q=q, cost=1, group="g", femit=femit), dict(kind="sym", q=0.8, cost=2)]
    return dict(family="EX", K=K, prior=np.array([0.5, 0.25, 0.25]), tools=tools, groups={"g": phi},
                loss=loss_matrix(K, 10, 30, 4), budget=6)

def ok_curve(sp, n_max=6, sequence=None):
    """P(benign) after n benign reports. sequence: tool index per step (default: always tool 0)."""
    mods = dict(true=build(sp), mmi=mmi(sp), unaware=fault_unaware(sp), split=split_true(sp))
    b = {k: m.prior.copy() for k, m in mods.items()}
    rows = [dict(n=0, **{k: float(mods[k].marg_s(v)[0]) for k, v in b.items()}, dedup=float(mods["true"].marg_s(b["true"])[0]))]
    first = None
    for n in range(1, n_max + 1):
        j = sequence[n - 1] if sequence else 0
        b = {k: mods[k].update(v, j, 0) for k, v in b.items()}
        if first is None: first = float(mods["true"].marg_s(b["true"])[0])
        rows.append(dict(n=n, tool="AB"[j] if sequence else "A", **{k: float(mods[k].marg_s(v)[0]) for k, v in b.items()},
                         dedup=first))
    return rows

if __name__ == "__main__":
    out = dict(tau=1 - 4 / 30,
               retry_phi03=ok_curve(example(0.3, 0.9)),
               retry_phi005_q075=ok_curve(example(0.05, 0.75)),
               shared_phi03=ok_curve(example(0.3, 0.9, shared=True), 4, [0, 1, 0, 1]))
    json.dump(out, open("results/mechanism.json", "w"), indent=1)
    for k in ["retry_phi03", "retry_phi005_q075", "shared_phi03"]:
        print(k); [print("  ", {a: (round(v, 3) if isinstance(v, float) else v) for a, v in r.items()}) for r in out[k]]
