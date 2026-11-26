"""Exact reference quantities for the re-query cases (plain presentation, correct-ish P(healthy), suboptimal re-query)."""
import json, numpy as np
from sim import Model
from diag_items import tools_for, EVIDENCE, REGIMES, PRIOR, LOSS, BUDGET, K, STATES
items = {it["id"]: it for it in json.load(open("diag/diag_items.json"))}
rows = [json.loads(l) for l in open("diag/results.jsonl", encoding="utf-8")]
latest = {}
for r in rows:
    if r.get("ok") and r.get("p_healthy") is not None and r.get("action"): latest[(r["id"], r["model"], r["sample"])] = r

def exact_detail(regime, pattern):
    phi, q = REGIMES[regime]["phi"], REGIMES[regime]["q"]
    names, tl, groups = tools_for(pattern, phi, q)
    m = Model(K, PRIOR, tl, groups, LOSS, BUDGET)
    b = m.prior.copy()
    for t, o in EVIDENCE[pattern]: b = m.update(b, names.index(t), o)
    ps = m.marg_s(b)
    # posterior that the source used so far is stale (first group)
    p_stale = float(sum(b[i] for i, (s, z) in enumerate(m.H) if len(z) and z[0] == 1)) if m.G else 0.0
    pred = {f"query_{nm}": [round(float(x), 4) for x in m.pred(b, j)] for j, nm in enumerate(names)}
    # what an agent would predict for the next same-source report if it ignored persistence (independent mixture)
    j = 0; marg_lik = np.array([(1 - phi) * np.r_[[q if o == s else (1 - q) / 2 for o in range(K)]] + phi * tl[0]["femit"] for s in range(K)])
    pred_indep = [round(float(x), 4) for x in ps @ marg_lik]
    return dict(post=[round(float(x), 4) for x in ps], p_stale=round(p_stale, 4), pred=pred, pred_same_if_independent=pred_indep)

cases = []
for (iid, model, smp), r in sorted(latest.items()):
    it = items[iid]
    if it["presentation"] != "plain": continue
    d = it["Q"][r["action"]] - it["best_value"]
    if d > 1e-9 and r["action"] in ("query_A", "query_B", "query_C") and abs(r["p_healthy"] - it["p_exact"]) <= 0.02:
        ex = exact_detail(it["regime"], it["pattern"])
        cases.append(dict(id=iid, model=model, sample=smp, p_model=r["p_healthy"], action=r["action"],
                          best=it["best_action"], loss=round(d, 3), Q={k: round(v, 3) for k, v in it["Q"].items()}, **ex))
json.dump(cases, open("diag/requery_cases.json", "w"), indent=1)
for c in cases:
    print(f"{c['model']:18s} {c['id']:24s} s{c['sample']} chose {c['action']} (best {c['best']}, loss {c['loss']})")
    print(f"   exact posterior H/D/Dn {c['post']}  P(source stale) {c['p_stale']}")
    print(f"   exact predictive next A {c['pred']['query_A']}  next V {c['pred']['query_V']}  | if A treated as fresh independent call {c['pred_same_if_independent']}")
    print(f"   exact Q {c['Q']}")
