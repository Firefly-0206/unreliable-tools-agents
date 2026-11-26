"""Structured-elicitation diagnostic (new experiment, not a re-reading of the first run).
For each plain item, the model is asked, BEFORE choosing, for every intermediate quantity of the decision:
full posterior, P(A's backend stale), predictive distribution of the next report from A and from V,
posterior after each possible report, expected total cost of every action, and the action.
Writes diag/elicit/elicit_items.json with exact targets for all of them."""
import json, os
from sim import Model
from diag_items import tools_for, EVIDENCE, REGIMES, PRIOR, LOSS, BUDGET, K, STATES, prompt

def exact_all(regime, pattern):
    phi, q = REGIMES[regime]["phi"], REGIMES[regime]["q"]
    names, tl, groups = tools_for(pattern, phi, q)
    m = Model(K, PRIOR, tl, groups, LOSS, BUDGET)
    b = m.prior.copy()
    for t, o in EVIDENCE[pattern]: b = m.update(b, names.index(t), o)
    gA = tl[0]["group"]; k = m.G.index(gA)
    p_stale = float(sum(b[i] for i, (s, z) in enumerate(m.H) if z[k] == 1))
    d = lambda v: {STATES[i]: float(v[i]) for i in range(K)}
    out = dict(posterior=d(m.marg_s(b)), p_stale_A=p_stale, predict_next={}, posterior_after={}, expected_cost={})
    for nm in ["A", "V"]:
        j = names.index(nm); pr = m.pred(b, j)
        out["predict_next"][f"query_{nm}"] = d(pr)
        out["posterior_after"][f"query_{nm}"] = {STATES[o]: d(m.marg_s(m.update(b, j, o))) for o in range(K)}
    ps = m.marg_s(b)
    for a, s in enumerate(STATES): out["expected_cost"][f"declare_{s}"] = float(LOSS[a] @ ps)
    out["expected_cost"]["escalate"] = float(LOSS[K] @ ps)
    for j, nm in enumerate(names):
        c = tl[j]["cost"]
        out["expected_cost"][f"query_{nm}"] = float(c + sum(p * m.V(m.update(b, j, o), BUDGET - c)[0]
                                                         for o, p in enumerate(m.pred(b, j)) if p > 1e-12))
    out["best_action"] = min(out["expected_cost"], key=out["expected_cost"].get)
    return out, names

ASK = """
Before choosing, write out your calculation, then report every quantity below. Report what you actually compute; do not round to make the numbers agree with your chosen action.
(a) posterior: your probability of each state given the observations so far.
(b) p_stale_A: your probability that tool A's backend is stale, given the observations so far.
(c) predict_next: for query_A and for query_V, the probability that the NEXT report from that tool is healthy / degraded / down.
(d) posterior_after: for query_A and for query_V, and for each possible next report, your posterior over the three states after seeing that report.
(e) expected_cost: for every action listed above, the expected total cost from now on if you take that action now and then act optimally within the remaining budget (include the query cost for queries).
(f) action: the single next action you choose.

End your answer with ONE line of JSON in exactly this shape (numbers are placeholders):
{"posterior": {"healthy": 0.0, "degraded": 0.0, "down": 0.0}, "p_stale_A": 0.0, "predict_next": {"query_A": {"healthy": 0.0, "degraded": 0.0, "down": 0.0}, "query_V": {"healthy": 0.0, "degraded": 0.0, "down": 0.0}}, "posterior_after": {"query_A": {"healthy": {"healthy": 0.0, "degraded": 0.0, "down": 0.0}, "degraded": {"healthy": 0.0, "degraded": 0.0, "down": 0.0}, "down": {"healthy": 0.0, "degraded": 0.0, "down": 0.0}}, "query_V": {"healthy": {"healthy": 0.0, "degraded": 0.0, "down": 0.0}, "degraded": {"healthy": 0.0, "degraded": 0.0, "down": 0.0}, "down": {"healthy": 0.0, "degraded": 0.0, "down": 0.0}}}, "expected_cost": {ACTIONS}, "action": "escalate"}"""

def elicit_prompt(regime, pattern, names):
    base = prompt(pattern, regime, "plain")
    base = base[:base.index("Give (1) your probability")]          # replace the original question
    acts = ["declare_healthy", "declare_degraded", "declare_down", "escalate"] + [f"query_{n}" for n in names]
    head = f"Possible actions now: {', '.join(acts)}.\n"
    return base + head + ASK.replace("{ACTIONS}", ", ".join(f'"{a}": 0.0' for a in acts).join("{}"))

if __name__ == "__main__":
    os.makedirs("diag/elicit", exist_ok=True)
    items = []
    for regime in REGIMES:
        for pattern in EVIDENCE:
            ex, names = exact_all(regime, pattern)
            items.append(dict(id=f"{regime}|{pattern}", regime=regime, pattern=pattern, exact=ex,
                              prompt=elicit_prompt(regime, pattern, names)))
    json.dump(items, open("diag/elicit/elicit_items.json", "w"), indent=1)
    print(len(items), "items")
    for it in items:
        e = it["exact"]
        print(f"{it['id']:22s} P(H)={e['posterior']['healthy']:.3f} P(stale)={e['p_stale_A']:.3f} "
              f"P(H|A=ok)={e['posterior_after']['query_A']['healthy']['healthy']:.3f} best={e['best_action']}")
