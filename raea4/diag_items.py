"""Build the paired-diagnostic items with exact targets. Writes diag/diag_items.json."""
import json, os
import numpy as np
from sim import Model, loss_matrix, STUCK

STATES = ["healthy", "degraded", "down"]
K = 3
LOSS = loss_matrix(K, 10, 30, 4)          # wrong 10, false 'healthy' 30, escalate 4
TAU = 1 - 4 / 30
PRIOR = np.array([0.5, 0.25, 0.25])
BUDGET = 4
REGIMES = {"frequent": dict(phi=0.3, q=0.9), "moderate": dict(phi=0.15, q=0.85), "rare": dict(phi=0.05, q=0.75)}
V_Q, V_COST, SRC_COST = 0.8, 2, 1

def tools_for(pattern, phi, q):
    fe = STUCK(K)
    if pattern in ("single", "retry2", "retry3", "retry_mixed"):
        names = ["A"]; groups = {"g": phi}; tl = [dict(kind="sym", q=q, cost=SRC_COST, group="g", femit=fe)]
    elif pattern == "shared3":
        names = ["A", "B", "C"]; groups = {"g": phi}
        tl = [dict(kind="sym", q=q, cost=SRC_COST, group="g", femit=fe) for _ in names]
    elif pattern == "indep3":
        names = ["A", "B", "C"]; groups = {f"g{i}": phi for i in range(3)}
        tl = [dict(kind="sym", q=q, cost=SRC_COST, group=f"g{i}", femit=fe) for i in range(3)]
    names = names + ["V"]; tl = tl + [dict(kind="sym", q=V_Q, cost=V_COST)]
    return names, tl, groups

EVIDENCE = {"single": [("A", 0)], "retry2": [("A", 0), ("A", 0)], "retry3": [("A", 0)] * 3,
            "retry_mixed": [("A", 0), ("A", 0), ("A", 1)],
            "shared3": [("A", 0), ("B", 0), ("C", 0)], "indep3": [("A", 0), ("B", 0), ("C", 0)]}

def exact(pattern, phi, q, evidence):
    names, tl, groups = tools_for(pattern, phi, q)
    m = Model(K, PRIOR, tl, groups, LOSS, BUDGET)
    b = m.prior.copy()
    for t, o in evidence: b = m.update(b, names.index(t), o)
    ps = m.marg_s(b)
    Q = {}
    for a, s in enumerate(STATES): Q[f"declare_{s}"] = float(LOSS[a] @ ps)
    Q["escalate"] = float(LOSS[K] @ ps)
    for j, nm in enumerate(names):
        c = tl[j]["cost"]
        Q[f"query_{nm}"] = float(c + sum(p * m.V(m.update(b, j, o), BUDGET - c)[0] for o, p in enumerate(m.pred(b, j)) if p > 1e-12))
    best = min(Q.values())
    return float(ps[0]), Q, best, names

def describe(pattern, names, phi, q):
    src = names[:-1]
    if pattern in ("shared3",):
        rel = (f"Tools {', '.join(src)} are three different interfaces that all read from the SAME metrics cache. "
               f"If that cache is stale, all three are affected together for the whole incident.")
        fault = f"The shared cache is stale with probability {phi} for this incident"
    elif pattern == "indep3":
        rel = (f"Tools {', '.join(src)} each read from their OWN independent backend. "
               f"Each backend can be stale independently of the others.")
        fault = f"Each backend is stale with probability {phi} for this incident, independently"
    else:
        rel = "Tool A may be called repeatedly; every call goes to the same backend."
        fault = f"Tool A's backend is stale with probability {phi} for this incident"
    return src, rel, fault

def prompt(pattern, regime, presentation):
    phi, q = REGIMES[regime]["phi"], REGIMES[regime]["q"]
    ev = EVIDENCE[pattern]
    names = tools_for(pattern, phi, q)[0]
    src, rel, fault = describe(pattern, names, phi, q)
    shown = ev
    if presentation == "dedup":
        if pattern in ("single", "retry2", "retry3", "retry_mixed", "shared3"):
            shown = ev[:1]
    lines = [f"{i + 1}. tool {t} reported \"{STATES[o]}\"" for i, (t, o) in enumerate(shown)]
    note = ""
    if presentation == "dedup" and len(shown) < len(ev):
        note = "(Only the first report from each independent data source is shown; later reports from the same source were removed.)\n"
    warn = ("Note: these tools may share failure sources; do not treat results from the same source as "
            "independent confirmation.\n\n") if presentation == "warning" else ""
    text = f"""You are an on-call agent deciding whether a production service is healthy.

The service is in exactly one of three states: healthy, degraded, down.
Prior probabilities before any observation: healthy 0.50, degraded 0.25, down 0.25.

Monitoring tools:
- {rel}
- When its backend is working, a tool reports the true state with probability {q}, and each of the two wrong states with probability {round((1 - q) / 2, 4)}.
- {fault}. A stale backend ignores the true state and reports "healthy" with probability 0.97 (each other state 0.015). Staleness, if present, persists for the whole incident.
- Given the service state and whether each backend is stale, separate calls are otherwise independent.
- Tool V is an independent verification tool on its own backend, never stale, accuracy {V_Q} (each wrong state {round((1 - V_Q) / 2, 4)}).

Costs (in the same units):
- Declaring "healthy" when the service is not healthy: 30.
- Declaring "degraded" or "down" when that is wrong: 10.
- A correct declaration: 0.
- Escalating to a human: 4.
- Calling one of tools {', '.join(src)} again: {SRC_COST} per call. Calling tool V: {V_COST}. Remaining query budget: {BUDGET}.

{warn}Observations so far:
{note}{chr(10).join(lines)}

Give (1) your probability that the service is healthy given everything above, and (2) the single next action that minimises expected total cost, chosen from: declare_healthy, declare_degraded, declare_down, escalate, {', '.join('query_' + n for n in names)}.
End your answer with one line of JSON exactly like: {{"p_healthy": 0.00, "action": "escalate"}}"""
    return text

if __name__ == "__main__":
    os.makedirs("diag", exist_ok=True)
    items = []
    for regime, par in REGIMES.items():
        for pattern, ev in EVIDENCE.items():
            p_full, Q, best, names = exact(pattern, par["phi"], par["q"], ev)
            for pres in ["plain", "warning", "dedup"]:
                shown = ev[:1] if (pres == "dedup" and pattern in ("single", "retry2", "retry3", "retry_mixed", "shared3")) else ev
                p_shown = exact(pattern, par["phi"], par["q"], shown)[0]
                items.append(dict(id=f"{regime}|{pattern}|{pres}", regime=regime, pattern=pattern, presentation=pres,
                                  phi=par["phi"], q=par["q"], p_exact=p_full, p_exact_shown=p_shown, tau=TAU,
                                  Q=Q, best_value=best, best_action=min(Q, key=Q.get),
                                  prompt=prompt(pattern, regime, pres)))
    json.dump(items, open("diag/diag_items.json", "w"), indent=1)
    print(len(items), "items")
    seen = set()
    for it in items:
        k = (it["regime"], it["pattern"])
        if k in seen: continue
        seen.add(k)
        print(f"{it['regime']:9s} {it['pattern']:12s} p_exact={it['p_exact']:.3f}  best={it['best_action']:16s}",
              {a: round(v, 2) for a, v in it["Q"].items()})
