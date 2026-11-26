"""Oracle-information intervention (mechanism localisation, not a deployable method).

Same 18 tasks (3 regimes x 6 evidence patterns, plain presentation), seven prompt conditions:
  L0  no added information (baseline; same prompt as the first run)
  L1  + exact joint posterior P(state, stale flags | history)
  L2  + L1, exact predictive distribution of every query and the exact posterior after each possible report
  L3  + L2, exact expected total cost of every action under the same budget
  C1, C2, C3  structure-matched controls: the same blocks with the same layout and wording,
              every number replaced by "?", and the instruction to fill them in.
Tasks are fixed and not filtered by earlier outcomes, so correct answers, wrong stops and wrong
switches are all kept. Writes diag/interv/interv_items.json with the exact action values per task."""
import json, os
from sim import Model
from diag_items import tools_for, EVIDENCE, REGIMES, PRIOR, LOSS, BUDGET, K, STATES, prompt, exact

def model_for(regime, pattern):
    phi, q = REGIMES[regime]["phi"], REGIMES[regime]["q"]
    names, tl, groups = tools_for(pattern, phi, q)
    m = Model(K, PRIOR, tl, groups, LOSS, BUDGET)
    b = m.prior.copy()
    for t, o in EVIDENCE[pattern]: b = m.update(b, names.index(t), o)
    return m, b, names, tl

def gname(pattern, g):
    if pattern == "shared3": return "shared cache"
    if pattern == "indep3": return f"backend of {'ABC'[int(g[1])]}"
    return "A's backend"

def f(x, blank):
    return "?" if blank else f"{x:.3f}"

def stale_line(m, b, pattern, blank):
    parts = []
    for k, g in enumerate(m.G):
        p = sum(b[i] for i, (s, z) in enumerate(m.H) if z[k] == 1)
        parts.append(f"P({gname(pattern, g)} stale) = {f(p, blank)}")
    return "; ".join(parts)

def block_joint(m, b, pattern, blank):
    lines = []
    for i, (s, z) in enumerate(m.H):
        flags = ", ".join(f"{gname(pattern, g)} {'stale' if z[k] else 'working'}" for k, g in enumerate(m.G))
        lines.append(f"  P({STATES[s]}, {flags}) = {f(b[i], blank)}")
    return "Joint posterior over (service state, staleness) given the observations so far:\n" + "\n".join(lines)

def block_branches(m, b, names, pattern, blank):
    out = ["For each possible query, the distribution of its next report and the posterior after each report:"]
    for j, nm in enumerate(names):
        pr = m.pred(b, j)
        out.append(f"  query_{nm}: next report " + ", ".join(f"{STATES[o]} {f(pr[o], blank)}" for o in range(K)))
        for o in range(K):
            bo = m.update(b, j, o); ps = m.marg_s(bo)
            out.append(f"    if it reports {STATES[o]}: posterior " + ", ".join(f"{STATES[s]} {f(ps[s], blank)}" for s in range(K))
                       + "; " + stale_line(m, bo, pattern, blank))
    return "\n".join(out)

def block_values(Q, blank):
    return ("Expected total cost from now on of each action, if taken now and followed by optimal decisions "
            "within the remaining budget (query costs included):\n" + "\n".join(f"  {a}: {f(v, blank)}" for a, v in Q.items()))

HEAD_ORACLE = "The following quantities were computed exactly from the model described above.\n"
HEAD_CTRL = ("Before deciding, work out the following quantities from the model described above "
             "and fill in every \"?\".\n")

def build_prompt(regime, pattern, level, ctrl):
    base = prompt(pattern, regime, "plain")
    if level == 0: return base
    m, b, names, tl = model_for(regime, pattern)
    Q = exact(pattern, REGIMES[regime]["phi"], REGIMES[regime]["q"], EVIDENCE[pattern])[1]
    blocks = [block_joint(m, b, pattern, ctrl)]
    if level >= 2: blocks.append(block_branches(m, b, names, pattern, ctrl))
    if level >= 3: blocks.append(block_values(Q, ctrl))
    cut = base.index("Give (1) your probability")
    return base[:cut] + (HEAD_CTRL if ctrl else HEAD_ORACLE) + "\n\n".join(blocks) + "\n\n" + base[cut:]

if __name__ == "__main__":
    os.makedirs("diag/interv", exist_ok=True)
    items = []
    conds = [("L0", 0, False), ("L1", 1, False), ("L2", 2, False), ("L3", 3, False), ("C1", 1, True), ("C2", 2, True), ("C3", 3, True)]
    for regime, par in REGIMES.items():
        for pattern, ev in EVIDENCE.items():
            p, Q, best, names = exact(pattern, par["phi"], par["q"], ev)
            for c, lv, ctrl in conds:
                items.append(dict(id=f"{regime}|{pattern}|{c}", task=f"{regime}|{pattern}", condition=c,
                                  p_exact=p, Q=Q, best_value=best, best_action=min(Q, key=Q.get),
                                  prompt=build_prompt(regime, pattern, lv, ctrl)))
    json.dump(items, open("diag/interv/interv_items.json", "w"), indent=1)
    import statistics
    for c, *_ in conds:
        L = [len(it["prompt"]) for it in items if it["condition"] == c]
        print(c, "prompt chars median", statistics.median(L))
