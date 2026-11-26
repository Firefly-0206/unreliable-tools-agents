"""Bench v2 core: scenarios, exact targets, naive-model choices, prompts, answer parsing, H1 planner.

One base task = (scenario, evidence pattern, phi, q, prior, severity). The math is the pilot's
(diag_items.py): 3 states, state 0 benign, a stale store reports the benign state w.p. 0.97,
budget 4, source tools cost 1, independent verifier V (q = 0.8, never stale) costs 2.
"""
import ast, json, math, operator, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import Model, loss_matrix, STUCK          # noqa: E402

K = 3
BUDGET = 4
SRC_COST, V_COST, V_Q = 1, 2, 0.8
ERR, ESC = 10.0, 4.0
PHIS = [0.05, 0.15, 0.3]
QS = [0.75, 0.85, 0.9]
PRIORS = {"p50": [0.5, 0.25, 0.25], "p70": [0.7, 0.15, 0.15]}
SEVS = {"sev3": 30.0, "sev6": 60.0}           # loss of a false benign declaration (other wrong = 10)

# structure: retry (one tool, persistent fault), shared (A,B,C on one store), indep (A,B,C own stores),
# transient (one tool, per-call independent staleness: repeats are genuinely informative)
PATTERNS = {
    "single":            ("retry",     [("A", 0)]),
    "retry2":            ("retry",     [("A", 0)] * 2),
    "retry3":            ("retry",     [("A", 0)] * 3),
    "retry_mixed":       ("retry",     [("A", 0), ("A", 0), ("A", 1)]),
    "retry_bad":         ("retry",     [("A", 1), ("A", 1)]),
    "shared3":           ("shared",    [("A", 0), ("B", 0), ("C", 0)]),
    "shared_mixed":      ("shared",    [("A", 0), ("B", 0), ("C", 1)]),
    "shared_then_indep": ("shared",    [("A", 0), ("B", 0), ("V", 0)]),
    "indep3":            ("indep",     [("A", 0), ("B", 0), ("C", 0)]),
    "indep_bad":         ("indep",     [("A", 2), ("B", 2), ("C", 0)]),
    "transient3":        ("transient", [("A", 0)] * 3),
    "transient_mixed":   ("transient", [("A", 0), ("A", 0), ("A", 1)]),
}
# items with several reports from one persistent source (over-counting is possible). retry_bad repeats a
# non-benign report, so over-confidence is measured in the direction of the repeated reports (dep_dir).
DEPENDENT = {"retry2", "retry3", "retry_mixed", "retry_bad", "shared3", "shared_mixed", "shared_then_indep"}

# Reporting categories of an action. A transient tool re-call hits a fresh node (its own source), so it is
# kept apart from re-querying a persistent source.
CATS = ["declare_benign", "declare_nonbenign", "escalate", "query_same_source", "query_repeat_transient",
        "query_new_source"]
# Selection strata of build_items.py (original definition, kept so the 140-item set is unchanged): a re-call of
# the transient tool counts as "query_same_source" there.
SEL_CATS = ["declare_benign", "declare_nonbenign", "escalate", "query_same_source", "query_new_source"]
POLICIES = ["fault_unaware", "mmi", "split", "dedup_history", "dedup_history_initial", "forbid_repeat", "dedup_both"]

SCEN = {
    "service": dict(
        states=["healthy", "degraded", "down"], entity="the service", Entity="The service",
        intro="You are an on-call agent deciding whether a production service is healthy.",
        tools_word="Monitoring tools", store="metrics cache", episode="incident",
        verifier="Tool V is an independent synthetic end-to-end probe on its own infrastructure"),
    "replication": dict(
        states=["replicated", "partial", "missing"], entity="the record", Entity="The record",
        intro=("You are a data-platform agent deciding whether a customer record has been fully replicated "
               "to the reporting database."),
        tools_word="Lookup tools", store="read-replica snapshot", episode="investigation",
        verifier="Tool V is an independent direct query against the primary database"),
    "payment": dict(
        states=["settled", "pending", "failed"], entity="the settlement", Entity="The settlement",
        intro="You are a payments-operations agent deciding the status of a merchant settlement.",
        tools_word="Status APIs", store="ledger snapshot", episode="case",
        verifier="Tool V is an independent bank-statement lookup through a separate provider"),
}

# ------------------------------------------------------------------ models ---------------------------
def tool_names(structure):
    return (["A"] if structure in ("retry", "transient") else ["A", "B", "C"]) + ["V"]

def sources(structure):
    """tool -> data-source id. For the transient structure 'S1' is a pool of nodes: each call is its own source
    (see call_src), so it is never a persistent source shared by two calls."""
    if structure == "shared": return {"A": "S1", "B": "S1", "C": "S1", "V": "SV"}
    if structure == "indep": return {"A": "S1", "B": "S2", "C": "S3", "V": "SV"}
    return {"A": "S1", "V": "SV"}

def is_transient(structure, t):
    return structure == "transient" and t != "V"

def call_src(structure, srcs, t, k):
    """source id of the k-th report (0-based) in a history."""
    return f"{srcs[t]}#{k + 1}" if is_transient(structure, t) else srcs[t]

def persistent_used(structure, srcs, hist):
    """persistent sources already queried (the ones forbid_repeat blocks)."""
    return frozenset(srcs[t] for t, _ in hist if not is_transient(structure, t))

def true_tools(structure, phi, q):
    fe = STUCK(K)
    if structure == "retry":
        tl, groups = [dict(kind="sym", q=q, cost=SRC_COST, group="g", femit=fe)], {"g": phi}
    elif structure == "transient":
        tl, groups = [dict(kind="sym", q=q, cost=SRC_COST, marg_phi=phi, femit=fe)], {}
    elif structure == "shared":
        tl, groups = [dict(kind="sym", q=q, cost=SRC_COST, group="g", femit=fe) for _ in range(3)], {"g": phi}
    else:
        tl = [dict(kind="sym", q=q, cost=SRC_COST, group=f"g{i}", femit=fe) for i in range(3)]
        groups = {f"g{i}": phi for i in range(3)}
    return tl + [dict(kind="sym", q=V_Q, cost=V_COST)], groups

def belief(M, names, evidence):
    b = M.prior.copy()
    for t, o in evidence:
        b = M.update(b, names.index(t), o)
        if b is None: return None
    return b

def V_forbid(M, b, B, used, srcl, memo):
    """Budgeted DP in which a source, once used, may not be queried again."""
    k = (tuple(np.round(b, 9)), round(B, 6), used)
    if k in memo: return memo[k]
    best = M.term(b)[0]
    for j, t in enumerate(M.tools):
        if t["cost"] > B + 1e-9 or (srcl[j] is not None and srcl[j] in used): continue
        v = t["cost"]
        u2 = used if srcl[j] is None else used | {srcl[j]}
        for o, p in enumerate(M.pred(b, j)):
            if p > 1e-12: v += p * V_forbid(M, M.update(b, j, o), B - t["cost"], u2, srcl, memo)
        best = min(best, v)
    memo[k] = best
    return best

def root_Q(M, b, names, srcs, used=frozenset(), forbid=False, B=BUDGET, structure=None):
    """Exact action values at the current belief: declarations, escalation, one query then optimal play.
    With forbid, persistent sources in `used` (and any queried later) may not be queried again; transient
    tool calls are never forbidden (each call is a fresh node)."""
    ps = M.marg_s(b)
    Q = {f"declare_{s}": float(M.loss[s] @ ps) for s in range(K)}
    Q["escalate"] = float(M.loss[K] @ ps)
    srcl = [None if is_transient(structure, n) else srcs[n] for n in names]; memo = {}
    for j, nm in enumerate(names):
        c = M.tools[j]["cost"]
        if c > B + 1e-9 or (forbid and srcl[j] is not None and srcl[j] in used): continue
        u2 = used if srcl[j] is None else used | {srcl[j]}
        v = c
        for o, p in enumerate(M.pred(b, j)):
            if p < 1e-12: continue
            b2 = M.update(b, j, o)
            v += p * (V_forbid(M, b2, B - c, u2, srcl, memo) if forbid else M.V(b2, B - c)[0])
        Q[f"query_{nm}"] = float(v)
    return Q

def argmin(Q):
    return min(Q, key=Q.get)

def category(a, hist, srcs, structure):
    """Reporting category of action a after history hist (see CATS)."""
    if a == "declare_0": return "declare_benign"
    if a.startswith("declare_"): return "declare_nonbenign"
    if a == "escalate": return "escalate"
    t = a[6:]
    if is_transient(structure, t): return "query_repeat_transient" if any(u == t for u, _ in hist) else "query_new_source"
    return "query_same_source" if srcs[t] in persistent_used(structure, srcs, hist) else "query_new_source"

def sel_category(a, hist, srcs):
    """Selection-stratum category (original definition: a transient re-call counts as query_same_source)."""
    c = category(a, hist, srcs, None)
    return c

def first_per_source(evidence, srcs, structure=None):
    """keep the first report of each source; transient calls are distinct sources, so all of them are kept."""
    seen, out = set(), []
    for k, (t, o) in enumerate(evidence):
        s = call_src(structure, srcs, t, k)
        if s not in seen: seen.add(s); out.append((t, o))
    return out

def naive_models(structure, phi, q, prior, loss):
    tl, groups = true_tools(structure, phi, q)
    out = {}
    out["fault_unaware"] = Model(K, prior, [dict(t, group=None, marg_phi=None) for t in tl], {}, loss, BUDGET)
    mm = [dict(t, marg_phi=groups[t["group"]], group=None) if t.get("group") else dict(t) for t in tl]
    out["mmi"] = Model(K, prior, mm, {}, loss, BUDGET)
    if structure == "shared":
        sp = [dict(t, group=f"{t['group']}{j}") if t.get("group") else dict(t) for j, t in enumerate(tl)]
        out["split"] = Model(K, prior, sp, {f"g{j}": phi for j in range(3)}, loss, BUDGET)
    return out

def make_policy(pol, M, Mn, names, srcs, structure, n0):
    """policy(hist, B) -> action. Each reference policy decides EVERY step from its own belief / constraint:
    fault_unaware, mmi, split  - their own (misspecified) model, belief from the full history, exact planner;
    dedup_history              - true model, belief from the first report of each source in the WHOLE history
                                 (later reports from an already-used persistent source, new ones included, are
                                 discarded), free planning;
    dedup_history_initial      - as dedup_history but only the initial history is deduplicated; reports
                                 obtained during the episode are all kept;
    forbid_repeat              - true model, full history, never queries a persistent source used before;
    dedup_both                 - dedup_history + forbid_repeat (the simulation's dedup).
    Transient tool calls are distinct sources: never discarded, never forbidden."""
    def f(hist, B):
        if pol == "optimal":
            return argmin(root_Q(M, belief(M, names, hist), names, srcs, B=B))
        if pol in ("fault_unaware", "mmi", "split"):
            b = belief(Mn, names, hist)
            return "escalate" if b is None else argmin(root_Q(Mn, b, names, srcs, B=B))
        if pol == "dedup_history_initial": sh = first_per_source(hist[:n0], srcs, structure) + list(hist[n0:])
        elif pol in ("dedup_history", "dedup_both"): sh = first_per_source(hist, srcs, structure)
        else: sh = hist
        return argmin(root_Q(M, belief(M, names, sh), names, srcs, persistent_used(structure, srcs, hist),
                             forbid=pol in ("forbid_repeat", "dedup_both"), B=B, structure=structure))
    return f

CL_FIELDS = ["loss", "call_cost", "p_escalate", "p_false_benign", "p_wrong_other", "n_calls"]

def closed_loop(M, names, f, hist, B):
    """Expected outcome under the TRUE model M when policy f picks every action until it stops:
    [total loss incl. call costs, call cost, P(escalate), P(false benign declaration), P(other wrong
    declaration), expected number of calls]."""
    a = f(hist, B); b = belief(M, names, hist)
    if not a.startswith("query_"):
        ps = M.marg_s(b); row = K if a == "escalate" else int(a[8:])
        out = np.zeros(6); out[0] = M.loss[row] @ ps
        if a == "escalate": out[2] = 1.0
        elif row == 0: out[3] = 1 - ps[0]
        else: out[4] = 1 - ps[row]
        return out
    j = names.index(a[6:]); c = M.tools[j]["cost"]
    out = np.array([c, c, 0, 0, 0, 1.0])
    for o, p in enumerate(M.pred(b, j)):
        if p > 1e-12: out = out + p * closed_loop(M, names, f, hist + [(a[6:], o)], B - c)
    return out

def solve_task(pattern, phi, q, prior_name, sev_name, closed=True):
    """All exact targets for one math task (scenario-independent)."""
    structure, ev = PATTERNS[pattern]
    prior = np.array(PRIORS[prior_name]); sev = SEVS[sev_name]
    loss = loss_matrix(K, ERR, sev, ESC)
    names, srcs = tool_names(structure), sources(structure)
    tl, groups = true_tools(structure, phi, q)
    M = Model(K, prior, tl, groups, loss, BUDGET)
    b = belief(M, names, ev)
    Q = root_Q(M, b, names, srcs)
    best = argmin(Q); srt = sorted(Q.values())
    ps = M.marg_s(b)
    joint = [dict(s=int(s), stale={g: int(z[k]) for k, g in enumerate(M.G)}, p=float(b[i]))
             for i, (s, z) in enumerate(M.H)]
    p_stale = {g: float(sum(b[i] for i, (s, z) in enumerate(M.H) if z[k])) for k, g in enumerate(M.G)}
    shown = first_per_source(ev, srcs, structure)
    b_sh = belief(M, names, shown)
    cats = {a: category(a, ev, srcs, structure) for a in Q}
    scats = {a: sel_category(a, ev, srcs) for a in Q}
    nm_models = naive_models(structure, phi, q, prior, loss)
    hist0 = [tuple(e) for e in ev]
    opt_cl = closed_loop(M, names, make_policy("optimal", M, None, names, srcs, structure, len(ev)), hist0, BUDGET) \
        if closed else None
    naive = {}
    for pol in POLICIES:
        if pol == "split" and structure != "shared": continue
        Mn = nm_models.get(pol)
        f = make_policy(pol, M, Mn, names, srcs, structure, len(ev))
        a = f(hist0, BUDGET)
        if Mn is not None: pb = Mn.marg_s(belief(Mn, names, ev))[0]
        elif pol in ("dedup_history", "dedup_history_initial", "dedup_both"): pb = M.marg_s(b_sh)[0]
        else: pb = ps[0]
        d = dict(action=a, category=cats[a], sel_category=scats[a], regret_1step=Q[a] - Q[best], p_benign=float(pb),
                 severe_1step=bool(a == "declare_0" and ps[0] < 1 - ESC / sev))
        if closed:
            cl = closed_loop(M, names, f, hist0, BUDGET)
            d.update(regret_closed=float(cl[0] - opt_cl[0]), **{f"cl_{k}": float(v) for k, v in zip(CL_FIELDS, cl)})
        naive[pol] = d
    # over-count direction on dependent items: +1 if the repeated persistent-source reports are mostly benign
    rep = [o for t, o in ev if sum(srcs[u] == srcs[t] for u, _ in ev) > 1 and not is_transient(structure, t)]
    dep_dir = (1 if sum(o == 0 for o in rep) * 2 >= len(rep) else -1) if rep else 0
    rep_q = [a for a in Q if cats[a] == "query_same_source"]
    srep_q = [a for a in Q if scats[a] == "query_same_source"]
    out = dict(pattern=pattern, structure=structure, persistent=structure != "transient", phi=phi, q=q,
               prior_name=prior_name, prior=list(prior),
               sev_name=sev_name, sev=sev, esc=ESC, err=ERR, tau=1 - ESC / sev, budget=BUDGET,
               evidence=ev, tools=names, sources=srcs, used_sources=sorted(persistent_used(structure, srcs, ev)),
               joint=joint, p_stale=p_stale, ps=[float(x) for x in ps], p_benign=float(ps[0]),
               p_benign_shown=float(M.marg_s(b_sh)[0]), shown_evidence=shown,
               Q=Q, best_action=best, best_value=Q[best], gap=srt[1] - srt[0], best_category=cats[best],
               sel_category=scats[best], categories=cats, sel_categories=scats, naive=naive,
               dependent=pattern in DEPENDENT, dep_dir=dep_dir,
               repeat_right=cats[best] == "query_same_source",
               repeat_right_transient=cats[best] == "query_repeat_transient",
               repeat_wrong=bool(rep_q) and cats[best] != "query_same_source"
               and any(naive[n]["category"] == "query_same_source" for n in ("fault_unaware", "mmi")),
               # selection-only flag (original definition), used by build_items.select
               sel_repeat_wrong=bool(srep_q) and scats[best] != "query_same_source"
               and any(naive[n]["sel_category"] == "query_same_source" for n in ("fault_unaware", "mmi")))
    if closed:
        out["closed_opt_check"] = float(opt_cl[0] - Q[best])
        out["closed_opt"] = {k: float(v) for k, v in zip(CL_FIELDS, opt_cl)}
    return out

def true_model(it):
    tl, groups = true_tools(it["structure"], it["phi"], it["q"])
    return Model(K, np.array(it["prior"]), tl, groups, loss_matrix(K, ERR, it["sev"], ESC), BUDGET)

# ------------------------------------------------------------------ prompts --------------------------
def fmt(x):
    return f"{x:g}"

def action_names(it):
    st = SCEN[it["scenario"]]["states"]
    return [f"declare_{s}" for s in st] + ["escalate"] + [f"query_{n}" for n in it["tools"]]

def to_scen(it, a):
    st = SCEN[it["scenario"]]["states"]
    return f"declare_{st[int(a[8:])]}" if a.startswith("declare_") else a

def gname(it, g):
    S = SCEN[it["scenario"]]
    if it["structure"] == "shared": return f"shared {S['store']}"
    if it["structure"] == "indep": return f"{S['store']} of {'ABC'[int(g[1])]}"
    return f"A's {S['store']}"

def task_text(it):
    S = SCEN[it["scenario"]]; st = S["states"]; phi, q = it["phi"], it["q"]
    src = it["tools"][:-1]; sto, epi = S["store"], S["episode"]
    w = fmt(round((1 - q) / 2, 4))
    if it["structure"] == "shared":
        rel = (f"Tools {', '.join(src)} are three different interfaces that all read from the SAME {sto}. "
               f"If that {sto} is stale, all three are affected together for the whole {epi}.")
        fault = f"The shared {sto} is stale with probability {fmt(phi)} for this {epi}"
        persist = f"Staleness, if present, persists for the whole {epi}."
    elif it["structure"] == "indep":
        rel = (f"Tools {', '.join(src)} each read from their OWN independent {sto}. "
               f"Each {sto} can be stale independently of the others.")
        fault = f"Each {sto} is stale with probability {fmt(phi)} for this {epi}, independently"
        persist = f"Staleness, if present, persists for the whole {epi}."
    elif it["structure"] == "transient":
        rel = f"Tool A may be called repeatedly; each call is served by a randomly chosen {sto} node."
        fault = f"Each call to tool A independently hits a stale {sto} node with probability {fmt(phi)}"
        persist = "Staleness does NOT persist: whether one call was stale says nothing about any other call."
    else:
        rel = f"Tool A may be called repeatedly; every call reads the same {sto}."
        fault = f"Tool A's {sto} is stale with probability {fmt(phi)} for this {epi}"
        persist = f"Staleness, if present, persists for the whole {epi}."
    pr = it["prior"]
    return f"""{S['intro']}

{S['Entity']} is in exactly one of three states: {st[0]}, {st[1]}, {st[2]}.
Prior probabilities before any observation: {st[0]} {pr[0]:.2f}, {st[1]} {pr[1]:.2f}, {st[2]} {pr[2]:.2f}.

{S['tools_word']}:
- {rel}
- When its {sto} is not stale, a tool reports the true state with probability {fmt(q)}, and each of the two wrong states with probability {w}.
- {fault}. A stale {sto} ignores the true state and reports "{st[0]}" with probability 0.97 (each other state 0.015). {persist}
- Given the true state and whether each {sto} is stale, separate calls are otherwise independent.
- {S['verifier']}, never stale, accuracy {fmt(V_Q)} (each wrong state {fmt(round((1 - V_Q) / 2, 4))}).

Costs (in the same units):
- Declaring "{st[0]}" when {S['entity']} is not {st[0]}: {fmt(it['sev'])}.
- Declaring "{st[1]}" or "{st[2]}" when that is wrong: {fmt(ERR)}.
- A correct declaration: 0.
- Escalating to a human: {fmt(ESC)}.
- Calling one of tools {', '.join(src)} (again): {SRC_COST} per call. Calling tool V: {V_COST}. Remaining query budget: {BUDGET}.
"""

WARNING = ("Note: these tools may share failure sources; do not treat results from the same source as "
           "independent confirmation.\n\n")

def obs_block(it, presentation):
    S = SCEN[it["scenario"]]; st = S["states"]; ev = [tuple(e) for e in it["evidence"]]
    if presentation == "grouped":
        # transient calls each hit a fresh node, so every call is its own source group
        order = {}
        for i, (t, o) in enumerate(ev): order.setdefault(call_src(it["structure"], it["sources"], t, i), []).append((i + 1, t, o))
        lines = ["Observations so far, grouped by data source (all reports are kept; the number in brackets is the "
                 "order in which the report arrived):"]
        for s, rows in order.items():
            mem = sorted({t for t, src in it["sources"].items() if src == s.split("#")[0]})
            if s == "SV": label = "tool V's own infrastructure"
            elif "#" in s: label = f"the randomly chosen {S['store']} node that served call {rows[0][0]} of tool {mem[0]}"
            elif len(mem) > 1: label = f"the shared {S['store']} behind tools {', '.join(mem)}"
            else: label = f"tool {mem[0]}'s {S['store']}"
            lines.append(f"- Source: {label} - {len(rows)} report{'s' if len(rows) > 1 else ''}: "
                         + "; ".join(f"[{i}] tool {t} reported \"{st[o]}\"" for i, t, o in rows))
        return "\n".join(lines)
    shown = [tuple(e) for e in it["shown_evidence"]] if presentation == "first" else ev
    lines = [f"{i + 1}. tool {t} reported \"{st[o]}\"" for i, (t, o) in enumerate(shown)]
    note = ""
    if presentation == "first" and len(shown) < len(ev):
        note = ("(Only the first report from each data source is shown; later reports from a source that had "
                "already reported were removed.)\n")
    return "Observations so far:\n" + note + "\n".join(lines)

ANSWER_TEMPLATE = '{"p_benign": <probability>, "action": "<action>"}'

def ask_text(it):
    S = SCEN[it["scenario"]]
    return (f"Give (1) your probability that {S['entity']} is {S['states'][0]} given everything above, and (2) the "
            f"single next action that minimises expected total cost, chosen from: {', '.join(action_names(it))}.\n"
            f"End your answer with one final line of JSON in this format, filled with your own values: "
            f"{ANSWER_TEMPLATE}\n"
            f"(p_benign is your probability that {S['entity']} is {S['states'][0]}, a number between 0 and 1; "
            f"action is one of the action names above.)")

def prompt_presentation(it, pres):
    warn = WARNING if pres == "warning" else ""
    p = "first" if pres == "first" else ("grouped" if pres == "grouped" else "plain")
    return task_text(it) + "\n" + warn + obs_block(it, p) + "\n\n" + ask_text(it)

# --- oracle-information blocks (build_interv.py wording, generalised) ---
def f3(x, blank):
    return "?" if blank else f"{x:.3f}"

def stale_line(it, M, b, blank):
    parts = []
    for k, g in enumerate(M.G):
        p = sum(b[i] for i, (s, z) in enumerate(M.H) if z[k] == 1)
        parts.append(f"P({gname(it, g)} stale) = {f3(p, blank)}")
    return "; ".join(parts)

def block_joint(it, M, b, blank):
    st = SCEN[it["scenario"]]["states"]; lines = []
    for i, (s, z) in enumerate(M.H):
        flags = ", ".join(f"{gname(it, g)} {'stale' if z[k] else 'not stale'}" for k, g in enumerate(M.G))
        lines.append(f"  P({st[s]}{', ' + flags if flags else ''}) = {f3(b[i], blank)}")
    head = ("Joint posterior over (state, staleness) given the observations so far:" if M.G else
            "Posterior over the state given the observations so far (staleness does not persist, so there is no "
            "persistent stale flag):")
    return head + "\n" + "\n".join(lines)

def block_branches(it, M, b, blank):
    st = SCEN[it["scenario"]]["states"]
    out = ["For each possible query, the distribution of its next report and the posterior after each report:"]
    for j, nm in enumerate(it["tools"]):
        pr = M.pred(b, j)
        out.append(f"  query_{nm}: next report " + ", ".join(f"{st[o]} {f3(pr[o], blank)}" for o in range(K)))
        for o in range(K):
            bo = M.update(b, j, o); ps = M.marg_s(bo); sl = stale_line(it, M, bo, blank)
            out.append(f"    if it reports {st[o]}: posterior " + ", ".join(f"{st[s]} {f3(ps[s], blank)}" for s in range(K))
                       + (f"; {sl}" if sl else ""))
    return "\n".join(out)

def block_values(it, blank):
    return ("Expected total cost from now on of each action, if taken now and followed by optimal decisions "
            "within the remaining budget (query costs included):\n"
            + "\n".join(f"  {to_scen(it, a)}: {f3(v, blank)}" for a, v in it["Q"].items()))

HEAD_ORACLE = "The following quantities were computed exactly from the model described above.\n"
HEAD_CTRL = "Before deciding, work out the following quantities from the model described above and fill in every \"?\".\n"

def prompt_oracle(it, level, ctrl, donor=None):
    """Lk: exact blocks under HEAD_ORACLE. Ck (ctrl, 'scaffold' control): same blocks with '?' and the
    instruction to fill them in. Xk (donor, 'placebo' control): HEAD_ORACLE wording and identical block shape,
    but the numbers are those of a different base task with the same evidence pattern (donor)."""
    base = prompt_presentation(it, "plain")
    src = it if donor is None else dict(donor, scenario=it["scenario"])
    M = true_model(src); b = belief(M, src["tools"], [tuple(e) for e in src["evidence"]])
    blocks = [block_joint(src, M, b, ctrl)]
    if level >= 2: blocks.append(block_branches(src, M, b, ctrl))
    if level >= 3: blocks.append(block_values(src, ctrl))
    cut = base.index("Give (1) your probability")
    return base[:cut] + (HEAD_CTRL if ctrl else HEAD_ORACLE) + "\n\n".join(blocks) + "\n\n" + base[cut:]

T1_NOTE = ("You have a tool calculate(expression) that evaluates an arithmetic expression exactly "
           "(+ - * / ** parentheses, and functions such as exp, log, sqrt, min, max). Use it for every numerical "
           "step instead of doing arithmetic in your head. ")

def prompt_tool(it):
    base = prompt_presentation(it, "plain")
    cut = base.index("Give (1) your probability")
    return base[:cut] + T1_NOTE + base[cut:]

def prompt_h1(it):
    S = SCEN[it["scenario"]]; st = S["states"]
    ex = {"sources": [{"tools": ["A"], "fault_prob": 0.1, "fault_persists": True}],
          "accuracy": {n: 0.8 for n in it["tools"]},
          "stale_report": {st[0]: 0.9, st[1]: 0.05, st[2]: 0.05}}
    return (task_text(it) + "\n" + obs_block(it, "plain") + "\n\n"
            "Do not choose an action yourself. An exact Bayesian planner will choose the action using the prior, costs, "
            "budget and observations above together with a model of the tools that YOU write down. Write that model as "
            "JSON with these fields:\n"
            "- \"sources\": a list of failure sources. Each has \"tools\" (the tools that read from it and therefore go "
            "stale together), \"fault_prob\" (probability that it is stale) and \"fault_persists\" (true if staleness "
            "lasts for the whole " + S["episode"] + ", false if each call is stale independently). A tool that can "
            "never be stale is not listed in any source.\n"
            "- \"accuracy\": for every tool, the probability that it reports the true state when not stale (the two "
            "wrong states share the rest equally).\n"
            "- \"stale_report\": the distribution of the report of a stale tool over the three states.\n"
            "Fill every number from the description above. Format example (illustrative numbers only):\n"
            + json.dumps(ex) + "\nEnd your answer with the JSON object between <model> and </model>.")

CONDITIONS = ["P0", "P1", "P2", "P3", "L1", "L2", "L3", "C1", "C2", "C3", "X1", "X2", "X3", "T1", "H1"]
CORE_CONDS = ["P0", "P1", "P2", "P3", "L1", "L2", "L3", "C1", "C2", "C3", "X1", "X2", "X3"]
COND_DOC = {
    "P0": "plain (= L0)", "P1": "one-sentence dependence warning", "P2": "all reports kept, grouped by data source",
    "P3": "first report per source only (history discarded)", "L1": "+ exact joint posterior",
    "L2": "+ L1 + predictive distributions and branch posteriors", "L3": "+ L2 + exact action values",
    "C1": "scaffold control for L1 (numbers -> ?, asked to fill them in)", "C2": "scaffold control for L2",
    "C3": "scaffold control for L3",
    "X1": "placebo control for L1 (L1 wording, numbers of a same-pattern donor task)", "X2": "placebo for L2",
    "X3": "placebo for L3",
    "T1": "plain + calculate() tool (function calling)", "H1": "LLM writes dependence model JSON; exact DP chooses"}

def build_prompt(it, cond, donor=None):
    if cond == "P0": return prompt_presentation(it, "plain")
    if cond == "P1": return prompt_presentation(it, "warning")
    if cond == "P2": return prompt_presentation(it, "grouped")
    if cond == "P3": return prompt_presentation(it, "first")
    if cond in ("L1", "L2", "L3"): return prompt_oracle(it, int(cond[1]), False)
    if cond in ("C1", "C2", "C3"): return prompt_oracle(it, int(cond[1]), True)
    if cond in ("X1", "X2", "X3"):
        if donor is None: raise ValueError("placebo conditions need a donor task")
        return prompt_oracle(it, int(cond[1]), False, donor=donor)
    if cond == "T1": return prompt_tool(it)
    if cond == "H1": return prompt_h1(it)
    raise ValueError(cond)

# ------------------------------------------------------------------ calculator tool ------------------
CALC_TOOL = {"type": "function", "function": {
    "name": "calculate", "description": "Evaluate an arithmetic expression exactly and return the number.",
    "parameters": {"type": "object", "properties": {"expression": {"type": "string",
                   "description": "e.g. 0.5*0.9**3/(0.5*0.9**3+0.25*0.05**3)"}}, "required": ["expression"]}}}

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos,
        ast.FloorDiv: operator.floordiv}
_FUN = {k: getattr(math, k) for k in ("exp", "log", "log2", "log10", "sqrt", "floor", "ceil", "fabs")}
_FUN.update(min=min, max=max, abs=abs, round=round, sum=lambda *a: sum(a))
_CONST = {"pi": math.pi, "e": math.e}

def safe_eval(expr):
    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            l, r = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(r) > 1000: raise ValueError("exponent too large")
            return _OPS[type(n.op)](l, r)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS: return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Name) and n.id in _CONST: return _CONST[n.id]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUN and not n.keywords:
            return _FUN[n.func.id](*[ev(a) for a in n.args])
        if isinstance(n, (ast.Tuple, ast.List)): return [ev(x) for x in n.elts]
        raise ValueError(f"unsupported expression element: {type(n).__name__}")
    if len(expr) > 2000: return "error: expression too long"
    try:
        v = ev(ast.parse(expr.replace("^", "**"), mode="eval"))
        return repr(round(v, 12)) if isinstance(v, float) else repr(v)
    except Exception as e:
        return f"error: {e}"

# ------------------------------------------------------------------ parsing --------------------------
def canon_action(it, a):
    if not isinstance(a, str): return None
    a = a.strip().strip("`'\" .").lower().replace("-", "_").replace(" ", "_")
    st = SCEN[it["scenario"]]["states"]
    for i, s in enumerate(st):
        if a in (f"declare_{s}", s): return f"declare_{i}"
    if a in ("escalate", "escalate_to_human"): return "escalate"
    for n in it["tools"]:
        if a in (f"query_{n.lower()}", f"query_tool_{n.lower()}", f"call_{n.lower()}"): return f"query_{n}"
    return None

def final_block(content):
    """The answer's tail with code fences, markdown emphasis and trailing punctuation removed."""
    lines = [l for l in (content or "").strip().splitlines() if not re.fullmatch(r"\s*(```\w*|`+)\s*", l)]
    return re.sub(r"[\s`*.]+$", "", "\n".join(lines))

def parse_answer(it, content):
    """-> dict(status, p_benign, action, raw_action).
    status: ok | no_json | template_copy | bad_action | bad_p. Only a JSON object that ENDS the answer is
    accepted (a template quoted mid-answer, e.g. in a truncated output, is ignored), and an exact copy of the
    old example line {"p_benign": 0.00, "action": "escalate"} is rejected as template_copy. Truncation
    (finish_reason == length) is handled by the caller (run.parse_into)."""
    benign = SCEN[it["scenario"]]["states"][0]
    keys = ("p_benign", f"p_{benign}")
    tail = final_block(content)
    m = re.search(r"\{[^{}]*\}$", tail)
    obj = None
    if m:
        try: obj = json.loads(m.group(0))
        except Exception: obj = None
        if obj is None:                       # e.g. unquoted action or trailing comma: fall back to fields
            a = re.findall(r"\"?action\"?\s*:\s*\"?([A-Za-z_ \-]+)\"?", m.group(0))
            p = re.findall(r"\"?p_(?:benign|" + re.escape(benign) + r")\"?\s*[:=]\s*([0-9.]+)", m.group(0))
            obj = {"action": a[-1], "p_benign": p[-1] if p else None} if a else None
    if not isinstance(obj, dict) or "action" not in obj:
        return dict(status="no_json", p_benign=None, action=None, raw_action=None)
    if re.search(r"\"p_benign\"\s*:\s*0\.00\s*,\s*\"action\"\s*:\s*\"escalate\"", m.group(0)):
        return dict(status="template_copy", p_benign=None, action=None, raw_action=obj.get("action"))
    raw = obj.get("action"); act = canon_action(it, raw)
    pv = next((obj[k] for k in keys if k in obj), None)
    try: pv = float(pv) if pv is not None else None
    except (TypeError, ValueError): pv = None
    if pv is not None and not 0 <= pv <= 1: pv = None
    if act is None: return dict(status="bad_action", p_benign=pv, action=None, raw_action=raw)
    if pv is None: return dict(status="bad_p", p_benign=None, action=act, raw_action=raw)
    return dict(status="ok", p_benign=pv, action=act, raw_action=raw)

def parse_h1(it, content):
    """LLM-written dependence model -> exact planner action. Prior, losses, costs, budget are the task's own."""
    content = content or ""
    m = re.findall(r"<model>(.*?)</model>", content, re.S)
    txt = m[-1] if m else None
    if txt is None:
        cands = re.findall(r"\{.*\}", content, re.S)
        txt = cands[-1] if cands else None
    if txt is None: return dict(status="no_json", p_benign=None, action=None, struct=None)
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    try: d = json.loads(txt)
    except Exception: return dict(status="bad_json", p_benign=None, action=None, struct=None)
    if not isinstance(d, dict): return dict(status="bad_json", p_benign=None, action=None, struct=None)
    try:
        M, names, fixes = h1_model(it, d)
        b = belief(M, names, [tuple(e) for e in it["evidence"]])
        if b is None: return dict(status="inconsistent", p_benign=None, action=None, struct=d, h1_fixes=fixes)
        Q = root_Q(M, b, names, it["sources"])
        # a model with missing / out-of-range fields is not "parsed": the planner's answer is kept for
        # diagnostics only (p_benign_planner, action_planner)
        if fixes: return dict(status="incomplete_model", p_benign=None, action=None, struct=d, h1_fixes=fixes,
                              p_benign_planner=float(M.marg_s(b)[0]), action_planner=argmin(Q))
        return dict(status="ok", p_benign=float(M.marg_s(b)[0]), action=argmin(Q), struct=d, h1_fixes=fixes)
    except Exception as e:
        return dict(status=f"bad_struct: {str(e)[:80]}", p_benign=None, action=None, struct=d)

def _tool_key(t):
    return str(t).strip().upper().replace("TOOL_", "").replace("TOOL ", "").strip()

def h1_model(it, d):
    """-> (Model, names, fixes). fixes lists every default / clip applied (empty = complete, in-range model)."""
    st = SCEN[it["scenario"]]["states"]; names = it["tools"]; fixes = []
    acc = {_tool_key(k): float(v) for k, v in (d.get("accuracy") or {}).items()}
    for n in names:
        if n not in acc: fixes.append(f"accuracy[{n}] missing -> 0.8"); acc[n] = 0.8
        elif not 1 / K <= acc[n] <= 1: fixes.append(f"accuracy[{n}]={acc[n]} clipped")
    sr = d.get("stale_report")
    if not isinstance(sr, dict) or any(s not in sr for s in st):
        fixes.append("stale_report missing or incomplete"); sr = sr if isinstance(sr, dict) else {}
    fe = np.array([float(sr.get(s, 0.0)) for s in st])
    if abs(fe.sum() - 1) > 0.02 or (fe < 0).any(): fixes.append(f"stale_report sums to {fe.sum():.3f}")
    fe = np.clip(fe, 0, None) + 1e-9
    fe = np.full(K, 1 / K) if fe.sum() < 1e-6 else fe / fe.sum()
    tools = [dict(kind="sym", q=float(np.clip(acc[n], 0.34, 0.9999)),
                  cost=V_COST if n == "V" else SRC_COST, femit=fe) for n in names]
    groups = {}
    srcs = d.get("sources")
    if not isinstance(srcs, list): fixes.append("sources missing"); srcs = []
    for i, s in enumerate(srcs):
        if "fault_prob" not in s: fixes.append(f"sources[{i}].fault_prob missing -> 0")
        if "fault_persists" not in s: fixes.append(f"sources[{i}].fault_persists missing -> true")
        phi0 = float(s.get("fault_prob", 0.0))
        if not 0 <= phi0 <= 1: fixes.append(f"sources[{i}].fault_prob={phi0} clipped")
        phi = float(np.clip(phi0, 0.0, 0.999))
        for t in [_tool_key(t) for t in s.get("tools", [])]:
            if t not in names: fixes.append(f"sources[{i}] unknown tool {t}"); continue
            j = names.index(t)
            if tools[j].get("group") or tools[j].get("marg_phi") is not None: continue
            if s.get("fault_persists", True): tools[j]["group"] = f"s{i}"
            else: tools[j]["marg_phi"] = phi
        if s.get("fault_persists", True): groups[f"s{i}"] = phi
    M = Model(K, np.array(it["prior"]), tools, groups, loss_matrix(K, ERR, it["sev"], ESC), BUDGET)
    return M, names, fixes

def true_struct(it):
    """The correct H1 answer (used by the mock model and as a check)."""
    st = SCEN[it["scenario"]]["states"]; s = it["structure"]
    if s == "shared": srcs = [{"tools": ["A", "B", "C"], "fault_prob": it["phi"], "fault_persists": True}]
    elif s == "indep": srcs = [{"tools": [t], "fault_prob": it["phi"], "fault_persists": True} for t in "ABC"]
    else: srcs = [{"tools": ["A"], "fault_prob": it["phi"], "fault_persists": s != "transient"}]
    return {"sources": srcs, "accuracy": {n: (V_Q if n == "V" else it["q"]) for n in it["tools"]},
            "stale_report": {st[0]: 0.97, st[1]: 0.015, st[2]: 0.015}}
