"""
K2 decomposition: elicitation, parsing, scoring and exact planning on a RECOVERED observation model.

  parse_docs(ep, text)        numbers that the tool-doc block (identical in every condition) states
  extract_json(text)          last {"observation_model": ...} object in a reply
  normalize_model(ep, obj)    recovered model -> spec, missing / invalid fields filled with DOCUMENTED defaults (recorded)
  truth_spec(ep)              the true observation model in the same format
  score_recovery(ep, spec)    sharing structure, phi, persistence, stale emission, accuracy errors vs the truth
  build_recovered_model(ep, spec)   sim.Model of the recovered model (true prior, costs, loss, budget are documented)
  plan_and_play(ep, spec, env)      EXACT DP planner (same machinery as the oracle) acts on the recovered model
  doc_reader_model(ep, text, link)  deterministic reader of the docs (used by the mock backends and the self-test)

spec = dict(groups=[dict(tools=[tool idx], shared_fault, phi, persistent, emission=[p_status0, p_status1, p_status2])],
            accuracy=[q_0..q_3], notes=[...], defaults=[...], clipped=[...], fallback=bool)
"""
import json, re
import numpy as np
from env import SCENARIOS, K, LOSS, BUDGET, F_STUCK, VPRIOR, N_TOOLS, marginal_stuck, stuck_emission
from env import HD_PHI_BINS, HD_ACC_BINS
from sim import Model

BENIGN_EMISSION = [F_STUCK] + [(1 - F_STUCK) / (K - 1)] * (K - 1)

# ------------------------------------------------------------------ what the docs state ----------
def parse_docs(ep, text):
    """Numbers stated in the 'Tools:' block of a system prompt (the same block in every condition). Returns one dict per
    tool: name, cost, backend (string), cache (bool), phi (None for non-replica tools), acc, stale_p, frozen_p."""
    if ep.get("doc_style") == "runbook":    # R experiment: the runbook states the true parameters as counts (env.runbook_doc,
        par = ep["params"]; sc = SCENARIOS[ep["scenario"]]      # checked by make_episodes_r.py); documented values = true values
        return [dict(name=sc["tools"][j]["name"], cost=float(t["cost"]), backend=sc["backends"][t["src"]], cache=t["src"] == "cache",
                     uniform=False, percall=False, phi=par["phi"] if t["src"] == "cache" else None,
                     stale_p=F_STUCK if t["src"] == "cache" else None, frozen_p=None, acc=t["q"]) for j, t in enumerate(par["tools"])]
    out = []
    for line in text.split("Tools:\n", 1)[1].split("\n\n", 1)[0].split("\n"):
        m = re.match(r"- (\w+)\(", line)
        if not m: continue
        hd = re.search(r"Accuracy(?: when the replica is fresh)?: (" + "|".join(HD_ACC_BINS) + r"); ", line)
        if hd:                                  # HD experiment: rates as words (env.hd_doc); "documented" value = middle of the
            mid = lambda r: round((r[0] + r[1]) / 2, 4)   # word's range (used only to fill fields a JSON leaves out)
            rep = "Served from read replica " in line
            d = dict(name=m.group(1), cost=float(re.search(r"Cost: ([0-9.]+)\.", line).group(1)), cache=rep, uniform=False,
                     percall=False, vague=True, phi=None, stale_p=None, frozen_p=None, acc=mid(HD_ACC_BINS[hd.group(1)]))
            if rep:
                d["backend"] = re.search(r"Served from read replica (\S+?)\. ", line).group(1)
                pw = re.search(r"(?:Replica staleness incidents: |On-call sessions during which this job is halted: )(" + "|".join(HD_PHI_BINS) + r")", line)
                d["phi"] = mid(HD_PHI_BINS[pw.group(1)]); d["stale_p"] = F_STUCK
            else:
                d["backend"] = (re.search(r"Served from (.+?) \(primary database", line) or re.search(r"Queries (.+?) directly on every call", line)).group(1)
            out.append(d); continue
        if " | backend: " in line:              # V experiment: terse field-list documentation (env.tool_doc)
            f = dict(x.split(": ", 1) for x in line.split(" | ")[1:]); rep = f["backend"].startswith("read replica ")
            d = dict(name=m.group(1), cost=float(f["cost"]), cache=rep, uniform=False, phi=None, stale_p=None, frozen_p=None,
                     acc=float(re.match(r"([0-9.]+) per call", f["accuracy while fresh" if rep else "accuracy"]).group(1)))
            if rep:
                d.update(backend=f["backend"][len("read replica "):], percall=False, phi=float(re.search(r"session, p = ([0-9.]+);", f["fault"]).group(1)),
                         stale_p=float(re.search(r"with p = ([0-9.]+)$", f["fault"]).group(1)))
            else:
                d["backend"] = re.match(r"(.+) \((?:primary database|queried directly on every call)\)$", f["backend"]).group(1)
            out.append(d); continue
        d = dict(name=m.group(1), cost=float(re.search(r"Cost: ([0-9.]+)\.", line).group(1)), cache=("read replica" in line or "aggregation feed" in line or "on wire service" in line), uniform="aggregation feed" in line,
                 phi=None, stale_p=None, frozen_p=None)
        d["acc"] = float(re.search(r"Accuracy(?: when the (?:replica is fresh|feed is up|wire item is correct))?: ([0-9.]+) per call", line).group(1))
        if d["cache"]:
            d["backend"] = re.search(r"(?:Served from (?:read replica|aggregation feed)|Republishes the item on wire service) (\S+?)\. ", line).group(1)
            mp = re.search(r"about (\d+)% of sessions", line)
            mc = re.search(r"with probability about (\d+)% \(per call", line)      # X experiment: per-call staleness
            d["percall"] = bool(mc)
            d["phi"] = int((mp or mc).group(1)) / 100 if (mp or mc) else 0.0
            ms = re.search(r"about (\d+)% of (?:calls|the time)", line); d["stale_p"] = int(ms.group(1)) / 100 if ms else None
            mf = re.search(r"in about (\d+)% of incidents", line); d["frozen_p"] = int(mf.group(1)) / 100 if mf else None
        else:
            mb = re.search(r"Served from (.+?) \(primary database", line) or re.search(r"Queries (.+?) directly on every call", line)
            d["backend"] = mb.group(1)
        out.append(d)
    assert len(out) == N_TOOLS, out
    if ep.get("order"):                 # Y experiment: tools are listed in a permuted order; return them by tool index
        pos = {j: p for p, j in enumerate(ep["order"])}; out = [out[pos[j]] for j in range(N_TOOLS)]
    return out

def doc_emission(d):
    """Stale emission implied by the docs of a replica tool: stale_p on the tool's first output value (= status 0); in the
    value-stuck variant the docs give only P(frozen = first value) = frozen_p, the rest is split evenly."""
    if d.get("uniform"): return [1 / K] * K          # second domain: a feed in outage reports an arbitrary value
    p = d["stale_p"] or F_STUCK
    def e(v): x = np.full(K, (1 - p) / (K - 1)); x[v] = p; return x
    if d["frozen_p"] is None: return e(0).tolist()
    rest = (1 - d["frozen_p"]) / (K - 1)
    return sum((d["frozen_p"] if v == 0 else rest) * e(v) for v in range(K)).tolist()

# ------------------------------------------------------------------ truth ----------------------------
def truth_spec(ep):
    """True model in spec format. For the value-stuck family the stale emission is the MARGINAL one (the spec class has
    one emission vector per group), so K2p on VALUE keeps a misspecification error even with a perfect recovery."""
    par = ep["params"]; srcs = [t["src"] for t in par["tools"]]; groups = []
    for src in dict.fromkeys(srcs):
        tools = [j for j, s in enumerate(srcs) if s == src]
        if src == "cache":
            groups.append(dict(tools=tools, shared_fault=par["phi"] > 0, phi=par["phi"], persistent=par["stuck"] != "percall",
                               emission=marginal_stuck(par).tolist()))
        elif src == "feed":
            groups.append(dict(tools=tools, shared_fault=par["phi2"] > 0, phi=par["phi2"], persistent=True, emission=[1 / K] * K))
        else:
            groups.append(dict(tools=tools, shared_fault=False, phi=0.0, persistent=True, emission=[1 / K] * K))
    return dict(groups=groups, accuracy=[t["q"] for t in par["tools"]], notes=[], defaults=[], clipped=[], fallback=False)

# ------------------------------------------------------------------ parsing ----------------------------
def extract_json(text):
    """The last balanced JSON object that has an "observation_model" key (placeholders echoed earlier are skipped)."""
    text = text or ""; idx = text.rfind('"observation_model"')
    while idx >= 0:
        start = text.rfind("{", 0, idx)
        if start >= 0:
            depth, ins, esc = 0, False, False
            for i in range(start, len(text)):
                c = text[i]
                if ins:
                    if esc: esc = False
                    elif c == "\\": esc = True
                    elif c == '"': ins = False
                    continue
                if c == '"': ins = True
                elif c == "{": depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[start:i + 1])
                            if isinstance(obj, dict) and "observation_model" in obj: return obj
                        except json.JSONDecodeError:
                            pass
                        break
        idx = text.rfind('"observation_model"', 0, idx)
    return None

def _num(x):
    if isinstance(x, bool) or x is None: return None
    if isinstance(x, (int, float)) and np.isfinite(x): return float(x)
    if isinstance(x, str):
        try: return float(x.strip().rstrip("%")) / (100 if x.strip().endswith("%") else 1)
        except ValueError: return None
    return None

def _bool(x):
    if isinstance(x, bool): return x
    if isinstance(x, str) and x.strip().lower() in ("true", "false"): return x.strip().lower() == "true"
    return None

def default_spec(ep, docs, note="no usable observation_model"):
    """Documented defaults only, no sharing inferred: every replica tool is its own persistent fault group with its
    documented incidence and stale emission; other tools are never faulty."""
    groups = []
    for j, d in enumerate(docs):
        if d["cache"] and d["phi"] > 0:
            groups.append(dict(tools=[j], shared_fault=True, phi=d["phi"], persistent=not d.get("percall"), emission=doc_emission(d)))
        else:
            groups.append(dict(tools=[j], shared_fault=False, phi=0.0, persistent=True, emission=[1 / K] * K))
    return dict(groups=groups, accuracy=[d["acc"] for d in docs], notes=[note], defaults=["ALL (documented default model)"],
                clipped=[], fallback=True)

def normalize_model(ep, obj, docs):
    """Recovered JSON -> spec. Anything missing or invalid is replaced by the documented value and recorded."""
    sc = SCENARIOS[ep["scenario"]]; names = [t["name"] for t in sc["tools"]]; states = sc["states"]
    alias = {}                                    # status name or any tool's output value -> status index
    for k, s in enumerate(states): alias[s] = k
    for t in sc["tools"]:
        for k, v in enumerate(t["values"]): alias.setdefault(v, k)
    om = obj.get("observation_model") if isinstance(obj, dict) else None
    tg = om.get("tool_groups") if isinstance(om, dict) else None
    if not isinstance(tg, list) or not tg: return default_spec(ep, docs, "tool_groups missing or empty")
    notes, defaults, clipped = [], [], []; assigned = set(); groups = []
    for gi, g in enumerate(tg):
        if not isinstance(g, dict): notes.append(f"group {gi} is not an object"); continue
        raw = g.get("tools") if isinstance(g.get("tools"), list) else []
        tools = []
        for t in raw:
            if t not in names: notes.append(f"unknown tool {t!r}"); continue
            j = names.index(t)
            if j in assigned or j in tools: notes.append(f"tool {t} listed twice (first group wins)"); continue
            tools.append(j)
        if not tools: continue
        cache_t = [j for j in tools if docs[j]["cache"]]
        phi, sf, per = _num(g.get("phi")), _bool(g.get("shared_fault")), _bool(g.get("persistent"))
        if sf is None:
            sf = (phi > 0) if phi is not None else bool(cache_t and docs[cache_t[0]]["phi"] > 0); defaults.append(f"g{gi}.shared_fault")
        if sf and phi is None:
            phi = docs[cache_t[0]]["phi"] if cache_t else 0.0; defaults.append(f"g{gi}.phi")
            if phi == 0.0: sf = False
        if not sf: phi = 0.0
        if 5 <= phi <= 100: clipped.append(f"g{gi}.phi={phi} read as percent"); phi = phi / 100
        if phi < 0 or phi > 0.95: clipped.append(f"g{gi}.phi={phi}"); phi = float(np.clip(phi, 0.0, 0.95))
        if per is None: per = not (cache_t and docs[cache_t[0]].get("percall")); defaults.append(f"g{gi}.persistent")
        em = g.get("stale_emission"); vec = None
        if isinstance(em, dict):
            v = [0.0] * K; ok = bool(em)
            for key, val in em.items():
                k = alias.get(key); x = _num(val)
                if k is None or x is None or x < 0: ok = False; break
                v[k] += x
            if ok and 0 < sum(v) < 1 - 1e-6:                      # partial emission: spread the remainder over unspecified statuses
                miss = [k for k in range(K) if v[k] == 0.0]
                if miss and cache_t:
                    base = np.array(doc_emission(docs[cache_t[0]]))[miss]; base = base / base.sum() if base.sum() > 0 else np.full(len(miss), 1 / len(miss))
                    rem = 1 - sum(v)
                    for k, b in zip(miss, base): v[k] = rem * float(b)
                    defaults.append(f"g{gi}.stale_emission (partial, remainder from documentation)")
            vec = v if ok and sum(v) > 0 else None
        elif isinstance(em, list) and len(em) == K and all(_num(x) is not None and _num(x) >= 0 for x in em) and sum(map(_num, em)) > 0:
            vec = [_num(x) for x in em]
        if sf:
            if vec is None:
                vec = doc_emission(docs[cache_t[0]]) if cache_t else BENIGN_EMISSION; defaults.append(f"g{gi}.stale_emission")
            vec = (np.array(vec) / np.sum(vec)).tolist()
        else:
            vec = [1 / K] * K
        groups.append(dict(tools=tools, shared_fault=bool(sf), phi=float(phi), persistent=bool(per), emission=vec)); assigned |= set(tools)
    if not groups: return default_spec(ep, docs, "no valid tool in tool_groups")
    for j in range(N_TOOLS):
        if j not in assigned:
            if docs[j]["cache"] and docs[j]["phi"] > 0:
                groups.append(dict(tools=[j], shared_fault=True, phi=docs[j]["phi"], persistent=not docs[j].get("percall"), emission=doc_emission(docs[j])))
                defaults.append(f"tool {names[j]} not in any group (own group, documented phi)")
            else:
                groups.append(dict(tools=[j], shared_fault=False, phi=0.0, persistent=True, emission=[1 / K] * K))
                defaults.append(f"tool {names[j]} not in any group (independent, never faulty)")
    acc = om.get("accuracy") if isinstance(om, dict) else None; accuracy = []
    for j in range(N_TOOLS):
        q = _num(acc.get(names[j])) if isinstance(acc, dict) else None
        if q is None: q = docs[j]["acc"]; defaults.append(f"accuracy.{names[j]}")
        elif not 0.02 <= q <= 0.98: clipped.append(f"accuracy.{names[j]}={q}"); q = float(np.clip(q, 0.02, 0.98))
        accuracy.append(q)
    groups.sort(key=lambda g: g["tools"][0])
    return dict(groups=groups, accuracy=accuracy, notes=notes, defaults=defaults, clipped=clipped, fallback=False)

# ------------------------------------------------------------------ scoring recovery ------------------
def _session_groups(spec):
    return [frozenset(g["tools"]) for g in spec["groups"] if g["shared_fault"] and g["phi"] > 0 and g["persistent"]]

def _phi_eff(spec):
    """per-tool probability of being on a faulty backend (persistent or not) in the spec."""
    e = [0.0] * N_TOOLS
    for g in spec["groups"]:
        if g["shared_fault"]:
            for j in g["tools"]: e[j] = g["phi"]
    return e

def _bits(spec):
    sg = _session_groups(spec)
    single = [any(j in g for g in sg) for j in range(N_TOOLS)]
    pairs = [any(i in g and j in g for g in sg) for i in range(N_TOOLS) for j in range(i + 1, N_TOOLS)]
    return single, pairs

def score_recovery(ep, spec):
    """Per-episode recovery metrics (None = not applicable). Truth: persistent shared fault of the replica iff phi > 0."""
    tr = truth_spec(ep); par = ep["params"]; cache = [j for j, t in enumerate(par["tools"]) if t["src"] == "cache"]
    s_h, p_h = _bits(spec); s_t, p_t = _bits(tr); bits = [a == b for a, b in zip(s_h + p_h, s_t + p_t)]
    pe_h, pe_t = _phi_eff(spec), _phi_eff(tr); faulty_t = par["phi"] > 0
    gA = next(g for g in spec["groups"] if 0 in g["tools"])
    r = dict(struct_exact=float(all(bits)), single_ok=float(all(a == b for a, b in zip(s_h, s_t))),
             pair_ok=float(all(a == b for a, b in zip(p_h, p_t))), bits_correct=float(np.mean(bits)),
             fault_present_ok=float(any(g["shared_fault"] and g["phi"] > 0 for g in spec["groups"] if set(g["tools"]) & set(cache)) == faulty_t),
             phi_abs_err=float(np.mean([abs(pe_h[j] - pe_t[j]) for j in cache])),
             phantom_fault=float(any(pe_h[j] > 0 for j in range(N_TOOLS) if j not in cache)),
             acc_mae=float(np.mean([abs(a - b) for a, b in zip(spec["accuracy"], tr["accuracy"])])),
             n_defaults=len(spec["defaults"]), fallback=float(spec["fallback"]),
             persistent_ok=None, emission_tv=None)
    if faulty_t:
        r["persistent_ok"] = float(gA["shared_fault"] and gA["phi"] > 0 and gA["persistent"])
        gt = next(g for g in tr["groups"] if 0 in g["tools"])
        if gA["shared_fault"]:       # undefined (None) when the agent says the replica has no fault: see fault_present_ok
            r["emission_tv"] = float(0.5 * np.abs(np.array(gA["emission"]) - np.array(gt["emission"])).sum())
    return r

# ------------------------------------------------------------------ exact planning on the recovered model ----
def build_recovered_model(ep, spec):
    par = ep["params"]; tools = [None] * N_TOOLS; groups = {}
    for gi, g in enumerate(spec["groups"]):
        faulty = g["shared_fault"] and g["phi"] > 0
        for j in g["tools"]:
            t = dict(kind=par["tools"][j].get("kind", "sym"), q=spec["accuracy"][j], cost=par["tools"][j]["cost"], group=None, femit=None)
            if faulty and g["persistent"]:
                t["group"] = f"g{gi}"; t["femit"] = np.array(g["emission"]); groups[f"g{gi}"] = g["phi"]
            elif faulty:                                   # every call faulty independently with probability phi
                t["marg_phi"] = g["phi"]; t["femit"] = np.array(g["emission"])
            tools[j] = t
    return Model(K, np.array(par["prior"]), tools, groups, LOSS, BUDGET)

def plan_and_play(ep, spec, env):
    """Exact Bayes-optimal DP of the RECOVERED model chooses every action (its belief is updated with the real outputs)."""
    M = build_recovered_model(ep, spec); b = M.prior.copy(); verdicts = env.sc["states"] + ["escalate"]
    for _ in range(4 * N_TOOLS + 4):
        if env.done: break
        a = M.V(b, env.budget)[1]
        if a[0] == "stop": env.submit(verdicts[a[1]]); break
        j = a[1]; env.call(env.names[j])
        if env.log[-1]["type"] != "call": break              # cannot happen (planner uses the true costs); guard only
        nb = M.update(b, j, env.hist[-1][1]); b = nb if nb is not None else M.prior.copy()
    return M

# ------------------------------------------------------------------ deterministic doc reader (mocks, tests) ----
def doc_reader_model(ep, system_text, link=True):
    """The model JSON a perfect reader of the tool docs would write. link=True groups tools by identical backend name;
    link=False reads each tool's doc separately (no cross-tool sharing). Uses only text visible in the prompt."""
    docs = parse_docs(ep, system_text); sc = SCENARIOS[ep["scenario"]]; st = sc["states"]; groups = []
    keys = [d["backend"] if link else j for j, d in enumerate(docs)]
    for key in dict.fromkeys(keys):
        idx = [j for j, k in enumerate(keys) if k == key]; d0 = docs[idx[0]]
        faulty = d0["cache"] and d0["phi"] > 0
        groups.append(dict(tools=[docs[j]["name"] for j in idx], shared_fault=bool(faulty), phi=d0["phi"] if faulty else 0.0,
                           persistent=True, stale_emission={s: round(p, 4) for s, p in zip(st, doc_emission(d0))} if faulty else {}))
    return {"observation_model": dict(tool_groups=groups, accuracy={d["name"]: d["acc"] for d in docs})}
