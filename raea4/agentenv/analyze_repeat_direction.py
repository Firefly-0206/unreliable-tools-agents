"""Closed-loop counterpart of bench/analyze_repeat_direction.py (offline, no model calls).

Do the LLM agents' errors on repeat-cache calls go in the direction predicted for MMI (over-valuing repeat calls of a
persistent source) in the closed-loop environment?

  python agentenv/analyze_repeat_direction.py                         # default logs, condition K1
  python agentenv/analyze_repeat_direction.py logs/new_run.jsonl [--conds K1 K3] [--out results/repeat_direction_loop]

Each logged LLM trajectory is replayed through the exact oracle (oracle.qvalues). At every decision point (each tool
call and the final submit) we record the LLM's action and, at the SAME history, the action the exact Bayes-optimal
planner, MMI-DP and FaultUnaware-DP would take (policies plan with their own model and belief, exactly as in
oracle.baseline_policies without forbid / discard). A cache-repeat (CR) action = a call to a tool on the cache source
('c') after the cache source was already used in the episode. Per step (all with the TRUE-model Q of the chosen action):
  over-repeat  = chose a CR action whose advantage is > 1e-6                  (eligible: a CR action is affordable and none is optimal)
  under-repeat = chose a non-CR action with advantage > 1e-6                  (eligible: some CR action is optimal)
  early stop   = submitted while a call was optimal;  late stop = called while submitting was optimal.
The same rates are computed for the planners at the LLM's histories. Episode-level rates are averaged over episodes
(eligible episodes only); CIs are the nested bootstrap of nestboot.py (scenarios, then episodes).
Trajectory level: the LLM's repeat-cache calls and regret vs. the exact EXPECTED values of Oracle-DP / MMI-DP /
FaultUnaware-DP on the same episodes (results/baselines.json per_episode), paired by episode.
Families: PERSIST, SHARED, MIXED, VALUE have a persistent cache fault; TRANSIENT (twin of SHARED) and TRANSIENT_P (twin
of PERSIST) have phi = 0, same parameters: there MMI-DP = Oracle-DP and a cache repeat is genuinely informative (control).
Regret split (estimates): cost part = realised call cost - Oracle-DP's expected call cost, verdict/stop part = regret - cost part
(for MMI-DP / FaultUnaware-DP both parts are exact expectations).
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from env import (BUDGET, CORE, TWINS, build_model, sim_spec, srcmap, load_episodes, load_truth)   # noqa: E402
from oracle import qvalues, true_model                                                            # noqa: E402
from nestboot import nested_ci, paired_ci                                                         # noqa: E402
from sim import mmi, fault_unaware                                                                # noqa: E402

EPS = 1e-6
FAULT = ["PERSIST", "SHARED", "MIXED", "VALUE"]; TWIN = ["TRANSIENT", "TRANSIENT_P"]
GROUPS = {"FAULT": FAULT, "TWIN": TWIN}
POLS = ["MMI-DP", "FaultUnaware-DP"]
DEFAULT_LOGS = ["logs/codex_terra_medium.jsonl", "logs/ds_chat_fc.jsonl"]

def pol_state(M):
    return [M.prior.copy()]

def replay(ep, log, truth):
    """-> list of per-step dicts for the LLM and the policies at the LLM's histories."""
    par = ep["params"]; sm = srcmap(par); T = true_model(ep)
    P = {"MMI-DP": mmi(sim_spec(par)), "FaultUnaware-DP": fault_unaware(sim_spec(par))}
    pb = {n: M.prior.copy() for n, M in P.items()}
    b = T.prior.copy(); B = float(T.budget); used = {}; steps = []; cost = 0.0
    def kind(a):
        if a[0] == "stop": return "stop"
        j = a[1]; s = sm[j]
        if s == "c": return "cache_repeat" if used.get("c", 0) > 0 else "cache_first"
        if s == "tool2": return "log_repeat" if used.get("tool2", 0) > 0 else "log_first"
        return "other_probe"
    for e in log:
        if e["type"] == "invalid": continue
        q = qvalues(T, b, B); V = min(q.values())
        cr_all = [k for k in q if k[0] == "probe" and sm[k[1]] == "c" and used.get("c", 0) > 0]
        rec = dict(V=V, B=B, cr_avail=bool(cr_all), cr_opt=bool(cr_all) and min(q[k] for k in cr_all) - V <= EPS,
                   opt_kind=kind(min(q, key=q.get)), opt_stop=min(q, key=q.get)[0] == "stop")
        acts = {"LLM": (("probe", e["j"]) if e["type"] == "call" else ("stop", e["a"]))}
        acts["Oracle-DP"] = min(q, key=q.get)
        for n, M in P.items(): acts[n] = M.V(pb[n], B)[1]
        for n, a in acts.items():
            a = tuple(a)
            if a not in q:                          # a policy action that is unaffordable cannot occur; guard anyway
                rec[n] = dict(kind=kind(a), adv=np.nan); continue
            rec[n] = dict(kind=kind(a), adv=q[a] - V)
        steps.append(rec)
        if e["type"] != "call": break
        j, o = e["j"], e["o"]
        used[sm[j]] = used.get(sm[j], 0) + 1; cost += T.tools[j]["cost"]
        b = T.update(b, j, o); B -= T.tools[j]["cost"]
        for n, M in P.items():
            b2 = M.update(pb[n], j, o); pb[n] = b2 if b2 is not None else M.prior.copy()
    return steps, cost

def episode_metrics(steps, who):
    """per-episode counts for one decision maker."""
    d = dict(over_elig=0, over=0, under_elig=0, under=0, early_elig=0, early=0, late_elig=0, late=0, n_cr=0, n_steps=len(steps))
    for s in steps:
        x = s[who]
        if np.isnan(x["adv"]): continue
        wrong = x["adv"] > EPS; cr = x["kind"] == "cache_repeat"
        if s["cr_avail"] and not s["cr_opt"]: d["over_elig"] += 1; d["over"] += int(cr and wrong)
        if s["cr_opt"]: d["under_elig"] += 1; d["under"] += int((not cr) and wrong)
        if not s["opt_stop"]: d["early_elig"] += 1; d["early"] += int(x["kind"] == "stop" and wrong)
        else: d["late_elig"] += 1; d["late"] += int(x["kind"] != "stop" and wrong)
        d["n_cr"] += int(cr)
    return d

def rate(ms, num, den):
    return {e: m[num] / m[den] for e, m in ms.items() if m[den] > 0}

def ci_of(vals, eps, sc):
    ids = list(vals)
    if not ids: return (np.nan, np.nan, np.nan, 0)
    m, lo, hi = nested_ci([vals[i] for i in ids], ids, [sc[i] for i in ids]); return (m, lo, hi, len(ids))

def fmt(t, pct=False, signed=False):
    m, lo, hi = t[0], t[1], t[2]
    if m is None or np.isnan(m): return "-"
    k = 100 if pct else 1; d = 0 if pct else 2; sg = "+" if signed else ""
    lo_hi = f" [{lo * k:{sg}.{d}f}, {hi * k:{sg}.{d}f}]" if not np.isnan(lo) else ""
    return f"{m * k:{sg}.{d}f}{lo_hi}"

def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y)[0]; return float(r)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("logs", nargs="*")
    ap.add_argument("--conds", nargs="+", default=["K1"])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "repeat_direction_loop"))
    ap.add_argument("--note", default=os.path.join(HERE, "results", "repeat_direction_loop_interpretation.md"))
    a = ap.parse_args()
    files = [f if os.path.isabs(f) or os.path.exists(f) else os.path.join(HERE, f) for f in (a.logs or DEFAULT_LOGS)]
    eps, truth = load_episodes(), load_truth(); sc = {i: e["scenario"] for i, e in eps.items()}
    per = json.load(open(os.path.join(HERE, "results", "baselines.json")))["per_episode"]
    recs = {}
    for f in files:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["status"].startswith("api_error") or r["condition"] not in a.conds: continue
            recs[(r["episode"], r["condition"], r["model"], r["protocol"], r.get("sample", 0))] = r
    cells = defaultdict(dict)           # (model, protocol, cond) -> episode -> data
    for k, r in recs.items():
        ep = eps[k[0]]; steps, cost = replay(ep, r["log"], truth[k[0]])
        from oracle import score_trajectory
        s = score_trajectory(ep, r["log"], truth[k[0]])
        info = sum(st["adv"] for st in s["steps"] if st["action"][0] == "probe") + sum(st["adv_stoptime"] for st in s["steps"] if st["action"][0] == "stop")
        cells[(k[2], k[3], k[1])][k[0]] = dict(steps=steps, cost=cost, regret=s["regret"], info=info,
                                              verdict=sum(st["adv_verdict"] for st in s["steps"] if st["action"][0] == "stop"),
                                              rep_cache=sum(1 for st in steps if st["LLM"]["kind"] == "cache_repeat"),
                                              rep_log=sum(1 for st in steps if st["LLM"]["kind"] == "log_repeat"),
                                              calls=sum(1 for st in steps if st["LLM"]["kind"] != "stop"),
                                              no_submit=bool(s["forced"]))
    out = dict(cells={}); md = ["# Closed loop: direction of cache-repeat errors (agentenv)\n",
                               "Generated by `agentenv/analyze_repeat_direction.py`; no model calls. See the script docstring for definitions. "
                               "CR = repeat call to the cache source; CIs: nested bootstrap over scenarios then episodes.\n"]
    for (model, prot, cond), E in sorted(cells.items()):
        ids = sorted(E)
        md += [f"\n## {model} / {prot} / {cond} ({len(ids)} episodes)\n"]
        cell = dict(n=len(ids), step_rates={}, trajectory={}, tests={})
        # -------------------------- step-level rates at the LLM's histories
        md += ["### Step level: errors at the LLM's own histories\n",
               "over-repeat = P(CR call with positive advantage | a CR call is affordable and none is optimal); under-repeat = P(non-CR action with positive advantage | a CR call is optimal); "
               "early stop = P(submit although a call was optimal); late stop = P(call although submitting was optimal). Episode-level means over eligible episodes.\n",
               "| Group | Who | n ep. (over-eligible) | Over-repeat % | n ep. (under-elig.) | Under-repeat % | Early stop % | Late stop % |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for gname, fams in list(GROUPS.items()) + [(f, [f]) for f in FAULT + TWIN]:
            gi = [i for i in ids if eps[i]["family"] in fams]
            if not gi: continue
            for who in ["LLM", "Oracle-DP"] + POLS:
                if who == "Oracle-DP": continue
                ms = {i: episode_metrics(E[i]["steps"], who) for i in gi}
                r = {nm: ci_of(rate(ms, num, den), eps, sc) for nm, num, den in
                     [("over", "over", "over_elig"), ("under", "under", "under_elig"), ("early", "early", "early_elig"), ("late", "late", "late_elig")]}
                cell["step_rates"][f"{gname}|{who}"] = r
                md.append(f"| {gname} | {'**LLM**' if who == 'LLM' else who} | {r['over'][3]} | {fmt(r['over'], True)} | {r['under'][3]} | {fmt(r['under'], True)} | {fmt(r['early'], True)} | {fmt(r['late'], True)} |")
        # -------------------------- trajectory level vs exact expectations of the policies
        md += ["\n### Trajectory level: repeat-cache calls and regret per episode (LLM realised, policies exact expectation)\n",
               "Paired differences LLM - policy over the same episodes. Cost/verdict split: see docstring (LLM parts are estimates).\n",
               "| Group | Who | rep. cache calls | d vs Oracle-DP | d vs MMI-DP | d vs FaultUnaware-DP | Regret | regret: cost part | regret: verdict/stop part | n |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for gname, fams in list(GROUPS.items()) + [(f, [f]) for f in FAULT + TWIN]:
            gi = [i for i in ids if eps[i]["family"] in fams]
            if not gi: continue
            scs = [sc[i] for i in gi]
            def pol_vals(pn, key): return {i: per[i][pn][key] for i in gi}
            llm_rc = {i: E[i]["rep_cache"] for i in gi}
            row = {"LLM": ci_of(llm_rc, eps, sc)}
            d = {pn: paired_ci(llm_rc, pol_vals(pn, "rep_cache"), sc) for pn in ["Oracle-DP", "MMI-DP", "FaultUnaware-DP"]}
            orc_cost = {i: per[i]["Oracle-DP"]["cost"] for i in gi}
            cost_part = {i: E[i]["cost"] - orc_cost[i] for i in gi}
            verd_part = {i: E[i]["regret"] - cost_part[i] for i in gi}
            reg = {i: E[i]["regret"] for i in gi}
            cell["trajectory"][f"{gname}|LLM"] = dict(rep_cache=row["LLM"], d_oracle=d["Oracle-DP"], d_mmi=d["MMI-DP"], d_fu=d["FaultUnaware-DP"],
                                                      regret=ci_of(reg, eps, sc), cost_part=ci_of(cost_part, eps, sc), verdict_part=ci_of(verd_part, eps, sc))
            c = cell["trajectory"][f"{gname}|LLM"]
            md.append(f"| {gname} | **LLM** | {fmt(c['rep_cache'])} | {fmt(d['Oracle-DP'], signed=True)} | {fmt(d['MMI-DP'], signed=True)} | {fmt(d['FaultUnaware-DP'], signed=True)} | "
                      f"{fmt(c['regret'])} | {fmt(c['cost_part'], signed=True)} | {fmt(c['verdict_part'], signed=True)} | {len(gi)} |")
            for pn in ["Oracle-DP"] + POLS:
                rc = {i: per[i][pn]["rep_cache"] for i in gi}
                rg = {i: per[i][pn]["total"] - per[i]["Oracle-DP"]["total"] for i in gi}
                cp = {i: per[i][pn]["cost"] - per[i]["Oracle-DP"]["cost"] for i in gi}
                vp = {i: rg[i] - cp[i] for i in gi}
                cell["trajectory"][f"{gname}|{pn}"] = dict(rep_cache=ci_of(rc, eps, sc), regret=ci_of(rg, eps, sc), cost_part=ci_of(cp, eps, sc), verdict_part=ci_of(vp, eps, sc))
                c = cell["trajectory"][f"{gname}|{pn}"]
                md.append(f"| {gname} | {pn} | {fmt(c['rep_cache'])} | | | | {fmt(c['regret'])} | {fmt(c['cost_part'], signed=True)} | {fmt(c['verdict_part'], signed=True)} | {len(gi)} |")
        # -------------------------- fault minus twin (difference in differences vs Oracle-DP), by pair
        md += ["\n### Fault family minus its phi = 0 twin, repeat-cache calls in excess of Oracle-DP (difference in differences, paired by episode index)\n",
               "excess(x, ep) = rep. cache calls of x - Oracle-DP's expected rep. cache calls on the same episode; value = excess in the fault episode - excess in its twin. "
               "MMI-DP is the prediction of the MMI mechanism; positive = more over-repeating when the fault is present. CAUTION: because Oracle-DP repeats LESS when the fault is present, "
               "any policy whose own repeat count does not change gets a positive value here; read it together with the raw change in each policy's own repeat count in the next table.\n",
               "| Pair | Who | Fault - twin excess [95% CI] | n pairs |", "| --- | --- | --- | --- |"]
        pair_sc = {e["pair"]: e["scenario"] for e in eps.values()}
        raw_rows = []
        for tw, base in TWINS.items():
            by_pair = {fam: {eps[i]["pair"]: i for i in ids if eps[i]["family"] == fam} for fam in (base, tw)}
            common = sorted(set(by_pair[base]) & set(by_pair[tw]))
            if not common: continue
            for who in ["LLM", "MMI-DP", "FaultUnaware-DP"]:
                def exc(i): return (E[i]["rep_cache"] if who == "LLM" else per[i][who]["rep_cache"]) - per[i]["Oracle-DP"]["rep_cache"]
                dd = {p: exc(by_pair[base][p]) - exc(by_pair[tw][p]) for p in common}
                m, lo, hi = nested_ci(list(dd.values()), list(dd), [pair_sc[p] for p in dd])
                cell["tests"][f"did_rep_cache|{base}-{tw}|{who}"] = (m, lo, hi, len(dd))
                md.append(f"| {base} - {tw} | {'**LLM**' if who == 'LLM' else who} | {fmt((m, lo, hi), signed=True)} | {len(dd)} |")
            raw_rows.append((base, tw, by_pair, common))
        md += ["\n### Raw change in each policy's OWN repeat-cache count, fault family minus its twin (paired by episode index; no reference policy subtracted)\n",
               "A policy that reacts to the fault like the oracle should show a negative value; MMI-DP's inflation shows as a positive value. An LLM near zero is insensitive to the fault.\n",
               "| Pair | Who | Fault - twin change in own rep. cache calls [95% CI] | n pairs |", "| --- | --- | --- | --- |"]
        for base, tw, by_pair, common in raw_rows:
            for who in ["LLM", "Oracle-DP", "MMI-DP", "FaultUnaware-DP"]:
                def rcn(i): return E[i]["rep_cache"] if who == "LLM" else per[i][who]["rep_cache"]
                dd = {p: rcn(by_pair[base][p]) - rcn(by_pair[tw][p]) for p in common}
                m, lo, hi = nested_ci(list(dd.values()), list(dd), [pair_sc[p] for p in dd])
                cell["tests"][f"raw_change_rep_cache|{base}-{tw}|{who}"] = (m, lo, hi, len(dd))
                md.append(f"| {base} - {tw} | {'**LLM**' if who == 'LLM' else who} | {fmt((m, lo, hi), signed=True)} | {len(dd)} |")
        # -------------------------- is the LLM worse where MMI-DP is wrong?
        md += ["\n### Is the LLM worse exactly on the episodes where MMI-DP is wrong?\n",
               "Episodes of the fault families, split by MMI-DP's exact expected regret (high = above the median over those episodes). Difference in the LLM's regret between the halves, nested bootstrap; "
               "placebo: the same split by FaultUnaware-DP's regret. Spearman across episodes (95% CI: bootstrap over episodes) between the LLM's regret and the policy's regret, and between the "
               "LLM's repeat-cache calls in excess of Oracle-DP and MMI-DP's excess.\n",
               "| Contrast | Value [95% CI] |", "| --- | --- |"]
        fi = [i for i in ids if eps[i]["family"] in FAULT]
        for pn in POLS:
            rg = {i: per[i][pn]["total"] - per[i]["Oracle-DP"]["total"] for i in fi}
            med = float(np.median(list(rg.values()))); hi_ = [i for i in fi if rg[i] > med]; lo_ = [i for i in fi if rg[i] <= med]
            da = {i: E[i]["regret"] for i in hi_}; db = {i: E[i]["regret"] for i in lo_}
            ma, mb = np.mean(list(da.values())), np.mean(list(db.values()))
            # difference of group means with a nested bootstrap over scenarios then episodes
            rng = np.random.default_rng(0); bs = []
            scn = sorted({sc[i] for i in fi})
            for _ in range(2000):
                xa, xb = [], []
                for s_ in rng.integers(0, len(scn), len(scn)):
                    ia = [i for i in hi_ if sc[i] == scn[s_]]; ib = [i for i in lo_ if sc[i] == scn[s_]]
                    if ia: xa += [da[i] for i in rng.choice(ia, len(ia))]
                    if ib: xb += [db[i] for i in rng.choice(ib, len(ib))]
                if xa and xb: bs.append(np.mean(xa) - np.mean(xb))
            t = (float(ma - mb), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(fi))
            cell["tests"][f"llm_regret_high_minus_low_{pn}"] = t
            md.append(f"| LLM regret: {pn}-regret high - low (median {med:.2f}; {len(hi_)} vs {len(lo_)} episodes) | {fmt(t, signed=True)} |")
            x = np.array([E[i]["regret"] for i in fi]); y = np.array([rg[i] for i in fi])
            rho = spearman(x, y); rng = np.random.default_rng(1)
            bs = [spearman(x[j], y[j]) for j in (rng.integers(0, len(x), len(x)) for _ in range(1000))]
            cell["tests"][f"spearman_regret_{pn}"] = (rho, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), len(x))
            md.append(f"| Spearman(LLM regret, {pn} regret), fault families | {fmt(cell['tests'][f'spearman_regret_{pn}'], signed=True)} |")
        x = np.array([E[i]["rep_cache"] - per[i]["Oracle-DP"]["rep_cache"] for i in fi]); y = np.array([per[i]["MMI-DP"]["rep_cache"] - per[i]["Oracle-DP"]["rep_cache"] for i in fi])
        rho = spearman(x, y); rng = np.random.default_rng(2)
        bs = [spearman(x[j], y[j]) for j in (rng.integers(0, len(x), len(x)) for _ in range(1000))]
        cell["tests"]["spearman_excess_rep_cache_MMI"] = (rho, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), len(x))
        md.append(f"| Spearman(LLM excess rep. cache calls, MMI-DP excess), fault families | {fmt(cell['tests']['spearman_excess_rep_cache_MMI'], signed=True)} |")
        out["cells"]["|".join((model, prot, cond))] = cell
    if os.path.exists(a.note): md += ["\n" + open(a.note, encoding="utf-8").read()]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out + ".md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump(out, open(a.out + ".json", "w"), indent=1, default=float)
    print("\n".join(md))

if __name__ == "__main__":
    main()
