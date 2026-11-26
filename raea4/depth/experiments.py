"""D1-D5. Usage: python experiments.py D1 D2 D3 D4 D5   (D5 reads results/D3_rows.json)"""
import json, sys, time
import numpy as np
from scipy.stats import spearmanr
from core import (Q, evaluate, instance, model, build, pmap, stack, summarize, boot, source_of, gross,
                  overvalue_max, overvalue_plan, tool_sources)

FAMS = ["RETRY", "SHARED"]
N_INST = {"D1": 300, "D2": 300, "D3": 300, "D4": 200}
SEED = {"D1": 401, "D2": 402, "D3": 403, "D4": 404}
PLANNERS = [("VOI-1", ("look", 1)), ("VOI-2", ("look", 2)), ("VOI-3", ("look", 3)), ("VOI-4", ("look", 4)), ("DP", ("dp",))]
BUDGETS = [2, 4, 6, 8]
D1_MODELS = ["true", "mmi", "fit_wrongmech", "fault_unaware", "split_true"]
PREFIX = {"RETRY": (0, 0, 0), "SHARED": (0, 1, 0)}       # inherited history with repeat-source reports

# ------------------------------------------------------------------ D1 ------------------------
def repeat_overvalue(M, true, sp):
    """Original (narrow) feature: model-minus-truth gross value of re-probing tool 0 after one benign
    report from tool 0. Blind to cross-tool repeats (tool 1 after tool 0 in SHARED); kept for reference."""
    return gross(M, M.update(M.prior, 0, 0), 0) - gross(true, true.update(true.prior, 0, 0), 0)

def job_D1(a):
    fam, idx = a
    sp0, _, S, O = instance(fam, SEED["D1"], idx, N_INST["D1"], n_cal=100)
    row, feat = {}, {}
    g = list(sp0["groups"].values())[0]
    feat.update(phi=g, pi0=float(sp0["prior"][0]))
    for B in BUDGETS:
        sp = dict(sp0, budget=B); true = build(sp)
        row[f"Oracle-DP@{B}"] = evaluate(true, true, ("dp",), sp)
        for m in D1_MODELS:
            M = model(m, sp, S, O)
            if B == 6:
                feat[f"ov_tool0|{m}"] = repeat_overvalue(M, true, sp)
                feat[f"ov_max|{m}"] = overvalue_max(M, true, sp)
            feat[f"ov_plan|{m}@{B}"] = overvalue_plan(true, M, ("dp",), sp)
            for pn, pol in PLANNERS:
                r = evaluate(true, M, pol, sp); own = evaluate(M, M, pol, sp)
                r.update(pred_total=own["total"], pred_repeat=own["repeat"], optimism=r["total"] - own["total"],
                         root=Q(M, {j: source_of(t, j) for j, t in enumerate(sp["tools"])}, M.prior, B,
                                None if pol[0] == "dp" else pol[1], frozenset(), False)[0])
                row[f"{m}|{pn}@{B}"] = r
    return row, feat

def D1():
    out = {}
    for fam in FAMS:
        res = pmap(job_D1, [(fam, i) for i in range(N_INST["D1"])])
        rows = [r for r, _ in res]; feats = [f for _, f in res]
        st = stack(rows); out[fam] = {"by_budget": {}, "analysis": {}}
        for B in BUDGETS:
            sub = {p.split("@")[0]: v for p, v in st.items() if p.endswith(f"@{B}")}
            out[fam]["by_budget"][str(B)] = summarize(sub)
        # amplification analysis: per-instance DP - VOI-1 regret vs over-valuation / planned repeats
        an = {}
        for m in D1_MODELS:
            ov0 = np.array([f[f"ov_tool0|{m}"] for f in feats]); ovm = np.array([f[f"ov_max|{m}"] for f in feats])
            a_m = {}
            for B in BUDGETS:
                ovp = np.array([f[f"ov_plan|{m}@{B}"] for f in feats])
                amp = st[f"{m}|DP@{B}"]["total"] - st[f"{m}|VOI-1@{B}"]["total"]
                rep = st[f"{m}|DP@{B}"]["pred_repeat"]
                z, t5 = rep <= 1e-9, rep <= 0.05
                a_m[str(B)] = dict(
                    amp=boot(amp),
                    frac_ov_tool0_pos=float((ov0 > 1e-6).mean()), frac_ov_max_pos=float((ovm > 1e-6).mean()),
                    frac_ov_plan_pos=float((ovp > 1e-6).mean()), mean_ov_plan=float(ovp.mean()),
                    spearman_amp_ov_tool0=_sp(amp, ov0), spearman_amp_ov_max=_sp(amp, ovm),
                    spearman_amp_ov_plan=_sp(amp, ovp),
                    spearman_amp_planned_repeats=_sp(amp, rep),
                    spearman_amp_any_planned_repeat=_sp(amp, (~z).astype(float)),
                    amp_if_zero_planned_repeat=_b(amp[z]), frac_zero_planned_repeat=float(z.mean()),
                    amp_if_planned_repeat=_b(amp[~z]),
                    amp_if_planned_repeat_le005=_b(amp[t5]), frac_planned_repeat_gt005=float((~t5).mean()),
                    max_abs_amp_if_zero_planned_repeat=float(np.abs(amp[z]).max()) if z.any() else None)
            an[m] = a_m
        out[fam]["analysis"] = an
        print("D1", fam, flush=True)
    return out

def _b(x):
    return boot(x) if len(x) > 1 else None

def _sp(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12: return None
    return float(spearmanr(x, y).correlation)

# ------------------------------------------------------------------ D2 ------------------------
# mode -> evaluate kwargs. "discard" is the naive variant (planner unaware of its own discard rule);
# "discard-consistent" plans under the discard dynamics; "group+forbid" keeps all reports, pooled per source.
MODES = {"neither": {}, "discard": dict(discard=True), "forbid": dict(forbid=True), "both": dict(discard=True, forbid=True)}
MODES_DP = {"discard-consistent": dict(discard=True, plan_discard=True), "group+forbid": dict(pool=True, forbid=True)}
MODES_TOOL = {"both(per-tool)": dict(discard=True, forbid=True), "group+forbid(per-tool)": dict(pool=True, forbid=True)}

def job_D2(a):
    fam, idx = a
    sp, true, S, O = instance(fam, SEED["D2"], idx, N_INST["D2"], n_cal=100)
    row = {}; ts = tool_sources(sp)
    for setting, pre in [("fresh", ()), ("inherited", PREFIX[fam])]:
        row[f"{setting}|Oracle-DP"] = evaluate(true, true, ("dp",), sp, prefix=pre)
        for m in ["true", "fit", "fit_indep", "fault_unaware", "mmi"]:
            M = model(m, sp, S, O)
            for mode, kw in MODES.items():
                for pn, pol in [("DP", ("dp",)), ("VOI-1", ("look", 1))]:
                    row[f"{setting}|{pn}[{m}]+{mode}"] = evaluate(true, M, pol, sp, prefix=pre, **kw)
            for mode, kw in MODES_DP.items():
                row[f"{setting}|DP[{m}]+{mode}"] = evaluate(true, M, ("dp",), sp, prefix=pre, **kw)
            if fam == "SHARED":
                for mode, kw in MODES_TOOL.items():
                    row[f"{setting}|DP[{m}]+{mode}"] = evaluate(true, M, ("dp",), sp, prefix=pre, asrc=ts, **kw)
    return row

def D2():
    out = {}
    for fam in FAMS:
        st = stack(pmap(job_D2, [(fam, i) for i in range(N_INST["D2"])]))
        out[fam] = {}
        for setting in ["fresh", "inherited"]:
            sub = {p.split("|", 1)[1]: v for p, v in st.items() if p.startswith(setting + "|")}
            out[fam][setting] = summarize(sub)
        print("D2", fam, flush=True)
    return out

# ------------------------------------------------------------------ D3 ------------------------
NS = [30, 100, 300]

def job_D3(a):
    fam, idx = a
    sp, true, S, O = instance(fam, SEED["D3"], idx, N_INST["D3"], n_cal=max(NS))
    ts = tool_sources(sp)
    row = {"Oracle-DP": evaluate(true, true, ("dp",), sp),
           "DP[fault_unaware]": evaluate(true, model("fault_unaware", sp), ("dp",), sp),
           "DP[fault_unaware]+first-report": evaluate(true, model("fault_unaware", sp), ("dp",), sp, True, True)}
    for n in NS:
        dep, ind, nof = model(f"fit@{n}", sp, S, O), model(f"fit_indep@{n}", sp, S, O), model(f"fit_nofault@{n}", sp, S, O)
        row[f"DP[dep@{n}]"] = evaluate(true, dep, ("dp",), sp)
        row[f"DP[dep@{n}]+forbid"] = evaluate(true, dep, ("dp",), sp, False, True)
        row[f"VOI-1[dep@{n}]"] = evaluate(true, dep, ("look", 1), sp)
        row[f"DP[indep@{n}]"] = evaluate(true, ind, ("dp",), sp)
        row[f"VOI-1[indep@{n}]"] = evaluate(true, ind, ("look", 1), sp)
        # fresh episodes: forbid == first-report (no repeat ever enters the history), so only one is run
        row[f"DP[indep@{n}]+first-report"] = evaluate(true, ind, ("dp",), sp, True, True)
        row[f"DP[indep@{n}]+first-report(per-tool)"] = evaluate(true, ind, ("dp",), sp, True, True, asrc=ts)
        row[f"DP[indep@{n}]+discard(naive)"] = evaluate(true, ind, ("dp",), sp, True, False)
        row[f"DP[symq-nofault@{n}]"] = evaluate(true, nof, ("dp",), sp)
        a3 = model(f"fit_indep_a0.001@{n}", sp, S, O)          # smoothing check for the indep-model trend
        row[f"DP[indep,alpha=1e-3@{n}]"] = evaluate(true, a3, ("dp",), sp)
    row["_cost"] = {"probe_cost_per_episode": float(3 * sum(t["cost"] for t in sp["tools"]))}
    return row

def D3():
    out, raw = {}, {}
    for fam in FAMS:
        rows = pmap(job_D3, [(fam, i) for i in range(N_INST["D3"])])
        raw[fam] = rows
        st = stack([{k: v for k, v in r.items() if k != "_cost"} for r in rows])
        out[fam] = summarize(st)
        print("D3", fam, flush=True)
    json.dump({f: [{k: {m: float(x) for m, x in v.items()} for k, v in r.items()} for r in rs] for f, rs in raw.items()},
              open("results/D3_rows.json", "w"))
    return out

# ------------------------------------------------------------------ D4 ------------------------
SEVS, ESCS = [1, 3, 10, 30], [2, 4, 8]
D4_POLS = {"Oracle-DP": (("dp",), "true", 0), "VOI-1[exact]": (("look", 1), "true", 0),
           "DP[MMI]": (("dp",), "mmi", 0), "DP[fault-unaware]": (("dp",), "fault_unaware", 0),
           "DP[oracle dedup]": (("dp",), "true", 1), "DP[fitted dep]": (("dp",), "fit", 0),
           "DP[fitted indep]": (("dp",), "fit_indep", 0), "DP[fitted indep]+first-report": (("dp",), "fit_indep", 1),
           "DP[fitted indep]+first-report(per-tool)": (("dp",), "fit_indep", 2),
           "Fixed-3[exact]": (("fixed", 3), "true", 0),
           "Thr-0.9[exact]": (("thr", 0.9), "true", 0)}

def job_D4(a):
    fam, idx = a; row = {}
    for sev in SEVS:
        for esc in ESCS:
            sp, true, S, O = instance(fam, SEED["D4"], idx, N_INST["D4"], n_cal=100, spec_kw=dict(sev_mult=sev, esc=esc))
            ms = {}
            for p, (pol, m, dd) in D4_POLS.items():
                if m not in ms: ms[m] = model(m, sp, S, O)
                row[f"{sev}|{esc}|{p}"] = evaluate(true, ms[m], pol, sp, bool(dd), bool(dd),
                                                   asrc=tool_sources(sp) if dd == 2 else None)
    return row

def D4():
    out = {}
    for fam in FAMS:
        st = stack(pmap(job_D4, [(fam, i) for i in range(N_INST["D4"])]))
        out[fam] = {}
        for sev in SEVS:
            for esc in ESCS:
                sub = {p.split("|", 2)[2]: v for p, v in st.items() if p.startswith(f"{sev}|{esc}|")}
                out[fam][f"sev{sev}_esc{esc}"] = summarize(sub)
        print("D4", fam, flush=True)
    return out

# ------------------------------------------------------------------ D5 ------------------------
C_LABEL = [0, 4, 30]

def D5():
    raw = json.load(open("results/D3_rows.json")); out = {}
    for fam, rows in raw.items():
        tot = lambda k: np.array([r[k]["total"] for r in rows])
        probe = rows[0]["_cost"]["probe_cost_per_episode"]
        out[fam] = {"probe_cost_per_calibration_episode": probe}
        for n in NS:
            dep = tot(f"DP[dep@{n}]")
            refs = {"fault-unaware DP (true q, no data)": (tot("DP[fault_unaware]"), 0),
                    "fitted-indep first-report rule, same n": (tot(f"DP[indep@{n}]+first-report"), n),
                    "fitted-indep first-report rule keyed on tool identity, same n":
                        (tot(f"DP[indep@{n}]+first-report(per-tool)"), n),
                    "fitted-indep first-report rule, n=30": (tot("DP[indep@30]+first-report"), 30),
                    "fitted-indep DP, same n": (tot(f"DP[indep@{n}]"), n)}
            o = {}
            for rn, (ref, n_ref) in refs.items():
                gain, lo, hi = boot(ref - dep)
                e = dict(gain_per_episode=gain, gain_lo=lo, gain_hi=hi)
                for cl in C_LABEL:
                    extra = (n - n_ref) * (probe + cl)
                    e[f"extra_cal_cost_clabel{cl}"] = extra
                    e[f"breakeven_episodes_clabel{cl}"] = (extra / gain if gain > 0 else None) if extra > 0 else 0.0
                    e[f"breakeven_episodes_clabel{cl}_worstCI"] = (extra / lo if lo > 0 else None) if extra > 0 else 0.0
                o[rn] = e
            out[fam][str(n)] = o
    return out

if __name__ == "__main__":
    for e in sys.argv[1:]:
        t = time.time(); res = globals()[e]()
        json.dump(res, open(f"results/{e}.json", "w"), indent=1)
        print("done", e, round(time.time() - t), flush=True)
