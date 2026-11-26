"""
Hard-description experiment: analysis (PLAN_hd_hard_description.md). No model call.

  python hd_analysis.py                 # the model runs listed in RUNS (arms found in each log)
  python hd_analysis.py --mock FILE     # any K2e log of the HD episodes (code check only)

Episodes: 50 draws x arms HN (rates as numbers), HQ (rates as words, persistence stated), HI (words, persistence left
to inference); same parameters and pre-drawn outputs in every arm of a draw. Condition K2e. Every trajectory is scored
under the TRUE model of its draw (x_analysis.parts). Executors in the same world:
  model     the LLM's own K2e trajectory
  planner   exact DP on the model's written JSON (K2r; k2decomp.plan_and_play)
  oracle    exact DP on the true model (first action stored in the episode)
  bayes     exact DP that reads every word as its range (hd_bayes.py; results/hd_bayes.json; HQ/HI words only)
"Within tolerance" (the subset of H2): HN, k2r_strict_subset.flags (JSON as written, every parameter within 0.02);
HQ / HI, in_range below (JSON as written, structure right, fault rate and every accuracy inside the range of its word,
stale emission on the stuck value at least 0.90).
"Own-plan pass-over" (H3): share of the model's decisions at which the planner on the model's own JSON, at the belief
that JSON implies after the same reports, would query the replica tool and the model does something else.
Nested bootstrap (scenarios, then draws), 95%; differences between word ranges (H4) resample scenarios, then draws
within each range. Output: results/hd_analysis.json, results/hd_analysis.txt
"""
import os
os.environ.setdefault("RAEA_EPISODES", "episodes_hd.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_hd.jsonl")
import argparse, json
from collections import defaultdict
import numpy as np
from env import HERE, BUDGET, HD_PHI_BINS, HD_ACC_BINS, load_episodes, load_truth
import k2decomp as K2D
from x_analysis import parts, replay, ci, fmt, REP
from k2r_strict_subset import flags

RUNS = {"flash": ("flash (non-thinking)", "hd_ds_chat_fc_k2e.jsonl"),
        "terra": ("gpt-6.1-terra (medium)", "hd_codex_terra_k2e.jsonl"),
        "sonnet": ("Sonnet 5.5 (high)", "hd_claude_s55_k2e.jsonl")}
ARMS = ["HN", "HQ", "HI"]; WORDS = list(HD_PHI_BINS); LOW = ["rare", "occasional"]
MIN_SUBSET = 20                 # a within-tolerance subset smaller than this is reported without a reading


def load(path):
    recs = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["condition"] != "K2e" or r["status"].startswith("api_error") or r.get("sample", 0) != 0: continue
        recs[r["episode"]] = r                      # a later successful rerun replaces an earlier row
    return recs


def inside(x, rng): return rng[0] - 1e-9 <= x <= rng[1] + 1e-9


def in_range(ep, spec):
    rec = K2D.score_recovery(ep, spec); w = ep["words"]
    as_written = (not spec["fallback"]) and not spec["defaults"] and not spec["clipped"] and not spec["notes"]
    g = next(x for x in spec["groups"] if REP in x["tools"])
    ok = (rec["struct_exact"] == 1.0 and rec["persistent_ok"] == 1.0 and rec["fault_present_ok"] == 1.0
          and rec["phantom_fault"] == 0.0 and g["shared_fault"] and inside(g["phi"], HD_PHI_BINS[w["phi"]]) and g["emission"][0] >= 0.90
          and all(inside(q, HD_ACC_BINS[a]) for q, a in zip(spec["accuracy"], w["acc"])))
    return as_written and ok


def own_plan(ep, spec, log):
    """Decisions at which the planner on the model's JSON would query the replica tool, and whether the model did."""
    M = K2D.build_recovered_model(ep, spec); b = M.prior.copy(); B = float(BUDGET); want = miss = n = 0
    for x in log:
        if x["type"] == "call": act = ("probe", x["j"])
        elif x["type"] == "submit": act = ("stop", None)
        else: continue
        a = M.V(b, B)[1]; n += 1
        if a[0] == "probe" and a[1] == REP:
            want += 1; miss += float(act != ("probe", REP))
        if x["type"] == "call":
            nb = M.update(b, x["j"], x["o"]); b = nb if nb is not None else M.prior.copy(); B = float(x["budget_after"])
        else: break
    return dict(steps=n, want=want, miss=miss, pass_share=miss / n if n else 0.0)


def rows_of(recs, eps, truth, ids, bayes):
    R = {}
    for i in ids:
        ep, r = eps[i], recs[i]; spec = r["elicit"]["spec"]; g = next(x for x in spec["groups"] if REP in x["tools"])
        k = ep["pair"].split("-")[1]; bz = bayes.get(f"HQ-{k}") if ep["arm"] != "HN" else None   # word-reading reference: HQ/HI only
        d = dict(arm=ep["arm"], word=ep["phi_word"], draw=ep["pair"], sc=ep["scenario"], m=parts(ep, r["log"], truth[i]),
                 r=parts(ep, replay(ep, spec), truth[i]), oracle_first=float(ep["oracle_first"] == ["probe", REP]),
                 bayes_first=bz["first"] if bz else None, bayes_regret=bz["regret"] if bz else None,
                 usable=float(bool(r["elicit"]["parse_ok"])), no_submit=float(r["status"] != "submitted"),
                 phi_written=float(g["phi"] if g["shared_fault"] else 0.0), persistent_written=float(bool(g["persistent"])),
                 acc_written=list(spec["accuracy"]), acc_words=list(ep["words"]["acc"]), invalid=sum(x["type"] == "invalid" for x in r["log"]))
        d["tol"] = float(flags(ep, spec)["as_written_correct"]) if ep["arm"] == "HN" else float(in_range(ep, spec))
        d["phi_in_range"] = float(inside(d["phi_written"], HD_PHI_BINS[d["word"]]))
        d["own"] = own_plan(ep, spec, r["log"])
        d["aw"] = bool((not spec["fallback"]) and not spec["defaults"] and not spec["clipped"] and not spec["notes"])
        R[i] = d
    return R


def group_diff(a, b, n_boot=2000, seed=0):
    """mean(a) - mean(b) for two disjoint sets of draws; a, b: lists of (value, draw, scenario)."""
    def struct(rows):
        by = defaultdict(lambda: defaultdict(list))
        for v, dr, sc in rows: by[sc][dr].append(v)
        return {sc: [float(np.mean(v)) for v in d.values()] for sc, d in by.items()}
    A, Bm = struct(a), struct(b); scs = sorted(set(A) | set(Bm)); rng = np.random.default_rng(seed); est = []
    mean = float(np.mean([v for v, _, _ in a]) - np.mean([v for v, _, _ in b]))
    for _ in range(n_boot):
        pick = rng.choice(scs, len(scs)); xa, xb = [], []
        for sc in pick:
            for G, X in ((A, xa), (Bm, xb)):
                if sc in G: X.extend(rng.choice(G[sc], len(G[sc])))
        if xa and xb: est.append(np.mean(xa) - np.mean(xb))
    lo, hi = np.percentile(est, [2.5, 97.5]); return dict(mean=mean, lo=float(lo), hi=float(hi), n=len(a) + len(b))


def analyse(path, eps, truth, bayes):
    recs = load(path); arms = [a for a in ARMS if any(eps[i]["arm"] == a for i in recs if i in eps)]
    ids = sorted(i for i in eps if eps[i]["arm"] in arms); missing = [i for i in ids if i not in recs]
    assert not missing, f"{len(missing)} episodes missing, e.g. {missing[:3]}"
    R = rows_of(recs, eps, truth, ids, bayes)
    sel = lambda arm, words=WORDS, pred=lambda d: True: [d for d in R.values() if d["arm"] == arm and d["word"] in words and pred(d)]
    C = lambda rows, f: ci([f(d) for d in rows], [d["draw"] for d in rows], [d["sc"] for d in rows])
    def P(arm1, arm2, f, words=WORDS):        # paired by draw
        a = {d["draw"]: d for d in sel(arm1, words)}; b = {d["draw"]: d for d in sel(arm2, words)}; k = sorted(set(a) & set(b))
        return ci([f(a[x]) - f(b[x]) for x in k], k, [a[x]["sc"] for x in k])
    out = dict(arms=arms, n={a: len(sel(a)) for a in arms}, cells={}, contrasts={}, readings={})
    for arm in arms:
        rows = sel(arm)
        out["cells"][arm] = dict(usable=int(sum(d["usable"] for d in rows)), no_submit=int(sum(d["no_submit"] for d in rows)),
                                 invalid=int(sum(d["invalid"] for d in rows)), n_tol=int(sum(d["tol"] for d in rows)), words={})
        for w in WORDS:
            rw = sel(arm, [w]); c = dict(n=len(rw))
            c["first_model"] = C(rw, lambda d: d["m"]["first"]); c["first_oracle"] = float(np.mean([d["oracle_first"] for d in rw]))
            c["first_planner"] = float(np.mean([d["r"]["first"] for d in rw]))
            bf = [d["bayes_first"] for d in rw]; c["first_bayes"] = float(np.mean(bf)) if all(x is not None for x in bf) else None
            for m in ["regret", "under", "over", "verdict", "calls"]: c[m] = C(rw, lambda d, m=m: d["m"][m])
            c["K2r"] = C(rw, lambda d: d["r"]["regret"]); c["tol"] = float(np.mean([d["tol"] for d in rw]))
            c["phi_written"] = C(rw, lambda d: d["phi_written"]); c["phi_in_range"] = float(np.mean([d["phi_in_range"] for d in rw]))
            c["own_pass"] = C(rw, lambda d: d["own"]["pass_share"])
            out["cells"][arm]["words"][w] = c
        acc = defaultdict(list)
        for d in rows:
            for q, a in zip(d["acc_written"], d["acc_words"]): acc[a].append(q)
        out["cells"][arm]["acc_written"] = {a: dict(mean=float(np.mean(v)), in_range=float(np.mean([inside(x, HD_ACC_BINS[a]) for x in v])), n=len(v))
                                            for a, v in acc.items()}
    D = out["contrasts"]
    for arm in arms:
        allr, low = sel(arm), sel(arm, LOW); tolr = sel(arm, pred=lambda d: d["tol"] == 1.0)
        loww = sel(arm, LOW, pred=lambda d: d["aw"])          # JSON used as written (no fallback, default, clip or note)
        D[f"{arm} H1 K2r regret"] = C(allr, lambda d: d["r"]["regret"]); D[f"{arm} K2e regret"] = C(allr, lambda d: d["m"]["regret"])
        D[f"{arm} H2 K2r-K2e|all"] = C(allr, lambda d: d["r"]["regret"] - d["m"]["regret"])
        if tolr: D[f"{arm} H2 K2r-K2e|within tolerance"] = dict(C(tolr, lambda d: d["r"]["regret"] - d["m"]["regret"]), n_ep=len(tolr))
        D[f"{arm} H3 first model-planner|low"] = C(low, lambda d: d["m"]["first"] - d["r"]["first"])
        D[f"{arm} H3 own-plan pass-over|low"] = C(low, lambda d: d["own"]["pass_share"])
        if loww:
            D[f"{arm} H3 first model-planner|low, as written"] = dict(C(loww, lambda d: d["m"]["first"] - d["r"]["first"]), n_ep=len(loww))
            D[f"{arm} H3 own-plan pass-over|low, as written"] = dict(C(loww, lambda d: d["own"]["pass_share"]), n_ep=len(loww))
        D[f"{arm} first model-oracle|low"] = C(low, lambda d: d["m"]["first"] - d["oracle_first"])
        if arm != "HN" and low and all(d["bayes_first"] is not None for d in low):
            D[f"{arm} first model-bayes|low"] = C(low, lambda d: d["m"]["first"] - d["bayes_first"])
        D[f"{arm} under|low"] = C(low, lambda d: d["m"]["under"])
        for who, key in (("model", "m"), ("planner", "r")):
            a = [(d[key]["first"], d["draw"], d["sc"]) for d in sel(arm, ["rare"])]; b = [(d[key]["first"], d["draw"], d["sc"]) for d in sel(arm, ["frequent"])]
            D[f"{arm} H4 first rare-frequent ({who})"] = group_diff(a, b)
    for arm in [x for x in arms if x != "HN"]:
        if "HN" not in arms: break
        D[f"{arm}-HN H1 K2r regret"] = P(arm, "HN", lambda d: d["r"]["regret"])
        D[f"{arm}-HN H5 K2e regret"] = P(arm, "HN", lambda d: d["m"]["regret"])
        D[f"{arm}-HN H5 K2r-K2e"] = P(arm, "HN", lambda d: d["r"]["regret"] - d["m"]["regret"])
        D[f"{arm}-HN H5 first model-planner|low"] = P(arm, "HN", lambda d: d["m"]["first"] - d["r"]["first"], LOW)
        D[f"{arm}-HN H5 first model|low"] = P(arm, "HN", lambda d: d["m"]["first"], LOW)
    # readings fixed in the plan (an interval that includes 0 is "not detected")
    below = lambda d: d["hi"] < 0; above = lambda d: d["lo"] > 0; Rd = out["readings"]
    for arm in [x for x in arms if x != "HN"]:
        k2r = D[f"{arm} H1 K2r regret"]
        Rd[f"{arm} describing costs something"] = (bool(k2r["mean"] > 0.05 and above(D[f"{arm}-HN H1 K2r regret"]))
                                                   if "HN" in arms else "no reading (HN not in this log)")
    for arm in arms:
        Rd[f"{arm} executor gap"] = below(D[f"{arm} H2 K2r-K2e|all"])
        t = D.get(f"{arm} H2 K2r-K2e|within tolerance")
        Rd[f"{arm} executor gap within tolerance"] = (below(t) if t["n_ep"] >= MIN_SUBSET else f"no reading (n={t['n_ep']} < {MIN_SUBSET})") if t else "no reading (n=0)"
        t = D.get(f"{arm} H3 first model-planner|low, as written")
        Rd[f"{arm} under-use against own description (as written)"] = (below(t) if t["n_ep"] >= MIN_SUBSET else f"no reading (n={t['n_ep']} < {MIN_SUBSET})") if t else "no reading (n=0)"
        Rd[f"{arm} under-use against own description (all low draws)"] = below(D[f"{arm} H3 first model-planner|low"])
        Rd[f"{arm} tracks the rate (model)"] = above(D[f"{arm} H4 first rare-frequent (model)"])
    for k, d in D.items():
        if "H5" in k: Rd[f"{k}: detected"] = bool(below(d) or above(d))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mock"); ap.add_argument("--runs", nargs="*", default=list(RUNS)); a = ap.parse_args()
    eps, truth = load_episodes(), load_truth()
    bp = os.path.join(HERE, "results", "hd_bayes.json"); bayes = json.load(open(bp)) if os.path.exists(bp) else {}
    out = {}; L = [] if bayes else ["(no results/hd_bayes.json: Bayes reference columns empty)"]
    jobs = [("mock", ("mock", a.mock))] if a.mock else [(k, RUNS[k]) for k in a.runs if os.path.exists(os.path.join(HERE, "logs", RUNS[k][1]))]
    for key, (label, fn) in jobs:
        o = analyse(fn if a.mock else os.path.join(HERE, "logs", fn), eps, truth, bayes); o["label"] = label; out[key] = o
        L += ["", f"== {label}   arms {', '.join(f'{x} n={o['n'][x]}' for x in o['arms'])}"]
        for arm in o["arms"]:
            c0 = o["cells"][arm]
            L += [f"  -- {arm}: usable JSON {c0['usable']}  within tolerance {c0['n_tol']}  no submit {c0['no_submit']}  invalid {c0['invalid']}",
                  f"  {'word':11s}{'first: model':>13s}{'oracle':>8s}{'planner':>9s}{'bayes':>7s}{'K2e':>7s}{'K2r':>7s}{'under':>7s}{'over':>7s}"
                  f"{'own pass':>9s}{'phi wr.':>8s}{'in rng':>7s}{'tol':>6s}"]
            for w, c in c0["words"].items():
                bz = f"{100 * c['first_bayes']:6.0f}%" if c["first_bayes"] is not None else f"{'-':>7s}"
                L.append(f"  {w:11s}{100 * c['first_model']['mean']:12.0f}%{100 * c['first_oracle']:7.0f}%{100 * c['first_planner']:8.0f}%{bz}"
                         f"{c['regret']['mean']:7.2f}{c['K2r']['mean']:7.2f}{c['under']['mean']:7.2f}{c['over']['mean']:7.2f}"
                         f"{100 * c['own_pass']['mean']:8.0f}%{c['phi_written']['mean']:8.2f}{100 * c['phi_in_range']:6.0f}%{100 * c['tol']:5.0f}%")
            L.append("  accuracy written, by word: " + "; ".join(f"{a_} {v['mean']:.2f} ({100 * v['in_range']:.0f}% in range)" for a_, v in c0["acc_written"].items()))
        for name, d in o["contrasts"].items():
            pct = "first" in name or "pass-over" in name
            L.append(f"  {name:44s} {fmt(d, pct=pct)}" + ("  (percentage points)" if pct else "") + (f"  [n={d['n_ep']}]" if "n_ep" in d else ""))
        L.append("  readings:")
        L += [f"    {k}: {v}" for k, v in o["readings"].items()]
    txt = "\n".join(L); print(txt)
    if not a.mock:
        json.dump(out, open(os.path.join(HERE, "results", "hd_analysis.json"), "w"), indent=1)
        open(os.path.join(HERE, "results", "hd_analysis.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
