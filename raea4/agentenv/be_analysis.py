"""
Belief elicitation by forking the logged runs: analysis (PLAN_be_belief_elicit.md). No model call.

  python be_analysis.py                               the runs in RUNS (logs/be_<stem>.jsonl)
  python be_analysis.py --mock MODEL FILE             a mock fork log of one model (code check only)

Points: results/be_points_<model>.jsonl (be_points.py), labels U (under-use step: the oracle would query a replica
tool, the model does something else), O (optimal step), R (rest). References at each point, from the logged reports:
the exact quantities under the TRUE model and under the model's OWN written JSON.
  B1  stated status posterior vs exact: total variation (TV) under the true model and under the JSON; absolute error of
      the stated replica-stale probability; log score of the stated posterior at the true status (clipped at 0.001);
      TV of the stated next-report law vs the JSON's, averaged over the tools stated
  B2  B1 at U points minus at O points (cluster bootstrap: scenarios, then episodes, all points of an episode together)
  B3  valuation: Spearman correlation of the stated value of calling each affordable tool with the exact multi-step value
      (JSON model; true model), points with >= 3 stated tools; at U points: share where the stated value of the replica
      tool the oracle would query is below the stated value of the logged action (stop = 0); stated minus exact value
      of that replica tool (JSON model; true model)
  B4  fork A: share whose action equals the logged action, all points and U points
  B5  fork C minus fork A, at U points, paired by point: share choosing a replica tool
Stated probability vectors are normalised; their raw sums are reported. Output: results/be_analysis.json, .txt
"""
import argparse, json, os
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata
from env import HERE, load_truth
from nestboot import nested_ci

RUNS = {"flash": "be_ds_chat_fc.jsonl", "terra": "be_codex_terra.jsonl", "sonnet": "be_claude_s55.jsonl"}
TV_OK, B2_MARGIN, AGREE_MIN = 0.10, 0.05, 0.70


def load_rows(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["status"] == "ok": rows[(r["pid"], r["fork"])] = r          # the last successful row wins
    return rows


def norm(v):
    v = np.clip(np.asarray(v, float), 0, None); s = v.sum(); return (v / s) if s > 0 else None


def tv(a, b): return float(0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum())


def spearman(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0: return None
    rx, ry = rankdata(x), rankdata(y); return float(np.corrcoef(rx, ry)[0, 1])        # average ranks for ties


def ci(vals, eps, scs):
    m, lo, hi = nested_ci(vals, eps, scs); return dict(mean=float(m), lo=float(lo), hi=float(hi), n=len(vals), n_ep=len(set(eps)))   # episode-weighted


def cluster_diff(rows, key, n_boot=2000, seed=0):
    """mean of key over U points minus over O points; resample scenarios, then episodes (points of an episode together)."""
    by = defaultdict(lambda: defaultdict(list))
    for d in rows:
        if d[key] is not None and d["label"] in ("U", "O"): by[d["sc"]][d["episode"]].append((d["label"], d[key]))
    def stat(groups):                       # episode-weighted: mean over episodes of each episode's U (O) mean
        u = [np.mean([v for l, v in g if l == "U"]) for g in groups if any(l == "U" for l, _ in g)]
        o = [np.mean([v for l, v in g if l == "O"]) for g in groups if any(l == "O" for l, _ in g)]
        return (np.mean(u) - np.mean(o)) if u and o else None
    allg = [g for sc in by.values() for g in sc.values()]; mean = stat(allg)
    if mean is None: return None
    rng = np.random.default_rng(seed); scs = sorted(by); est = []
    for _ in range(n_boot):
        groups = []
        for sc in rng.choice(scs, len(scs)):
            eps = list(by[sc].values()); groups += [eps[i] for i in rng.integers(0, len(eps), len(eps))]
        x = stat(groups)
        if x is not None: est.append(x)
    lo, hi = np.percentile(est, [2.5, 97.5]); return dict(mean=float(mean), lo=float(lo), hi=float(hi))


def per_point(p, A, B, C, s_true):
    d = dict(pid=p["pid"], episode=p["episode"], sc=p["scenario"], label=p["label"], cache=set(p["cache"]))
    d["agree"] = (float(A["action"] == p["logged"]) if A["action"] is not None else 0.0) if A else None
    d["A_parsed"] = (A["action"] is not None) if A else None
    d["A_rep"] = (float(A["action"][0] == "probe" and A["action"][1] in d["cache"]) if A["action"] is not None else None) if A else None
    d["A_single"] = (A.get("n_calls", 1) <= 1) if A else None
    d["C_rep"] = float(C["action"] is not None and C["action"][0] == "probe" and C["action"][1] in d["cache"]) if C and C.get("action") is not None else None
    for k in ("tv_true", "tv_json", "stale_err_true", "stale_err_json", "logscore", "pred_tv", "rho_json", "rho_true",
              "rank_below", "rep_bias_json", "rep_bias_true", "status_sum"):
        d[k] = None
    d["B_found"] = bool(B and B.get("belief_found")); d["B_acted"] = bool(B and B.get("acted_in_B"))
    d["B_repaired"] = bool(B and B.get("belief_repaired")); d["C_repaired"] = bool(C and C.get("belief_repaired"))   # amendment
    if not d["B_found"]: return d
    bel = B["belief"]; T = p["true"]; H = p["json"] if p.get("parse_ok", True) else None    # JSON reference only for usable JSONs
    if bel["status"] is not None:
        s = norm(bel["status"]); d["status_sum"] = float(sum(bel["status"]))
        d["tv_true"] = tv(s, T["post"]); d["tv_json"] = tv(s, H["post"]) if H else None
        d["logscore"] = float(np.log(max(s[s_true], 1e-3)))
    if bel["stale"] is not None:
        d["stale_err_true"] = abs(bel["stale"] - T["stale"]) if T["stale"] is not None else None
        d["stale_err_json"] = abs(bel["stale"] - H["stale"]) if H and H["stale"] is not None else None
    R_ = H or T
    pt = [tv(norm(v), R_["pred"][j]) for j, v in bel["next_report"].items() if j in R_["pred"] and norm(v) is not None]
    d["pred_tv"] = float(np.mean(pt)) if pt else None                       # vs the JSON's law (true law if JSON unusable)
    js = [j for j in bel["value"] if j in T["voi"]]
    if len(js) >= 3:
        x = [bel["value"][j] for j in js]; d["rho_true"] = spearman(x, [T["voi"][j] for j in js])
        d["rho_json"] = spearman(x, [H["voi"][j] for j in js]) if H else None
    if p["label"] == "U":
        jr = str(T["best"][1])                                   # the replica tool the oracle would query
        if jr in bel["value"]:
            la = p["logged"]; chosen = bel["value"].get(str(la[1])) if la[0] == "probe" else 0.0
            if chosen is not None: d["rank_below"] = float(bel["value"][jr] < chosen)
            d["rep_bias_json"] = bel["value"][jr] - H["voi"][jr] if H and jr in H["voi"] else None; d["rep_bias_true"] = bel["value"][jr] - T["voi"][jr]
    return d


def analyse(model, path, truth):
    pts = [json.loads(l) for l in open(os.path.join(HERE, "results", f"be_points_{model}.jsonl"), encoding="utf-8")]
    pts = [p for p in pts if p["subset"]]; rows = load_rows(path)
    forks = sorted({f for _, f in rows}); missing = {f: sum((p["pid"], f) not in rows for p in pts) for f in forks}
    D = [per_point(p, rows.get((p["pid"], "A")), rows.get((p["pid"], "B")), rows.get((p["pid"], "C")), truth[p["episode"]]["s"]) for p in pts]
    sel = lambda lab=None, key=None: [d for d in D if (lab is None or d["label"] == lab) and (key is None or d[key] is not None)]
    C = lambda rows_, key: ci([d[key] for d in rows_], [d["episode"] for d in rows_], [d["sc"] for d in rows_]) if rows_ else None
    out = dict(n=len(pts), labels={l: sum(d["label"] == l for d in D) for l in "UOR"}, forks=forks, missing=missing, stats={}, readings={})
    S = out["stats"]
    S["B found"] = sum(d["B_found"] for d in D); S["B acted"] = sum(d["B_acted"] for d in D)
    S["B repaired (closing braces appended)"] = sum(d["B_repaired"] for d in D); S["C repaired"] = sum(d["C_repaired"] for d in D)
    S["status sum (raw), mean"] = float(np.mean([d["status_sum"] for d in D if d["status_sum"] is not None])) if S["B found"] else None
    for key in ("tv_true", "tv_json", "stale_err_true", "stale_err_json", "logscore", "pred_tv", "rho_json", "rho_true"):
        for lab in (None, "U", "O"):
            S[f"B1 {key} | {lab or 'all'}"] = C(sel(lab, key), key)
        S[f"B2 {key} U-O"] = cluster_diff(D, key)
    for key in ("rank_below", "rep_bias_json", "rep_bias_true"): S[f"B3 {key} | U"] = C(sel("U", key), key)
    if "A" in forks:
        S["B4 agree | all"] = C(sel(None, "agree"), "agree"); S["B4 agree | U"] = C(sel("U", "agree"), "agree"); S["B4 agree | O"] = C(sel("O", "agree"), "agree")
        S["fork A parsed"] = sum(bool(d["A_parsed"]) for d in D)
        S["fork A several calls"] = sum(d["A_single"] is False for d in D)
        S["B4 agree | all, single-call replies"] = C([d for d in sel(None, "agree") if d["A_single"]], "agree")
        S["logged turns with several calls"] = sum(p.get("n_parallel", 1) > 1 for p in pts)
        S["points with unusable JSON"] = sum(not p.get("parse_ok", True) for p in pts)
    if "A" in forks and "C" in forks:
        pr = [d for d in sel("U", "C_rep") if d["A_rep"] is not None]; S["B5 n paired (both parsed)"] = len(pr)
        S["B5 replica C-A | U"] = ci([d["C_rep"] - d["A_rep"] for d in pr], [d["episode"] for d in pr], [d["sc"] for d in pr]) if pr else None
        S["fork C replica | U"] = C(pr, "C_rep"); S["fork A replica | U"] = C(pr, "A_rep")
    R = out["readings"]; g = lambda k: S.get(k)
    tvU, b2 = g("B1 tv_json | U"), g("B2 tv_json U-O")
    R["belief tracks the exact posterior at U (own JSON)"] = bool(tvU and b2 and tvU["mean"] <= TV_OK and b2["lo"] <= B2_MARGIN) if tvU and b2 else "no reading (no data)"
    R["belief error larger at U than at O (status, own JSON)"] = bool(b2["lo"] > 0) if b2 else "no reading"
    b2s = g("B2 stale_err_json U-O"); R["stale-probability error larger at U than at O (own JSON)"] = bool(b2s["lo"] > 0) if b2s else "no reading"
    rb = g("B3 rep_bias_json | U"); R["replica valued below its exact value at U (own JSON)"] = bool(rb["hi"] < 0) if rb else "no reading"
    ag = g("B4 agree | all"); R["forks stand in for the logged turn (agreement >= 70%)"] = bool(ag["mean"] >= AGREE_MIN) if ag else "no reading (no fork A)"
    b5 = g("B5 replica C-A | U"); R["stating beliefs changes replica choice at U"] = (bool(b5["lo"] > 0 or b5["hi"] < 0) if b5 else "no reading (no fork C)")
    return out


def fmt(d):
    if d is None: return "-"
    return f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]" + (f"  n={d['n']}" if "n" in d else "")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mock", nargs=2, metavar=("MODEL", "FILE")); a = ap.parse_args()
    truth = load_truth(); out, L = {}, []
    jobs = [(a.mock[0], a.mock[1])] if a.mock else [(m, os.path.join(HERE, "logs", f)) for m, f in RUNS.items() if os.path.exists(os.path.join(HERE, "logs", f))]
    for m, path in jobs:
        o = analyse(m, path, truth); out[m] = o
        L += ["", f"== {m}  points {o['n']} {o['labels']}  forks {o['forks']}  missing {o['missing']}"]
        L += [f"  {k:46s} {fmt(v) if isinstance(v, dict) or v is None else v}" for k, v in o["stats"].items()]
        L += ["  readings:"] + [f"    {k}: {v}" for k, v in o["readings"].items()]
    txt = "\n".join(L); print(txt)
    if not a.mock:
        json.dump(out, open(os.path.join(HERE, "results", "be_analysis.json"), "w"), indent=1)
        open(os.path.join(HERE, "results", "be_analysis.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
