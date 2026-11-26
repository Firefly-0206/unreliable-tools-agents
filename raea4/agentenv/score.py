"""
Score closed-loop runs exactly.  python score.py [logs/*.jsonl ...] [--out results/score.json]

Per model x protocol x family x condition:
  realised loss; exact expected regret (sum of per-step oracle advantages) split into
    info  = probe-step advantages + stop-timing part (stopping while the oracle would still probe)
    verdict = wrong verdict given the belief at the stop;
  severe-error rate, escalation, accuracy, calls, cost, repeat calls to the cache and log sources, P(used a
  cache tool) and regret conditional on using / not using a cache tool; invalid actions; tokens; latency.
  Reference policies on the SAME episodes (exact, from results/baselines.json): Probe-once, Oracle-VOI1,
  MMI-DP, Oracle+ForbidAny-DP; paired difference model - Probe-once.
Paired condition contrasts (K2, K3, K4, K4c vs K1, and K4 vs K4c) per family and pooled over the core families.
All CIs: nested bootstrap over scenarios, then episodes within scenario, samples of an episode kept together.
Rows with status api_error are excluded (reported as counts). Runs without a submit (turn cap) stay in, are
flagged no_submit and are scored with the WORST verdict at the final belief, not as a cheap escalation.
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np
from env import HERE, CORE, TWINS, EXTRA_CONDITIONS, load_episodes, load_truth
from oracle import score_trajectory
from nestboot import nested_ci, paired_ci
import k2decomp as K2D

REFS = ["Probe-once", "Oracle-VOI1", "MMI-DP", "Oracle+ForbidAny-DP"]
CONTRASTS = [("K2", "K1"), ("K3", "K1"), ("K4", "K1"), ("K4c", "K1"), ("K4", "K4c")]
# K2 decomposition (K2_DECOMP.md); a contrast is only reported when both conditions are in the logs, so the output for
# logs without the new conditions is unchanged.
CONTRASTS += [("K2a", "K1"), ("K2a", "K2"), ("K2e", "K2"), ("K2p", "K2"), ("K2p", "K1"), ("K1e", "K1"), ("K2e", "K1e"),
              ("K2p", "K2e")]
# K2r = same-JSON executor replay (k2e_replay.py): the JSON of each K2e episode, exact planner acts. Same rule: only when present.
CONTRASTS += [("K2r", "K2e"), ("K2r", "K1"), ("K2r", "K2p"), ("K2r", "K2"), ("K2r", "K1e")]
KEY_DEFAULT_PREFIXES = (".shared_fault", ".phi", ".persistent", ".stale_emission")
CMETRICS = ["regret", "info", "verdict", "severe", "rep_cache", "realised"]
RECOVERY_METRICS = ["struct_exact", "pair_ok", "single_ok", "bits_correct", "fault_present_ok", "persistent_ok",
                    "phi_abs_err", "phantom_fault", "emission_tv", "acc_mae", "n_defaults", "fallback"]

def recovery_summary(rows):
    """rows: [(episode id, regret, elicit dict)] -> mean of each recovery metric over the episodes where it is defined."""
    ms = [(eid, reg, el, K2D.score_recovery(EPS[eid], el["spec"])) for eid, reg, el in rows]
    out = {}
    for m in RECOVERY_METRICS:
        v = [x[3][m] for x in ms if x[3][m] is not None]
        out[m] = float(np.mean(v)) if v else None; out["n_" + m] = len(v)
    out["parse_ok"] = float(np.mean([x[2]["parse_ok"] for x in ms])); out["attempts"] = float(np.mean([x[2]["attempts"] for x in ms]))
    for name, flag in [("exact", 1.0), ("wrong", 0.0)]:        # regret split by whether the sharing structure was recovered
        sub = [x[1] for x in ms if x[3]["struct_exact"] == flag]
        out["regret_if_struct_" + name] = float(np.mean(sub)) if sub else None; out["n_struct_" + name] = len(sub)
    return out

EPS = {}

def per_run(rec, ep, truth):
    s = score_trajectory(ep, rec["log"], truth)
    calls = [st for st in s["steps"] if st["action"][0] == "probe"]; stops = [st for st in s["steps"] if st["action"][0] == "stop"]
    inv = defaultdict(int)
    for e in rec["log"]:
        if e["type"] == "invalid": inv[e["reason"]] += 1
    first = s["steps"][0]
    adv_probe = sum(st["adv"] for st in calls); st_t = sum(st["adv_stoptime"] for st in stops)
    return dict(realised=s["realised"], regret=s["regret"], V0=s["V0"],
                adv_probe=adv_probe, adv_stoptime=st_t, verdict=sum(st["adv_verdict"] for st in stops),
                info=adv_probe + st_t,
                stop_when_oracle_probes=float(any(st["action"][0] == "stop" and st["oracle"][0] == "probe" for st in s["steps"])),
                severe=float(s["severe"]), esc=float(s["esc"]), correct=float(s["correct"]), no_submit=float(s["forced"]),
                calls=len(calls), cost=s["cost"], used_cache=float(s["used_cache"]),
                rep_cache=sum(st["repeat_src"] and st["src"] == "c" for st in calls),
                rep_log=sum(st["repeat_src"] and st["src"] == "tool2" for st in calls),
                first_agree=float(first["action"] == first["oracle"]),
                invalid=sum(inv.values()), insufficient_budget=inv["insufficient_budget"],
                truncated=sum(fr == "length" for fr in rec.get("finish_reasons", [])),
                tok_in=rec["tokens"]["prompt"], tok_out=rec["tokens"]["completion"], tok_reason=rec["tokens"]["reasoning"],
                n_api=rec["n_api"], latency=rec["latency"], p_benign_at_stop=s["steps"][-1]["post"][0])

def load_refs():
    p = os.path.join(HERE, "results", "baselines.json")
    if not os.path.exists(p): return {}
    per = json.load(open(p))["per_episode"]
    return {i: {n: r[n]["total"] - r["Oracle-DP"]["total"] for n in REFS} for i, r in per.items()}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("logs", nargs="*")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "score.json")); a = ap.parse_args()
    files = a.logs or sorted(glob.glob(os.path.join(HERE, "logs", "*.jsonl")))
    eps, truth, refs = load_episodes(), load_truth(), load_refs()
    EPS.update(eps)
    sc = {i: e["scenario"] for i, e in eps.items()}
    elic = defaultdict(list)                  # group -> [(episode, regret, elicit record)] for K1e / K2e / K2p
    recs, errors = {}, defaultdict(int)
    for f in files:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line); k = (r["episode"], r["condition"], r["model"], r["protocol"], r.get("sample", 0))
            if r["status"].startswith("api_error"):
                errors[(r["model"], r["protocol"], r["family"], r["condition"])] += 1; continue
            recs[k] = r                      # a later successful rerun replaces an earlier row
    groups = defaultdict(list)
    for k, r in recs.items():
        g = (r["model"], r["protocol"], r["family"], r["condition"]); pr = per_run(r, eps[k[0]], truth[k[0]])
        groups[g].append((k[0], pr))
        if r.get("elicit"): elic[g].append((k[0], pr["regret"], r["elicit"]))
    out = dict(cells={}, contrasts={}, api_errors_excluded={"|".join(k): v for k, v in errors.items()})
    print(f"{'model':16s}{'prot':5s}{'family':12s}{'cond':5s}{'n':>4s}{'loss':>7s}{'regret [95% CI]':>22s}{'info/verd':>11s}"
          f"{'severe':>7s}{'esc':>6s}{'acc':>6s}{'calls':>6s}{'useC':>6s}{'repC':>6s}{'repL':>6s}{'noSub':>6s}{'Probe1':>7s}{'d-Probe1':>9s}")
    for g in sorted(groups):
        rows = groups[g]; cl = [c for c, _ in rows]; scl = [sc[c] for c in cl]
        M = {k: float(np.mean([r[k] for _, r in rows])) for k in rows[0][1]}
        for m in ["regret", "realised", "info", "verdict"]:
            _, M[m + "_lo"], M[m + "_hi"] = nested_ci([r[m] for _, r in rows], cl, scl)
        for flag, name in [(1.0, "regret_if_used_cache"), (0.0, "regret_if_no_cache")]:
            sub = [r["regret"] for _, r in rows if r["used_cache"] == flag]; M[name] = float(np.mean(sub)) if sub else None
            M["n_" + name] = len(sub)
        sub = [r["regret"] for _, r in rows if not r["no_submit"]]; M["regret_submitted_only"] = float(np.mean(sub)) if sub else None
        if refs:
            for n in REFS: M["ref_" + n] = float(np.mean([refs[c][n] for c in cl]))
            mine = defaultdict(list)
            for c, r in rows: mine[c].append(r["regret"])
            d, lo, hi, _ = paired_ci(mine, {c: refs[c]["Probe-once"] for c in mine}, sc)
            M["minus_probe_once"], M["minus_probe_once_lo"], M["minus_probe_once_hi"] = d, lo, hi
        M["n"] = len(rows); M["n_episodes"] = len(set(cl)); M["n_scenarios"] = len(set(scl))
        M["api_errors_excluded"] = errors.get(g, 0)
        if g in elic: M["recovery"] = recovery_summary(elic[g])
        out["cells"]["|".join(g)] = M
        print(f"{g[0][:16]:16s}{g[1]:5s}{g[2]:12s}{g[3]:5s}{M['n']:4d}{M['realised']:7.2f}"
              f"{M['regret']:8.2f} [{M['regret_lo']:5.2f},{M['regret_hi']:5.2f}]{M['info']:5.2f}/{M['verdict']:5.2f}"
              f"{M['severe']:7.2f}{M['esc']:6.2f}{M['correct']:6.2f}{M['calls']:6.2f}{M['used_cache']:6.2f}{M['rep_cache']:6.2f}"
              f"{M['rep_log']:6.2f}{M['no_submit']:6.2f}{M.get('ref_Probe-once', np.nan):7.2f}{M.get('minus_probe_once', np.nan):9.2f}")
    # paired condition contrasts, per family and pooled over the core families
    by = defaultdict(lambda: defaultdict(list))        # (model, protocol, family, cond) -> episode -> [run metrics]
    for g, rows in groups.items():
        for c, r in rows: by[g][c].append(r)
    mp = sorted({(g[0], g[1]) for g in groups})
    print("\nPaired condition contrasts (A - B over common episodes) [95% nested CI]")
    for model, prot in mp:
        fams = sorted({g[2] for g in groups if g[:2] == (model, prot)})
        for fam in fams + ["CORE"]:
            fl = CORE if fam == "CORE" else [fam]
            for A, B in CONTRASTS:
                res = {}
                for m in CMETRICS:
                    da = {c: [x[m] for x in v] for f in fl for c, v in by[(model, prot, f, A)].items()}
                    db = {c: [x[m] for x in v] for f in fl for c, v in by[(model, prot, f, B)].items()}
                    if not set(da) & set(db): continue
                    d, lo, hi, n = paired_ci(da, db, sc); res[m] = dict(diff=d, lo=lo, hi=hi, n=n)
                if not res: continue
                out["contrasts"][f"{model}|{prot}|{fam}|{A}-{B}"] = res
                print(f"{model[:16]:16s}{prot:5s}{fam:12s}{A:>4s}-{B:4s} n={res['regret']['n']:3d}  " + "  ".join(
                    f"{m} {v['diff']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}]" for m, v in res.items() if m in ("regret", "info", "severe", "rep_cache")))
    if elic:
        print("\nRecovery of the observation model (K1e / K2e / K2p); rates over episodes where defined, errors are means")
        print(f"{'model':16s}{'prot':5s}{'family':12s}{'cond':5s}{'n':>4s}{'parse':>6s}{'struct':>7s}{'pair':>6s}{'single':>7s}"
              f"{'fault':>6s}{'persist':>8s}{'phiErr':>7s}{'phantom':>8s}{'emTV':>6s}{'accMAE':>7s}{'dflt':>5s}{'reg|exact':>10s}{'reg|wrong':>10s}")
        f = lambda x, w, d=2: f"{x:{w}.{d}f}" if x is not None else " " * (w - 1) + "-"
        for g in sorted(elic):
            R = out["cells"]["|".join(g)]["recovery"]
            print(f"{g[0][:16]:16s}{g[1]:5s}{g[2]:12s}{g[3]:5s}{len(elic[g]):4d}{f(R['parse_ok'], 6)}{f(R['struct_exact'], 7)}"
                  f"{f(R['pair_ok'], 6)}{f(R['single_ok'], 7)}{f(R['fault_present_ok'], 6)}{f(R['persistent_ok'], 8)}"
                  f"{f(R['phi_abs_err'], 7)}{f(R['phantom_fault'], 8)}{f(R['emission_tv'], 6)}{f(R['acc_mae'], 7, 3)}"
                  f"{f(R['n_defaults'], 5, 1)}{f(R['regret_if_struct_exact'], 10)}{f(R['regret_if_struct_wrong'], 10)}")
        pooled = defaultdict(list)                    # pooled over the core families
        for g, rows in elic.items():
            if g[2] in CORE: pooled[(g[0], g[1], g[3])] += rows
        out["recovery_core_pooled"] = {}
        for k, rows in sorted(pooled.items()):
            R = recovery_summary(rows); out["recovery_core_pooled"]["|".join(k)] = R
            print(f"{k[0][:16]:16s}{k[1]:5s}{'CORE':12s}{k[2]:5s}{len(rows):4d}{f(R['parse_ok'], 6)}{f(R['struct_exact'], 7)}"
                  f"{f(R['pair_ok'], 6)}{f(R['single_ok'], 7)}{f(R['fault_present_ok'], 6)}{f(R['persistent_ok'], 8)}"
                  f"{f(R['phi_abs_err'], 7)}{f(R['phantom_fault'], 8)}{f(R['emission_tv'], 6)}{f(R['acc_mae'], 7, 3)}"
                  f"{f(R['n_defaults'], 5, 1)}{f(R['regret_if_struct_exact'], 10)}{f(R['regret_if_struct_wrong'], 10)}")
    newc = sorted({g[3] for g in groups} & set(EXTRA_CONDITIONS))
    if newc:                                          # fault minus phi = 0 twin, paired by episode index (episode['pair'])
        print("\nTwin contrasts (fault family minus its phi=0 twin, same parameters) [95% nested CI]; conditions of runs that include a K2 decomposition condition")
        out["twin_contrasts_k2decomp"] = {}
        pair_sc = {e["pair"]: e["scenario"] for e in eps.values()}
        for model, prot in mp:
            if not {g[3] for g in groups if g[:2] == (model, prot)} & set(EXTRA_CONDITIONS): continue
            for cond in sorted({g[3] for g in groups if g[:2] == (model, prot)}):
                for key, prs in [("Shared", [("SHARED", "TRANSIENT")]), ("Persist", [("PERSIST", "TRANSIENT_P")]),
                                 ("Both", [("SHARED", "TRANSIENT"), ("PERSIST", "TRANSIENT_P")])]:
                    res = {}
                    for m in ["regret", "rep_cache"]:
                        da, db = {}, {}
                        for fam, tw in prs:
                            for c, v in by[(model, prot, fam, cond)].items(): da[eps[c]["pair"]] = [x[m] for x in v]
                            for c, v in by[(model, prot, tw, cond)].items(): db[eps[c]["pair"]] = [x[m] for x in v]
                        if set(da) & set(db):
                            d, lo, hi, n = paired_ci(da, db, pair_sc); res[m] = dict(diff=d, lo=lo, hi=hi, n=n)
                    if not res: continue
                    out["twin_contrasts_k2decomp"][f"{model}|{prot}|{key}|{cond}"] = res
                    print(f"{model[:16]:16s}{prot:5s}{key:8s}{cond:>4s} n={res['regret']['n']:3d}  " + "  ".join(
                        f"{m} {v['diff']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}]" for m, v in res.items()))
    if any(g[3] == "K2r" for g in groups):           # K2r (same-JSON replay) split by how the JSON was completed (K2_DECOMP.md)
        def kind(sp):
            if sp["fallback"]: return "fallback"
            key = [d for d in sp["defaults"] if any(p in d for p in KEY_DEFAULT_PREFIXES) or "not in any group" in d]
            return "key_defaults" if key else ("other_defaults" if sp["defaults"] else "clean")
        out["k2r_split"] = {}
        print("\nK2r vs K2e by completion of the JSON (fallback = no usable JSON; key_defaults = phi / shared_fault / persistent / emission / group membership filled; other_defaults = accuracy only; clean = nothing filled)")
        for model, prot in mp:
            for fam in sorted({g[2] for g in groups if g[:2] == (model, prot) and g[3] == "K2r"}) + ["CORE"]:
                fl = CORE if fam == "CORE" else [fam]
                e2 = {c: x[0] for f in fl for c, x in by[(model, prot, f, "K2e")].items()}
                rr = {c: x[0] for f in fl for c, x in by[(model, prot, f, "K2r")].items()}
                kd = {}
                for f in fl:
                    for c, reg, el in elic[(model, prot, f, "K2e")]: kd[c] = kind(el["spec"])
                if not rr: continue
                res = {}
                for k in ["clean", "other_defaults", "key_defaults", "fallback", "ANY_DEFAULT_OR_FALLBACK"]:
                    ids = [c for c in rr if c in e2 and (kd[c] != "clean" if k == "ANY_DEFAULT_OR_FALLBACK" else kd[c] == k)]
                    res[k] = dict(n=len(ids), k2r=float(np.mean([rr[c]["regret"] for c in ids])) if ids else None,
                                  k2e=float(np.mean([e2[c]["regret"] for c in ids])) if ids else None)
                out["k2r_split"][f"{model}|{prot}|{fam}"] = res
                print(f"{model[:16]:16s}{prot:5s}{fam:12s}" + "  ".join(
                    f"{k} n={v['n']} K2r={v['k2r']:.3f} K2e={v['k2e']:.3f}" for k, v in res.items() if v["n"]))
    if errors: print("\napi_error rows excluded:", dict(out["api_errors_excluded"]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
