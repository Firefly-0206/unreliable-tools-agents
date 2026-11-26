"""
Sensitivity of the main comparisons to the six surface scenarios and to multiplicity. No model call. Post hoc
(added in response to a review; not part of any plan fixed before the runs).

  python scenario_sensitivity.py                 # first domain: CORE, held-out confirmation, fault-rate sweep
  python scenario_sensitivity.py --d2 [--workers 4]   # second domain (exact solves: about 25 minutes per run)
  python scenario_sensitivity.py --d3 [--workers 4]   # third domain (fact-checking; added 2026-11-17, also post hoc)
  python scenario_sensitivity.py --summarise          # recompute every summary from the cached values (no solves)
Each mode caches the per-unit values of its comparisons in results/scenario_sensitivity_values.json, and every run
recomputes ALL summaries from that cache, so the Bonferroni level always uses the full family size M.

Every interval in the paper resamples scenarios first, and there are only six. For each comparison this script reports
  est, lo, hi     the estimate and nested 95% interval as in the paper (nestboot.nested_ci, 2000 draws, seed 0)
  by_scenario     the estimate within each scenario; same_sign = how many of the six have the sign of the pooled estimate
  loso            the estimate and nested 95% interval with each scenario left out in turn (five scenarios remain)
  sign_p          exact two-sided sign-flip test on the six scenario means (smallest attainable value 2/64 = 0.031)
  bonf            nested interval at level 1 - 0.05/M (20000 draws), M = number of comparisons listed in FAMILY below

Comparisons (FAMILY):
  core_gap|run        K2r - K2e regret, CORE episodes, four runs                      (main text, Table 2)
  confirm_gap|run     K2r - K2e regret, held-out CORE episodes, three runs             (main text, held-out confirmation)
  x_first|run|arm     model - oracle share of episodes opening with the replica tool,
                      fault rates 0.10 and 0.30, per arm, three runs                    (main text, fault-rate sweep)
  d2_gap|run          K2r - K2e regret, second domain, all episodes, three runs        (main text, second domain)
  d2_first|run        model - oracle first-call share of the replica tool, second domain, fault rates 0.10 and 0.30
  d3_gap|run          K2r - K2e regret, third domain, all episodes, three runs
  d3_first|run        model - oracle share of episodes opening with an outlet, third domain, fault rates 0.10 and 0.30

Output: results/scenario_sensitivity.json (the two modes update the same file), results/scenario_sensitivity.txt
"""
import argparse, itertools, json, os, sys

D2, D3 = "--d2" in sys.argv, "--d3" in sys.argv
if D2:
    os.environ["RAEA_DOMAIN"] = "2"
    os.environ.setdefault("RAEA_EPISODES", "episodes_d2.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_d2.jsonl")
if D3:
    os.environ["RAEA_DOMAIN"] = "3"
    os.environ.setdefault("RAEA_EPISODES", "episodes_d3.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_d3.jsonl")
import numpy as np
from nestboot import nested_ci

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(HERE, *a)
FAMILY = ([f"core_gap|{r}" for r in ("flash", "terra", "terra_high", "sonnet")] + [f"confirm_gap|{r}" for r in ("flash", "terra", "sonnet")]
          + [f"x_first|{r}|{a}" for r in ("flash", "terra", "sonnet") for a in ("XP", "XI")]
          + [f"d2_gap|{r}" for r in ("flash", "terra", "sonnet")] + [f"d2_first|{r}" for r in ("flash", "terra", "sonnet")]
          + [f"d3_gap|{r}" for r in ("flash", "terra", "sonnet")] + [f"d3_first|{r}" for r in ("flash", "terra", "sonnet")])
M = len(FAMILY); BONF_LEVEL = 100 * (1 - 0.05 / M)


def summarise(vals, ids, scs):
    """vals[i] belongs to cluster ids[i] (episode or parameter draw) in scenario scs[i]."""
    vals = [float(v) for v in vals]; names = sorted(set(scs))
    est, lo, hi = nested_ci(vals, ids, scs, weight="episode")
    by = {}
    for s in names:
        cl = {}
        for v, i, c in zip(vals, ids, scs):
            if c == s: cl.setdefault(i, []).append(v)
        by[s] = float(np.mean([x for v in cl.values() for x in v]))
    sgn = np.sign(est)
    loso = {}
    for s in names:
        k = [j for j, c in enumerate(scs) if c != s]
        e, l, h = nested_ci([vals[j] for j in k], [ids[j] for j in k], [scs[j] for j in k], weight="episode"); loso[s] = dict(est=e, lo=l, hi=h)
    m = np.array([by[s] for s in names]); obs = abs(m.mean())
    flips = [abs(float(np.mean(m * np.array(f)))) for f in itertools.product((1, -1), repeat=len(m))]
    sign_p = float(np.mean([f >= obs - 1e-12 for f in flips]))
    _, bl, bh = nested_ci(vals, ids, scs, n_boot=20000, level=BONF_LEVEL, weight="episode")
    excl = lambda l, h: bool(l > 0 or h < 0)
    return dict(est=est, lo=lo, hi=hi, excl0=excl(lo, hi), n=len(set(ids)), by_scenario=by,
                same_sign=int(sum(np.sign(by[s]) == sgn for s in names)), n_scenarios=len(names), loso=loso,
                loso_min=min(v["est"] for v in loso.values()), loso_max=max(v["est"] for v in loso.values()),
                loso_all_excl0=all(excl(v["lo"], v["hi"]) for v in loso.values()),
                loso_same_sign=all(np.sign(v["est"]) == sgn for v in loso.values()),
                sign_p=sign_p, bonf=dict(level=BONF_LEVEL, lo=bl, hi=bh, excl0=excl(bl, bh)))


def V(vals, ids, scs):
    """Raw values of one comparison (summarised later, from the cache, at the full family size)."""
    return dict(vals=[float(v) for v in vals], ids=list(ids), scs=list(scs))


def domain1():
    from env import CORE, load_episodes, load_truth, build_model
    from score import per_run
    from oracle import qvalues
    import k2v_summary as KV
    import anatomy_confirm as AC
    out = {}
    eps, truth = load_episodes(), load_truth()
    for run, files in KV.RUNS.items():
        recs = KV.load(files)
        ids = sorted(e for e in eps if eps[e]["family"] in CORE and (e, "K2e") in recs and (e, "K2r") in recs)
        assert len(ids) == 100, (run, len(ids))
        d = [per_run(recs[(e, "K2r")], eps[e], truth[e])["regret"] - per_run(recs[(e, "K2e")], eps[e], truth[e])["regret"] for e in ids]
        out[f"core_gap|{run}"] = V(d, ids, [eps[e]["scenario"] for e in ids])
    for run, (_, ce, cr, _, _, _) in AC.RUNS.items():
        e_, r_ = AC.load(ce, "K2e", 25, 50), AC.load(cr, "K2r", 25, 50); ids = sorted(e_)
        assert len(ids) == 100 and sorted(r_) == ids, (run, len(ids))
        d = [per_run(r_[e], eps[e], truth[e])["regret"] - per_run(e_[e], eps[e], truth[e])["regret"] for e in ids]
        out[f"confirm_gap|{run}"] = V(d, ids, [eps[e]["scenario"] for e in ids])
    # fault-rate sweep: own episode files (x_analysis sets nothing at import that domain 1 does not already use)
    import x_analysis as X
    ex, tx = load_episodes(P("episodes_x.jsonl")), load_truth(P("ground_truth_x.jsonl"))
    of = {}
    for i, ep in ex.items():
        if ep["phi"] > 0.31: continue
        Mo = build_model(ep["params"]); q = qvalues(Mo, Mo.prior.copy(), float(Mo.budget)); of[i] = float(min(q, key=q.get) == ("probe", X.REP))
    for run, (_, fn) in X.RUNS.items():
        recs = X.load(P("logs", fn))
        for arm in X.ARMS:
            ids = sorted(i for i in of if ex[i]["arm"] == arm); assert len(ids) == 50 and all(i in recs for i in ids), (run, arm, len(ids))
            first = lambda r: float(bool(c := [x["j"] for x in r["log"] if x["type"] == "call"]) and c[0] == X.REP)
            out[f"x_first|{run}|{arm}"] = V([first(recs[i]) - of[i] for i in ids], [ex[i]["pair"] for i in ids], [ex[i]["scenario"] for i in ids])
    return out


def domain2(workers, runs):
    from multiprocessing import Pool
    from env import load_episodes, load_truth
    import d2_analysis as D
    EP, TR = load_episodes(P("episodes_d2.jsonl")), load_truth(P("ground_truth_d2.jsonl")); ORC = json.load(open(P("d2_oracle.json")))
    out = {}
    for run in runs:
        recs = {}
        for line in open(P("logs", f"d2_{D.RUNS[run][1]}_k2e.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if r["condition"] == "K2e" and not r["status"].startswith("api_error") and r.get("sample", 0) == 0: recs[r["episode"]] = r
        ids = sorted(EP); assert all(i in recs for i in ids)
        with Pool(workers) as pool: res = dict(pool.imap_unordered(D.work, [(EP[i], TR[i], recs[i]) for i in ids], chunksize=1))
        out[f"d2_gap|{run}"] = V([res[i]["r"]["regret"] - res[i]["m"]["regret"] for i in ids], [EP[i]["pair"] for i in ids],
                                         [EP[i]["scenario"] for i in ids])
        lo = [i for i in ids if EP[i]["part"] == "DX" and EP[i]["phi"] < 0.31]
        out[f"d2_first|{run}"] = V([float(res[i]["fa"] == D.REP) - float(ORC[i]["first"] == D.REP) for i in lo], [EP[i]["pair"] for i in lo],
                                           [EP[i]["scenario"] for i in lo])
        print(run, "done", flush=True)
    return out


def domain3(workers, runs):
    """As domain2; an outlet (tool 0 or 1) is the fault-capable source."""
    from multiprocessing import Pool
    from env import load_episodes, load_truth
    import d3_analysis as D
    EP, TR = load_episodes(P("episodes_d3.jsonl")), load_truth(P("ground_truth_d3.jsonl")); ORC = json.load(open(P("d3_oracle.json")))
    out = {}
    for run in runs:
        recs = {}
        for line in open(P("logs", f"d3_{D.RUNS[run][1]}_k2e.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if r["condition"] == "K2e" and not r["status"].startswith("api_error") and r.get("sample", 0) == 0: recs[r["episode"]] = r
        ids = sorted(EP); assert all(i in recs for i in ids)
        with Pool(workers) as pool: res = dict(pool.imap_unordered(D.work, [(EP[i], TR[i], recs[i]) for i in ids], chunksize=1))
        out[f"d3_gap|{run}"] = V([res[i]["r"]["regret"] - res[i]["m"]["regret"] for i in ids], [EP[i]["pair"] for i in ids],
                                 [EP[i]["scenario"] for i in ids])
        lo = [i for i in ids if EP[i]["part"] == "DX" and EP[i]["phi"] < 0.31]
        out[f"d3_first|{run}"] = V([float(res[i]["fa"] in D.OUT) - float(ORC[i]["first"] in D.OUT) for i in lo], [EP[i]["pair"] for i in lo],
                                   [EP[i]["scenario"] for i in lo])
        print(run, "done", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--d2", action="store_true"); ap.add_argument("--d3", action="store_true")
    ap.add_argument("--summarise", action="store_true"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--runs", nargs="*", default=["flash", "terra", "sonnet"]); a = ap.parse_args()
    cf = P("results", "scenario_sensitivity_values.json")
    cache = json.load(open(cf, encoding="utf-8")) if os.path.exists(cf) else {}
    if not a.summarise:
        cache.update(domain2(a.workers, a.runs) if a.d2 else domain3(a.workers, a.runs) if a.d3 else domain1())
        json.dump(cache, open(cf, "w", encoding="utf-8"))
    allr = {k: summarise(cache[k]["vals"], cache[k]["ids"], cache[k]["scs"]) for k in FAMILY if k in cache}
    json.dump(allr, open(P("results", "scenario_sensitivity.json"), "w", encoding="utf-8"), indent=1)
    L = [f"family of {M} comparisons; Bonferroni level {BONF_LEVEL:.3f}%; smallest attainable scenario-level sign-flip p = {2 / 64:.3f}",
         f"{'comparison':24s}{'estimate [95% nested]':>30s}{'same sign':>11s}{'leave-one-out range':>22s}{'LOO CIs excl. 0':>17s}{'sign p':>8s}{'Bonferroni interval':>26s}"]
    for k, r in allr.items():
        pct = "first" in k; s = 100 if pct else 1; f = (lambda x: f"{s * x:+.0f}") if pct else (lambda x: f"{x:+.2f}")
        L.append(f"{k:24s}{f(r['est']) + ' [' + f(r['lo']) + ', ' + f(r['hi']) + ']':>30s}{str(r['same_sign']) + '/' + str(r['n_scenarios']):>11s}"
                 f"{f(r['loso_min']) + ' to ' + f(r['loso_max']):>22s}{'yes' if r['loso_all_excl0'] else 'no':>17s}{r['sign_p']:8.3f}"
                 f"{'[' + f(r['bonf']['lo']) + ', ' + f(r['bonf']['hi']) + ']' + ('' if r['bonf']['excl0'] else ' incl. 0'):>26s}")
    missing = [k for k in FAMILY if k not in allr]
    if missing: L.append("not yet computed: " + ", ".join(missing))
    txt = "\n".join(L); print(txt); open(P("results", "scenario_sensitivity.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
