"""Two-stage (nested) cluster bootstrap: resample scenarios (surface tasks), then episodes within each drawn
scenario, then (implicitly) all samples of an episode together. Used for means and for paired differences."""
from collections import defaultdict
import os
import numpy as np

def nested_ci(vals, episodes, scenarios, n_boot=2000, seed=0, level=95, weight="cluster"):
    """vals[i] belongs to episode episodes[i], which belongs to scenario scenarios[i].
    Returns (mean, lo, hi). weight="cluster" (default): per-cluster values are averaged first (samples of an episode stay
    together) and every cluster counts once. weight="episode": clusters are still resampled whole, but every value counts
    once, so mean and interval refer to the plain mean of vals (needed when clusters have unequal sizes, e.g. parameter
    draws with one or four episodes). With equal cluster sizes both give identical numbers (same random draws)."""
    assert weight in ("cluster", "episode")
    ep_vals = defaultdict(list); ep_sc = {}
    for v, e, s in zip(vals, episodes, scenarios): ep_vals[e].append(v); ep_sc[e] = s
    by_sc = defaultdict(list); by_n = defaultdict(list)
    for e, vs in ep_vals.items():
        if weight == "cluster": by_sc[ep_sc[e]].append(float(np.mean(vs))); by_n[ep_sc[e]].append(1.0)
        else: by_sc[ep_sc[e]].append(float(np.sum(vs))); by_n[ep_sc[e]].append(float(len(vs)))
    if os.environ.get("RAEA_CI_AUDIT") and weight == "cluster":       # audit only: record calls whose clusters differ in size
        sizes = sorted({len(vs) for vs in ep_vals.values()})
        if len(sizes) > 1:
            import inspect, json
            st = inspect.stack(); fr = st[1]; up = st[2] if len(st) > 2 else st[1]; vm = float(np.mean(vals)); cm = float(np.mean([np.mean(vs) for vs in ep_vals.values()]))
            with open(os.environ["RAEA_CI_AUDIT"], "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(caller=f"{os.path.basename(fr.filename)}:{fr.lineno}<-{os.path.basename(up.filename)}:{up.lineno}", sizes=sizes, n=len(vals),
                                        value_mean=vm, cluster_mean=cm)) + "\n")
    scs = [np.array(by_sc[s]) for s in sorted(by_sc)]; ns = [np.array(by_n[s]) for s in sorted(by_sc)]
    mean = float(np.concatenate(scs).sum() / np.concatenate(ns).sum())
    if len(scs) < 2: return mean, float("nan"), float("nan")
    rng = np.random.default_rng(seed); est = np.empty(n_boot)
    for b in range(n_boot):
        tot = cnt = 0.0
        for i in rng.integers(0, len(scs), len(scs)):
            j = rng.integers(0, len(scs[i]), len(scs[i])); tot += scs[i][j].sum(); cnt += ns[i][j].sum()
        est[b] = tot / cnt
    a = (100 - level) / 2
    return mean, float(np.percentile(est, a)), float(np.percentile(est, 100 - a))

def paired_ci(a, b, scenarios, **kw):
    """a, b: dicts episode -> value (or list of values); difference a - b over common episodes."""
    common = sorted(set(a) & set(b))
    d = [float(np.mean(a[e])) - float(np.mean(b[e])) for e in common]
    return nested_ci(d, common, [scenarios[e] for e in common], **kw) + (len(common),)
