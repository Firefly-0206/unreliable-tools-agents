"""Two-stage (nested) cluster bootstrap: resample scenarios (surface tasks), then episodes within each drawn
scenario, then (implicitly) all samples of an episode together. Used for means and for paired differences."""
from collections import defaultdict
import numpy as np

def nested_ci(vals, episodes, scenarios, n_boot=2000, seed=0, level=95):
    """vals[i] belongs to episode episodes[i], which belongs to scenario scenarios[i].
    Returns (mean, lo, hi). Per-episode values are averaged first (samples of an episode stay together)."""
    ep_vals = defaultdict(list); ep_sc = {}
    for v, e, s in zip(vals, episodes, scenarios): ep_vals[e].append(v); ep_sc[e] = s
    by_sc = defaultdict(list)
    for e, vs in ep_vals.items(): by_sc[ep_sc[e]].append(float(np.mean(vs)))
    scs = [np.array(by_sc[s]) for s in sorted(by_sc)]
    mean = float(np.mean(np.concatenate(scs)))
    if len(scs) < 2: return mean, float("nan"), float("nan")
    rng = np.random.default_rng(seed); est = np.empty(n_boot)
    for b in range(n_boot):
        tot = cnt = 0.0
        for i in rng.integers(0, len(scs), len(scs)):
            x = scs[i][rng.integers(0, len(scs[i]), len(scs[i]))]; tot += x.sum(); cnt += len(x)
        est[b] = tot / cnt
    a = (100 - level) / 2
    return mean, float(np.percentile(est, a)), float(np.percentile(est, 100 - a))

def paired_ci(a, b, scenarios, **kw):
    """a, b: dicts episode -> value (or list of values); difference a - b over common episodes."""
    common = sorted(set(a) & set(b))
    d = [float(np.mean(a[e])) - float(np.mean(b[e])) for e in common]
    return nested_ci(d, common, [scenarios[e] for e in common], **kw) + (len(common),)
