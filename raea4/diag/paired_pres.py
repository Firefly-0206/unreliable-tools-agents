"""Task-level paired comparison of presentations, and Dedup scored against the visible-information posterior."""
import json, random
from collections import defaultdict
items = {it["id"]: it for it in json.load(open("diag_items.json"))}
latest = {}
for l in open("results.jsonl"):
    r = json.loads(l)
    if r.get("ok") and r.get("p_healthy") is not None and r.get("action"): latest[(r["id"], r["model"], r["sample"])] = r
by = defaultdict(list)   # (model, task, pres) -> list of per-answer dicts
for (iid, m, s), r in latest.items():
    it = items[iid]; t = f"{it['regime']}|{it['pattern']}"
    dep = it["pattern"] in ("retry2", "retry3", "shared3", "retry_mixed")
    by[(m, t, it["presentation"])].append(dict(
        dloss=it["Q"][r["action"]] - it["best_value"],
        over=float(dep and r["p_healthy"] - it["p_exact"] > 0.10),
        err_full=abs(r["p_healthy"] - it["p_exact"]), err_vis=abs(r["p_healthy"] - it["p_exact_shown"]),
        opt=float(r["action"] == it["best_action"])))
tasks = sorted({k[1] for k in by})
mean = lambda v: sum(v) / len(v)
def boot(d, n=10000, seed=0):
    rng = random.Random(seed); bs = sorted(mean([rng.choice(d) for _ in d]) for _ in range(n))
    return mean(d), bs[int(.025 * n)], bs[int(.975 * n) - 1]
out = {}
for m in ["deepseek-chat", "deepseek-reasoner"]:
    for pres in ["warning", "dedup"]:
        for key in ["dloss", "opt", "over"]:
            d = [mean([x[key] for x in by[(m, t, pres)]]) - mean([x[key] for x in by[(m, t, "plain")]]) for t in tasks]
            mu, lo, hi = boot(d); out[f"{m}|{pres}-plain|{key}"] = (mu, lo, hi)
            print(f"{m:18s} {pres}-plain {key:6s} {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]  (18 tasks; per-task mean over samples)")
    for pres in ["plain", "warning", "dedup"]:
        ev = mean([x["err_vis"] for t in tasks for x in by[(m, t, pres)]]); ef = mean([x["err_full"] for t in tasks for x in by[(m, t, pres)]])
        print(f"{m:18s} {pres:8s} mean |err| vs visible-info posterior {ev:.3f}; vs full-history posterior {ef:.3f}")
json.dump(out, open("paired_pres.json", "w"), indent=1)
