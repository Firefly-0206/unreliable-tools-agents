"""Score elicit_results.jsonl against the exact targets in elicit_items.json. Standard library only.

For each answer, every elicited quantity is compared with its exact value, and the answer is assigned the
FIRST stage, in decision order, whose numbers depart from exact by more than the tolerance:
  1 belief         posterior (L1 > 0.06) or P(A stale) (abs > 0.05)
  2 prediction     predictive distribution of the next report from A or V (L1 > 0.06)
  3 anticipated    posterior after a hypothetical report: P(healthy) error averaged over outcomes,
                   weighted by the exact predictive (> 0.03)
  4 value          any stated expected cost off by more than 0.3, or the stated costs rank a
                   different action first than the exact costs
  5 selection      chosen action is not the cheapest under the model's OWN stated costs
  - none           all within tolerance and the choice follows them
Usage: python score_elicit.py [elicit_results.jsonl]   -> prints tables, writes elicit_scored.json
"""
import json, os, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
items = {it["id"]: it for it in json.load(open(os.path.join(HERE, "elicit_items.json"), encoding="utf-8"))}
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "elicit_results.jsonl")
latest = {}
for line in open(path, encoding="utf-8"):
    r = json.loads(line)
    if r.get("ok") and r.get("parsed"):
        latest[(r["id"], r["model"], r["sample"])] = r
S = ["healthy", "degraded", "down"]
TOL = dict(post=0.06, stale=0.05, pred=0.06, after=0.03, cost=0.3)

def l1(a, b):
    return sum(abs(float(a.get(s, float("nan"))) - b[s]) for s in S)

def stages(d, ex):
    """Return dict of per-stage errors (None if the model omitted the field) and the first failing stage."""
    e = {}
    try: e["post"] = l1(d["posterior"], ex["posterior"])
    except Exception: e["post"] = None
    try: e["stale"] = abs(float(d["p_stale_A"]) - ex["p_stale_A"])
    except Exception: e["stale"] = None
    try: e["pred"] = max(l1(d["predict_next"][a], ex["predict_next"][a]) for a in ("query_A", "query_V"))
    except Exception: e["pred"] = None
    try:
        e["after"] = max(sum(ex["predict_next"][a][o] * abs(float(d["posterior_after"][a][o]["healthy"])
                                                            - ex["posterior_after"][a][o]["healthy"]) for o in S)
                         for a in ("query_A", "query_V"))
    except Exception: e["after"] = None
    try:
        sc = {a: float(v) for a, v in d["expected_cost"].items() if a in ex["expected_cost"]}
        e["cost"] = max(abs(sc[a] - ex["expected_cost"][a]) for a in ex["expected_cost"])
        e["rank"] = min(sc, key=sc.get) != ex["best_action"]
        e["own_best"] = min(sc, key=sc.get)
    except Exception: e["cost"] = e["rank"] = e["own_best"] = None
    order = [("belief", (e["post"], TOL["post"]), (e["stale"], TOL["stale"])),
             ("prediction", (e["pred"], TOL["pred"])),
             ("anticipated", (e["after"], TOL["after"])),
             ("value", (e["cost"], TOL["cost"]))]
    first = None
    for name, *checks in order:
        if any(v is None for v, _ in checks): first = first or f"{name} (missing)"; break
        if any(v > t for v, t in checks) or (name == "value" and e["rank"]): first = name; break
    if first is None:
        first = "selection" if d.get("action") != e["own_best"] else "none"
    return e, first

recs = []
for (iid, model, smp), r in sorted(latest.items()):
    it = items[iid]; ex = it["exact"]; d = r["parsed"]
    e, first = stages(d, ex)
    a = d.get("action")
    recs.append(dict(id=iid, model=model, sample=smp, action=a, best=ex["best_action"],
                     dloss=(ex["expected_cost"][a] - ex["expected_cost"][ex["best_action"]]) if a in ex["expected_cost"] else None,
                     requery=(a == "query_A" and ex["best_action"] != "query_A"), first=first, **{f"err_{k}": v for k, v in e.items()},
                     has_reasoning=bool(r.get("reasoning_content"))))

def fmt(x): return "–" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
STG = ["belief", "prediction", "anticipated", "value", "selection", "none"]
for model in sorted(set(x["model"] for x in recs)):
    R = [x for x in recs if x["model"] == model]
    print(f"\n## {model}: {len(R)} parsed answers\n")
    print("| Subset | n | " + " | ".join(STG) + " | missing field |")
    print("| --- | --- |" + " --- |" * (len(STG) + 1))
    for name, sub in [("all answers", R), ("decision loss > 0", [x for x in R if (x["dloss"] or 0) > 1e-9]),
                      ("re-queried A when not optimal", [x for x in R if x["requery"]])]:
        c = Counter(x["first"] for x in sub)
        miss = sum(v for k, v in c.items() if "missing" in k)
        print(f"| {name} | {len(sub)} | " + " | ".join(str(c.get(s, 0)) for s in STG) + f" | {miss} |")
    print("\n| Item | s | action | exact best | loss | first departure | post L1 | P(stale) err | pred L1 | after err | max cost err |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for x in R:
        print(f"| {x['id']} | {x['sample']} | {x['action']} | {x['best']} | {fmt(x['dloss'])} | {x['first']} | "
              f"{fmt(x['err_post'])} | {fmt(x['err_stale'])} | {fmt(x['err_pred'])} | {fmt(x['err_after'])} | {fmt(x['err_cost'])} |")
json.dump(recs, open(os.path.join(HERE, "elicit_scored.json"), "w"), indent=1)
print("\nNote: asking for every intermediate quantity changes the task, so action rates here are not comparable"
      " with the first run; this experiment locates errors, it does not re-measure them.")
