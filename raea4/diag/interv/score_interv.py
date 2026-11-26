"""Score interv_results.jsonl. Standard library only.
Regret = Q(chosen) - Q(best) under the exact model, per answer; averaged over samples within a task,
then over the 18 tasks. Paired differences between conditions are bootstrapped over tasks (10,000 draws).
Usage: python score_interv.py [interv_results.jsonl]"""
import json, os, random, sys
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
items = {it["id"]: it for it in json.load(open(os.path.join(HERE, "interv_items.json"), encoding="utf-8"))}
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "interv_results.jsonl")
latest, fails = {}, Counter()
for line in open(path, encoding="utf-8"):
    r = json.loads(line)
    if r.get("ok") and r.get("p_healthy") is not None and r.get("action") in items[r["id"]]["Q"]:
        latest[(r["id"], r["model"], r["sample"])] = r
    elif not r.get("ok"): fails[(r["model"], r.get("error", "")[:8])] += 1
    else: fails[(r["model"], f"unparsed, finish_reason={r.get('finish_reason')}")] += 1

def kind(it, a):
    best = it["best_action"]
    if a == best: return "optimal"
    if a.startswith("query_") and best.startswith("query_"): return "wrong query"
    if a.startswith("query_"): return "query instead of stopping"
    if best.startswith("query_"): return "stopped too early"
    return "wrong terminal action"

per = defaultdict(list)          # (model, condition, task) -> regrets
kinds = defaultdict(Counter)
for (iid, m, s), r in latest.items():
    it = items[iid]; a = r["action"]
    per[(m, it["condition"], it["task"])].append(it["Q"][a] - it["best_value"])
    kinds[(m, it["condition"])][kind(it, a)] += 1
mean = lambda v: sum(v) / len(v)
def boot(d, n=10000, seed=0):
    rng = random.Random(seed); bs = sorted(mean([rng.choice(d) for _ in d]) for _ in range(n))
    return mean(d), bs[int(.025 * n)], bs[int(.975 * n) - 1]

CONDS = ["L0", "C1", "L1", "C2", "L2", "C3", "L3"]
PAIRS = [("L1", "L0", "joint posterior"), ("L2", "L1", "predictions and branch posteriors"),
         ("L3", "L2", "exact action values"), ("C1", "L0", "control 1 (structure only)"),
         ("L1", "C1", "joint posterior vs its control"), ("L2", "C2", "branch quantities vs control"),
         ("L3", "C3", "action values vs control")]
KINDS = ["optimal", "stopped too early", "wrong terminal action", "query instead of stopping", "wrong query"]
for m in sorted({k[0] for k in per}):
    tasks = sorted({k[2] for k in per if k[0] == m})
    print(f"\n## {m}\n\n| Condition | answers | mean regret | " + " | ".join(KINDS) + " |")
    print("| --- | --- | --- |" + " --- |" * len(KINDS))
    for c in CONDS:
        R = [x for t in tasks for x in per.get((m, c, t), [])]
        if R:
            print(f"| {c} | {len(R)} | {mean([mean(per[(m, c, t)]) for t in tasks if per.get((m, c, t))]):.3f} | "
                  + " | ".join(str(kinds[(m, c)][k]) for k in KINDS) + " |")
    print("\n| Contrast | what it isolates | paired difference in regret [95% CI over tasks] |\n| --- | --- | --- |")
    for a, b, what in PAIRS:
        common = [t for t in tasks if per.get((m, a, t)) and per.get((m, b, t))]
        if len(common) >= 5:
            mu, lo, hi = boot([mean(per[(m, a, t)]) - mean(per[(m, b, t)]) for t in common])
            print(f"| {a} − {b} | {what} | {mu:+.3f} [{lo:+.3f}, {hi:+.3f}] (n = {len(common)} tasks) |")
print("\nFailed or unparsed calls (kept separate: no-balance errors vs truncation):", dict(fails) or "none")
print("Oracle conditions locate mechanisms; they are not the performance of a deployable method.")
