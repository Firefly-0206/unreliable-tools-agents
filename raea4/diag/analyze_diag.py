"""Score results.jsonl against the exact targets in diag_items.json. Prints markdown tables, writes summary.json."""
import json, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
items = {it["id"]: it for it in json.load(open(os.path.join(HERE, "diag_items.json"), encoding="utf-8"))}
rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results.jsonl"), encoding="utf-8")]
# keep the latest parsed answer per (item, model, sample); earlier truncated attempts are dropped
latest = {}
for r in rows:
    if r.get("ok") and r.get("p_healthy") is not None and r.get("action"):
        latest[(r["id"], r["model"], r["sample"])] = r
rows = list(latest.values())

recs = []
for r in rows:
    it = items[r["id"]]; p = r["p_healthy"]; a = r["action"]
    valid_p = p is not None and 0 <= p <= 1
    valid_a = a in it["Q"]
    recs.append(dict(model=r["model"], pres=it["presentation"], regime=it["regime"], pattern=it["pattern"],
                     p=p if valid_p else np.nan, p_exact=it["p_exact"], tau=it["tau"],
                     err=(p - it["p_exact"]) if valid_p else np.nan,
                     cross=(valid_p and ((p >= it["tau"]) != (it["p_exact"] >= it["tau"]))),
                     dloss=(it["Q"][a] - it["best_value"]) if valid_a else np.nan,
                     act=a, best=it["best_action"], parsed=valid_p and valid_a))

def agg(sel):
    if not sel: return None
    e = np.array([x["err"] for x in sel]); d = np.array([x["dloss"] for x in sel])
    return dict(n=len(sel), mean_abs_err=float(np.nanmean(np.abs(e))), mean_err=float(np.nanmean(e)),
                cross_rate=float(np.mean([x["cross"] for x in sel])), decision_loss=float(np.nanmean(d)),
                parsed=float(np.mean([x["parsed"] for x in sel])))

out = {}
models = sorted(set(x["model"] for x in recs))
print("## Overall by model and presentation\n")
print("| Model | Presentation | n | Mean abs. error in P(healthy) | Mean signed error | Crosses τ* wrongly | Mean decision loss |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for m in models:
    for pr in ["plain", "warning", "dedup"]:
        a = agg([x for x in recs if x["model"] == m and x["pres"] == pr])
        if a:
            out[f"{m}|{pr}"] = a
            print(f"| {m} | {pr} | {a['n']} | {a['mean_abs_err']:.3f} | {a['mean_err']:+.3f} | {a['cross_rate']*100:.0f}% | {a['decision_loss']:.2f} |")

print("\n## By evidence pattern (plain presentation): mean reported P(healthy) vs exact\n")
pats = ["single", "retry2", "retry3", "shared3", "indep3", "retry_mixed"]
for m in models:
    print(f"\n**{m}**\n")
    print("| Regime | " + " | ".join(pats) + " |")
    print("| --- |" + " --- |" * len(pats))
    for rg in ["frequent", "moderate", "rare"]:
        cells = []
        for pt in pats:
            sel = [x for x in recs if x["model"] == m and x["pres"] == "plain" and x["regime"] == rg and x["pattern"] == pt]
            if sel:
                cells.append(f"{np.nanmean([x['p'] for x in sel]):.2f} (exact {sel[0]['p_exact']:.2f})")
            else:
                cells.append("–")
        print(f"| {rg} | " + " | ".join(cells) + " |")

print("\n## Key contrasts (mean over samples)\n")
print("| Model | Presentation | Regime | retry3 − single: model / exact | indep3 − shared3: model / exact | Decision loss on retry3+shared3 | on indep3 |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for m in models:
    for pr in ["plain", "warning", "dedup"]:
        for rg in ["frequent", "moderate", "rare"]:
            g = lambda pt: [x for x in recs if x["model"] == m and x["pres"] == pr and x["regime"] == rg and x["pattern"] == pt]
            if not g("single"): continue
            mp = lambda pt: np.nanmean([x["p"] for x in g(pt)]); ex = lambda pt: g(pt)[0]["p_exact"]
            dl_dep = np.nanmean([x["dloss"] for x in g("retry3") + g("shared3")]); dl_ind = np.nanmean([x["dloss"] for x in g("indep3")])
            print(f"| {m} | {pr} | {rg} | {mp('retry3') - mp('single'):+.2f} / {ex('retry3') - ex('single'):+.2f} | "
                  f"{mp('indep3') - mp('shared3'):+.2f} / {ex('indep3') - ex('shared3'):+.2f} | {dl_dep:.2f} | {dl_ind:.2f} |")
json.dump(dict(summary=out, records=recs), open(os.path.join(HERE, "summary.json"), "w"), indent=1, default=float)
