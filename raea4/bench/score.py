"""Score bench v2 answers against the exact targets.

  python bench/score.py bench/results/deepseek_deepseek-chat.jsonl [more.jsonl ...] [--name report]

Per answer (Q = exact action values within the remaining budget, full history):
  regret      one-step regret (optimal continuation): Q(chosen) - Q(best), i.e. the cost of the chosen FIRST
              action when the true-model optimum plays the rest. It is not a closed-loop regret (a single-
              decision benchmark cannot measure one). Unparsed / truncated / failed answers fall back to
              `escalate`; regret_parsed uses parsed answers only. The fallback flatters items whose optimum is
              escalate, so always read it next to `parsed`.
  optimal     parsed and exactly optimal (unparsed counts as not optimal). severe/escalate/query_cost/cbwa
              use parsed answers only.
  severe      chose to declare the benign state although the exact P(benign) < tau* = 1 - esc/sev.
  escalate    chose escalate.        query_cost  cost of the chosen query (0 for declarations/escalation).
  overcount   on items with several reports from one persistent source: the reported p_benign is more than
              0.10 beyond the exact value IN THE DIRECTION of the repeated reports (dep_dir: up if they are
              benign, down for retry_bad); overconf2 = |p - exact| > 0.10 in either direction; signed_err = p - exact.
  cbwa        "correct belief, wrong action": |reported - exact| <= 0.02 and the action is suboptimal
              (split by the category of the chosen action). For P3 also against the shown-evidence posterior.
Record selection: one answer per run key (run.run_key). The FIRST record of a key that is ok or a model-side
failure (timeout) is scored; ok=False records count as failed answers (status api_error / timeout); later
attempts (--retry-unparsed, re-runs after failures) are reported separately and never replace the first.
Answers are averaged within base task; CIs are 95% percentile bootstrap over base tasks; contrasts are paired at
the base-task level within one run; a pattern-cluster bootstrap CI (12 evidence patterns) is given as a
sensitivity check. P2-P0 and P3-P0 are computed on persistent-structure tasks (P2/P3 change nothing on
transient tasks except the grouping labels) and, separately, on transient tasks.
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core                                             # noqa: E402
from run import price_of, parse_into, run_key, load_items  # noqa: E402
from reference import boot                              # noqa: E402

# (A, B, subset)
CONTRASTS = [("P1", "P0", "all"), ("P2", "P0", "persistent"), ("P3", "P0", "persistent"),
             ("P2", "P0", "transient"), ("P3", "P0", "transient"),
             ("L1", "P0", "all"), ("L2", "P0", "all"), ("L3", "P0", "all"), ("L2", "L1", "all"), ("L3", "L2", "all"),
             ("L1", "C1", "all"), ("L2", "C2", "all"), ("L3", "C3", "all"),
             ("L1", "X1", "all"), ("L2", "X2", "all"), ("L3", "X3", "all"),
             ("C1", "P0", "all"), ("X1", "P0", "all"), ("T1", "P0", "all"), ("H1", "P0", "all")]
METRICS = ["regret", "optimal", "severe", "escalate", "query_cost", "parsed"]
QCOST = {"V": core.V_COST}

def is_timeout(r):
    e = (r.get("error") or "").lower()
    return not r.get("ok") and ("timed out" in e or "timeout" in e)

def score_answer(it, r):
    a = r.get("action") if r.get("parse_status") == "ok" else None
    p = r.get("p_benign") if r.get("parse_status") == "ok" else None
    ae = a or "escalate"
    reg = it["Q"][ae] - it["best_value"]
    dep = it["dependent"] and p is not None
    d = dict(parsed=float(a is not None), regret=reg, regret_parsed=reg if a else np.nan,
             optimal=float(a is not None and reg <= 1e-6), severe=float(a == "declare_0" and it["p_benign"] < it["tau"]),
             escalate=float(a == "escalate"),
             query_cost=float(QCOST.get(a[6:], core.SRC_COST)) if a and a.startswith("query_") else 0.0,
             chosen_cat=it["categories"][ae] if a else "unparsed",
             abs_err=abs(p - it["p_benign"]) if p is not None else np.nan,
             overcount=float(it["dep_dir"] * (p - it["p_benign"]) > 0.10) if dep else np.nan,
             overconf2=float(abs(p - it["p_benign"]) > 0.10) if dep else np.nan,
             signed_err=float(p - it["p_benign"]) if dep else np.nan,
             cbwa=float(abs(p - it["p_benign"]) <= 0.02 and reg > 1e-6) if p is not None else np.nan,
             cbwa_shown=float(abs(p - it["p_benign_shown"]) <= 0.02 and reg > 1e-6) if p is not None else np.nan)
    return d

def task_means(rows, key):
    by = defaultdict(list)
    for r in rows: by[r["item"]].append(r[key])
    return {i: float(np.nanmean(v)) if not all(np.isnan(v)) else np.nan for i, v in by.items()}

def tb(d, items, cluster=False):
    """bootstrap over task means (dict item -> value), optionally clustered by evidence pattern."""
    ks = [k for k in d if not np.isnan(d[k])]
    return boot([d[k] for k in ks], [items[k]["pattern"] for k in ks] if cluster else None)

def fmt_ci(t, pct=False):
    m, lo, hi, n = t
    if np.isnan(m): return "-"
    k = 100 if pct else 1
    return f"{m * k:.{0 if pct else 2}f} [{lo * k:.{0 if pct else 2}f}, {hi * k:.{0 if pct else 2}f}]"

def load_records(files):
    """-> (primary record per run key, later attempts per key)."""
    by = defaultdict(list)
    for pat in files:
        for fn in glob.glob(pat):
            for l in open(fn, encoding="utf-8"):
                r = json.loads(l); by[run_key(r)].append(r)
    prim, later = {}, {}
    for k, rs in by.items():
        el = [r for r in rs if r.get("ok") or is_timeout(r)]
        prim[k] = el[0] if el else rs[-1]
        later[k] = [r for r in rs if r is not prim[k] and r.get("ok") and rs.index(r) > rs.index(prim[k])]
    return prim, later

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--name", default=None)
    ap.add_argument("--items-file", default=None, help="alternate items jsonl (default bench/items.jsonl)")
    a = ap.parse_args()
    items = load_items(a.items_file)
    sev_names = ["sev3", "sev6"] if not a.items_file else sorted({t["sev_name"] for t in items.values()})
    prim, later = load_records(a.files)
    rows = []
    for k, r in prim.items():
        it = items[r["item"]]
        if r.get("ok"): parse_into(r, it, r["cond"])          # re-parse with the current rules; raw content is kept
        else: r.update(parse_status="timeout" if is_timeout(r) else "api_error", action=None, p_benign=None)
        d = score_answer(it, r)
        u = r.get("usage") or {}
        tag = r.get("tag") or ""
        d.update(item=r["item"], cond=r["cond"], run=f"{r['model']}|cap{r['max_tokens']}" + (f"|{tag}" if tag else ""),
                 model=r["model"], cap=r["max_tokens"], status=r.get("parse_status"), finish=r.get("finish_reason"),
                 out_tok=u.get("completion_tokens", np.nan), in_tok=u.get("prompt_tokens", np.nan),
                 reason_tok=u.get("reasoning_tokens", np.nan), latency=r["latency"] if r.get("latency") is not None else np.nan,
                 best_cat=it["best_category"], pattern=it["pattern"], sev=it["sev_name"], persistent=it["persistent"],
                 repeat_right=it["repeat_right"], repeat_right_transient=it["repeat_right_transient"],
                 repeat_wrong=it["repeat_wrong"], n_later=len(later[k]),
                 later_parsed=float(any(parse_into(dict(x), it, x["cond"])["parse_status"] == "ok" for x in later[k])),
                 call_errors=len(r.get("call_errors") or []))
        rows.append(d)
    out = dict(conditions={}, contrasts={}, breakdown={}, cbwa_by_action={}, fixed_budget={}, cost={}, unparsed={},
               retries={}, reference_policies={})
    md = [f"# Bench v2 score ({len(rows)} answers, {len({r['item'] for r in rows})} base tasks)\n",
          "Regret = one-step regret (optimal continuation): Q(chosen first action) - Q(best), the rest played "
          "optimally (unparsed -> escalate). Means with 95% bootstrap CI over base tasks.\n"]
    runs = sorted({r["run"] for r in rows})
    used_items = sorted({r["item"] for r in rows})
    md += ["## Reference policies (exact, no LLM) on the same base tasks\n",
           "Category-balanced selection, not a population estimate (full grid: results/reference_policies.md).\n",
           "| Policy | One-step regret (opt. continuation) | Closed-loop regret | First action optimal | Closed-loop optimal | One-step severe |",
           "| --- | --- | --- | --- | --- | --- |"]
    for nm in core.POLICIES:
        sel = [items[i] for i in used_items if nm in items[i]["naive"]]
        if not sel: continue
        r1 = [t["naive"][nm]["regret_1step"] for t in sel]; rc = [t["naive"][nm]["regret_closed"] for t in sel]
        sv = [float(t["naive"][nm]["severe_1step"]) for t in sel]
        res = dict(regret_1step=boot(r1), regret_closed=boot(rc), opt_1step=boot([float(x <= 1e-6) for x in r1]),
                   opt_closed=boot([float(x <= 1e-6) for x in rc]), severe_1step=boot(sv))
        out["reference_policies"][nm] = res
        md.append(f"| {nm} (n={len(sel)}) | {fmt_ci(res['regret_1step'])} | {fmt_ci(res['regret_closed'])} | "
                  f"{fmt_ci(res['opt_1step'], True)}% | {fmt_ci(res['opt_closed'], True)}% | {fmt_ci(res['severe_1step'], True)}% |")
    for run in runs:
        R = [r for r in rows if r["run"] == run]
        conds = [c for c in core.CONDITIONS if any(r["cond"] == c for r in R)]
        md += [f"\n## Run {run}\n", "| Cond | n ans | One-step regret | Regret (parsed only) | Optimal % | Severe % | Escalate % | "
               "Query cost | Parsed % | |p-exact| | Over-count % (dir.) | Over-conf. % (2-sided) | Signed err | CBWA % |",
               "| --- |" + " --- |" * 14]
        for c in conds:
            S = [r for r in R if r["cond"] == c]
            res = {m: tb(task_means(S, m), items) for m in METRICS + ["abs_err", "overcount", "overconf2", "signed_err",
                                                                       "cbwa", "cbwa_shown", "regret_parsed"]}
            out["conditions"][f"{run}|{c}"] = dict(n_answers=len(S), **res)
            md.append(f"| {c} | {len(S)} | {fmt_ci(res['regret'])} | {fmt_ci(res['regret_parsed'])} | {fmt_ci(res['optimal'], True)} | "
                      f"{fmt_ci(res['severe'], True)} | {fmt_ci(res['escalate'], True)} | {res['query_cost'][0]:.2f} | "
                      f"{res['parsed'][0] * 100:.0f} | {res['abs_err'][0]:.3f} | {fmt_ci(res['overcount'], True)} | "
                      f"{fmt_ci(res['overconf2'], True)} | {res['signed_err'][0]:+.3f} | {fmt_ci(res['cbwa'], True)} |")
        md += ["\nPaired contrasts (task-level; difference A - B; last column: pattern-cluster bootstrap CI of d regret):\n",
               "| A - B | Tasks | n | d Regret | d Optimal (pts) | d Severe (pts) | d Regret, cluster CI |",
               "| --- | --- | --- | --- | --- | --- | --- |"]
        for A, B, sub in CONTRASTS:
            if A not in conds or B not in conds: continue
            keep = lambda r: sub == "all" or r["persistent"] == (sub == "persistent")
            res = {}
            for m in ["regret", "optimal", "severe"]:
                ta = task_means([r for r in R if r["cond"] == A and keep(r)], m)
                tb_ = task_means([r for r in R if r["cond"] == B and keep(r)], m)
                diff = {i: ta[i] - tb_[i] for i in sorted(set(ta) & set(tb_))}
                res[m] = tb(diff, items)
                if m == "regret": res["regret_cluster"] = tb(diff, items, cluster=True)
            if res["regret"][3] == 0: continue
            out["contrasts"][f"{run}|{A}-{B}|{sub}"] = res
            rc = res["regret_cluster"]
            md.append(f"| {A} - {B} | {sub} | {res['regret'][3]} | {fmt_ci(res['regret'])} | {fmt_ci(res['optimal'], True)} | "
                      f"{fmt_ci(res['severe'], True)} | [{rc[1]:.2f}, {rc[2]:.2f}] |")
        flags = ["repeat_right", "repeat_right_transient", "repeat_wrong"]
        md += ["\nOne-step regret by exact-best category (task means; repeat right = best action re-queries a used "
               "persistent source; repeat right (transient) = best action re-calls the transient tool):\n",
               "| Cond | " + " | ".join(core.CATS) + " | repeat right (persistent) | repeat right (transient) | repeat wrong |",
               "| --- |" + " --- |" * (len(core.CATS) + 3)]
        for c in conds:
            S = [r for r in R if r["cond"] == c]; cells = []
            for cat in core.CATS:
                cells.append(f"{tb(task_means([r for r in S if r['best_cat'] == cat], 'regret'), items)[0]:.2f}")
            for fl in flags:
                cells.append(f"{tb(task_means([r for r in S if r[fl]], 'regret'), items)[0]:.2f}")
            out["breakdown"][f"{run}|{c}"] = dict(zip(core.CATS + flags, cells))
            md.append(f"| {c} | " + " | ".join(cells) + " |")
        md += ["\nCorrect belief, wrong action: share of parsed answers, by chosen action category:\n",
               "| Cond | " + " | ".join(core.CATS) + " |", "| --- |" + " --- |" * len(core.CATS)]
        for c in conds:
            S = [r for r in R if r["cond"] == c and r["parsed"]]
            cells = {cat: float(np.mean([r["cbwa"] == 1 and r["chosen_cat"] == cat for r in S])) if S else np.nan for cat in core.CATS}
            out["cbwa_by_action"][f"{run}|{c}"] = cells
            md.append(f"| {c} | " + " | ".join(f"{v * 100:.1f}%" for v in cells.values()) + " |")
        md += ["\nTrade-off by severity variant (severe % / escalate % / mean query cost / one-step regret):\n",
               "| Cond | " + " | ".join(sev_names) + " |", "| --- |" + " --- |" * len(sev_names)]
        for c in conds:
            cells = []
            for sv in sev_names:
                S = [r for r in R if r["cond"] == c and r["sev"] == sv]
                if not S: cells.append("-"); continue
                g = lambda m: tb(task_means(S, m), items)[0]
                cells.append(f"{g('severe') * 100:.0f} / {g('escalate') * 100:.0f} / {g('query_cost'):.2f} / {g('regret'):.2f}")
            md.append(f"| {c} | " + " | ".join(cells) + " |")
        st = defaultdict(int)
        for r in R:
            if r["status"] != "ok": st[f"{r['cond']}:{r['status']}"] += 1
        out["unparsed"][run] = dict(st)
        out["retries"][run] = dict(keys_with_later_attempts=sum(r["n_later"] > 0 for r in R),
                                   later_attempt_parsed_where_first_did_not=sum(r["later_parsed"] > 0 and not r["parsed"] for r in R),
                                   answers_after_failed_calls=sum(r["call_errors"] > 0 for r in R))
    md += ["\n## Fixed-budget view (per model and output cap)\n",
           "| Model | Cap | Cond | n | Success % (finished + parsed) | One-step regret | Median out tok | Median reasoning tok | Median latency s |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (m, cap, c) in sorted({(r["model"], r["cap"], r["cond"]) for r in rows}, key=lambda x: (x[0], x[1], core.CONDITIONS.index(x[2]))):
        S = [r for r in rows if r["model"] == m and r["cap"] == cap and r["cond"] == c]
        succ = tb(task_means([dict(r, ok_=float(r["parsed"] and r["finish"] != "length")) for r in S], "ok_"), items)
        reg = tb(task_means(S, "regret"), items)
        mo = float(np.nanmedian([r["out_tok"] for r in S])); mr = float(np.nanmedian([r["reason_tok"] for r in S]))
        ml = float(np.nanmedian([r["latency"] for r in S]))
        out["fixed_budget"][f"{m}|{cap}|{c}"] = dict(success=succ, regret=reg, median_out=mo, median_reason=mr, median_latency=ml, n=len(S))
        md.append(f"| {m} | {cap} | {c} | {len(S)} | {fmt_ci(succ, True)} | {fmt_ci(reg)} | {mo:.0f} | {mr:.0f} | {ml:.1f} |")
    md += ["\n## Cost (USD at off-peak list price, no cache discount; x2 at peak)\n",
           "| Model | Answers | Input tok | Output tok (incl. reasoning) | Reasoning tok | USD | USD per answer |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for m in sorted({r["model"] for r in rows}):
        S = [r for r in rows if r["model"] == m]
        ti = float(np.nansum([r["in_tok"] for r in S])); to = float(np.nansum([r["out_tok"] for r in S]))
        tr = float(np.nansum([r["reason_tok"] for r in S])); pin, pout = price_of(m)
        usd = (ti * pin + to * pout) / 1e6
        out["cost"][m] = dict(answers=len(S), input=ti, output=to, reasoning=tr, usd=usd)
        md.append(f"| {m} | {len(S)} | {ti:.0f} | {to:.0f} | {tr:.0f} | {usd:.4f} | {usd / max(len(S), 1):.5f} |")
    md += ["\nUnparsed / failed answers by condition and status (first attempts): " + json.dumps(out["unparsed"]),
           "\nLater attempts (never scored): " + json.dumps(out["retries"])]
    name = a.name or os.path.splitext(os.path.basename(a.files[0]))[0]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    open(os.path.join(HERE, "results", f"score_{name}.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump(out, open(os.path.join(HERE, "results", f"score_{name}.json"), "w"), indent=1, default=float)
    print("\n".join(md))

if __name__ == "__main__":
    main()
