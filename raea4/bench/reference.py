"""Reference (non-LLM) policies: one-step vs closed-loop regret, on the selected items and on the full grid.

  python bench/reference.py [--workers 4]   -> results/reference_policies.md / .json

one-step regret (optimal continuation)  Q(first action) - Q(best): the policy picks only the FIRST action, the
                                        true-model optimum plays the rest. This is also what the LLM single-
                                        decision score measures.
closed-loop regret                      the policy picks EVERY action from its own belief / constraint until it
                                        stops; expected total loss under the true model minus the optimum
                                        (core.make_policy / core.closed_loop).
The 140 selected items are category-balanced and over-sample cases where naive models fail; the full grid
(432 tasks, every evidence pattern x parameter combination once) is the population reference. CIs: 95%
percentile bootstrap over tasks, and over the 12 evidence patterns (cluster bootstrap) as a sensitivity check.
"""
import argparse, json, os, sys
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core                                                 # noqa: E402

POL = core.POLICIES
TOOL_LEVEL = ["dedup_history", "dedup_history_initial", "forbid_repeat", "dedup_both"]

def boot(vals, clusters=None, n_boot=2000, seed=0):
    """mean and 95% CI; clusters: list of cluster labels (resample clusters with replacement)."""
    x = np.asarray(vals, float)
    if len(x) == 0: return (np.nan, np.nan, np.nan, 0)
    rng = np.random.default_rng(seed)
    if clusters is None:
        b = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    else:
        labs = sorted(set(clusters)); idx = {c: np.where(np.asarray(clusters) == c)[0] for c in labs}
        b = np.empty(n_boot)
        for i in range(n_boot):
            pick = rng.integers(0, len(labs), len(labs))
            sel = np.concatenate([idx[labs[k]] for k in pick]); b[i] = x[sel].mean()
    return (float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(x))

def f2(t, pct=False):
    m, lo, hi, _ = t
    if np.isnan(m): return "-"
    k, d = (100, 0) if pct else (1, 2)
    return f"{m * k:.{d}f} [{lo * k:.{d}f}, {hi * k:.{d}f}]"

def tool_level(t):
    """Closed-loop regret of the dedup / forbid policies when the agent treats a transient tool as ONE source
    (it does not know that each call hits a fresh node)."""
    M = core.true_model(t); names, srcs = t["tools"], t["sources"]; ev = [tuple(e) for e in t["evidence"]]
    out = {}
    for pol in TOOL_LEVEL:
        f = core.make_policy(pol, M, None, names, srcs, None, len(ev))
        out[pol] = float(core.closed_loop(M, names, f, ev, core.BUDGET)[0] - t["best_value"])
    return out

def table(tasks, title, md, out, key):
    pats = [t["pattern"] for t in tasks]
    md += [f"\n### {title} (n = {len(tasks)})\n",
           "| Policy | n | One-step regret (optimal continuation) | Closed-loop regret | Closed-loop regret, pattern-cluster CI | "
           "First action optimal % | Closed-loop optimal % | One-step severe % | CL P(false benign) % | CL P(escalate) % | "
           "CL call cost |", "| --- |" + " --- |" * 10]
    opt = dict(regret_closed=(0.0,) * 3 + (len(tasks),))
    co = lambda k: float(np.mean([t["closed_opt"][k] for t in tasks]))
    md.append(f"| optimal | {len(tasks)} | 0 | 0 | - | 100 | 100 | - | {co('p_false_benign') * 100:.1f} | "
              f"{co('p_escalate') * 100:.1f} | {co('call_cost'):.2f} |")
    out[key] = dict(n=len(tasks), optimal=dict(p_false_benign=co("p_false_benign"), p_escalate=co("p_escalate"),
                                               call_cost=co("call_cost"), loss=co("loss")))
    for pol in POL:
        S = [t for t in tasks if pol in t["naive"]]
        if not S: continue
        g = lambda k: [t["naive"][pol][k] for t in S]
        r1, rc = g("regret_1step"), g("regret_closed"); cl = [t["pattern"] for t in S]
        res = dict(regret_1step=boot(r1), regret_closed=boot(rc), regret_closed_cluster=boot(rc, cl),
                   regret_1step_cluster=boot(r1, cl),
                   opt_1step=boot([float(x <= 1e-6) for x in r1]), opt_closed=boot([float(x <= 1e-6) for x in rc]),
                   severe_1step=boot([float(x) for x in g("severe_1step")]),
                   cl_p_false_benign=boot(g("cl_p_false_benign")), cl_p_escalate=boot(g("cl_p_escalate")),
                   cl_call_cost=boot(g("cl_call_cost")))
        out[key][pol] = res
        md.append(f"| {pol} | {len(S)} | {f2(res['regret_1step'])} | {f2(res['regret_closed'])} | "
                  f"{f2(res['regret_closed_cluster'])} | {res['opt_1step'][0] * 100:.0f} | {res['opt_closed'][0] * 100:.0f} | "
                  f"{res['severe_1step'][0] * 100:.0f} | {res['cl_p_false_benign'][0] * 100:.1f} | "
                  f"{res['cl_p_escalate'][0] * 100:.1f} | {res['cl_call_cost'][0]:.2f} |")
    # paired closed-loop differences among the dedup family
    md += ["\nPaired closed-loop regret differences (task bootstrap; pattern-cluster bootstrap):\n",
           "| A - B | d closed-loop regret | cluster CI |", "| --- | --- | --- |"]
    for A, B in [("dedup_history", "forbid_repeat"), ("dedup_history_initial", "forbid_repeat"),
                 ("dedup_history", "dedup_history_initial"), ("dedup_both", "forbid_repeat"), ("mmi", "dedup_history"),
                 ("mmi", "dedup_history_initial"), ("fault_unaware", "mmi")]:
        d = [t["naive"][A]["regret_closed"] - t["naive"][B]["regret_closed"] for t in tasks]
        r, rcl = boot(d), boot(d, pats)
        out[key][f"{A}-{B}"] = dict(task=r, cluster=rcl)
        md.append(f"| {A} - {B} | {f2(r)} | [{rcl[1]:.2f}, {rcl[2]:.2f}] |")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    grid = json.load(open(os.path.join(HERE, "grid_all.json")))
    items = [json.loads(l) for l in open(os.path.join(HERE, "items.jsonl"), encoding="utf-8")]
    md = ["# Reference policies (exact, no LLM)\n",
          "One-step regret (optimal continuation) = Q(first action) - Q(best): the policy chooses only the first "
          "action and the true-model optimum plays the rest (the quantity the single-decision LLM score measures). "
          "Closed-loop regret = the policy chooses every action from its own belief / constraint until it stops; "
          "expected total loss under the true model minus the optimum. The 140 selected items are "
          "category-balanced and over-sample cases where naive models fail; the full grid (432 tasks) is the "
          "population reference. 95% CIs: bootstrap over tasks, and over the 12 evidence patterns (cluster "
          "bootstrap) as a sensitivity check.\n",
          "Policies (core.make_policy): fault_unaware / mmi / split plan with their own model at every step. "
          "dedup_history discards every later report from an already-used persistent source, including reports "
          "obtained during the episode; dedup_history_initial discards repeat-source reports in the initial history "
          "only and keeps every new report; forbid_repeat keeps all reports and never re-queries a used persistent "
          "source; dedup_both = dedup_history + forbid_repeat. Transient tool calls are separate sources (never "
          "discarded or forbidden). CL = closed loop.\n"]
    out = {}
    table(items, "Selected 140 base tasks (category-balanced; NOT a population estimate)", md, out, "selected")
    table(grid, "Full grid (all 432 math tasks)", md, out, "grid")
    table([t for t in grid if t["gap"] >= 0.05], "Full grid without near-ties (gap >= 0.05)", md, out, "grid_no_ties")
    table([t for t in grid if t["persistent"]], "Full grid, persistent structures only", md, out, "grid_persistent")
    table([t for t in items if t["persistent"]], "Selected tasks, persistent structures only", md, out, "selected_persistent")
    # transient tasks under a tool-level (source-unaware) dedup / forbid
    tr = [t for t in grid if not t["persistent"]]
    with Pool(a.workers) as p: tl = p.map(tool_level, tr)
    md += ["\n### Transient tasks: source-aware vs tool-level dedup / forbid (closed-loop regret, full grid, "
           f"n = {len(tr)})\n",
           "Tool-level = the agent treats every call of the transient tool as the same source (the definition "
           "used before this revision).\n", "| Policy | Source-aware | Tool-level |", "| --- | --- | --- |"]
    out["transient_tool_level"] = {}
    for pol in TOOL_LEVEL:
        sa = boot([t["naive"][pol]["regret_closed"] for t in tr]); tlv = boot([x[pol] for x in tl])
        out["transient_tool_level"][pol] = dict(source_aware=sa, tool_level=tlv)
        md.append(f"| {pol} | {f2(sa)} | {f2(tlv)} |")
    md.append(f"\nCheck: closed-loop optimal play reproduces the exact best value on every grid task "
              f"(max |error| = {max(abs(t['closed_opt_check']) for t in grid):.1e}).")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    open(os.path.join(HERE, "results", "reference_policies.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump(out, open(os.path.join(HERE, "results", "reference_policies.json"), "w"), indent=1, default=float)
    print("\n".join(md))

if __name__ == "__main__":
    main()
