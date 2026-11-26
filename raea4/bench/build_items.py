"""Build bench v2: base tasks with exact targets (items.jsonl) and one prompt per task x condition (prompts.jsonl).

Full grid: 12 evidence patterns x phi{.05,.15,.3} x q{.75,.85,.9} x 2 priors x 2 severities = 432 math tasks.
Selection (deterministic): drop near-ties (best-action gap < MIN_GAP), then for each exact-best action category
take up to PER_CAT tasks, round-robin over evidence patterns in a seeded order; each selected math task gets one
scenario, assigned round-robin within its category so that scenarios are balanced per stratum.
Usage: python bench/build_items.py [--per-cat 28] [--workers 4]
"""
import argparse, hashlib, itertools, json, os, sys
from collections import Counter, defaultdict
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core                                       # noqa: E402

MIN_GAP = 0.05

def _solve(args):
    return core.solve_task(*args)

def h(s):
    return hashlib.sha256(s.encode()).hexdigest()

def select(tasks, per_cat):
    """Category-balanced selection on the original strata (sel_category, sel_repeat_wrong). It over-samples
    the cases where naive models fail, so numbers on the selected set are NOT population estimates; the
    full-grid numbers are in results/reference_policies.md."""
    by_cat = defaultdict(lambda: defaultdict(list))
    for t in tasks:
        if t["gap"] >= MIN_GAP: by_cat[t["sel_category"]][t["pattern"]].append(t)
    chosen = []
    for cat in core.SEL_CATS:
        pats = sorted(by_cat[cat], key=lambda p: h(cat + p))
        queues = {p: sorted(by_cat[cat][p], key=lambda t: h(t["math_id"])) for p in pats}
        # within a pattern, alternate repeat-wrong / other tasks so both kinds survive the cap
        for p in pats:
            rw = [t for t in queues[p] if t["sel_repeat_wrong"]]; ot = [t for t in queues[p] if not t["sel_repeat_wrong"]]
            queues[p] = [x for pair in itertools.zip_longest(rw, ot) for x in pair if x is not None]
        got = []
        while len(got) < per_cat and any(queues.values()):
            for p in pats:
                if queues[p] and len(got) < per_cat: got.append(queues[p].pop(0))
        for i, t in enumerate(sorted(got, key=lambda t: h("scen" + t["math_id"]))):
            t["scenario"] = sorted(core.SCEN)[i % len(core.SCEN)]
        chosen += got
    return chosen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=28)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    grid = list(itertools.product(core.PATTERNS, core.PHIS, core.QS, core.PRIORS, core.SEVS))
    with Pool(a.workers) as pool:
        tasks = pool.map(_solve, grid, chunksize=4)
    for t in tasks:
        t["math_id"] = f"{t['pattern']}|phi{t['phi']}|q{t['q']}|{t['prior_name']}|{t['sev_name']}"
    json.dump(tasks, open(os.path.join(HERE, "grid_all.json"), "w"), default=float)
    print(f"{len(tasks)} math tasks; near-ties dropped: {sum(t['gap'] < MIN_GAP for t in tasks)}")
    print("exact-best category over the full grid:", dict(Counter(t["best_category"] for t in tasks)))
    items = select(tasks, a.per_cat)
    for t in items:
        t["id"] = f"{t['scenario']}|{t['math_id']}"
    items.sort(key=lambda t: t["id"])
    # placebo donor (X1-X3): another selected task with the same evidence pattern (identical block shape),
    # chosen by a seeded hash; whether its best action agrees with the item's is recorded
    byid = {t["id"]: t for t in items}
    for t in items:
        cand = sorted((u for u in items if u["pattern"] == t["pattern"] and u["math_id"] != t["math_id"]),
                      key=lambda u: h("donor" + t["id"] + u["id"]))
        d = cand[0]
        t["placebo_donor"] = d["id"]; t["placebo_same_best"] = d["best_action"] == t["best_action"]
        t["placebo_donor_p_benign"] = d["p_benign"]
    with open(os.path.join(HERE, "items.jsonl"), "w", encoding="utf-8") as f:
        for t in items: f.write(json.dumps(t, default=float) + "\n")
    n = 0
    with open(os.path.join(HERE, "prompts.jsonl"), "w", encoding="utf-8") as f:
        for t in items:
            for c in core.CONDITIONS:
                pr = core.build_prompt(t, c, donor=byid[t["placebo_donor"]])
                f.write(json.dumps(dict(pid=f"{t['id']}|{c}", item=t["id"], cond=c, prompt=pr,
                                        tools=c == "T1"), ensure_ascii=False) + "\n"); n += 1
    # strata report
    rep = dict(n_items=len(items), n_prompts=n,
               best_category=Counter(t["best_category"] for t in items),
               scenario=Counter(t["scenario"] for t in items), pattern=Counter(t["pattern"] for t in items),
               phi=Counter(str(t["phi"]) for t in items), q=Counter(str(t["q"]) for t in items),
               prior=Counter(t["prior_name"] for t in items), sev=Counter(t["sev_name"] for t in items),
               sel_category=Counter(t["sel_category"] for t in items),
               structure=Counter(t["structure"] for t in items),
               repeat_right_persistent=sum(t["repeat_right"] for t in items),
               repeat_right_transient=sum(t["repeat_right_transient"] for t in items),
               repeat_right_by_pattern=Counter(t["pattern"] for t in items if t["repeat_right"] or t["repeat_right_transient"]),
               repeat_wrong=sum(t["repeat_wrong"] for t in items),
               dependent=sum(t["dependent"] for t in items),
               dependent_dir=Counter(str(t["dep_dir"]) for t in items if t["dependent"]),
               placebo_same_best=sum(t["placebo_same_best"] for t in items),
               cat_x_scenario=Counter(f"{t['sel_category']}|{t['scenario']}" for t in items),
               grid_best_category=Counter(t["best_category"] for t in tasks),
               grid_sel_category=Counter(t["sel_category"] for t in tasks),
               closed_opt_check_max=max(abs(t["closed_opt_check"]) for t in tasks),
               naive_mean_regret_1step={m: round(sum(t["naive"][m]["regret_1step"] for t in items if m in t["naive"]) /
                                                 max(1, sum(m in t["naive"] for t in items)), 3) for m in core.POLICIES},
               naive_mean_regret_closed={m: round(sum(t["naive"][m]["regret_closed"] for t in items if m in t["naive"]) /
                                                  max(1, sum(m in t["naive"] for t in items)), 3) for m in core.POLICIES})
    json.dump(rep, open(os.path.join(HERE, "items_strata.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))

if __name__ == "__main__":
    main()
