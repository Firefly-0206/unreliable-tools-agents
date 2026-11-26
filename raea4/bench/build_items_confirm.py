"""Build the held-out CONFIRMATION task set (no model is called).

Same generator, strata and selection code as build_items.py (imported, not copied), but on a parameter grid that
shares NO value with the original grid:

    original grid: phi {0.05, 0.15, 0.3}   q {0.75, 0.85, 0.9}   prior {p50: .5/.25/.25, p70: .7/.15/.15}   sev {30, 60}
    confirm grid : phi {0.08, 0.22, 0.38}  q {0.72, 0.82, 0.92}  prior {p60: .6/.2/.2,    p80: .8/.1/.1}     sev {40, 80}

(same 12 evidence patterns, same 3 scenarios, same costs: wrong non-benign declaration 10, escalation 4, source
call 1, verifier V 2 with q = 0.8, budget 4). Selection = build_items.select with a fixed seed salt, so the five
selection strata (declare_benign / declare_nonbenign / escalate / re-query a used tool / query_new_source) are
balanced exactly as before and scenarios are balanced within each stratum.

Writes (all in bench/):  items_confirm.jsonl, prompts_confirm.jsonl (conditions P0, L1, L2, H1 only),
items_strata_confirm.json (counts + comparison with the original 140).
Usage: python bench/build_items_confirm.py [--per-cat 28] [--workers 4]
"""
import argparse, hashlib, itertools, json, os, sys
from collections import Counter
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core                                             # noqa: E402
import build_items as bi                                # noqa: E402

SEED = "confirm-v1"                                     # fixed seed: salts every hash used for ordering / assignment
CONFIRM_CONDS = ["P0", "L1", "L2", "H1"]
PHIS = [0.08, 0.22, 0.38]
QS = [0.72, 0.82, 0.92]
PRIORS = {"p60": [0.6, 0.2, 0.2], "p80": [0.8, 0.1, 0.1]}
SEVS = {"sev4": 40.0, "sev8": 80.0}

# registered at import time so that spawned worker processes see the same tables
core.PRIORS.update(PRIORS)
core.SEVS.update(SEVS)

def h(s):
    return hashlib.sha256((SEED + "|" + s).encode()).hexdigest()
bi.h = h                                                # select() and the donor choice use the salted hash

def tuple_of(t):
    return (t["scenario"], t["pattern"], t["phi"], t["q"], tuple(t["prior"]), t["sev"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=28)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    grid = list(itertools.product(core.PATTERNS, PHIS, QS, PRIORS, SEVS))
    with Pool(a.workers) as pool:
        tasks = pool.map(bi._solve, grid, chunksize=4)
    for t in tasks:
        t["math_id"] = f"{t['pattern']}|phi{t['phi']}|q{t['q']}|{t['prior_name']}|{t['sev_name']}"
    print(f"{len(tasks)} math tasks; near-ties dropped: {sum(t['gap'] < bi.MIN_GAP for t in tasks)}")
    items = bi.select(tasks, a.per_cat)
    for t in items:
        t["id"] = f"{t['scenario']}|{t['math_id']}"
    items.sort(key=lambda t: t["id"])
    byid = {t["id"]: t for t in items}
    for t in items:
        cand = sorted((u for u in items if u["pattern"] == t["pattern"] and u["math_id"] != t["math_id"]),
                      key=lambda u: h("donor" + t["id"] + u["id"]))
        d = cand[0]
        t["placebo_donor"] = d["id"]; t["placebo_same_best"] = d["best_action"] == t["best_action"]
        t["placebo_donor_p_benign"] = d["p_benign"]
    with open(os.path.join(HERE, "items_confirm.jsonl"), "w", encoding="utf-8") as f:
        for t in items: f.write(json.dumps(t, default=float) + "\n")
    n = 0
    with open(os.path.join(HERE, "prompts_confirm.jsonl"), "w", encoding="utf-8") as f:
        for t in items:
            for c in CONFIRM_CONDS:
                pr = core.build_prompt(t, c, donor=byid[t["placebo_donor"]])
                f.write(json.dumps(dict(pid=f"{t['id']}|{c}", item=t["id"], cond=c, prompt=pr,
                                        tools=False), ensure_ascii=False) + "\n"); n += 1
    # ---- overlap check against the original 140 and distribution comparison
    orig = [json.loads(l) for l in open(os.path.join(HERE, "items.jsonl"), encoding="utf-8")]
    o_tuples = {tuple_of(t) for t in orig}
    n_tuple_overlap = sum(tuple_of(t) in o_tuples for t in items)
    o5 = {(t["pattern"], t["phi"], t["q"], tuple(t["prior"]), t["sev"]) for t in orig}
    n_math_overlap = sum((t["pattern"], t["phi"], t["q"], tuple(t["prior"]), t["sev"]) in o5 for t in items)
    o_ids = {t["id"] for t in orig}
    shared_values = {k: sorted(set(map(str, (t[k] for t in items))) & set(map(str, (t[k] for t in orig))))
                     for k in ("phi", "q", "sev", "prior_name")}
    shared_values["prior_vec"] = sorted({str(tuple(t["prior"])) for t in items} & {str(tuple(t["prior"])) for t in orig})
    C = lambda rows, k: dict(Counter(str(t[k]) for t in rows))
    rep = dict(
        seed=SEED, grid=dict(phi=PHIS, q=QS, prior=PRIORS, sev=SEVS),
        n_items=len(items), n_prompts=n, conditions=CONFIRM_CONDS,
        overlap=dict(scenario_pattern_phi_q_prior_sev_tuples_shared_with_original=n_tuple_overlap,
                     pattern_phi_q_prior_sev_tuples_shared=n_math_overlap,
                     item_ids_shared=len({t["id"] for t in items} & o_ids),
                     parameter_values_shared_with_original=shared_values),
        best_category=dict(confirm=C(items, "best_category"), original=C(orig, "best_category")),
        sel_category=dict(confirm=C(items, "sel_category"), original=C(orig, "sel_category")),
        scenario=dict(confirm=C(items, "scenario"), original=C(orig, "scenario")),
        pattern=dict(confirm=C(items, "pattern"), original=C(orig, "pattern")),
        structure=dict(confirm=C(items, "structure"), original=C(orig, "structure")),
        phi=C(items, "phi"), q=C(items, "q"), prior=C(items, "prior_name"), sev=C(items, "sev_name"),
        best_action_state=dict(confirm=Counter(("declare_" + t["best_action"][8:]) if t["best_action"].startswith("declare_")
                                               else t["best_action"].split("_")[0] for t in items),
                               original=Counter(("declare_" + t["best_action"][8:]) if t["best_action"].startswith("declare_")
                                                else t["best_action"].split("_")[0] for t in orig)),
        cat_x_scenario=C([dict(k=f"{t['sel_category']}|{t['scenario']}") for t in items], "k"),
        dependent=dict(confirm=sum(t["dependent"] for t in items), original=sum(t["dependent"] for t in orig)),
        repeat_right_persistent=dict(confirm=sum(t["repeat_right"] for t in items), original=sum(t["repeat_right"] for t in orig)),
        repeat_right_transient=dict(confirm=sum(t["repeat_right_transient"] for t in items), original=sum(t["repeat_right_transient"] for t in orig)),
        repeat_wrong=dict(confirm=sum(t["repeat_wrong"] for t in items), original=sum(t["repeat_wrong"] for t in orig)),
        p_benign_mean=dict(confirm=sum(t["p_benign"] for t in items) / len(items), original=sum(t["p_benign"] for t in orig) / len(orig)),
        gap_min=dict(confirm=min(t["gap"] for t in items), original=min(t["gap"] for t in orig)),
        gap_median=dict(confirm=sorted(t["gap"] for t in items)[len(items) // 2], original=sorted(t["gap"] for t in orig)[len(orig) // 2]),
        grid_best_category=dict(Counter(t["best_category"] for t in tasks)),
        closed_opt_check_max=max(abs(t["closed_opt_check"]) for t in tasks),
        naive_mean_regret_1step={m: round(sum(t["naive"][m]["regret_1step"] for t in items if m in t["naive"]) /
                                          max(1, sum(m in t["naive"] for t in items)), 3) for m in core.POLICIES},
        naive_mean_regret_1step_original={m: round(sum(t["naive"][m]["regret_1step"] for t in orig if m in t["naive"]) /
                                                   max(1, sum(m in t["naive"] for t in orig)), 3) for m in core.POLICIES})
    json.dump(rep, open(os.path.join(HERE, "items_strata_confirm.json"), "w"), indent=1, default=float)
    print(json.dumps(rep, indent=1, default=float))

if __name__ == "__main__":
    main()
