"""Direction of the models' repeat errors on the single-decision bench (offline, no model calls).

Question: do LLM errors on repeat decisions go in the direction the theory predicts for MMI (marginal-matched
independence)? MMI treats repeated reports of one persistent source as independent, so it OVER-values a repeat query
of a persistent source ("query_same_source", PR); the fault-unaware model ignores faults altogether.

  python bench/analyze_repeat_direction.py                       # default sets: main (140 items) + confirmation (134)
  python bench/analyze_repeat_direction.py runA.jsonl runB.jsonl [--items-file items_confirm.jsonl] [--label NAME]
  python bench/analyze_repeat_direction.py --set LABEL ITEMS_FILE RUN.jsonl [RUN.jsonl ...] [--set ...]   (repeatable)
  options: --conds P0 L1 L2   --out results/repeat_direction   --note results/repeat_direction_interpretation.md

Everything is read from existing answer files (records are selected / re-parsed exactly as in score.py) and from the
exact targets stored in the items files; MMI action values are recomputed with core.root_Q (checked against the stored
MMI action). Nothing is written except results/repeat_direction.{json,md}.

Definitions (per base task; PR = action category query_same_source = re-query of an already used persistent source;
TR = query_repeat_transient = re-call of the per-call-independent tool)
  an action is optimal if its exact one-step regret is <= 1e-6; unparsed answers are excluded from the rates below
  (parse rates are printed), but score.py's fallback (unparsed -> escalate) is kept for the regret columns.
  over-repeat  = P(choose a PR action | PR available and no PR action is optimal)
  under-repeat = P(choose a non-optimal non-PR action | some PR action is optimal)
  The same two rates are given for every category c in CATS (over-choice / under-choice of c).
  Reference policies fault_unaware and mmi choose exactly one action per task (stored in items[*]["naive"]); they are
  evaluated on the same tasks as each model/condition.
Statistics: answers are averaged within base task; CIs are 95% percentile bootstrap over base tasks (2000 draws, seed 0).
Permutation tests permute the grouping label within evidence-pattern strata (4000 permutations, seed 0).
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import core                                                  # noqa: E402
from run import parse_into, load_items                       # noqa: E402
from score import load_records, is_timeout, score_answer     # noqa: E402

EPS = 1e-6
PR, TR = "query_same_source", "query_repeat_transient"
REFS = ["mmi", "fault_unaware"]
N_BOOT, N_PERM = 2000, 4000
DEFAULT_SETS = [
    ("main", "items.jsonl", ["results/ds_chat_xfam.jsonl", "results/codex_terra_medium.jsonl"]),
    ("confirm", "items_confirm.jsonl", ["results/confirm_flash.jsonl", "results/confirm_terra.jsonl"]),
]

# ------------------------------------------------------------------------------------------ item geometry
def item_geometry(it):
    """Exact and MMI action values by category group."""
    cats = it["categories"]; Q = it["Q"]
    pr = [a for a in Q if cats[a] == PR]
    best = it["best_value"]
    g = dict(pr_avail=bool(pr),
             pr_opt=bool(pr) and min(Q[a] for a in pr) - best <= EPS,
             tr_avail=any(cats[a] == TR for a in Q),
             tr_opt=any(cats[a] == TR and Q[a] - best <= EPS for a in Q))
    st = it["structure"]; names = core.tool_names(st); srcs = core.sources(st)
    Mn = core.naive_models(st, it["phi"], it["q"], np.array(it["prior"]),
                           core.loss_matrix(core.K, core.ERR, it["sev"], core.ESC))["mmi"]
    b = core.belief(Mn, names, [tuple(e) for e in it["evidence"]])
    Qm = core.root_Q(Mn, b, names, srcs, B=core.BUDGET)
    g["mmi_check"] = core.argmin(Qm) == it["naive"]["mmi"]["action"]
    if pr:
        npr = [a for a in Q if cats[a] != PR]
        g["d_exact"] = min(Q[a] for a in pr) - min(Q[a] for a in npr)    # <0: a PR action is the best (exact)
        g["d_mmi"] = min(Qm[a] for a in pr) - min(Qm[a] for a in npr)    # <0: MMI thinks a PR action is the best
        g["misvalue"] = g["d_exact"] - g["d_mmi"]                        # >0: MMI values PR higher (relative) than the truth
    return g

# ------------------------------------------------------------------------------------------ statistics
def boot_idx(n, seed=0):
    return np.random.default_rng(seed).integers(0, n, size=(N_BOOT, n))

def ci(draws):
    return float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))

def mean_ci(x, seed=0):
    """x: task-level values (nan dropped) -> (mean, lo, hi, n)."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return (np.nan, np.nan, np.nan, 0)
    lo, hi = ci(x[boot_idx(len(x), seed)].mean(1))
    return (float(x.mean()), lo, hi, len(x))

def diff_ci(x, grp, seed=0):
    """mean(x | grp) - mean(x | ~grp), bootstrap over tasks (draws with an empty group are dropped)."""
    x = np.asarray(x, float); grp = np.asarray(grp, bool); ok = ~np.isnan(x); x, grp = x[ok], grp[ok]
    if grp.sum() == 0 or (~grp).sum() == 0: return (np.nan, np.nan, np.nan, int(grp.sum()), int((~grp).sum()))
    I = boot_idx(len(x), seed); xs, gs = x[I], grp[I]
    n1, n0 = gs.sum(1), (~gs).sum(1); m1 = (xs * gs).sum(1) / np.maximum(n1, 1); m0 = (xs * ~gs).sum(1) / np.maximum(n0, 1)
    d = np.where((n1 > 0) & (n0 > 0), m1 - m0, np.nan)
    lo, hi = ci(d)
    return (float(x[grp].mean() - x[~grp].mean()), lo, hi, int(grp.sum()), int((~grp).sum()))

def cluster_diff_ci(x, grp, clusters, seed=0, n_boot=2000):
    """mean(x | grp) - mean(x | ~grp); percentile CI from a cluster bootstrap (whole evidence patterns resampled)."""
    x = np.asarray(x, float); grp = np.asarray(grp, bool); ok = ~np.isnan(x); x, grp = x[ok], grp[ok]
    cl = np.asarray(clusters)[ok]; names = sorted(set(cl)); idx = [np.where(cl == c)[0] for c in names]
    rng = np.random.default_rng(seed); d = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx), len(idx)); ii = np.concatenate([idx[j] for j in pick])
        g = grp[ii]
        d.append(x[ii][g].mean() - x[ii][~g].mean() if g.any() and (~g).any() else np.nan)
    d = np.array(d)
    if np.isnan(d).all(): return (np.nan, np.nan)
    return float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))

def strat_perm_p(x, grp, strata, seed=0):
    """two-sided p of mean(x|grp)-mean(x|~grp), grp labels permuted within strata."""
    x = np.asarray(x, float); grp = np.asarray(grp, bool); ok = ~np.isnan(x); x, grp = x[ok], grp[ok]
    st = np.asarray(strata)[ok]
    if grp.sum() == 0 or (~grp).sum() == 0: return np.nan
    if np.ptp(x) == 0: return 1.0
    obs = x[grp].mean() - x[~grp].mean(); rng = np.random.default_rng(seed)
    groups = [np.where(st == s)[0] for s in sorted(set(st))]
    cnt = 0
    for _ in range(N_PERM):
        g2 = grp.copy()
        for ix in groups: g2[ix] = grp[ix][rng.permutation(len(ix))]
        d = x[g2].mean() - x[~g2].mean(); cnt += abs(d) >= abs(obs) - 1e-12
    return float((cnt + 1) / (N_PERM + 1))

def rank(x):
    x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x))
    for v in np.unique(x):                           # average ranks for ties
        m = x == v; r[m] = r[m].mean()
    return r

def spearman(x, y):
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

def partial_spearman(x, y, z):
    rx, ry, rz = rank(x), rank(y), rank(z)
    if rz.std() > 0:
        A = np.c_[np.ones(len(rz)), rz]
        rx = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]; ry = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    if rx.std() < 1e-12 or ry.std() < 1e-12: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

def corr_ci(fn, cols, seed=0):
    cols = [np.asarray(c, float) for c in cols]; ok = np.all([~np.isnan(c) for c in cols], axis=0)
    cols = [c[ok] for c in cols]
    if len(cols[0]) < 5: return (np.nan, np.nan, np.nan, len(cols[0]))
    I = boot_idx(len(cols[0]), seed)
    d = np.array([fn(*[c[i] for c in cols]) for i in I[:1000]])
    lo, hi = ci(d)
    return (fn(*cols), lo, hi, len(cols[0]))

def f2(t, pct=False, signed=False):
    m, lo, hi = t[0], t[1], t[2]
    if m is None or np.isnan(m): return "-"
    k = 100 if pct else 1; d = 0 if pct else 2
    s = "+" if signed else ""
    return f"{m * k:{s}.{d}f} [{lo * k:{s}.{d}f}, {hi * k:{s}.{d}f}]"

# ------------------------------------------------------------------------------------------ loading answers
def load_answers(files, items):
    """-> {run: {cond: {item: [answer dict]}}} using score.py's record selection and parsing."""
    prim, _ = load_records(files)
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for k, r in prim.items():
        it = items[r["item"]]
        if r.get("ok"): parse_into(r, it, r["cond"])
        else: r.update(parse_status="timeout" if is_timeout(r) else "api_error", action=None, p_benign=None)
        d = score_answer(it, r)
        tag = r.get("tag") or ""
        run = f"{r['model']}|cap{r['max_tokens']}" + (f"|{tag}" if tag else "")
        out[run][r["cond"]][r["item"]].append(dict(parsed=d["parsed"], regret=d["regret"], regret_raw=it["Q"][r["action"]] - it["best_value"] if d["parsed"] else np.nan,
                                                   action=r.get("action") if d["parsed"] else None))
    return out

def answers_to_tasks(ans, items, ids):
    """per-task vectors (nan where nothing parsed) for one run x condition."""
    T = {m: np.full(len(ids), np.nan) for m in ["parsed", "regret", "regret_parsed"]}
    for c in core.CATS:
        T["chose:" + c] = np.full(len(ids), np.nan); T["wrong:" + c] = np.full(len(ids), np.nan); T["under:" + c] = np.full(len(ids), np.nan)
    T["optimal"] = np.full(len(ids), np.nan); T["has"] = np.zeros(len(ids), bool)
    for n, i in enumerate(ids):
        A = ans.get(i)
        if not A: continue
        it = items[i]; T["has"][n] = True
        T["parsed"][n] = np.mean([a["parsed"] for a in A]); T["regret"][n] = np.mean([a["regret"] for a in A])
        P = [a for a in A if a["parsed"]]
        if not P: continue
        T["regret_parsed"][n] = np.mean([a["regret_raw"] for a in P])
        T["optimal"][n] = np.mean([a["regret_raw"] <= EPS for a in P])
        for c in core.CATS:
            T["chose:" + c][n] = np.mean([it["categories"][a["action"]] == c for a in P])
            T["wrong:" + c][n] = np.mean([it["categories"][a["action"]] == c and a["regret_raw"] > EPS for a in P])
            T["under:" + c][n] = np.mean([it["categories"][a["action"]] != c and a["regret_raw"] > EPS for a in P])
    return T

def ref_to_tasks(pol, items, ids, keep):
    """reference policy as a one-answer-per-task 'run' restricted to tasks in keep."""
    T = {m: np.full(len(ids), np.nan) for m in ["parsed", "regret", "regret_parsed"]}
    for c in core.CATS:
        T["chose:" + c] = np.full(len(ids), np.nan); T["wrong:" + c] = np.full(len(ids), np.nan); T["under:" + c] = np.full(len(ids), np.nan)
    T["optimal"] = np.full(len(ids), np.nan); T["has"] = np.asarray(keep, bool).copy()
    for n, i in enumerate(ids):
        if not keep[n]: continue
        nv = items[i]["naive"][pol]; reg = nv["regret_1step"]
        T["parsed"][n] = 1.0; T["regret"][n] = T["regret_parsed"][n] = reg; T["optimal"][n] = float(reg <= EPS)
        for c in core.CATS:
            T["chose:" + c][n] = float(nv["category"] == c); T["wrong:" + c][n] = float(nv["category"] == c and reg > EPS)
            T["under:" + c][n] = float(nv["category"] != c and reg > EPS)
    return T

# ------------------------------------------------------------------------------------------ analyses
def under_vec(T, G, cat):
    """per-task under-choice indicator (task mean over answers): category cat has an optimal action, the answer is
    not optimal and not in cat."""
    opt = np.array([g["pr_opt"] if cat == PR else g["tr_opt"] if cat == TR else g["cat_opt"][cat] for g in G])
    return np.where(opt, T["under:" + cat], np.nan)

def cat_geometry(it):
    cats = it["categories"]; Q = it["Q"]; best = it["best_value"]
    return ({c: any(cats[a] == c for a in Q) for c in core.CATS},
            {c: any(cats[a] == c and Q[a] - best <= EPS for a in Q) for c in core.CATS})

def analyse_set(label, items, runs_ans, conds):
    ids = sorted(items)
    G = []
    for i in ids:
        g = item_geometry(items[i]); g["cat_avail"], g["cat_opt"] = cat_geometry(items[i]); G.append(g)
    pattern = np.array([items[i]["pattern"] for i in ids]); persistent = np.array([items[i]["persistent"] for i in ids])
    res = dict(label=label, n_items=len(ids), mmi_action_check=bool(all(g["mmi_check"] for g in G)),
               strata=dict(pr_avail=int(sum(g["pr_avail"] for g in G)), pr_optimal=int(sum(g["pr_opt"] for g in G)),
                           pr_avail_not_optimal=int(sum(g["pr_avail"] and not g["pr_opt"] for g in G)),
                           tr_avail=int(sum(g["tr_avail"] for g in G)), tr_optimal=int(sum(g["tr_opt"] for g in G)),
                           best_category={c: int(sum(items[i]["best_category"] == c for i in ids)) for c in core.CATS}),
               rows={}, tests={}, confusion={}, misvalue={})
    mis = np.array([g.get("misvalue", np.nan) for g in G]); d_ex = np.array([g.get("d_exact", np.nan) for g in G])
    d_mmi = np.array([g.get("d_mmi", np.nan) for g in G])
    avail_not_opt = np.array([g["pr_avail"] and not g["pr_opt"] for g in G])
    # --- MMI over-valuation of PR (model-side check of the premise)
    res["misvalue"] = dict(
        n_pr_avail=int(np.sum([g["pr_avail"] for g in G])),
        mean_misvalue=mean_ci(mis[[g["pr_avail"] for g in G]]),
        share_mmi_values_pr_higher=mean_ci((mis[[g["pr_avail"] for g in G]] > 1e-9).astype(float)),
        mean_misvalue_pr_not_optimal=mean_ci(mis[avail_not_opt]),
        mmi_prefers_pr=mean_ci(np.array([float(items[i]["naive"]["mmi"]["category"] == PR) for i in ids])[[g["pr_avail"] for g in G]]),
        fu_prefers_pr=mean_ci(np.array([float(items[i]["naive"]["fault_unaware"]["category"] == PR) for i in ids])[[g["pr_avail"] for g in G]]),
        exact_prefers_pr=mean_ci(np.array([float(g["pr_opt"]) for g in G])[[g["pr_avail"] for g in G]]))
    # MMI-over-repeat flag per task
    mmi_pr = np.array([items[i]["naive"]["mmi"]["category"] == PR for i in ids])
    fu_pr = np.array([items[i]["naive"]["fault_unaware"]["category"] == PR for i in ids])
    mmi_reg = np.array([items[i]["naive"]["mmi"]["regret_1step"] for i in ids])
    fu_reg = np.array([items[i]["naive"]["fault_unaware"]["regret_1step"] for i in ids])
    mmi_over = avail_not_opt & mmi_pr; fu_over = avail_not_opt & fu_pr
    mmi_wrong = mmi_reg > EPS; fu_wrong = fu_reg > EPS
    res["strata"].update(mmi_over_repeat=int(mmi_over.sum()), fu_over_repeat=int(fu_over.sum()),
                         mmi_wrong=int(mmi_wrong.sum()), fu_wrong=int(fu_wrong.sum()))
    for run in sorted(runs_ans):
        for cond in conds:
            if cond not in runs_ans[run]: continue
            T = answers_to_tasks(runs_ans[run][cond], items, ids)
            keep = T["has"] & ~np.isnan(T["optimal"])
            if keep.sum() == 0: continue
            key = f"{run}|{cond}"
            rows = {"LLM": T};
            for p in REFS: rows[p] = ref_to_tasks(p, items, ids, keep)
            out = dict(n_tasks=int(T["has"].sum()), parse_rate=float(np.nanmean(T["parsed"][T["has"]])))
            for who, X in rows.items():
                kk = X["has"] & ~np.isnan(X["optimal"])
                r = dict(optimal=mean_ci(np.where(kk, X["optimal"], np.nan)), regret=mean_ci(np.where(kk, X["regret"], np.nan)),
                         chose_pr_all=mean_ci(np.where(kk, X["chose:" + PR], np.nan)),
                         chose_tr_all=mean_ci(np.where(kk, X["chose:" + TR], np.nan)))
                for cat in core.CATS:
                    ov = np.where(kk & np.array([g["pr_avail"] if cat == PR else g["tr_avail"] if cat == TR else g["cat_avail"][cat] for g in G])
                                  & ~np.array([g["pr_opt"] if cat == PR else g["tr_opt"] if cat == TR else g["cat_opt"][cat] for g in G]),
                                  X["wrong:" + cat], np.nan)
                    un = np.where(kk, under_vec(X, G, cat), np.nan)
                    r["over:" + cat] = mean_ci(ov); r["under:" + cat] = mean_ci(un)
                # signed bias of the share of PR / TR choices relative to the exact share
                for cat, ex in [(PR, np.array([float(g["pr_opt"]) for g in G])), (TR, np.array([float(g["tr_opt"]) for g in G]))]:
                    r["bias:" + cat] = mean_ci(np.where(kk, X["chose:" + cat] - ex, np.nan))
                out[who] = r
            # --- error-type mix and agreement with the reference policies' errors
            def chosen_cat(X, n):    # modal chosen category of a task (for the policies: the single one)
                return None
            conf = {}
            kk = T["has"] & ~np.isnan(T["optimal"])
            for who, wrong_mask in [("mmi", mmi_wrong), ("fault_unaware", fu_wrong)]:
                m = kk & wrong_mask
                # P(LLM chooses the same category as the policy | policy wrong), task-mean over answers
                same = np.array([np.nan if not m[n] else T["chose:" + items[i]["naive"][who]["category"]][n] for n, i in enumerate(ids)])
                conf["agree_with_" + who + "_when_it_errs"] = mean_ci(same)
                # baseline: P(LLM chooses that category) on the tasks where the policy is right
                r_ = kk & ~wrong_mask
                same0 = np.array([np.nan if not r_[n] else T["chose:" + items[i]["naive"][who]["category"]][n] for n, i in enumerate(ids)])
                conf["agree_with_" + who + "_when_it_is_right"] = mean_ci(same0)
                # accuracy-free version: among the LLM's WRONG answers on tasks where the policy is wrong, the share in the policy's category,
                # against the chance share if the LLM picked uniformly among the available non-optimal categories
                wn = np.where(m, 1 - T["optimal"], 0.0); wn = np.nan_to_num(wn)
                hit = np.nan_to_num(np.array([0.0 if not m[n] else T["wrong:" + items[i]["naive"][who]["category"]][n] for n, i in enumerate(ids)]))
                alt = np.array([max(1, sum(g["cat_avail"][c] and not g["cat_opt"][c] for c in core.CATS)) for g in G], float)
                conf["agree_given_both_wrong_" + who] = dict(
                    share=float(hit.sum() / wn.sum()) if wn.sum() > 0 else np.nan,
                    chance=float((wn / alt).sum() / wn.sum()) if wn.sum() > 0 else np.nan, wrong_mass=float(wn.sum()))
            # error mix: among the LLM's wrong answers, share by chosen category (answer level pooled over tasks)
            mix = {}
            tot = np.nansum(np.where(kk, 1 - T["optimal"], 0))
            for c in core.CATS: mix[c] = float(np.nansum(np.where(kk, T["wrong:" + c], 0)) / tot) if tot > 0 else np.nan
            conf["llm_error_mix"] = mix
            for who in REFS:
                m_ = {}; X = rows[who]; kw = kk & (X["optimal"] == 0); tw = kw.sum()
                for c in core.CATS: m_[c] = float(np.nansum(np.where(kw, X["wrong:" + c], 0)) / tw) if tw else np.nan
                conf[who + "_error_mix"] = m_
            out["agreement"] = conf
            # --- matched over-repeat test: among tasks where PR is available and not optimal, does the LLM over-repeat
            #     more where MMI (FU) over-repeats?  (bootstrap diff + permutation stratified by pattern)
            tests = {}
            for who, flag in [("mmi", mmi_over), ("fault_unaware", fu_over)]:
                el = kk & avail_not_opt
                x = np.where(el, T["wrong:" + PR], np.nan)
                d = diff_ci(x[el], flag[el], seed=1)
                p = strat_perm_p(x[el], flag[el], pattern[el], seed=1)
                tests["over_repeat_llm|" + who + "_over_repeats_minus_not"] = dict(diff=d, perm_p=p, cluster_ci=cluster_diff_ci(x[el], flag[el], pattern[el], seed=1))
            # --- LLM regret where MMI is wrong vs right (all tasks, and PR-error subset), + placebo by FU
            for who, wmask in [("mmi", mmi_wrong), ("fault_unaware", fu_wrong)]:
                x = np.where(kk, T["regret"], np.nan)
                d = diff_ci(x[kk], wmask[kk], seed=2); p = strat_perm_p(x[kk], wmask[kk], pattern[kk], seed=2)
                tests["llm_regret|" + who + "_wrong_minus_right"] = dict(diff=d, perm_p=p, cluster_ci=cluster_diff_ci(x[kk], wmask[kk], pattern[kk], seed=2))
            # MMI wrong by over-repeat vs MMI right: regret of the LLM
            sub = kk & (mmi_over | ~mmi_wrong)
            x = np.where(sub, T["regret"], np.nan); d = diff_ci(x[sub], mmi_over[sub], seed=3)
            tests["llm_regret|mmi_over_repeat_minus_right"] = dict(diff=d, perm_p=strat_perm_p(x[sub], mmi_over[sub], pattern[sub], seed=3), cluster_ci=cluster_diff_ci(x[sub], mmi_over[sub], pattern[sub], seed=3))
            # MMI wrong in other ways (not an over-repeat) vs right
            oth = mmi_wrong & ~mmi_over
            sub = kk & (oth | ~mmi_wrong)
            x = np.where(sub, T["regret"], np.nan); d = diff_ci(x[sub], oth[sub], seed=4)
            tests["llm_regret|mmi_wrong_other_minus_right"] = dict(diff=d, perm_p=strat_perm_p(x[sub], oth[sub], pattern[sub], seed=4), cluster_ci=cluster_diff_ci(x[sub], oth[sub], pattern[sub], seed=4))
            # regret correlation across tasks (persistent structures only to avoid the transient control dominating)
            pm = kk & persistent
            tests["spearman_llm_regret_vs_mmi_regret_persistent"] = corr_ci(spearman, [T["regret"][pm], mmi_reg[pm]])
            tests["spearman_llm_regret_vs_fu_regret_persistent"] = corr_ci(spearman, [T["regret"][pm], fu_reg[pm]])
            # --- dose-response: LLM PR-choice vs MMI's relative over-valuation of PR, where PR is NOT optimal
            el = kk & avail_not_opt
            tests["n_pr_error_tasks"] = int(np.nansum(T["wrong:" + PR][el] > 0))
            tests["dose_response_spearman(PR_wrong, misvalue)"] = corr_ci(spearman, [T["wrong:" + PR][el], mis[el]])
            tests["dose_response_partial_spearman(PR_wrong, misvalue | d_exact)"] = corr_ci(partial_spearman, [T["wrong:" + PR][el], mis[el], d_ex[el]])
            tests["dose_response_spearman(PR_wrong, d_exact)"] = corr_ci(spearman, [T["wrong:" + PR][el], d_ex[el]])
            out["tests"] = tests
            bp = {}
            for pt in sorted(set(pattern[avail_not_opt])):
                m = avail_not_opt & (pattern == pt)
                bp[pt] = dict(n=int(m.sum()), llm_over_pr=float(np.nanmean(np.where(kk & m, T["wrong:" + PR], np.nan))) if (kk & m).any() else np.nan,
                              mmi_over_pr=float(mmi_over[m].mean()), fu_over_pr=float(fu_over[m].mean()))
            out["by_pattern_pr_over"] = bp
            pl = [bp[p]["llm_over_pr"] for p in bp]
            out["pattern_corr"] = dict(
                n_patterns=len(bp),
                pearson_with_mmi=float(np.corrcoef(pl, [bp[p]["mmi_over_pr"] for p in bp])[0, 1]) if np.std(pl) > 0 else np.nan,
                pearson_with_fu=float(np.corrcoef(pl, [bp[p]["fu_over_pr"] for p in bp])[0, 1]) if np.std(pl) > 0 else np.nan)
            res["rows"][key] = out
    return res

# ------------------------------------------------------------------------------------------ report
def f2g(t):
    return "n/a" if t["share"] != t["share"] else f"{t['share'] * 100:.0f}% ({t['chance'] * 100:.0f}%; {t['wrong_mass']:.1f} wrong ans.)"

def short(run):
    p = run.split("|")
    return (p[0].replace("codex:gpt-6.1-terra@medium", "terra").replace("deepseek-chat", "flash") + "".join("/" + x for x in p[2:]))

def render(sets_res):
    md = ["# Direction of repeat errors (MMI-direction hypothesis)\n",
          "Generated by `bench/analyze_repeat_direction.py` from existing answer files; no model calls. PR = re-query of an already used "
          "persistent source (`query_same_source`), TR = re-call of the per-call-independent tool. Rates are task-level means with 95% bootstrap "
          "CIs over base tasks; reference policies (`mmi`, `fault_unaware`) are evaluated on the same tasks as the model/condition in each block. "
          "Definitions are in the script docstring.\n"]
    for res in sets_res:
        s = res["strata"]
        md += [f"\n## Task set `{res['label']}` ({res['n_items']} base tasks)\n",
               f"Exact strata: PR available on {s['pr_avail']} tasks, PR optimal on {s['pr_optimal']}, PR available but not optimal on "
               f"{s['pr_avail_not_optimal']}; TR available on {s['tr_avail']}, optimal on {s['tr_optimal']}. MMI over-repeats (chooses PR when it is not optimal) "
               f"on {s['mmi_over_repeat']} tasks, fault-unaware on {s['fu_over_repeat']}; MMI's first action is wrong on {s['mmi_wrong']}, fault-unaware's on {s['fu_wrong']}. "
               f"Stored MMI action reproduced from the recomputed MMI values: {res['mmi_action_check']}.\n",
               "Premise check (is MMI's valuation of a PR action too high?). misvalue = [Q_exact(best PR) - Q_exact(best non-PR)] - "
               "[Q_MMI(best PR) - Q_MMI(best non-PR)]; > 0 means MMI ranks PR relatively better than the truth does.\n",
               "| Quantity (tasks with PR available) | Value [95% CI] |", "| --- | --- |"]
        mv = res["misvalue"]
        md += [f"| mean misvalue (all) | {f2(mv['mean_misvalue'], signed=True)} (n={mv['mean_misvalue'][3]}) |",
               f"| mean misvalue, PR not optimal | {f2(mv['mean_misvalue_pr_not_optimal'], signed=True)} (n={mv['mean_misvalue_pr_not_optimal'][3]}) |",
               f"| share with misvalue > 0 | {f2(mv['share_mmi_values_pr_higher'], True)}% |",
               f"| exact picks PR / MMI picks PR / fault-unaware picks PR | {mv['exact_prefers_pr'][0]*100:.0f}% / {mv['mmi_prefers_pr'][0]*100:.0f}% / {mv['fu_prefers_pr'][0]*100:.0f}% |"]
        keys = list(res["rows"])
        md += ["\n### Over- and under-repeat of persistent sources (PR)\n",
               "over = P(choose PR | PR available, not optimal); under = P(non-optimal non-PR answer | PR optimal); bias = P(choose PR) - P(PR optimal) over all tasks. "
               "Few tasks have PR optimal (see strata), so the under-repeat CIs are wide.\n",
               "| Run | Cond | n | parsed | Who | Over-repeat % | Under-repeat % | PR bias (pts) | One-step regret | Optimal % |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for k in keys:
            r = res["rows"][k]; run, cond = k.rsplit("|", 1)
            for who in ["LLM", "mmi", "fault_unaware"]:
                x = r[who]
                md.append(f"| {short(run) if who == 'LLM' else ''} | {cond if who == 'LLM' else ''} | {r['n_tasks'] if who == 'LLM' else ''} | "
                          f"{r['parse_rate'] * 100:.0f}% | {'**LLM**' if who == 'LLM' else who} | {f2(x['over:' + PR], True)} | {f2(x['under:' + PR], True)} | "
                          f"{f2(x['bias:' + PR], True, True)} | {f2(x['regret'])} | {f2(x['optimal'], True)} |")
        md += ["\n### Control: repeats of the per-call-independent (transient) tool (TR), where repeating is genuinely informative\n",
               "| Run | Cond | Who | Over-TR % | Under-TR % | TR bias (pts) |", "| --- | --- | --- | --- | --- | --- |"]
        for k in keys:
            r = res["rows"][k]; run, cond = k.rsplit("|", 1)
            for who in ["LLM", "mmi", "fault_unaware"]:
                x = r[who]
                md.append(f"| {short(run) if who == 'LLM' else ''} | {cond if who == 'LLM' else ''} | {'**LLM**' if who == 'LLM' else who} | "
                          f"{f2(x['over:' + TR], True)} | {f2(x['under:' + TR], True)} | {f2(x['bias:' + TR], True, True)} |")
        md += ["\n### Other error directions (all categories; over = chosen when available but not optimal, under = optimal but not chosen)\n",
               "| Run | Cond | Who | " + " | ".join(f"{c} over / under" for c in core.CATS if c not in (PR, TR)) + " |",
               "| --- | --- | --- |" + " --- |" * len([c for c in core.CATS if c not in (PR, TR)])]
        for k in keys:
            r = res["rows"][k]; run, cond = k.rsplit("|", 1)
            for who in ["LLM", "mmi", "fault_unaware"]:
                x = r[who]
                md.append(f"| {short(run) if who == 'LLM' else ''} | {cond if who == 'LLM' else ''} | {'**LLM**' if who == 'LLM' else who} | " +
                          " | ".join(f"{x['over:' + c][0] * 100:.0f} / {x['under:' + c][0] * 100:.0f}" for c in core.CATS if c not in (PR, TR)) + " |")
        md += ["\n### Do the LLM's errors coincide with the reference policies' errors?\n",
               "Agreement = share of the LLM's answers falling in the same action category as the policy's, on tasks where the policy is wrong (vs right). "
               "The first agreement columns are dominated by the LLM's accuracy (a model that is mostly right cannot agree with a wrong policy); the 'share of the LLM's wrong answers' columns are accuracy-free: "
               "among the LLM's wrong answers on tasks where the policy is wrong, the fraction in the policy's wrong category, with the share expected if the LLM picked uniformly among the available non-optimal categories in brackets. "
               "Error mix = distribution of the chosen category among wrong answers.\n",
               "| Run | Cond | agree w/ MMI when MMI wrong | when MMI right | agree w/ FU when FU wrong | when FU right | MMI cat. share of LLM's wrong answers, MMI wrong (chance) | same for FU | LLM error mix | MMI error mix | FU error mix |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        ab = {"declare_benign": "dB", "declare_nonbenign": "dN", "escalate": "esc", "query_same_source": "PR", "query_repeat_transient": "TR", "query_new_source": "new"}
        mixs = lambda m: " ".join(f"{ab[c]} {m[c] * 100:.0f}" for c in core.CATS if m[c] == m[c] and m[c] > 0.005)
        for k in keys:
            r = res["rows"][k]; run, cond = k.rsplit("|", 1); a = r["agreement"]
            md.append(f"| {short(run)} | {cond} | {f2(a['agree_with_mmi_when_it_errs'], True)}% | {f2(a['agree_with_mmi_when_it_is_right'], True)}% | "
                      f"{f2(a['agree_with_fault_unaware_when_it_errs'], True)}% | {f2(a['agree_with_fault_unaware_when_it_is_right'], True)}% | "
                      f"{f2g(a['agree_given_both_wrong_mmi'])} | {f2g(a['agree_given_both_wrong_fault_unaware'])} | {mixs(a['llm_error_mix'])} | {mixs(a['mmi_error_mix'])} | {mixs(a['fault_unaware_error_mix'])} |")
        md += ["\n### Is the LLM worse exactly where MMI is wrong?\n",
               "Difference in the LLM's mean one-step regret (or over-repeat rate) between task groups, bootstrap CI over tasks and a permutation p-value with the group label permuted within evidence-pattern strata "
               "(controls for pattern-level difficulty). 'fault_unaware wrong' is a placebo grouping. The task-level bootstrap CI treats tasks as independent although they cluster in "
               "evidence patterns and is therefore anti-conservative; the pattern-cluster bootstrap CI (whole patterns resampled) and the stratified permutation p are the authoritative ones.\n",
               "| Run | Cond | Contrast | n(group) / n(rest) | Difference [task-bootstrap CI] | Pattern-cluster bootstrap CI | perm. p |", "| --- | --- | --- | --- | --- | --- | --- |"]
        lab = {"llm_regret|mmi_wrong_minus_right": "LLM regret: MMI wrong - MMI right (all tasks)",
               "llm_regret|fault_unaware_wrong_minus_right": "LLM regret: FU wrong - FU right (placebo)",
               "llm_regret|mmi_over_repeat_minus_right": "LLM regret: MMI over-repeats - MMI right",
               "llm_regret|mmi_wrong_other_minus_right": "LLM regret: MMI wrong, not over-repeat - MMI right",
               "over_repeat_llm|mmi_over_repeats_minus_not": "LLM over-repeat rate (PR avail., not optimal): MMI over-repeats - MMI does not",
               "over_repeat_llm|fault_unaware_over_repeats_minus_not": "LLM over-repeat rate (PR avail., not optimal): FU over-repeats - FU does not"}
        for k in keys:
            r = res["rows"][k]; run, cond = k.rsplit("|", 1)
            for t, nm in lab.items():
                v = r["tests"][t]; d = v["diff"]
                md.append(f"| {short(run)} | {cond} | {nm} | {d[3]} / {d[4]} | {f2(d, signed=True)} | [{v['cluster_ci'][0]:+.2f}, {v['cluster_ci'][1]:+.2f}] | {v['perm_p']:.3f} |")
        md += ["\n### Dose-response and rank agreement\n",
               "Among tasks where PR is available but not optimal: Spearman correlation across tasks between the LLM's rate of non-optimal PR choices and (a) MMI's relative over-valuation of PR "
               "(misvalue), also partialling out the exact PR advantage d_exact = Q_exact(best PR) - Q_exact(best non-PR) (so that tasks where repeating is merely close to optimal are not confounded), "
               "(b) d_exact alone. Positive (a) supports the MMI direction. The last two columns: Spearman of the LLM's one-step regret with MMI's / fault-unaware's regret over persistent-structure tasks.\n",
               "| Run | Cond | tasks with a PR error | rho(PR error, misvalue) | partial rho given d_exact | rho(PR error, d_exact) | rho(regret, MMI regret) | rho(regret, FU regret) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for k in keys:
            r = res["rows"][k]; run, cond = k.rsplit("|", 1); t = r["tests"]
            md.append(f"| {short(run)} | {cond} | {t['n_pr_error_tasks']} | {f2(t['dose_response_spearman(PR_wrong, misvalue)'], signed=True)} (n={t['dose_response_spearman(PR_wrong, misvalue)'][3]}) | "
                      f"{f2(t['dose_response_partial_spearman(PR_wrong, misvalue | d_exact)'], signed=True)} | {f2(t['dose_response_spearman(PR_wrong, d_exact)'], signed=True)} | "
                      f"{f2(t['spearman_llm_regret_vs_mmi_regret_persistent'], signed=True)} | {f2(t['spearman_llm_regret_vs_fu_regret_persistent'], signed=True)} |")
    for res in sets_res:
        c0 = None
        for k in res["rows"]:
            run, cond = k.rsplit("|", 1)
            if c0 is None: c0 = cond
        if c0 is None: continue
        ks = [k for k in res["rows"] if k.endswith("|" + c0)]
        md += [f"\n### Where the over-repeats happen (`{res['label']}`, condition {c0}): rate of non-optimal PR choices by evidence pattern, tasks with PR available and not optimal\n",
               "| Pattern | n tasks | MMI | fault_unaware | " + " | ".join(short(k.rsplit("|", 1)[0]) for k in ks) + " |",
               "| --- | --- | --- | --- | " + " --- |" * len(ks)]
        pats = sorted(res["rows"][ks[0]]["by_pattern_pr_over"])
        for pt in pats:
            b0 = res["rows"][ks[0]]["by_pattern_pr_over"][pt]
            md.append(f"| {pt} | {b0['n']} | {b0['mmi_over_pr'] * 100:.0f}% | {b0['fu_over_pr'] * 100:.0f}% | " +
                      " | ".join(f"{res['rows'][k]['by_pattern_pr_over'][pt]['llm_over_pr'] * 100:.0f}%" for k in ks) + " |")
        pc = lambda v: "-" if v != v else f"{v:+.2f}"
        md.append("\nPearson correlation across the evidence patterns (descriptive only, no CI; " + str(len(pats)) + " patterns) between the model's per-pattern over-repeat rate and MMI's / fault-unaware's: " +
                  "; ".join(f"{short(k.rsplit('|', 1)[0])}: MMI {pc(res['rows'][k]['pattern_corr']['pearson_with_mmi'])}, FU {pc(res['rows'][k]['pattern_corr']['pearson_with_fu'])}" for k in ks) + ".")
    return md

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="run files (one set, scored against --items-file)")
    ap.add_argument("--items-file", default=None); ap.add_argument("--label", default="custom")
    ap.add_argument("--set", nargs="+", action="append", default=[], metavar="LABEL ITEMS RUN", help="LABEL ITEMS_FILE RUN [RUN ...]")
    ap.add_argument("--conds", nargs="+", default=["P0", "L1", "L2"])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "repeat_direction"))
    ap.add_argument("--note", default=os.path.join(HERE, "results", "repeat_direction_interpretation.md"),
                    help="optional hand-written interpretation appended to the report (not touched by this script)")
    a = ap.parse_args()
    sets = []
    for s in a.set:
        if len(s) < 3: ap.error("--set needs LABEL ITEMS_FILE RUN [RUN ...]")
        sets.append((s[0], s[1], s[2:]))
    if a.files: sets.append((a.label, a.items_file, a.files))
    if not sets: sets = DEFAULT_SETS
    out_sets = []
    for label, itf, files in sets:
        items = load_items(itf)
        files = [f if os.path.isabs(f) or os.path.exists(f) else os.path.join(HERE, f) for f in files]
        ans = load_answers(files, items)
        out_sets.append(analyse_set(label, items, ans, a.conds))
    md = render(out_sets)
    if os.path.exists(a.note):
        md += ["\n" + open(a.note, encoding="utf-8").read()]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out + ".md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump(out_sets, open(a.out + ".json", "w"), indent=1, default=float)
    print("\n".join(md))

if __name__ == "__main__":
    main()
