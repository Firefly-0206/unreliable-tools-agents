"""
First decisions matched on decision difficulty: is the first action more often suboptimal when the best first query is
the fault-capable replica tool than when it is another tool, at the same exact value margin? No model call. Post hoc
(added in response to a review; not part of any plan fixed before the runs; the margin bins were fixed before looking at
the outcome and are the only ones tried).

Episodes: the fault-rate sweep (episodes_x.jsonl, 200) and the accuracy sweep in which the best first query flips
(episodes_w.jsonl, 175), K2e, sample 0, three runs. For each episode, under the TRUE model at the root:
  best      the oracle's first action;  margin = Q(second-best first action) - Q(best), the exact cost of the cheapest
            first-step mistake (small margin = hard decision)
  group R   best is the replica tool (the only fault-capable tool);  group N  best is another tool
  subopt    the model's first action is not the oracle's;  loss = Q(model's first action) - Q(best)
Episodes whose best first action is to stop are left out (none of them is a tool choice).

Matching: margin bins EDGES; within every bin that holds both groups, the R-minus-N difference of the suboptimal share
and of the first-step loss; bins are averaged with weights min(n_R, n_N) (overlap weights). Intervals: nested bootstrap
(scenarios, then parameter draws), the whole matched estimate recomputed in each draw; 2000 draws, seed 0.

Reading: a difference near 0 is what a difficulty-only account predicts; a positive difference says that, at equal
margin, mistakes concentrate where the fault-capable tool is the right first query. Neither reading identifies an
internal mechanism, and the margin is one measure of difficulty among several (it ignores planning depth).

Output: results/difficulty_matched.json, results/difficulty_matched.txt
"""
import json, os
import numpy as np
from env import HERE, load_episodes, build_model
from oracle import qvalues
import x_analysis as X
from y_analysis import load as load_s

P = lambda *a: os.path.join(HERE, *a)
RUNS = {"flash": "ds_chat_fc", "terra": "codex_terra", "sonnet": "claude_s55"}
EDGES = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1e9]
REP = X.REP


def rows_for(eps, recs):
    R = []
    for i, ep in eps.items():
        M = build_model(ep["params"]); q = qvalues(M, M.prior.copy(), float(M.budget))
        stop = min(v for k, v in q.items() if k[0] == "stop"); acts = {k: v for k, v in q.items() if k[0] == "probe"}; acts[("stop",)] = stop
        order = sorted(acts.items(), key=lambda kv: kv[1]); best, qb = order[0]; margin = order[1][1] - qb
        if best[0] == "stop": continue
        calls = [x["j"] for x in recs[i]["log"] if x["type"] == "call"]; a = ("probe", calls[0]) if calls else ("stop",)
        R.append(dict(id=i, sc=ep["scenario"], draw=ep["pair"], grp="R" if best[1] == REP else "N", margin=float(margin),
                      subopt=float(acts[a] - qb > 1e-9), loss=float(acts[a] - qb), chose_rep=float(a == ("probe", REP)), bin=int(np.searchsorted(EDGES, margin, side="right") - 1)))
    return R


def matched(rows):
    num = {"subopt": 0.0, "loss": 0.0}; lev = {g + k: 0.0 for g in "RN" for k in num}; W = 0.0
    for b in range(len(EDGES) - 1):
        r = [d for d in rows if d["bin"] == b and d["grp"] == "R"]; n = [d for d in rows if d["bin"] == b and d["grp"] == "N"]
        w = min(len(r), len(n))
        if not w: continue
        W += w
        for k in num:
            mr, mn = np.mean([d[k] for d in r]), np.mean([d[k] for d in n]); num[k] += w * (mr - mn); lev["R" + k] += w * mr; lev["N" + k] += w * mn
    if not W: return None
    return dict({k: v / W for k, v in num.items()}, **{k: v / W for k, v in lev.items()}, weight=W)


def boot(rows, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); by = {}
    for d in rows: by.setdefault(d["sc"], {}).setdefault(d["draw"], []).append(d)
    scs = sorted(by); est = {"subopt": [], "loss": []}
    for _ in range(n_boot):
        samp = []
        for s in rng.integers(0, len(scs), len(scs)):
            dr = sorted(by[scs[s]])
            for k in rng.integers(0, len(dr), len(dr)): samp += by[scs[s]][dr[k]]
        m = matched(samp)
        if m:
            for k in est: est[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in est.items()}


def main():
    EX, EW = load_episodes(P("episodes_x.jsonl")), load_episodes(P("episodes_w.jsonl")); out, L = {}, []
    for run, stem in RUNS.items():
        rx = X.load(P("logs", f"x_{stem}_k2e.jsonl")); rw = {i: r for (i, s), r in load_s(P("logs", f"w_{stem}_k2e.jsonl"), [0]).items()}
        rows = rows_for(EX, rx) + rows_for(EW, rw)
        m = matched(rows); ci = boot(rows)
        bins = []
        for b in range(len(EDGES) - 1):
            r = [d for d in rows if d["bin"] == b and d["grp"] == "R"]; n = [d for d in rows if d["bin"] == b and d["grp"] == "N"]
            bins.append(dict(lo=EDGES[b], hi=EDGES[b + 1], nR=len(r), nN=len(n), subR=float(np.mean([d["subopt"] for d in r])) if r else None,
                             subN=float(np.mean([d["subopt"] for d in n])) if n else None))
        raw = {g: dict(n=sum(d["grp"] == g for d in rows), subopt=float(np.mean([d["subopt"] for d in rows if d["grp"] == g])),
                       loss=float(np.mean([d["loss"] for d in rows if d["grp"] == g])), margin=float(np.median([d["margin"] for d in rows if d["grp"] == g]))) for g in "RN"}
        n_err_rep = float(np.mean([d["chose_rep"] for d in rows if d["grp"] == "N" and d["subopt"]])) if any(d["grp"] == "N" and d["subopt"] for d in rows) else None
        out[run] = dict(n=len(rows), raw=raw, matched=m, ci=ci, bins=bins, N_errors_choosing_replica=n_err_rep)
        L += [f"== {run}: {len(rows)} first decisions; unmatched suboptimal share R {100 * raw['R']['subopt']:.0f}% (n={raw['R']['n']}, median margin {raw['R']['margin']:.2f}) "
              f"N {100 * raw['N']['subopt']:.0f}% (n={raw['N']['n']}, median margin {raw['N']['margin']:.2f})",
              f"   matched on margin (overlap weight {m['weight']:.0f}): suboptimal share R {100 * m['Rsubopt']:.0f}%  N {100 * m['Nsubopt']:.0f}%  "
              f"difference {100 * m['subopt']:+.0f} [{100 * ci['subopt'][0]:+.0f}, {100 * ci['subopt'][1]:+.0f}] points; "
              f"first-step loss R {m['Rloss']:.2f}  N {m['Nloss']:.2f}  difference {m['loss']:+.2f} [{ci['loss'][0]:+.2f}, {ci['loss'][1]:+.2f}]",
              "   bins (margin: n_R / n_N, suboptimal R / N): " + "; ".join(
                  f"{b['lo']:g}-{'inf' if b['hi'] > 1e8 else format(b['hi'], 'g')}: {b['nR']}/{b['nN']}, "
                  + (f"{100 * b['subR']:.0f}%" if b['subR'] is not None else "-") + "/" + (f"{100 * b['subN']:.0f}%" if b['subN'] is not None else "-") for b in bins)]
        if n_err_rep is not None: L.append(f"   of the suboptimal first actions in group N, {100 * n_err_rep:.0f}% choose the replica tool")
    json.dump(out, open(P("results", "difficulty_matched.json"), "w"), indent=1)
    txt = "\n".join(L); print(txt); open(P("results", "difficulty_matched.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
