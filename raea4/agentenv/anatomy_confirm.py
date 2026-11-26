"""
Held-out confirmation of the same-JSON executor gap and of the regret attribution (PLAN_anatomy_confirm.md).
No model call: reads the confirmation K2e logs and their offline K2r replays, scores with oracle.score_trajectory and
k2e_anatomy.split (the functions used for the exploratory analysis; both frozen by hash in the plan).

  python anatomy_confirm.py                      # confirmation episodes (index 25..49 of the CORE families)
  python anatomy_confirm.py --selftest           # same code on the ORIGINAL logs (index 0..24); must reproduce k2e_anatomy.json

Criteria (fixed in the plan before any confirmation run):
  C1  executor gap      paired nested 95% CI of K2r - K2e regret is below 0                     (flash, terra, Sonnet)
  C2  attribution       nested 95% CI of the per-episode difference (regret at steps where a
                        replica tool is passed over) - (all other regret) is above 0            (terra, Sonnet)
  C3  use of the source fault families: share of episodes whose first call is a replica tool,
                        model (K2e) minus exact planner on the same JSON (K2r); paired nested
                        95% CI below 0                                                          (terra, Sonnet)
A model confirms if all of its criteria are met. Everything is reported whether or not it is met.
Output: results/anatomy_confirm.json, results/anatomy_confirm.txt (or *_selftest.* with --selftest).
"""
import argparse, json, os
import numpy as np
from env import HERE, CORE, load_episodes, load_truth, srcmap
from oracle import score_trajectory
from nestboot import nested_ci
from k2e_anatomy import split, FAULT

RUNS = {  # key: (label, confirmation K2e log, confirmation K2r replay, original K2e log, original K2r replay, criteria)
    "flash": ("flash (non-thinking)", "confirm_ds_chat_fc_k2e.jsonl", "confirm_ds_chat_fc_k2e_replay.jsonl",
              "ds_chat_fc_k2decomp.jsonl", "ds_chat_fc_k2e_replay.jsonl", ["C1"]),
    "terra": ("gpt-6.1-terra (medium)", "confirm_codex_terra_k2e.jsonl", "confirm_codex_terra_k2e_replay.jsonl",
              "codex_terra_medium_k2decomp.jsonl", "codex_terra_k2e_replay.jsonl", ["C1", "C2", "C3"]),
    "sonnet": ("Sonnet 5.5 (high)", "confirm_claude_s55_k2e.jsonl", "confirm_claude_s55_k2e_replay.jsonl",
               "claude_s55.jsonl", "claude_s55_k2e_replay.jsonl", ["C1", "C2", "C3"]),
}


def load(fn, cond, lo, hi):
    recs = {}
    for line in open(os.path.join(HERE, "logs", fn), encoding="utf-8"):
        r = json.loads(line)
        k = int(r["episode"].rsplit("-", 1)[1])
        if r["condition"] != cond or r["family"] not in CORE or r["status"].startswith("api_error") or not lo <= k < hi:
            continue
        if r.get("sample", 0) != 0: continue
        recs[r["episode"]] = r  # a later successful rerun replaces an earlier row (as in score.py)
    return recs


def first_cache(rec, ep):
    src = srcmap(ep["params"]); calls = [src[x["j"]] for x in rec["log"] if x["type"] == "call"]
    return float(bool(calls) and calls[0] == "c"), float("c" in calls)


def ci(v, ids, scl):
    m, lo, hi = nested_ci(v, ids, scl); return dict(mean=float(np.mean(v)), lo=float(lo), hi=float(hi))


def analyse(key, lo, hi, selftest, eps, truth):
    label, ce, cr, oe, orr, crit = RUNS[key]
    e = load(oe if selftest else ce, "K2e", lo, hi); r = load(orr if selftest else cr, "K2r", lo, hi)
    ids = sorted(e); assert len(ids) == 100 and sorted(r) == ids, (key, len(ids), len(r))
    scl = [eps[i]["scenario"] for i in ids]
    sp = lambda rec, i: split(score_trajectory(eps[i], rec[i]["log"], truth[i]), srcmap(eps[i]["params"]))
    E = [sp(e, i) for i in ids]; R = [sp(r, i) for i in ids]
    out = dict(label=label, n=len(ids), criteria=crit,
               usable=int(sum(bool(e[i]["elicit"]["parse_ok"]) for i in ids)),
               no_submit=int(sum(e[i]["status"] != "submitted" for i in ids)))
    out["K2e"] = ci([x["regret"] for x in E], ids, scl); out["K2r"] = ci([x["regret"] for x in R], ids, scl)
    out["K2r-K2e"] = ci([b["regret"] - a["regret"] for a, b in zip(E, R)], ids, scl)
    for m in ["pass_cache", "pass_cache_other", "pass_cache_stop", "rest", "over", "wrong", "early", "verdict", "calls"]:
        out[m] = ci([x[m] for x in E], ids, scl)
    out["pass-rest"] = ci([x["pass_cache"] - x["rest"] for x in E], ids, scl)
    out["pass_share"] = out["pass_cache"]["mean"] / out["K2e"]["mean"] if out["K2e"]["mean"] > 0 else None
    out["planner_calls"] = float(np.mean([x["calls"] for x in R]))
    for sub, fams in [("FAULT", FAULT), ("TWIN", ["TRANSIENT"])]:
        sel = [k for k, i in enumerate(ids) if eps[i]["family"] in fams]; sid = [ids[k] for k in sel]; ssc = [scl[k] for k in sel]
        fm = [first_cache(e[i], eps[i]) for i in sid]; fp = [first_cache(r[i], eps[i]) for i in sid]
        out["first|" + sub] = dict(n=len(sel), model=ci([a[0] for a in fm], sid, ssc), planner=ci([a[0] for a in fp], sid, ssc),
                                   diff=ci([a[0] - b[0] for a, b in zip(fm, fp)], sid, ssc),
                                   used_model=float(np.mean([a[1] for a in fm])), used_planner=float(np.mean([a[1] for a in fp])),
                                   pass_cache_other=float(np.mean([E[k]["pass_cache_other"] for k in sel])),
                                   regret=float(np.mean([E[k]["regret"] for k in sel])))
    met = dict(C1=out["K2r-K2e"]["hi"] < 0, C2=out["pass-rest"]["lo"] > 0, C3=out["first|FAULT"]["diff"]["hi"] < 0)
    out["met"] = {k: bool(v) for k, v in met.items()}
    out["confirms"] = bool(all(met[c] for c in crit))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--runs", nargs="*", default=list(RUNS)); a = ap.parse_args()
    lo, hi = (0, 25) if a.selftest else (25, 50)
    eps, truth = load_episodes(), load_truth()
    out = {k: analyse(k, lo, hi, a.selftest, eps, truth) for k in a.runs}
    f = lambda d: f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
    L = [f"episodes: index {lo}..{hi - 1} of {', '.join(CORE)} (n=100 per run)" + ("   SELFTEST on the original logs" if a.selftest else "")]
    for k, o in out.items():
        L += ["", f"== {o['label']}   usable JSON {o['usable']}/100, no submit {o['no_submit']}",
              f"  K2e {f(o['K2e'])}   K2r {f(o['K2r'])}",
              f"  C1  K2r-K2e {f(o['K2r-K2e'])}  -> {'met' if o['met']['C1'] else 'NOT met'}",
              f"  C2  pass_cache {f(o['pass_cache'])}  rest {f(o['rest'])}  pass-rest {f(o['pass-rest'])}  share "
              f"{100 * o['pass_share']:.0f}%  -> {'met' if o['met']['C2'] else 'NOT met'}" + ("" if "C2" in o["criteria"] else " (not a criterion for this run)"),
              f"  C3  first call replica, fault families: model {100 * o['first|FAULT']['model']['mean']:.0f}%  planner "
              f"{100 * o['first|FAULT']['planner']['mean']:.0f}%  diff {f(o['first|FAULT']['diff'])}  -> "
              f"{'met' if o['met']['C3'] else 'NOT met'}" + ("" if "C3" in o["criteria"] else " (not a criterion for this run)"),
              f"      twin: model {100 * o['first|TWIN']['model']['mean']:.0f}%  planner {100 * o['first|TWIN']['planner']['mean']:.0f}%;"
              f"  pass_cache_other fault {o['first|FAULT']['pass_cache_other']:.3f}  twin {o['first|TWIN']['pass_cache_other']:.3f}",
              f"  parts: over {o['over']['mean']:.3f}  wrong {o['wrong']['mean']:.3f}  early {o['early']['mean']:.3f}  verdict {o['verdict']['mean']:.3f};"
              f"  calls {o['calls']['mean']:.2f} (planner {o['planner_calls']:.2f})",
              f"  => {'CONFIRMS' if o['confirms'] else 'does NOT confirm'} ({', '.join(o['criteria'])})"]
    txt = "\n".join(L); print(txt)
    tag = "anatomy_confirm_selftest" if a.selftest else "anatomy_confirm"
    json.dump(out, open(os.path.join(HERE, "results", tag + ".json"), "w"), indent=1)
    open(os.path.join(HERE, "results", tag + ".txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
